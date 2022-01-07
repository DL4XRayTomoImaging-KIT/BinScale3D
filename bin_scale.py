import numpy as np
import dask.array as da
from skimage.transform import rescale
from functools import partial
from univread import read as imread
import argparse
import tifffile
from flexpand import Expander, Matcher
import os
from tqdm.auto import tqdm
import warnings

from sklearn.cluster import KMeans

import yaml
from skimage.exposure import rescale_intensity, adjust_sigmoid
from skimage.morphology import binary_dilation, ball, disk
from skimage.filters.rank import entropy

from joblib import Parallel, delayed
from copy import deepcopy


def get_random_voxels(img, count):
    return img[tuple([np.random.randint(0, img.shape[i], count) for i in range(img.ndim)])]

def get_random_outter_voxels(img, count):
    return img[tuple([(np.random.beta(0.4, 0.4, count)*(img.shape[i]-1)).astype(int) for i in range(img.ndim)])]

from skimage.measure import label

def select_top_k_connected_areas(markup, k):
    connected_regions = label(markup)
    region_id, region_size = np.unique(connected_regions, return_counts=True)
    regions_order = (np.argsort(region_size[1:]) + 1)[::-1] # ordering without zero
    return np.isin(connected_regions, regions_order[:k])

def get_disjoint_thresholds(rs, ors, f=1, t=99.95):
    inbox = (np.percentile(np.concatenate([rs, ors]), f), np.percentile(np.concatenate([rs, ors]), t))
    bins = np.linspace(*inbox, num=100)
    digitized_rs = np.bincount(np.digitize(rs, bins=bins))
    digitized_ors = np.bincount(np.digitize(ors, bins=bins))

    comp = 2*(digitized_rs-digitized_ors)/(digitized_ors+digitized_rs+1) > 1
    is_different = np.where(select_top_k_connected_areas(comp, 1))[0]

    _f = is_different[0]
    _t = is_different[-1]

    if _t == 100:
        _t = 99

    _f = bins[_f]
    _t = bins[_t]

    return (_f, _t)

CHUNK_SIZE = 256
from functools import wraps

def dasked(f):
    @wraps(f)
    def wrap(self, img, *args, **kwargs):
        img = da.from_array(img, chunks=CHUNK_SIZE)
        outps = f(self, img, *args, **kwargs)
        if isinstance(outps, (list, tuple)):
            outps = list(outps)
            outps[0] = np.array(outps[0], dtype=outps[0].dtype)
        else:
            outps = np.array(outps, dtype=outps.dtype)
        return outps
    return wrap


class Scaler:
    def __init__(self, scale):
        self.scale = scale
        self.prefix = f'scaled_{self.scale}'

    @dasked
    def __call__(self, img):
        dd = img.dtype.type
        if issubclass(dd, np.integer):
            img = img.astype(np.float32)
        scaler = partial(rescale, scale=self.scale, preserve_range=True)
        img = da.overlap.overlap(img,
                                depth={0:int(1/self.scale), 1:int(1/self.scale), 2:int(1/self.scale)},
                                boundary={0:'nearest', 1:'nearest', 2:'nearest'})
        img = img.map_blocks(scaler, dtype=img.dtype)
        img = da.overlap.trim_internal(img, {0:1, 1:1, 2:1})
        if issubclass(dd, np.integer):
            img = img.map_blocks(np.rint, dtype=img.dtype)
            img = img.astype(dd)
        return img, None


class Converter:
    def __init__(self, from_percentile=1, to_percentile=99.95, apply_sigmoid=False, disjoint_distributions=False):
        self._f = from_percentile
        self._t = to_percentile
        self._s = apply_sigmoid
        self._dis = disjoint_distributions
        self.prefix = '8bit'
        if self._s:
            self.prefix += '_sigmoid'
        if self._dis:
            self.prefix += '_disjoint'

    @dasked
    def _scale(self, img, rs, ors):
        if self._dis:
            f, t = get_disjoint_thresholds(rs, ors, self._f, self._t)
        else:
            f = np.percentile(rs, self._f)
            t = np.percentile(rs, self._t)

        if self._s:
            scaler = partial(rescale_intensity, in_range=(f,t), out_range=(0,1))
        else:
            scaler = partial(rescale_intensity, in_range=(f,t), out_range=(0,255))
        img = img.map_blocks(scaler, dtype=img.dtype)

        if self._s:
            scaled_sample = scaler(rs)
            _m = np.median(scaled_sample)
            corrector = partial(adjust_sigmoid, cutoff=_m)
            img = img.map_blocks(corrector, dtype=img.dtype)
            img = img * 255

        return img.astype(np.uint8), (f, t)

    def __call__(self, img):
        if self._dis:
            rs = get_random_voxels(img, 10_000_000)
            ors = get_random_outter_voxels(img, 10_000_000)
        else:
            rs = get_random_voxels(img, 10_000_000)
            ors = None

        img, p = self._scale(img, rs, ors)
        return img, str(p)

from scipy.spatial import ConvexHull, Delaunay
from einops import rearrange

def fill_convex(mask):
    mask = select_top_k_connected_areas(mask, 1)

    mp = np.moveaxis(np.stack(np.where(mask)), 0, 1) # marked points
    ap = rearrange(np.indices(mask.shape), 'c l w h -> (l w h) c') # all volume points

    hull = ConvexHull(mp)
    hp = hull.points[hull.vertices] # hull vertice coordinates
    hull = Delaunay(hp) # new hull version
    is_in_hull = hull.find_simplex(ap) > 0 # checking for each point if it lies on 3d simplex

    mask = rearrange(is_in_hull, '(l w h) -> l w h', l=mask.shape[0], w=mask.shape[1], h=mask.shape[2])
    return mask

def get_bracket_along_axis(mask, axis):
    is_masked = np.where(mask.sum(tuple([i for i in range(mask.ndim) if i != axis])) > 0)[0]
    return (is_masked[0], is_masked[-1])

multiply_slices = lambda slc, m: [(f*m, t*m) for f,t in slc]

def select_sample_kmeans(vol, ignore_zeroes=True, n_clusters=2, sample_cluster=-1):
    vol_flat = rearrange(vol, 'h w d -> (h w d) 1')
    
    if ignore_zeroes:
        vol_to_train = vol_flat[vol_flat != 0].reshape(-1, 1)
    else:
        vol_to_train = vol_flat
    
    model = KMeans(n_clusters=n_clusters)
    model.fit(vol_to_train)
    sample_class = np.argsort(model.cluster_centers_.flatten())[sample_cluster]
    
    mask = (model.predict(vol_flat) == sample_class)
    if ignore_zeroes:
        mask[vol_flat[:, 0] == 0] = False
        
    mask = rearrange(mask, '(h w d) -> h w d', h=vol.shape[0], w=vol.shape[1], d=vol.shape[2])
    return mask

def select_sample_threshold(vol, area_percent=5):
    _a = 100 - area_percent
    mask = vol > np.percentile(vol.flatten(), _a)
    return mask

def parallel_entropy(img):
    rs = get_random_voxels(img, 10_000_000)
    f = np.percentile(rs, 0.05)
    t = np.percentile(rs, 99.95)
    img = da.from_array(img, chunks=256)
    scaler = partial(rescale_intensity, in_range=(f,t), out_range=(0,255))
    img = img.map_blocks(scaler, dtype=img.dtype)
    img = np.array(img, dtype=np.uint8)
    
    entroper = lambda x: entropy(deepcopy(x), disk(2))
    img = Parallel(n_jobs=32)(delayed(entroper)(i) for i in list(img))
    return np.stack(img)

class Cropper:
    def __init__(self, scaling_coefficient=16, mask=False, dilation=None, 
                 sample_localisation_function='select_sample_threshold', sample_localisation_kwargs=None, 
                 preprocessing_function=None):
        
        self.sample_localisation_kwargs = sample_localisation_kwargs or {}
        self.sample_localisation_function = globals()[sample_localisation_function]
        
        self.preprocessing_function = globals()[preprocessing_function] if preprocessing_function is not None else None
        
        self._s = scaling_coefficient
        self._m = mask
        self._d = dilation
        self.prefix = 'masked' if mask else 'cropped'

    @dasked
    def _downscale(self, img):
        dd = img.dtype.type
        if issubclass(dd, np.integer):
            img = img.astype(np.float32)


        scaler = partial(rescale, scale=1/self._s, preserve_range=True)
        img = da.overlap.overlap(img,
                                depth={0:int(self._s), 1:int(self._s), 2:int(self._s)},
                                boundary={0:'nearest', 1:'nearest', 2:'nearest'})
        img = img.map_blocks(scaler, dtype=img.dtype)
        img = da.overlap.trim_internal(img, {0:1, 1:1, 2:1})
        if issubclass(dd, np.integer):
            img = img.map_blocks(np.rint, dtype=img.dtype)
            img = img.astype(dd)

        return img

    @dasked
    def _upscale(self, img):
        dd = img.dtype.type
        if issubclass(dd, np.integer):
            img = img.astype(np.float32)
        scaler = partial(rescale, scale=self._s, preserve_range=True)
        img = img.map_blocks(scaler, dtype=img.dtype)
        if issubclass(dd, np.integer):
            img = img.map_blocks(np.rint, dtype=img.dtype)
            img = img.astype(dd)
        return img

    def __call__(self, img):
        img = img[tuple([slice(0, (i // self._s) * self._s) for i in img.shape])] # those will be cropped out anyways

        if self.preprocessing_function is not None:
            pimg = self.preprocessing_function(img)
        else:
            pimg = img

        dimg = self._downscale(pimg) #downscale

        # get mask & convert it to convex hull
        mask = self.sample_localisation_function(dimg, **self.sample_localisation_kwargs)
        mask = fill_convex(mask)
        if self._d is not None:
            mask = binary_dilation(mask, selem=ball(self._d))

        # crop regions of both mask and image
        mask_bb = [get_bracket_along_axis(mask, i) for i in range(mask.ndim)]
        mask = mask[tuple([slice(*b) for b in mask_bb])]
        img_bb = multiply_slices(mask_bb, self._s)
        img = img[tuple([slice(*b) for b in img_bb])]

        # if masking required upscale and multiply else return cropped image
        if self._m:
            mask = self._upscale(mask)
            img *= mask

        return img, {'bounding_box': str(img_bb)}


class PageLoader:
    def __init__(self):
        self.prefix = ""

    def __call__(self, input_addr):
        img = imread(input_addr)
        return img, {'shape': img.shape}


def get_filename(addr, depth=-1):
    file_name, file_ext = os.path.splitext(os.path.basename(addr))
    if depth == -1:
        return file_name + file_ext
    else:
        depth_counter = -1 * depth - 1
        cur_addr = addr
        for i in range(depth_counter):
            cur_addr = os.path.dirname(cur_addr)
        return os.path.basename(cur_addr) + file_ext



def load_n_process(input_addr, output_addr, converters):
    log = {'input_addr': input_addr, 'output_addr': output_addr}
    try:
        img = input_addr
        for c in converters:
            img, log_chunk = c(img)
            log[c.__class__.__name__] = log_chunk
        tifffile.imsave(output_addr, img)
    except Exception as e:
        log['error'] = input_addr+' err'+str(e)
        return log
    return log

def get_conversions(conv_conf):
    conv_dict = {'scale': Scaler, '8bit': Converter, 'crop': Cropper, 'paginate': PageLoader}

    conversions = []
    if conv_conf:
        for config_key, conversion_class in conv_dict.items():
            if (config_key in conv_conf.keys()) and (conv_conf[config_key] is not None): # is configured
                if conv_conf[config_key] == True: # default parameters
                    conversions.append(conversion_class())
                else: # passed parameters
                    conversions.append(conversion_class(**conv_conf[config_key]))
    else:
        warnings.warn("the only conversion defined is loading")

    # add PageLoader if not configured
    loader = next((x for x in conversions if isinstance(x,PageLoader)), None)
    if loader is None:
        conversions.insert(0,PageLoader())
    else:
        indx = conversions.index(loader)      
        if indx != 0:
            conversions[0], conversions[indx] = loader, conversions[0]

    return conversions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Rescale and/or convert to 8bit 3d TIFF volumes')

    parser.add_argument('--conversion-config', help='YAML configuration file for the conversion operations.')
    parser.add_argument('--data-config', help='YAML configuration file for the dataset and results saving.')

    parser.add_argument('--force', default=False, const=True, action='store_const', help='If file with the same name found it will be overwrited. By default this file will not be processed.')
    parser.add_argument('--multithread', default=0, type=int, help='Number of threads to process files. By default everything is done in one thread.')

    args = parser.parse_args()

    # find out what to do
    with open(args.conversion_config) as f:
        conv_conf = yaml.safe_load(f)
    conversions = get_conversions(conv_conf) 


    # load data config
    with open(args.data_config) as f:
         data_conf = yaml.safe_load(f)

    # get input file space
    fle = Expander(files_only=False)
    input_files = fle(**data_conf['input'])

    # get output file space
    mtch = Matcher()
    matcher_config = {'prefix': '_'.join([c.prefix for c in conversions if c.prefix])}
    matcher_config_addendum = data_conf['output'] if ('output' in data_conf.keys()) else {}
    matcher_config.update(matcher_config_addendum)
    io_files = mtch(input_files, **matcher_config)

    log = {'conversion_config': conv_conf, 'threads': args.multithread}

    # for each file in list load, process and save
    if args.multithread:
        results = Parallel(n_jobs=args.multithread, verbose=20)(delayed(load_n_process)(i, o, conversions) for i,o in io_files)
    else:
        results = [load_n_process(i, o, conversions) for i,o in tqdm(io_files)]

    log['individual_files'] = results

    if ('log' in data_conf.keys()) and (data_conf['log'] is not None):
        with open(data_conf['log'], 'w') as f:
            yaml.safe_dump(log, f)
    else:
        print(log)
