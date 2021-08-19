import numpy as np
import dask.array as da
from skimage.transform import rescale
from functools import partial
import tifffile
from concert.readers import TiffSequenceReader
import argparse
from flexpand import Expander, add_args
import os
from tqdm.auto import tqdm

import yaml
from skimage.exposure import rescale_intensity, adjust_sigmoid
from sklearn.mixture import GaussianMixture
from skimage.morphology import binary_dilation, ball

def file_load(addr, paginated=False):
    if paginated:
        imgobj = tifffile.TiffFile(addr)
        img = np.zeros((len(imgobj.pages), *imgobj.pages[0].shape), dtype=imgobj.pages[0].asarray().dtype)
        for i, page in enumerate(imgobj.pages):
            img[i] = page.asarray()
    else:
        img = tifffile.imread(addr)

    return img

def multifile_load(inp_dir, ext="tiff"):
    seq_reader = TiffSequenceReader(inp_dir, ext=ext)
    pg_0 = seq_reader.read(0)

    img = np.zeros((seq_reader.num_images, *pg_0.shape), dtype=pg_0.dtype)
    for i in range(seq_reader.num_images):
        img[i] = seq_reader.read(i)

    return img


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

    mask = rearrange(is_in_hull, '(l w h) -> l w h', l=mask.shape[0], w=mask.shape[1], h=mask.shape[1])
    return mask

def get_bracket_along_axis(mask, axis):
    is_masked = np.where(mask.sum(tuple([i for i in range(mask.ndim) if i != axis])) > 0)[0]
    return (is_masked[0], is_masked[-1])

multiply_slices = lambda slc, m: [(f*m, t*m) for f,t in slc]

class Cropper:
    def __init__(self, area_percent=5, scaling_coefficient=16, mask=False, dilation=None):
        self._a = 100 - area_percent
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
        dimg = self._downscale(img) #downscale

        # get mask & convert it to convex hull
        mask = dimg > np.percentile(dimg.flatten(), self._a)
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
    def __init__(self, pagination_type="singlepage"):
        self.pagination_type = pagination_type
        self.prefix = ""

    def __call__(self, input_addr, is_paginated):
        if self.pagination_type == "singlepage":
            img = file_load(input_addr, is_paginated)
        elif self.pagination_type == "multipage":
            img = file_load(input_addr, is_paginated)

        return img, self.pagination_type


def get_io_pairs(input_files, output_folder, conversions, force=False):
    if output_folder is None:
        # warn if only transformation is PageLoader and no output dir provided
        if len(conversions) == 1 and isinstance(conversions[0], PageLoader):
            output_files = input_files
            while "the choice is invalid":
                choice = str(input("overwrite? [y/n]: "))
                if choice == 'y':
                    force = True
                    break
                elif choice == 'n':
                    force = False
                    break
        else:
            prefixes = [c.prefix for c in conversions]
            output_files = [os.path.join(os.path.split(a)[0], '_'.join(prefixes)+'_'+os.path.split(a)[1]) for a in input_files]

    elif output_folder.endswith('txt'):
        with open(output_folder) as f:
            output_files = f.read().split('\n')
        if len(output_files) != len(input_files):
            raise ValueError('Inconsistent input and output filenames')

    elif os.path.exists(output_folder) and os.path.isdir(output_folder):
        output_files = [os.path.join(output_folder, os.path.basename(a)) for a in input_files]

    else:
        raise ValueError('No legal output files provided!')

    if force:
        pairs = list(zip(input_files, output_files))
    else:
        pairs = [(i,o) for i,o in zip(input_files, output_files) if not os.path.exists(o)]
    return pairs


def load_n_process(input_folder, output_folder, converters, is_paginated, force):
    input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
    # get output file space
    io_files = get_io_pairs(input_files, output_folder, converters, force)

    # handle loading multifiles
    for c in converters:
        if isinstance(c,PageLoader):
            if c.pagination_type == "multifile":
               img =  multifile_load(input_folder)
               tifffile.imsave(output_folder + '/merged.tiff' , img)

    logs = {"individual_files" : []}
    for input_addr, output_addr in tqdm(io_files):
        log = {'input_addr': input_addr, 'output_addr': output_addr}
        try:
            img = file_load(input_addr, is_paginated)
            for c in converters:
                if isinstance(c,PageLoader): # pass the input address
                    if c.pagination_type != "multifile":
                        img, log_chunk = c(input_addr, is_paginated)
                else:
                    img, log_chunk = c(img)
                log[c.__class__.__name__] = log_chunk
            tifffile.imsave(output_addr, img)
        except Exception as e:
            log['error'] = str(e)
        logs["individual_files"].append(log)
    return logs

def get_conversions(conv_conf):
    conv_dict = {'scale': Scaler, '8bit': Converter, 'crop': Cropper, 'paginate': PageLoader}

    conversions = []
    for config_key, conversion_class in conv_dict.items():
        if (config_key in conv_conf.keys()) and (conv_conf[config_key] is not None): # is configured
            if conv_conf[config_key] == True: # default parameters
                conversions.append(conversion_class())
            else: # passed parameters
                conversions.append(conversion_class(**conv_conf[config_key]))
    if len(conversions) < 1:
        raise ValueError('No conversions cofigured!')
    return conversions


from joblib import Parallel, delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Rescale and/or convert to 8bit 3d TIFF volumes')

    parser.add_argument('--conversion-config', help='YAML configuration file for the conversion operations.')

    input_group = parser.add_argument_group('Input files to be processed with this util')
    add_args(input_group)

    parser.add_argument('--output-files', default=None, help='Files to output the result of processing. If folder is provided will be saved with the same name as input files. If nothing provided will be saved with prefix alongside with input files.')
    parser.add_argument('--force', default=False, const=True, action='store_const', help='If file with the same name found it will be overwrited. By default this file will not be processed.')

    parser.add_argument('--is-paginated', default=False, const=True, action='store_const', help='Use if the saved tiff file is paginated and will not be loaded whole by default.')

    parser.add_argument('--multithread', default=0, type=int, help='Number of threads to process files. By default everything is done in one thread.')

    parser.add_argument('--log-file', default=None, help='Where to store final results log.')

    args = parser.parse_args()

    # find out what to do
    with open(args.conversion_config) as f:
        conv_conf = yaml.safe_load(f)
    conversions = get_conversions(conv_conf)

    # add PageLoader if not configured
    for c in conversions:
        if isinstance(c,PageLoader): break
    else:
        conversions.append(PageLoader())    

    # get input file space
    fle = Expander(verbosity=True, files_only=False)
    input_folders = fle(args=args)

    is_paginated = args.is_paginated

    log = {'conversion_config': conv_conf, 'is_paginated': is_paginated, 'threads': args.multithread}

    # for each file in list load, process and save
    if args.multithread:
        results = Parallel(n_jobs=args.multithread, verbose=20)(delayed(load_n_process)(input_folders, args.output_files, conversions, is_paginated, args.force) for input_folder in input_folders)
    else:
        results = [load_n_process(input_folder, args.output_files, conversions, is_paginated, args.force) for input_folder in tqdm(input_folders)]

    log['individual_folders'] = results

    if args.log_file is not None:
        with open(args.log_file, 'w') as f:
            yaml.safe_dump(log, f)