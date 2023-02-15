from .helpers import dasked, select_top_k_connected_areas, get_random_voxels
import numpy as np
from functools import partial
import dask.array as da
from skimage.transform import rescale

from scipy.spatial import ConvexHull, Delaunay
from einops import rearrange

from skimage.morphology import binary_dilation, ball, disk

from sklearn.cluster import KMeans

from skimage.exposure import rescale_intensity
from skimage.filters.rank import entropy
from copy import deepcopy

from joblib import Parallel, delayed

def get_bracket_along_axis(mask, axis):
    is_masked = np.where(mask.sum(tuple([i for i in range(mask.ndim) if i != axis])) > 0)[0]
    return (is_masked[0], is_masked[-1])

multiply_slices = lambda slc, m: [(f*m, t*m) for f,t in slc]

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

def entropy_and_brightness(img):
    ent = parallel_entropy(img)
    return img*ent



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
