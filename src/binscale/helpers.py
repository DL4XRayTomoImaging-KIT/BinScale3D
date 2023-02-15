import dask.array as da
import numpy as np

from skimage.measure import label

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


def select_top_k_connected_areas(markup, k):
    connected_regions = label(markup)
    region_id, region_size = np.unique(connected_regions, return_counts=True)
    regions_order = (np.argsort(region_size[1:]) + 1)[::-1] # ordering without zero
    return np.isin(connected_regions, regions_order[:k])

def get_random_voxels(img, count):
    return img[tuple([np.random.randint(0, img.shape[i], count) for i in range(img.ndim)])]
