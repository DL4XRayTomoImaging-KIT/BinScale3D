from .helpers import dasked
from functools import partial
from skimage.transform import rescale
import numpy as np
import dask.array as da


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
