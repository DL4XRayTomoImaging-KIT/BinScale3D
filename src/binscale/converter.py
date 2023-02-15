import numpy as np
from .helpers import dasked, select_top_k_connected_areas, get_random_voxels
from functools import partial
from skimage.exposure import rescale_intensity, adjust_sigmoid

def get_random_outter_voxels(img, count):
    return img[tuple([(np.random.beta(0.4, 0.4, count)*(img.shape[i]-1)).astype(int) for i in range(img.ndim)])]

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

class Converter:
    def __init__(self, from_percentile=1, to_percentile=99.95, apply_sigmoid=False, disjoint_distributions=False, to_format='uint8', autoscale=True):
        self._f = from_percentile
        self._t = to_percentile
        self._s = apply_sigmoid
        self._dis = disjoint_distributions
        self.to_format = to_format
        self.prefix = self.to_format
        self.autoscale = autoscale
        
        if np.dtype(self.to_format).name.startswith('float'):
            self.scale = (-1, 1)
        else:
            self.scale = (np.iinfo(np.dtype(self.to_format)).min, np.iinfo(np.dtype(self.to_format)).max)
            
        if self._s:
            self.prefix += '_sigmoid'
            self.autoscale = True
        if self._dis:
            self.prefix += '_disjoint'

    @dasked
    def _scale(self, img, rs, ors):
        if self._dis:
            f, t = get_disjoint_thresholds(rs, ors, self._f, self._t)
        else:
            f = np.percentile(rs, self._f)
            t = np.percentile(rs, self._t)

        if self.autoscale:
            if self._s:
                scaler = partial(rescale_intensity, in_range=(f,t), out_range=(0,1))
                rescaler = partial(rescale_intensity, in_range=(0, 1), out_range=self.scale)
            else:
                scaler = partial(rescale_intensity, in_range=(f,t), out_range=self.scale)
        else:
            scaler = partial(np.clip, a_min=f, a_max=t)
                
        img = img.map_blocks(scaler, dtype=img.dtype)

        if self._s:
            scaled_sample = scaler(rs)
            _m = np.median(scaled_sample)
            corrector = partial(adjust_sigmoid, cutoff=_m)
            img = img.map_blocks(corrector, dtype=img.dtype)
            img = rescaler(img)

        return img.astype(np.dtype(self.to_format)), (f, t)

    def __call__(self, img):
        if self._dis:
            rs = get_random_voxels(img, 10_000_000)
            ors = get_random_outter_voxels(img, 10_000_000)
        else:
            rs = get_random_voxels(img, 10_000_000)
            ors = None

        img, p = self._scale(img, rs, ors)
        return img, str(p)
