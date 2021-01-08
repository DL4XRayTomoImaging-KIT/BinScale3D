import numpy as np
import dask.array as da
from skimage.transform import rescale
from functools import partial
import tifffile
import argparse
from FileListExpander import Expander
import os
from tqdm.auto import tqdm

def file_load(addr, paginated=False):
    if paginated:
        imgobj = tifffile.TiffFile(addr)
        img = np.zeros((len(imgobj.pages), *imgobj.pages[0].shape))
        for i, page in enumerate(imgobj.pages):
            img[i] = page.asarray()
    else:
        img = tifffile.imread(addr)

    return img


def get_random_voxels(img, count):
    return img[tuple([np.random.randint(0, img.shape[i], count) for i in range(img.ndim)])]


def get_f_t(img, f=1, t=99.95):
    ti = get_random_voxels(img, 10_000_000)
    return np.percentile(ti, 1), np.percentile(ti, 99.95)


def dask_rescale(img, scale):
    scaler = partial(rescale, scale=scale, preserve_range=True)
    img = da.overlap.overlap(img,
                             depth={0:int(1/scale), 1:int(1/scale), 2:int(1/scale)},
                             boundary={0:'nearest', 1:'nearest', 2:'nearest'})
    img = img.map_blocks(scaler, dtype=img.dtype)
    img = da.overlap.trim_internal(img, {0:1, 1:1, 2:1})
    return img


def convert_8_bit(img, f, t):
    img = (img - f) / (t - f) * 255
    clipper = partial(np.clip, a_min=0, a_max=255)
    img = img.map_blocks(clipper, dtype=img.dtype)
    return img.astype(np.uint8)


def convert_scale(img, convert_to_8bit=True, scale=None, chunk_size=256):
    if convert_to_8bit:
        f, t = get_f_t(img)
    img = da.from_array(img, chunks=(chunk_size, chunk_size, chunk_size))
    if convert_to_8bit:
        img = convert_8_bit(img, f, t)

    if scale is not None:
        img = dask_rescale(img, scale)
    return np.array(img, dtype=img.dtype)


def get_output_file_space(input_files, output_folder):
    if output_folder.endswith('txt'):
        with open(output_folder) as f:
            output_files = f.read().split('\n')
        if len(output_files) != len(input_files):
            raise ValueError('Inconsistent input and output filenames')

    elif os.path.exists(output_folder) and os.path.isdir(output_folder):
        output_files = [os.path.join(output_folder, os.path.basename(a)) for a in input_files]

    elif output_folder is None:
        prefix = ''
        if to_8bit:
            prefix += '8bit_'
        if rescale is not None:
            prefix += f'rescaled_{rescale}'
        output_files = [os.path.join(os.path.split(a)[0], prefix+os.path.split(a)[1]) for a in input_files]

    else:
        raise ValueError('No legal output files provided!')

    return output_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Rescale and/or convert to 8bit 3d TIFF volumes')

    parser.add_argument('-convert-to-8bit', const=True, default=False,  action='store_const', help='Will convert tiff to the 8bit.')
    parser.add_argument('-rescale', default=None, type=float, help='Rescale result, should provide rescaling coefficient. Proper work guaranteed if (1/n is int) or (n is int). This is because of tiling paralellisation.')

    parser.add_argument('--input-files', help='Files to process. As list, directory, glob or file containing addresses')
    parser.add_argument('--regexp', default=None, help='RegExp to filter files from --input-files.')
    parser.add_argument('--regexp-mode', default='includes', help='Mode of RegExp interpretation. Possible ones are [includes, matches, not_includes, not_matches]. Default is includes.')

    parser.add_argument('--output-files', default=None, help='Files to output the result of processing. If folder is provided will be saved with the same name as input files. If nothing provided will be saved with prefix alongside with input files.')

    parser.add_argument('--is-paginated', default=False, const=True, action='store_const', help='Use if the saved tiff file is paginated and will not be loaded whole by default.')


    args = parser.parse_args()

    # find out what to do
    to_8bit = args.convert_to_8bit
    to_rescale = args.rescale
    if (not to_8bit) and (to_rescale is None):
        raise ValueError('No meaningful actions required! Use cp if you need to copy files, please.')

    # get input file space
    fle = Expander()
    input_files = fle(args.input_files, args.regexp, args.regexp_mode)

    # get output file space
    output_files = get_output_file_space(input_files, args.output_files)

    # for each file in list load, process and save
    for input_addr, output_addr in tqdm(zip(input_files, output_files), total=len(input_files)):
        img = file_load(input_addr, paginated=args.is_paginated)
        img = convert_scale(img, convert_to_8bit=to_8bit, scale=to_rescale)
        tifffile.imsave(output_addr, img)
