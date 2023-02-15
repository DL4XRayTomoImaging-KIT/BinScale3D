#!/usr/bin/env python3

from univread import read as imread
import argparse
import tifffile
from flexpand import Expander, Matcher
import os
from tqdm.auto import tqdm
import warnings

from binscale import Converter, Cropper, Scaler

import yaml
from joblib import Parallel, delayed



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
    conv_dict = {'scale': Scaler, '8bit': Converter, 'rescale': Converter, 'crop': Cropper, 'paginate': PageLoader}

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
