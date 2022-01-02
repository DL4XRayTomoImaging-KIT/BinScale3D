# BinScale3D
Parallelized tool for batch processing tomography 3D data: scaling, cropping and binning. 
Just as a bonus: unifying formats.

It's more or less basic operations but we tried to make them faster with dask and joblib.
Also we took care about different file formats and path patterns you tend to store you files in, to enable batch processing out of box.
Most of the parameters are stored within YAML configs, so reproducibility is also included.

## Why can I need this?
If you have tons of files in separate folders, want them all to be uniformly processed without scripting conversions yourselve and worrying about naming patterns.
This tool can do perform in range from simpliest operations like saving in 8bit and down to rough background removal.
You need more CV operations to be added to this batch processing tool? Just drop us a line.

## Ok, how van I install it?
Up to now, this is not an installable package. Just clone the repository wherever and install requirements.
Just like that:
```
cd ~
git clone git@github.com:DL4XRayTomoImaging-KIT/BinScale3D.git
cd BinScale3D
pip install -r requirements.txt
```

Now you should be good to go.

## Ok, now I want a simple test, that it's alive

Luckily we have minimal example included.
It will just crop files, using entropy of pixel as guidance to find the sample.
New file will be saved alongside with the original one, with prefix "cropped_".
If you are on Imaging Group servers, you should be able to run it like this:
```
nice -n 5 python bin_scale.py --conversion-config=crop_only.yaml --data-config=xeno_9.yaml
```
otherwise, you will need to go through the configuration files and either alter `xeno_9.yaml` by your own intuition, or follow our docs regarding configurations.

## Ok, so how do I configure my job more precisely?

You need two config files: one for operations you want to make and one for the files you want to operate with. 
There will be also 2 extra-parameters which you could add from comand line only.
If you never had encountered YAML files earlier, you can basically read about them here: https://en.wikipedia.org/wiki/YAML or more example-driven here: https://keleshev.com/yaml-quick-introduction

### Operations config

Up to now we have implemented three operations: 
- binning to 8bit with contrast adjustment included
- cropping with automatically found threshold either by pixel value, or by neighbourhood entropy.
- scaling, mainly by factor of 2, and mainly concentrated on downscaling. Which, however, covers most of the use cases.

Each of them has it's own name and config parameters, which we will cover now in short. 
You can combine all of them and even several times, while the order in which they are included into config will define the order of execution.

#### Cropping
Name of the operation in config: 
