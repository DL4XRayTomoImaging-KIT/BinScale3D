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
```bash
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
```bash
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

Important parts of the cropping are:

1. preprocessing, e.g. take entropy of the image
2. downscaling image by a large number
3. defining thresholds of the pixel values related to sample
4. morphological operations to remove false-positive pixels

<details>
  <summary>Read more detailed algorithm</summary>
  
  Cropper now has the following algorithm:
  
  1. if preprocessing function defined, image is preprocessed. Now the only available function is entropy.
  2. image is downscaled a lot (now x16) to speed-up all computations and filter out small noise.
  3. sample localisation function is applied, either selecting all values brighter than threshold, or by clustering all available values.
  4. largest connected area selected (supposed sample).
  5. pixels related to the sample are wrapped in a convex hull, now all pixels within this hull are denoted as pixels containing the sample
  6. binary dilation is applied to the sample pixels, therefore adding safety margin around the sample.
  7. mask is upscaled back to the original image size.
  8. bounding box is defined to fit the sample tightly and the image is cropped.
  9. if configured, all pixels not marked as sample will be reassigned to zero.
</details>

Let go through the reference for all parameters of configuration:

- `scaling_coefficient`: int [16], defines how much image is downscaled before finding embryo. Default value is 16.
- `mask`: bool [False], if set to True, will not only crop, but try to roughly remove background on the cropped image as well.
- `dilation`: int [0], if more than 0, will add expand found sample area by `scaling_coefficient * dilation` by binary dilation operation.
- `sample_localisation_function`: str ["select_sample_threshold"], defines the method of sample localisation. Could be either "select_sample_threshold" or "select_sample_kmeans".  
   "select_sample_threshold" defines that sample as simple as top k% of brightest pixels.  
   "select_sample_kmeans" acts a bit more complicated, it clusterizes all pixel values to several m clusters and selects j-th cluster from top bright as a sample.
- `sample_localisation_kwargs`: dict, hyper-parameters for the `sample_localisation_function`.  
   For "select_sample_threshold":
   - `area_percent`: int [5], how much percents of area to select as sample (from brightest to darkest).
   
   For "select_sample_kmeans":
   - `ignore_zeroes` bool [True], if set to True function ignores values exactly equal to 0. Very useful when operating with already rotated or padded data.
   - `n_clusters` int [2], number of clusters to use to localize sample. 2 is for background-vs-sample, but sometimes, it's worthy to add more, e.g. for sample holder.
   - `sample_cluster` int [-1], id of cluster in sorted (from darkest to brightest) array of clusters (python slicing notation). Typically, sample is the brightest, but it could differ.
 - `preprocessing_function` str, function to pre-process image for cropping. Could be either ignored or set to "parallel_entropy". **NB!** saved volume will be anyway original one, pre-processing is only for cropping.

Example of minmal working configuration:
```yaml
crop: True
```
this will just crop volume with all default parameters.

Example of simple configuration:
```yaml
crop:
  sample_localisation_function: "select_sample_kmeans"
  preprocessing_function: "parallel_entropy"
  dilation: 2
```
this will first take local entropy of the image, then localise sample as the brightest part of the image (after defining thresholds of the 2 clusters) and then add safety margin of 32 pixels.

Example of quite expanded configuration:
```yaml
crop:
  sample_localisation_function: "select_sample_kmeans"
  sample_localisation_kwargs:
    n_clusters: 3
    sample_cluster: -2
  scaling_coefficient: 8
  dilation: 2
  mask: True
```
this will define thresholds of the pixel values by brightness and with clustering to 3 clusters. For sample will be taken the second brightest area. Downscaling will be only x8 instead of x16, therefore results probably will be more precise, but will require more computation. The safety margin of 16 pixels will be added around sample. After cropping, all areas which are not selected as sample will be turned to 0.
