## Flare7K: A Phenomenological Nighttime Flare Removal Dataset

**Yuekun Dai, Chongyi Li, Shangchen Zhou, Ruicheng Feng, Chen Change Loy**

**homepage:** [link](https://nukaliad.github.io/projects/Flare7K)

**paper:** [link](https://openreview.net/forum?id=Proso5bUa&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FTrack%2FDatasets_and_Benchmarks%2FAuthors%23your-submissions))

### Abstract

> Artificial lights commonly leave strong lens flare artifacts on the images when one captures images at night. Nighttime flare not only affects the visual quality but also degrades the performance of vision algorithms. Different from sunshine flare, the nighttime flare has its unique luminance and spectrum of artificial lights and diverse patterns. Models trained on existing sunshine flare removal datasets, however, cannot cope with nighttime flare. Existing flare removal methods mainly focus on the removal of daytime flares while they fail in removing nighttime flares. Nighttime flare removal is challenging because of the unique luminance and spectrum of artificial lights and the diverse patterns and image degradation of the flares captured at night. The scarcity of the nighttime flare removal dataset limits the research on this paramount task. In this paper, we introduce, Flare7K,  the first nighttime flare removal dataset, which is generated based on the observation and statistic of real-world nighttime lens flares. It offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares. The 7,000 flare patterns can be randomly added to the flare-free images, forming the flare-corrupted and flare-free image pairs. With the paired data, deep models can effectively restore the flare-corrupted images taken in real world. Apart from sufficient flare patterns, we also provide rich annotations, including the light source, glare with shimmer, reflective flare, and streak, which are frequently absent from existing datasets. Thus, our dataset can facilitate new work in nighttime flare removal and more. Extensive experiments demonstrate that our dataset can complement the diversity of existing flare datasets and push the frontier of nighttime flare removal. 

### DataLoader

We provide a dataloader function and a flare-corrupted/flare-free pairs generation script in this repository. To use this function, please put the Flare7K dataset and 24K Flickr dataset on the same path with the `generate_flare.ipynb` file.

If you only want to generate the flare-corrupted image without reflective flare, you can comment out the following line:

`#flare_image_loader.load_reflective_flare('Flare7K','Flare7k/Reflective_Flare')`

### Dataset

Our Flare7K dataset is under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license and can be downloaded from:

https://drive.google.com/file/d/1PPXWxn7gYvqwHX301SuWmjI7IUUtqxab/view

The 24K flare-free Flickr images are sampled from [*Single Image Reflection Removal with Perceptual Losses*](https://people.eecs.berkeley.edu/~cecilia77/project-pages/reflection.html) (Zhang et al., CVPR 2018). We filter our most of the flare-corrupted images and overexposed images. You can download our training dataset at:

https://drive.google.com/file/d/1GNFGWfUbgXfELx5fZtjTjU2qqWnEa-Lr/view

### Code and Pre-trained model

Training code and test code with pretrained models will be released soon.