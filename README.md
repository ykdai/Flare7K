## Flare7K: A Phenomenological Nighttime Flare Removal Dataset

[Paper](https://openreview.net/pdf?id=Proso5bUa) | [Project Page](https://nukaliad.github.io/projects/Flare7K) | [Video](https://youtu.be/CR3VFj4NOQM)


[Yuekun Dai](https://www.linkedin.com/in/%E6%9C%88%E5%9D%A4-%E6%88%B4-19b33421a/), [Chongyi Li](https://li-chongyi.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Ruicheng Feng](https://jnjaby.github.io/),  [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

### Flare7K
Flare7K, the first nighttime flare removal dataset, which is generated based on the observation and statistic of real-world nighttime lens flares. It offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares. The 7,000 flare patterns can be randomly added to the flare-free images, forming the flare-corrupted and flare-free image pairs.

<img src="assets/flare7k.png" width="800px"/>

#### Data download:
(all data is hosted on Google Drive)
|     | Link | Number | Description
| :--- | :--: | :----: | :---- | 
| Flares | [link](https://drive.google.com/file/d/1PPXWxn7gYvqwHX301SuWmjI7IUUtqxab/view) | 7,000 | We offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares.
| Background Images| [link](https://drive.google.com/file/d/1GNFGWfUbgXfELx5fZtjTjU2qqWnEa-Lr/view) | 24,000 | The background images are sampled from [[Single Image Reflection Removal with Perceptual Losses, Zhang et al., CVPR 2018]](https://people.eecs.berkeley.edu/~cecilia77/project-pages/reflection.html). We filter our most of the flare-corrupted images and overexposed images.

### Paired Data Generation

We provide a on-the-fly dataloader function and a flare-corrupted/flare-free pairs generation script in this repository. To use this function, please put the Flare7K dataset and 24K Flickr dataset on the same path with the `generate_flare.ipynb` file.

If you only want to generate the flare-corrupted image without reflective flare, you can comment out the following line:
```
# flare_image_loader.load_reflective_flare('Flare7K','Flare7k/Reflective_Flare')
```


### Code and Model

The code and pretrained models will be released soon. Please stay tuned! :hugs:

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This dataset and code are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


### Contact
If you have any question, please feel free to reach me out at `ydai005@e.ntu.edu.sg`.