## Flare7K: A Phenomenological Nighttime Flare Removal Dataset (NeurIPS 2022)

[Paper](https://openreview.net/pdf?id=Proso5bUa) | [Project Page](https://nukaliad.github.io/projects/Flare7K) | [Video](https://youtu.be/CR3VFj4NOQM)


[Yuekun Dai](https://www.linkedin.com/in/%E6%9C%88%E5%9D%A4-%E6%88%B4-19b33421a/), [Chongyi Li](https://li-chongyi.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Ruicheng Feng](https://jnjaby.github.io/),  [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

### Flare7K
Flare7K, the first nighttime flare removal dataset, which is generated based on the observation and statistic of real-world nighttime lens flares. It offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares. The 7,000 flare patterns can be randomly added to the flare-free images, forming the flare-corrupted and flare-free image pairs.

<img src="assets/flare7k.png" width="800px"/>

### Update

- **2022.10.11**: Upload the dataset and pretrained model in Baidu Netdisk.
- **2022.10.9**: Update baseline inference code for flare removal.
- **2022.09.16**: Our paper *Flare7K: A Phenomenological Nighttime Flare Removal Dataset* is accepted by the NeurIPS 2022 Track Datasets and Benchmarks. 🤗
- **2022.08.27**: Update dataloader for our dataset.
- **2022.08.25**: Increase the number of test images from 20 to 100. Please download the latest version of our Flare7K dataset.
- **2022.08.19**: This repo is created.

### Data Download

|     | Baidu Netdisk | Google Drive | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| Flares | [link](https://pan.baidu.com/s/1fiuOcygs0bfhWBvYndkVtw?pwd=x420) | [link](https://drive.google.com/file/d/1PPXWxn7gYvqwHX301SuWmjI7IUUtqxab/view) | 7,000 | We offers 5,000 scattering flare images and 2,000 reflective flare images, consisting of 25 types of scattering flares and 10 types of reflective flares.|
| Background Images| [link](https://pan.baidu.com/s/1BYPRCNSsVmn4VvuU4y4C-Q?pwd=zoyv) | [link](https://drive.google.com/file/d/1GNFGWfUbgXfELx5fZtjTjU2qqWnEa-Lr/view) | 23,949 | The background images are sampled from [[Single Image Reflection Removal with Perceptual Losses, Zhang et al., CVPR 2018]](https://people.eecs.berkeley.edu/~cecilia77/project-pages/reflection.html). We filter our most of the flare-corrupted images and overexposed images.|
| Flare-corrupted images |  |  | 645 | We offer an extra flare-corrupted dataset without ground truth. It contains |

### Paired Data Generation

We provide a on-the-fly dataloader function and a flare-corrupted/flare-free pairs generation script in this repository. To use this function, please put the Flare7K dataset and 24K Flickr dataset on the same path with the `generate_flare.ipynb` file.

If you only want to generate the flare-corrupted image without reflective flare, you can comment out the following line:
```
# flare_image_loader.load_reflective_flare('Flare7K','Flare7k/Reflective_Flare')
```


### Pretrained Model

The inference code based on Uformer is released Now. Your can download the pretrained checkpoints on [[GoogleDrive](https://drive.google.com/file/d/1uFzIBNxfq-82GTBQZ_5EE9jgDh79HVLy/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1EJSYIbbQe5SZYiNIcvrmNQ?pwd=xui4 )]. Please place it under the `experiments` folder and unzip it, then you can run the `deflare.ipynb` for inference. We provide two models, the model in the folder `uformer` can help remove both the reflective flares and scattering flares. The `uformer_noreflection` one can only help remove the scattering flares but is more robust. 

### TODO

- [ ] Add training code and config files.
- [ ] Add a Colab version of our flare removal baseline model.
- [ ] Add a test dataset with around 600 real-world flare-corrupted images (without ground truth).
- [x] Upload a Baidu Netdisk version for our dataset and pretrained model. 

### License

This project is licensed under <a rel="license" href="https://github.com/ykdai/Flare7K/blob/main/LICENSE">S-Lab License 1.0</a>. Redistribution and use of the dataset and code for non-commercial purposes should follow this license.

### Contact
If you have any question, please feel free to reach me out at `ydai005@e.ntu.edu.sg`.