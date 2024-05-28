Code for reproducing results in paper "Equivariant Imaging for Self-supervised Hyperspectral Image Inpainting"

<!-- GETTING STARTED -->
## Getting Started

1. To reporduce the results, download the Chikusei test set (samples) from this link: [Chikusei_Test_5_images.mat](https://drive.google.com/file/d/1hsE4uxQgHTZK-0amcCYIzFTAz5JRnipj/view?usp=share_link). Or alternatively, you could download the whole dataset at [Chikusei Full Image](https://naotoyokoya.com/Download.html) and randomly crop it to the desired size.
2. put the .mat file into the "Test_HSI_Samples" folder.


<!-- USAGE EXAMPLES -->
## Model training
1. Before training, specify the mask type in both closure/ei.py and in physics/inpainting.py. I've created some examples for you in the "Mask_Samples" folder, these are the strips masks masking off all the spectral bands at that location.
```bash
$ python train.py
```


## Testing
```bash
$ python test_inpainting.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
