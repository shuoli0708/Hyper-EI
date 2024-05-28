Code for reproducing results in paper "Equivariant Imaging for Self-supervised Hyperspectral Image Inpainting"

<!-- GETTING STARTED -->
## Getting Started

1. To reporduce the results, download the Chikusei test set (samples) at this link: [Chikusei_Test_5_images.mat](https://drive.google.com/file/d/1hsE4uxQgHTZK-0amcCYIzFTAz5JRnipj/view?usp=share_link). Or alternatively, you could download the whole dataset at [Chikusei Full Image](https://naotoyokoya.com/Download.html), and randomly crop the image 
2. put the .mat file into the "Test_HSI_Samples" folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Model training
```bash
$ python train.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Testing
```bash
$ python test_inpainting.py
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
