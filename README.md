# PhotoWCT with closed form matting

Simplified Torch implementation of the papers [Universal Style Transfer](https://arxiv.org/pdf/1705.08086.pdf) and [A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/abs/1802.06474)

This is an unofficial implementation and it is heavily relied on [Photo WCT](https://github.com/vidursatija/PhotoWCT)

The original implementation of [Universal Style Transfer](https://github.com/Yijunmaverick/UniversalStyleTransfer) and [A Closed-form Solution to Photorealistic Image Stylization](https://github.com/NVIDIA/FastPhotoStyle) are there <-

## How to run it
1. Download the decoders from the directories
2. For no segmentation ```python3 run_wct.py --x 4 --style <path to style> --content <path to content> --output <output file name> --decoder decoder_1/dec_gi1849.pkl,decoder_2/dec_1849.pkl,decoder_3/dec_1849.pkl,decoder_4/dec_1849.pkl --smooth gif --encoder 2```
 
3. for segmentation file: 
    ```python3 run_wct.py --x 4 --style examples/style/in34.png  --content examples/content/in34.png --style_seg examples/style_segment/in34.png  --content_seg examples/content_segment/in34.png --output outputs/in34 --decoder decoder_10epoch/dec_1_18490.pkl,decoder_10epoch/dec_2_18490.pkl,decoder_10epoch/dec_3_18490.pkl,decoder_10epoch/dec_4_18490.pkl --smooth gif --encoder=2```

       - note that prompt `encoder=2` means that we use FastPhotoStyle's official saved pth's encoder to transfer photo. Note to change the pth file location in inference file (`run_wct_segmentation` or `run_wct_noseg`). 

## How to train it
1. Get the 2017 MS COCO train and validation datasets and unzip them
2. Download PyTorch VGG16 model
```wget https://download.pytorch.org/models/vgg16-397923af.pth```
3. For every layer(x = 1 to 4) train the decoder. It is recommended to run training twice with starting lr 0.001 and then 0.0001
```python3 --x <layer number> --batch_size <64> --decoder <saved checkpoint if any> --optimizer <optimized checkpoint if any>```
*Note: all decoders & optimizers are saved in the dir `decoder_<x>`*

