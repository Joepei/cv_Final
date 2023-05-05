import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model import PhotoWCT
from fast_vgg16 import VGGEncoder, VGGDecoder
from core import __feature_wct, image_loader, load_segment, compute_label_info
import custom_vgg16 as cvgg16
import mat_transforms

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--decoder', type=str, default=None,
                    help='Decoder path')
parser.add_argument('--x', type=int, default=2,
                    help='Num layers to transform')
parser.add_argument('--style', type=str, default=None,
                    help='Style image path')
parser.add_argument('--content', type=str, default=None,
                    help='Content image path')

parser.add_argument('--style_seg', type=str, default=None,
                    help='Style image segmentation path')
parser.add_argument('--content_seg', type=str, default=None,
                    help='Content image segmentation path')

parser.add_argument('--output', type=str, default='stylized.png',
                    help='Output image path')
parser.add_argument('--smooth', type=str, help='apply gif smoothing or mat transform')
parser.add_argument('--encoder', type=int, default=2, help='options for encoders: 1: vgg-16 encoder; 2: FastphotoStyle encoder; 3: serial encoder')


args = parser.parse_args()



transform = transforms.Compose([
#      transforms.RandomResizedCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

reverse_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1./0.229, 1./0.224, 1./0.225])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
print(args.encoder)
decoder_paths = args.decoder.split(",")

encoders = []
decoders = []
if args.encoder == 3:
    print("in serial backbone")
    encoders = [cvgg16.vgg16_enc(x=j+1, pretrained=True).to(device) for j in range(args.x)]
    decoders = [cvgg16.vgg16_dec(x=j+1, pretrained=True, pretrained_path=decoder_paths[j]).to(device) for j in range(args.x)]
else: 
    print("in fast backbone")
    p_wct = PhotoWCT()
    p_wct.load_state_dict(torch.load('/scratch/mc8895/FastPhotoStyle/PhotoWCTModels/photo_wct.pth'))
    encoders_pwct = [p_wct.e1.to(device), p_wct.e2.to(device), p_wct.e3.to(device), p_wct.e4.to(device)]

    for i in range(args.x):
        encoder = VGGEncoder(level=i+1)
        if args.encoder == 1:
            encoder.load_state_dict(torch.load("vgg16-397923af.pth"), strict=False)
        if args.encoder == 2:
            encoder = encoders_pwct[i]
        for p in encoder.parameters():
            p.requires_grad = False
        
        encoder.train(False)
        encoder.eval()
        encoder.to(device)
        encoders.append(encoder)
        
        decoder = VGGDecoder(level=i+1).to(device)
        #load in saved decoder path
        decoder.load_state_dict(torch.load(decoder_paths[i]))
        for p in decoder.parameters():
            p.requires_grad = False
            decoder.train(False)
            decoder.eval()
        decoder.to(device)
        decoders.append(decoder)

content_image = image_loader(transform, args.content).to(device)
style_image = image_loader(transform, args.style).to(device)
_, _, ccw, cch = content_image.shape
_, _, ssw, ssh = style_image.shape

content_seg = load_segment(args.content_seg)
style_seg = load_segment(args.style_seg) 


print(content_image.shape)
print(content_seg.shape)

print(style_image.shape)
print(style_seg.shape)


label_set, label_indicator = compute_label_info(content_seg, style_seg)
print("label contnet")
print(label_set)
print(label_indicator)



for j in range(args.x, 0, -1):

    if args.encoder != 3:
        cF, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = encoders[j-1](content_image.to(device)) # (1, C, H, W)
        # z_style, _ = encoders[j-1](style_image) # (1, C, H, W)
        sF, _, _, _, _, _, _ = encoders[j-1](style_image.to(device))
        content_feat = cF.data.squeeze(0).to(device)
        style_feat = sF.data.squeeze(0).to(device)

        ccsF = __feature_wct(content_feat, style_feat, content_seg, style_seg, label_set, label_indicator, device)
        content_image = decoders[j-1](ccsF, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3) # (1, C, H, W)

    else:

        z_content, maxpool_content = encoders[j-1](content_image.to(device))
        z_style, _ = encoders[j-1](style_image)
        content_feat = z_content.data.squeeze(0).to(device)
        style_feat = z_style.data.squeeze(0).to(device)
        ccsF = __feature_wct(content_feat, style_feat, content_seg, style_seg, label_set, label_indicator, device)
        content_image = decoders[j-1](ccsF.to(device), maxpool_content)


new_image = content_image.squeeze(0) # (C, H, W)
new_image = reverse_normalize(new_image) # (C, H, W)
new_image = torch.transpose(new_image, 0, 1) # (H, C, W)
new_image = torch.transpose(new_image, 1, 2) # (H, W, C)

new_image = np.maximum(np.minimum(new_image.cpu().detach().numpy(), 1.0), 0.0)

result = Image.fromarray((new_image * 255).astype(np.uint8))
result.save(args.output + '.png')

if args.smooth and args.smooth == "mat":
    result = mat_transforms.smoothen(args.output+".png", args.content)
    result.save(args.output+"_smooth_mat.png")
elif args.smooth and args.smooth == "gif":
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
    result = p_pro.process(args.output+".png", args.content)
    result.save(args.output+"_smooth_gif.png")

