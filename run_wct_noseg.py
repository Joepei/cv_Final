import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from fast_vgg16 import VGGEncoder, VGGDecoder
import mat_transforms
from model import PhotoWCT

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
parser.add_argument('--output', type=str, default='stylized.png',
                    help='Output image path')
parser.add_argument('--smooth', type=str, help='apply gif smoothing or mat transform')
parser.add_argument('--encoder', type=int, help='options for encoders: 1: vgg-16 encoder; 2: FastphotoStyle encoder')

args = parser.parse_args()



def image_loader(loader, image_name):
    img = Image.open(image_name).convert("RGB")
    h, w, c = np.array(img).shape
    h = (h//8)*8
    w = (w//8)*8
    img = Image.fromarray(np.array(img)[:h, :w])
    img = loader(img).float()
    img = img.clone().detach() # torch.tensor(image, requires_grad=True)
    img = img.unsqueeze(0)
    return img


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

decoder_paths = args.decoder.split(",")

# encoders = [cvgg16.vgg16_enc(x=j+1, pretrained=True).to(device) for j in range(args.x)]
# decoders = [cvgg16.vgg16_dec(x=j+1, pretrained=True, pretrained_path=decoder_paths[j]).to(device) for j in range(args.x)]
encoders = []
decoders = []


# Testing for FastPhoto Style Official pth migration
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('/scratch/mc8895/FastPhotoStyle/PhotoWCTModels/photo_wct.pth'))
encoders_pwct = [p_wct.e1.to(device), p_wct.e2.to(device), p_wct.e3.to(device), p_wct.e4.to(device)]
# decoders = [p_wct.d1.to(device), p_wct.d2.to(device), p_wct.d3.to(device), p_wct.d4.to(device)]
for i in range(args.x):
    encoder = VGGEncoder(level=i+1)
    if args.encoder == 1: #vgg19
        encoder.load_state_dict(torch.load("vgg16-397923af.pth"), strict=False)
    if args.encoder == 2: #fastphotostyle pth
        encoder = encoders_pwct[i]
    for p in encoder.parameters():
        p.requires_grad = False
        # print(p.data)
    
    encoder.train(False)
    encoder.eval()
    encoder.to(device)
    encoders.append(encoder)


    # print(torch.load(decoder_paths[i]).keys())
    decoder = VGGDecoder(level=i+1).to(device)
    #load in saved decoder path
    decoder.load_state_dict(torch.load(decoder_paths[i]))
    for p in decoder.parameters():
        p.requires_grad = False
        decoder.train(False)
        decoder.eval()
    decoder.to(device)
    # print(decoder)
    decoders.append(decoder)



content_image = image_loader(transform, args.content).to(device)
style_image = image_loader(transform, args.style).to(device)

for j in range(args.x, 0, -1):
    # z_content, maxpool_content = encoders[j-1](content_image) # (1, C, H, W)
    cF, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = encoders[j-1](content_image)
    sF, _, _, _, _, _, _ = encoders[j-1](style_image)
    
    # z_style, _ = encoders[j-1](style_image) # (1, C, H, W)
    n_channels = cF.size()[1] # C
    n_1 = cF.size()[2] # H
    n_2 = cF.size()[3] # W
    print(n_channels, n_1, n_2)

    z_content = cF.squeeze(0).view([n_channels, -1]) # (C, HW)
    z_style = sF.squeeze(0).view([n_channels, -1]) # (C, HW)
    print("z_content", z_content.shape)
    print("z_style", z_style.shape)
    white_content = mat_transforms.whitening(z_content) # (C, HW)
    color_content = mat_transforms.colouring(z_style, white_content) # (C, HW)
    print("white_content", white_content.shape)
    print("color_content", color_content.shape)

    alpha = 0.8
    color_content = alpha*color_content + (1.-alpha)*z_content

    color_content = color_content.view([1, n_channels, n_1, n_2]) # (1, C, H, W)
    print(color_content.shape)
    # color_content = color_content.unsqueeze(0) # (1, C, H, W)

    content_image = decoders[j-1](color_content.to(device), cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3) # (1, C, H, W)

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