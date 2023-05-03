import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model import PhotoWCT
from fast_vgg16 import VGGEncoder, VGGDecoder
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

encoders = []
decoders = []
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

def compute_label_info(cont_seg, styl_seg):
    if cont_seg.size == False or styl_seg.size == False:
        return None, None
    # print('content seg size', cont_seg)
    max_label = np.max(cont_seg) + 1
    label_set = np.unique(cont_seg)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        # if l==0:
        #   continue
        is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
        o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
        o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
        label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
    
    return label_set, label_indicator


def __wct_core(cont_feat, styl_feat):
    cFSize = cont_feat.size()
    c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
    cont_feat = cont_feat - c_mean
    
    iden = torch.eye(cFSize[0])  # .double()
    # if self.is_cuda:
    iden = iden.cuda()
    
    contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
    # print(contentConv)
    has_non_finite = torch.isnan(contentConv) | torch.isinf(contentConv)

    # Replace non-finite values with a specified value (e.g., 0)
    contentConv[has_non_finite] = 0.000001
    # del iden
    c_u, c_e, c_v = torch.svd(contentConv, some=False)
    # c_e2, c_v = torch.eig(contentConv, True)
    # c_e = c_e2[:,0]
    
    k_c = cFSize[0]
    for i in range(cFSize[0] - 1, -1, -1):
        if c_e[i] >= 0.00001:
            k_c = i + 1
            break
    
    sFSize = styl_feat.size()
    s_mean = torch.mean(styl_feat, 1)
    styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
    styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)

    has_non_finite = torch.isnan(styleConv) | torch.isinf(styleConv)
    # Replace non-finite values with a specified value (e.g., 0)
    styleConv[has_non_finite] = 0.000001
    s_u, s_e, s_v = torch.svd(styleConv, some=False)
    
    k_s = sFSize[0]
    for i in range(sFSize[0] - 1, -1, -1):
        if s_e[i] >= 0.00001:
            k_s = i + 1
            break
    
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cont_feat)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature

def __feature_wct(cont_feat, styl_feat, cont_seg, styl_seg, label_set, label_indicator):
    cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
    styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
    cont_feat_view = cont_feat.view(cont_c, -1).clone()
    styl_feat_view = styl_feat.view(styl_c, -1).clone()

    if cont_seg.size == False or styl_seg.size == False:
        target_feature = __wct_core(cont_feat_view, styl_feat_view)
    else:
        target_feature = cont_feat.view(cont_c, -1).clone()
        if len(cont_seg.shape) == 2:
            t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
        else:
            t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
        if len(styl_seg.shape) == 2:
            t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
        else:
            t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))

        for l in label_set:
            if label_indicator[l] == 0:
                continue
            cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
            styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
            if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                continue

            cont_indi = torch.LongTensor(cont_mask[0])
            styl_indi = torch.LongTensor(styl_mask[0])
            # if self.is_cuda:
            cont_indi = cont_indi.cuda(0)
            styl_indi = styl_indi.cuda(0)

            cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
            sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
            # print(len(cont_indi))
            # print(len(styl_indi))
            tmp_target_feature = __wct_core(cFFG, sFFG)
            # print(tmp_target_feature.size())
            if torch.__version__ >= "0.4.0":
                # This seems to be a bug in PyTorch 0.4.0 to me.
                new_target_feature = torch.transpose(target_feature, 1, 0)
                new_target_feature.index_copy_(0, cont_indi, \
                        torch.transpose(tmp_target_feature,1,0))
                target_feature = torch.transpose(new_target_feature, 1, 0)
            else:
                target_feature.index_copy_(1, cont_indi, tmp_target_feature)

    target_feature = target_feature.view_as(cont_feat)
    ccsF = target_feature.float().unsqueeze(0)
    return ccsF


content_image = image_loader(transform, args.content)
style_image = image_loader(transform, args.style)
_, _, ccw, cch = content_image.shape
_, _, ssw, ssh = style_image.shape

try:
    content_seg = Image.open(args.content_seg)
    content_seg = np.asarray(content_seg)
    style_seg = Image.open(args.style_seg)
    style_seg = np.asarray(style_seg)
    # the black and white segementation is werid 
    # the following code is for that mask
    if content_seg.ndim == 3:
        content_seg = np.asarray(content_seg[:ccw, :cch, -1])
        style_seg = np.asarray(style_seg[:ssw, :ssh, -1])
    else: 
        content_seg = np.asarray(content_seg[:ccw, :cch])
        style_seg = np.asarray(style_seg[:ssw, :ssh])
except:
    content_seg = np.array([])
    style_seg = np.array([])


# style_seg
# debugging purpose

print(content_image.shape)
print(content_seg.shape)
# print(content_seg)

print(style_image.shape)
print(style_seg.shape)
# print(style_seg)


label_set, label_indicator = compute_label_info(content_seg, style_seg)


sF4, sF3, sF2, sF1 = encoders[-1].forward_multiple(style_image.to(device))
# style_Fs = [sF1, sF2, sF3,sF4]

for j in range(args.x, 0, -1):
    cF, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = encoders[j-1](content_image.to(device)) # (1, C, H, W)
    # z_style, _ = encoders[j-1](style_image) # (1, C, H, W)
    sF, _, _, _, _, _, _ = encoders[j-1](style_image.to(device))

    content_feat = cF.data.squeeze(0).to(device)
    # print(content_feat.shape)
    style_feat = sF.data.squeeze(0).to(device)

    ccsF = __feature_wct(content_feat, style_feat, content_seg, style_seg, label_set, label_indicator)
    content_image = decoders[j-1](ccsF, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3) # (1, C, H, W)


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

