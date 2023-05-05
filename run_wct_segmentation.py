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
parser.add_argument('--encoder', type=int, default=2, help='options for encoders: 1: vgg-16 encoder; 2: FastphotoStyle encoder')


args = parser.parse_args()

import torchvision.models as models
'''
#####################################################################################
'''
model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}
# Load pre-trained VGG19 model
# so you do not have to download model to some unknown black hole
vgg19_model = models.vgg19(weights=None) 
# load and then delete later
vgg19_model.load_state_dict(torch.load("vgg19-dcbb9e9d.pth"))

def load_vgg_weight(level=4, model=vgg19_model):
    encoder = VGGEncoder(level=level)
    conv0 = torch.tensor([[[[  0.]],

            [[  0.]],

            [[255.]]],


            [[[  0.]],

            [[255.]],

            [[  0.]]],


            [[[255.]],

            [[  0.]],

            [[  0.]]]], requires_grad=True)
    if level == 1:
        encoder.conv0.weight.data = conv0.data
        encoder.conv1_1.weight.data = model.features[0].weight.data
        return encoder
    elif level == 2:
        encoder.conv0.weight.data = conv0.data
        encoder.conv1_1.weight.data = model.features[0].weight.data
        encoder.conv1_2.weight.data = model.features[2].weight.data
        encoder.conv2_1.weight.data = model.features[5].weight.data
        return encoder
    elif level == 3:
        encoder.conv0.weight.data = conv0.data
        encoder.conv1_1.weight.data = model.features[0].weight.data
        encoder.conv1_2.weight.data = model.features[2].weight.data
        encoder.conv2_1.weight.data = model.features[5].weight.data
        encoder.conv2_2.weight.data = model.features[7].weight.data
        encoder.conv3_1.weight.data = model.features[10].weight.data
        return encoder
    elif level == 4:
        encoder.conv0.weight.data = conv0.data
        encoder.conv1_1.weight.data = model.features[0].weight.data
        encoder.conv1_2.weight.data = model.features[2].weight.data
        encoder.conv2_1.weight.data = model.features[5].weight.data
        encoder.conv2_2.weight.data = model.features[7].weight.data
        encoder.conv3_1.weight.data = model.features[10].weight.data
        encoder.conv3_2.weight.data = model.features[12].weight.data
        encoder.conv3_3.weight.data = model.features[14].weight.data
        encoder.conv3_4.weight.data = model.features[16].weight.data
        encoder.conv4_1.weight.data = model.features[19].weight.data
        return encoder
    print("failed to load!!!!!!")
    return None
'''
#####################################################################################
'''

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
p_wct.load_state_dict(torch.load('photo_wct.pth'))
encoders_pwct = [p_wct.e1.to(device), p_wct.e2.to(device), p_wct.e3.to(device), p_wct.e4.to(device)]

for i in range(args.x):
    encoder = VGGEncoder(level=i+1)
    if args.encoder == 1:
        encoder = load_vgg_weight(level=i+1)
        # encoder.load_state_dict(torch.load("vgg19-dcbb9e9d.pth"), strict=False)
        # encoder.load_state_dict(torch.load("vgg16-397923af.pth"), strict=False)
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


def compute_label_info(content_segment, style_segment):
    if not content_segment.size or not style_segment.size:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        content_mask = np.where(content_segment.reshape(content_segment.shape[0] * content_segment.shape[1]) == l)
        style_mask = np.where(style_segment.reshape(style_segment.shape[0] * style_segment.shape[1]) == l)

        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if c_size > 10 and s_size > 10 and c_size / s_size < 100 and s_size / c_size < 100:
            label_indicator[l] = True
        else:
            label_indicator[l] = False
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

def __wct_core_segment(content_feat, style_feat, content_segment, style_segment,
                     label_set, label_indicator, weight=1, registers=None,
                     device='cpu'):
    def resize(feat, target):
        size = (target.size(2), target.size(1))
        if len(feat.shape) == 2:
            return np.asarray(Image.fromarray(feat).resize(size, Image.NEAREST))
        else:
            return np.asarray(Image.fromarray(feat, mode='RGB').resize(size, Image.NEAREST))

    def get_index(feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0]).to(device)

    squeeze_content_feat = content_feat.squeeze(0)
    squeeze_style_feat = style_feat.squeeze(0)

    content_feat_view = squeeze_content_feat.view(squeeze_content_feat.size(0), -1).clone()
    style_feat_view = squeeze_style_feat.view(squeeze_style_feat.size(0), -1).clone()

    resized_content_segment = resize(content_segment, squeeze_content_feat)
    resized_style_segment = resize(style_segment, squeeze_style_feat)

    target_feature = content_feat_view.clone()
    for label in label_set:
        if not label_indicator[label]:
            continue
        content_index = get_index(resized_content_segment, label)
        style_index = get_index(resized_style_segment, label)
        if content_index is None or style_index is None:
            continue
        masked_content_feat = torch.index_select(content_feat_view, 1, content_index)
        masked_style_feat = torch.index_select(style_feat_view, 1, style_index)
        _target_feature = __wct_core(masked_content_feat, masked_style_feat)
        if torch.__version__ >= '0.4.0':
            # XXX reported bug in the original repository
            new_target_feature = torch.transpose(target_feature, 1, 0)
            new_target_feature.index_copy_(0, content_index,
                                           torch.transpose(_target_feature, 1, 0))
            target_feature = torch.transpose(new_target_feature, 1, 0)
        else:
            target_feature.index_copy_(1, content_index, _target_feature)
    return target_feature

def __feature_wct(cont_feat, styl_feat, cont_seg, styl_seg, label_set, label_indicator):
    cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
    styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
    cont_feat_view = cont_feat.view(cont_c, -1).clone()
    styl_feat_view = styl_feat.view(styl_c, -1).clone()

    if cont_seg.size == False or styl_seg.size == False:
        target_feature = __wct_core(cont_feat_view, styl_feat_view)
    else:
        target_feature = __wct_core_segment(cont_feat, styl_feat, cont_seg, styl_seg,
                                          label_set, label_indicator, weight=1, registers=None, device=device)

    target_feature = target_feature.view_as(cont_feat)
    ccsF = target_feature.float().unsqueeze(0)
    return ccsF


content_image = image_loader(transform, args.content)
style_image = image_loader(transform, args.style)
_, _, ccw, cch = content_image.shape
_, _, ssw, ssh = style_image.shape

def change_seg(seg):
    color_dict = {
        (0, 0, 255): 3,  # blue
        (0, 255, 0): 2,  # green
        (0, 0, 0): 0,  # black
        (255, 255, 255): 1,  # white
        (255, 0, 0): 4,  # red
        (255, 255, 0): 5,  # yellow
        (128, 128, 128): 6,  # grey
        (0, 255, 255): 7,  # lightblue
        (255, 0, 255): 8  # purple
    }
    arr_seg = np.asarray(seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y-1, :]
                        except Exception:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)

def load_segment(image_path, image_size=512):
    if not image_path:
        return np.asarray([])
    image = Image.open(image_path)
    if image_size is not None:
        transform = transforms.Resize(image_size, interpolation=Image.NEAREST)
        image = transform(image)
    w, h = image.size
    transform = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))
    image = transform(image)
    if len(np.asarray(image).shape) == 3:
        image = change_seg(image)
    return np.asarray(image)


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


# sF4, sF3, sF2, sF1 = encoders[-1].forward_multiple(style_image.to(device))

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

