import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from fast_vgg16 import VGGEncoder, VGGDecoder
from tqdm import tqdm

import argparse
from model import PhotoWCT
# import perceptual_loss

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--decoder', type=str, default=None,
                    help='Decoder path')
parser.add_argument('--optimizer', type=str, default=None,
                    help='Optimizer path')
parser.add_argument('--x', type=int, default=1,
                    help='Number of VGG16 layers to use')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of VGG16 layers to use')
parser.add_argument('--save_path', type=str, default="decoder_",
                    help='saved decoder path loc')
parser.add_argument('--epoch', type=int, default=2,
                    help='number of epoches')

args = parser.parse_args()

transform = transforms.Compose(
    [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.CocoDetection(root='./train2017', annFile="./annotations_trainval2017/instances_train2017.json",
                                              transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1, collate_fn=lambda x: x )
num_batches = len(trainloader)
print(f"train loader num batches {num_batches}")
valset = torchvision.datasets.CocoDetection(root='./val2017', annFile="./annotations_trainval2017/instances_val2017.json",
                                             transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1, collate_fn=lambda x: x )

num_batches = len(valloader)
print(f"val loader num batches {num_batches}")

num_layers = args.x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('/scratch/mc8895/FastPhotoStyle/PhotoWCTModels/photo_wct.pth'))
encoders = [p_wct.e1.to(device), p_wct.e2.to(device), p_wct.e3.to(device), p_wct.e4.to(device)]
# decoders = [p_wct.d1.to(device), p_wct.d2.to(device), p_wct.d3.to(device), p_wct.d4.to(device)]

#encoder is pertrained
# encoder = VGGEncoder(level=num_layers)
# encoder.load_state_dict(torch.load("vgg16-397923af.pth"), strict=False)
encoder = encoders[num_layers-1]
for p in encoder.parameters():
    p.requires_grad = False
encoder.train(False)
encoder.eval()
encoder.to(device)

decoder = VGGDecoder(level=num_layers).to(device)
# decoder = decoders[num_layers-1].to(device)

print(encoder)
print(decoder)
criterion = nn.MSELoss().to(device)
# criterion_perceptual = perceptual_loss.VGGPerceptualLoss()
optimizer = optim.Adam(decoder.parameters(), lr=0.0001) # .to(device)
if args.optimizer:
    optimizer.load_state_dict(torch.load(args.optimizer))
z_loss = 0.01

step = 0
for epoch in range(args.epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    i = 0
    for data in tqdm(trainloader):
        # get the inputs; data is a list of tuple(inputs, labels)
        try:
            # shape: (64, 3, 224, 224)
            inputs = torch.stack([d[0] for d in data], dim = 0).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3  = encoder(inputs)
            out = decoder(cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)
            cF4_hat, _, _, _, _, _, _ = encoder(out)

            reconstruction_loss = (criterion(out, inputs) + z_loss*criterion(cF4_hat, cF4))/(1+z_loss)
            perceptual_loss = 0
            loss = perceptual_loss + reconstruction_loss

            loss.backward()
            optimizer.step()
            step += 1
        except Exception as e:
            print(e)
            break

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / 400))
            running_loss = 0.0
            torch.save(decoder.state_dict(), args.save_path+"/dec_"+str(num_layers)+"_"+str(step)+".pkl")
            torch.save(optimizer.state_dict(), args.save_path+"/opt_"+str(num_layers)+"_"+str(step)+".pkl")
        i += 1

    torch.save(decoder.state_dict(), args.save_path+"/dec_"+str(num_layers)+"_"+str(step)+".pkl")
    torch.save(optimizer.state_dict(), args.save_path+"/opt_"+str(num_layers)+"_"+str(step)+".pkl")
    i = 0
    running_loss = 0.0


    for data in tqdm(valloader):
        # get the inputs; data is a list of [inputs, labels]

        try:
            inputs = torch.stack([d[0] for d in data], dim = 0).to(device)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            # z, maxpool = encoder(inputs)
            # inputs_hat = decoder(z.to(device), maxpool)
            # z_hat, _ = encoder(inputs_hat.to(device))
            cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3  = encoder(inputs)
            out = decoder(cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)
            cF4_hat, _, _, _, _, _, _ = encoder(out)
            loss = criterion(out, inputs) + 0.01*criterion(cF4_hat, cF4)
            # loss.backward()
            # optimizer.step()
        except Exception as e:
            print(e)
            continue
        i += 1
        # print statistics
        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / i))

print('Finished Training')
