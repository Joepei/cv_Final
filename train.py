import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import custom_vgg16 as cvgg16
from tqdm import tqdm

import argparse

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
                                          shuffle=True, num_workers=12, collate_fn=lambda x: x )

valset = torchvision.datasets.CocoDetection(root='./val2017', annFile="./annotations_trainval2017/instances_val2017.json",
                                             transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=12, collate_fn=lambda x: x )

num_layers = args.x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

encoder = cvgg16.vgg16_enc(x=num_layers, pretrained=True).to(device)
if args.decoder:
    decoder = cvgg16.vgg16_dec(x=num_layers, pretrained=True, pretrained_path=args.decoder).to(device)
else:
    decoder = cvgg16.vgg16_dec(x=num_layers, pretrained=False).to(device)

criterion = nn.MSELoss().to(device)
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
            z, maxpool = encoder(inputs)
            inputs_hat = decoder(z.to(device), maxpool)
            z_hat, _ = encoder(inputs_hat.to(device))
            loss = (criterion(inputs_hat, inputs) + z_loss*criterion(z_hat, z))/(1+z_loss)
            loss.backward()
            optimizer.step()
            step += 1
        except Exception as e:
            print(e)
            break

        # print statistics
        running_loss += loss.item()
        if i % 400 == 399:    # print every 2000 mini-batches
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
            z, maxpool = encoder(inputs)
            inputs_hat = decoder(z.to(device), maxpool)
            z_hat, _ = encoder(inputs_hat.to(device))
            loss = criterion(inputs_hat, inputs) + 0.01*criterion(z_hat, z)
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
