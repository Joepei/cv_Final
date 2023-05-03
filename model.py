import torch.nn as nn
from fast_vgg16 import VGGEncoder, VGGDecoder

class PhotoWCT(nn.Module):
    def __init__(self):
        super(PhotoWCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)
    def forward(self, *input):
        pass 