import numpy as np
from util import resize
from skimage.metrics import structural_similarity as ssim
import cv2
from tqdm import tqdm

import argparse


# parser.add_argument('--content', type=str,
#                     help='original content image')
# parser.add_argument('--output', type=str,
#                     help='transferred image')
# args = args = parser.parse_args()


# test the ssim index for transfered image after edge detection
def ssim_index_loss(content_path, output_path):
    content = cv2.imread(content_path)
    output = cv2.imread(output_path)
    # print(content.shape, output.shape)
    output, content = resize(output, content)
    print(content.shape, output.shape)

    content_gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ssim_idx = ssim(content_gray, output_gray,
                  data_range=output.max() - output.min())
    print(ssim_idx)
    return ssim_idx

if __name__=="__main__":
    content_p = "/scratch/mc8895/cv_Final/examples/content_edge/eval_results/imgs_epoch_013/"
    output_p = '/scratch/mc8895/cv_Final/outputs/outputs_edge/eval_results/imgs_epoch_013/'
    file_name = 'in14.png'
    file_name_out ='in14_smooth_gif.png'
    ssim_index_loss(content_p+file_name, output_p+file_name_out)