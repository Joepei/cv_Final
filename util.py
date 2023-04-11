import numpy as np

def resize(cont_img, init_img):
    '''
    Notes: assume cont_img and init_img are arrays with 3-channel
    '''
    lc, hc, _ = cont_img.shape
    li, hi, _ = init_img.shape
    if lc > li: 
        cont_img = cont_img[int(np.floor((lc-li)/2)):int(-np.ceil((lc-li)/2)),:,:]
    if lc < li: 
        init_img = init_img[int(np.floor((li-lc)/2)):int(-np.ceil((li-lc)/2)),:,:]
    if hc > hi:
        cont_img = cont_img[:,int(np.floor((hc-hi)/2)):int(-np.ceil((hc-hi)/2)),:]
    if hc < hi: 
        init_img = init_img[:,int(np.floor((hi-hc)/2)):int(-np.ceil((hi-hc)/2)),:]
    return cont_img, init_img
