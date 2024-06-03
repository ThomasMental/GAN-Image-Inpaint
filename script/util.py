import random
import torch
import torchvision.transforms as transforms
import numpy as np



# PARAM:  hole_size: (hole width, hole height)
# RETURN: (hole width, hole height, upleft corner x, upleft corner y)
def get_hole(hole_size):
    return hole_size + (random.randint(0,255-hole_size[0]), random.randint(0,255-hole_size[1]))

# generate a mask for a batch of input image
# PARAM: shape: (batch_num, channels, width, height)
#        hole: (hole width, hole height, upleft corner x, upleft corner y)
# RETURN: mask of same shape with to-complete region be 1
def get_mask(shape, hole):
    mask = torch.zeros(shape)
    hw, hh, x, y = hole
    mask[:,:,y:y+hh,x:x+hw] = 1.0
    return mask

def crop(batch, area):
    hw, hh, x, y = area
    return batch[:,:,y:y+hh, x:x+hw]
