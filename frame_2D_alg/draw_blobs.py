from time import time
from collections import deque, defaultdict
import numpy as np
from comp_pixel import comp_pixel
from utils import *
from frame_blobs import *
import argparse

# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])

frame = image_to_blobs(image)

# Draw blobs --------------------------------------------------------------

from matplotlib import pyplot as plt

IMAGE_PATH = "./images/raccoon_eye.jpeg"
image = imread(IMAGE_PATH)

def draw_blobs(frame, param):

    img_blob_ = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2]))
    box_ = []

    # loop across blobs
    for i, blob in enumerate(frame['blob__']):

        if False in blob['dert__'][0].mask:  # if there is unmasked dert

            dert__ = blob['dert__'][param].data
            mask_index = np.where(blob['dert__'][0].mask == True)
            dert__[mask_index] = 0
            dert__ = dert__*255

            # draw blobs into image
            img_blob_[ blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3] ] += dert__
            box_.append(blob['box'])

#     uncomment to enable draw animation
#     plt.figure(dert__select); plt.clf()
#     plt.imshow(img_blobs.astype('uint8'))
#     plt.title('Blob Number ' + str(i))
#     plt.pause(0.001)

    return img_blob_.astype('uint8'), box_

# select dert params
# 0 = i
# 1 = g
# 2 = dy
# 3 = dx
'''
Why are we drawing param values? 
Images should only show box masks, black for masked, white for unmasked 
'''

iblob_, ibox_ = draw_blobs(frame, param=0)
gblob_, gbox_ = draw_blobs(frame, param=1)


for i in range(len(gbox_)):

    # cv2 rectangle (xstart,ystart) (xstop, ystop)
    # box = [ystart, ystop, xstart,xstop]
    iblob_ = cv2.rectangle(iblob_, (ibox_[i][2], ibox_[i][0]), (ibox_[i][3], ibox_[i][1]), color=(750, 0 ,0),
                  thickness=1)
    gblob_ = cv2.rectangle(gblob_, (gbox_[i][2], gbox_[i][0]), (gbox_[i][3], gbox_[i][1]), color=(750, 0 ,0),
                  thickness=1)

# save to disk
cv2.imwrite("images/blobs0.png", iblob_)
cv2.imwrite("images/blobs1.png", gblob_)


# Save each individual blobs and their mask to disk  --------------------------

for i in range(len(frame['blob__'])):
    
    # masked = black
    # unmasked = white
    mask = 255-(frame['blob__'][i]['dert__'].mask[0] * 255)
    img_mask = mask.astype('uint8')
    
    cv2.imwrite("images/blob_mask/masks"+str(i)+".bmp", img_mask)







