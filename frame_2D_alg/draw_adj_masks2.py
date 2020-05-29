from time import time
from collections import deque, defaultdict
import numpy as np
from comp_pixel import comp_pixel
from utils import *
from frame_blobs_adj2 import *
import argparse


# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_eye.jpeg')
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])

frame = image_to_blobs(image)


# Draw blobs --------------------------------------------------------------
# draw internal, external and current blob
# blue = external blob
# red = current blob
# green = internal blob

for i, blob in enumerate(frame['blob__']):

    # initialize image with 3 channels (colour image)
    img_blob_ = np.zeros((frame['dert__'].shape[1], frame['dert__'].shape[2], 3))
    img_blob_box = img_blob_.copy()

    dert__mask = ~blob['dert__'][0].mask # get inverted mask value (we need plot mask = false)
    dert__mask = dert__mask*155 # set intensity of colour

    # draw current blob into image
    img_blob_[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3],2] += dert__mask
    img_blob_box[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3],2] += dert__mask
    # draw bounding box of current blob
    cv2.rectangle(img_blob_box, (blob['box'][2], blob['box'][0]),
                                            (blob['box'][3], blob['box'][1]),
                                            color=(0, 0 ,255), thickness=1)

    # internal blob
    for j, adj_blob in enumerate(blob['adj_blob_']):

        adj_dert__mask = ~adj_blob['dert__'][0].mask # get inverted mask value (we need plot mask = false)
        adj_dert__mask = adj_dert__mask*155 # set intensity of colour

        # draw blobs into image
        img_blob_[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3],1] += adj_dert__mask
        img_blob_box[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3],1] += adj_dert__mask

        # draw bounding box
        cv2.rectangle(img_blob_box, (adj_blob['box'][2], adj_blob['box'][0]),
                                    (adj_blob['box'][3], adj_blob['box'][1]),
                                    color=(0, 255 ,0), thickness=1)

      # external blob
    for k, adj_blob_ext in enumerate(blob['adj_blob_ext_']):

        adj_dert_ext__mask = ~adj_blob_ext['dert__'][0].mask # get inverted mask value (we need plot mask = false)
        adj_dert_ext__mask = adj_dert_ext__mask*155 # set intensity of colour

        # draw blobs into image
        img_blob_[adj_blob_ext['box'][0]:adj_blob_ext['box'][1], adj_blob_ext['box'][2]:adj_blob_ext['box'][3],0] += adj_dert_ext__mask
        img_blob_box[adj_blob_ext['box'][0]:adj_blob_ext['box'][1], adj_blob_ext['box'][2]:adj_blob_ext['box'][3],0] += adj_dert_ext__mask

        # draw bounding box
        cv2.rectangle(img_blob_box, (adj_blob_ext['box'][2], adj_blob_ext['box'][0]),
                                    (adj_blob_ext['box'][3], adj_blob_ext['box'][1]),
                                    color=(255, 0 ,0), thickness=1)


    cv2.imwrite("images/adj_blob_mask2/mask_adj_blob_"+str(i)+".png", img_blob_.astype('uint8'))
    cv2.imwrite("images/adj_blob_mask2/mask_adj_blob_"+str(i)+"_box.png", img_blob_box.astype('uint8'))

