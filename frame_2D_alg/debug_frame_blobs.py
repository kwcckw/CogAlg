''' 
debug and check the results from different frame_blobs approaches
work in progress

'''

import numpy as np

from utils import imread



from frame_blobs import comp_pixel, derts2blobs
from xy_blobs import image_to_blobs
from matplotlib import pyplot as plt


if __name__ == "__main__":
    
    image = imread('./images/raccoon_eye.jpeg')
    img_blobs_ff = np.zeros((image.shape[0]-1,image.shape[1]-1,))
    img_blobs_xy = np.zeros((image.shape[0]-1,image.shape[1]-1,))
    
    ## flood fill method ## 
    dert__ = comp_pixel(image)
    frame_ff = derts2blobs(dert__,
                        verbose=0,
                        render=0,
                        use_c=0)

    blob_ff_ = frame_ff.blob_ # flood fill blobs
    fmatch_ff_ = np.zeros((len(blob_ff_))) # flag to know whether the blob is matched
    ind_ff2xy = np.zeros((len(blob_ff_)))-1; # index of matched blob xy to blob ff
    
    ## dert-P-stack-blob method ##
    frame_xy = image_to_blobs(image,
                              verbose=0,
                                render=0)

    blob_xy_ = frame_xy['blob__'] # xy blobs
    fmatch_xy_ = np.zeros((len(blob_xy_))) # flag to know whether the blob is matched
    ind_xy2ff = np.zeros((len(blob_xy_)))-1; # index of matched blob ff to blob xy
    
    
    ## check and compare each blob
    repeat_index_ff  = []
    repeat_index_xy  = []
    
    # find same blob based on their bounding box    
    for iff, blob_ff in enumerate(blob_ff_):
        for ixy, blob_xy in enumerate(blob_xy_):            
            if blob_ff.box == blob_xy.box:
                ind_ff2xy[iff] = ixy
                ind_xy2ff[ixy] = iff
                fmatch_ff_[iff] = 1
                fmatch_xy_[ixy] = 1
                break
    
    # draw ff blobs
    for blob_ff in blob_ff_:
        ymask = 0
        for y in range(blob_ff.box[0],blob_ff.box[1]):
            xmask = 0
            for x in range(blob_ff.box[2],blob_ff.box[3]):
                if blob_ff.mask[ymask,xmask] == False:
                    if blob_ff.sign:
                       img_blobs_ff[y,x] = 255 
                    else:
                        img_blobs_ff[y,x] = 128
                xmask+=1
            ymask+=1
            
    # draw xy blobs
    for blob_xy in blob_xy_:
        ymask = 0
        for y in range(blob_xy.box[0],blob_xy.box[1]):
            xmask = 0
            for x in range(blob_xy.box[2],blob_xy.box[3]):
                if blob_xy.mask[ymask,xmask] == False:
                    if blob_xy.sign:
                       img_blobs_xy[y,x] = 255 
                    else:
                        img_blobs_xy[y,x] = 128
                xmask+=1
            ymask+=1

    # difference in blobs
    ind_dif = np.where(abs(img_blobs_ff-img_blobs_xy))
    
    img_blobs_dif = np.zeros((image.shape[0]-1,image.shape[1]-1,3),dtype='uint8')
    img_blobs_dif[:,:,0] = img_blobs_ff
    img_blobs_dif[:,:,1] = img_blobs_ff
    img_blobs_dif[:,:,2] = img_blobs_ff
    img_blobs_dif[ind_dif[0],ind_dif[1],0] = 255
    img_blobs_dif[ind_dif[0],ind_dif[1],1] = 0
    img_blobs_dif[ind_dif[0],ind_dif[1],2] = 0
    
    
    ## visualization
    plt.figure()
    plt.imshow(img_blobs_ff)
    plt.title('flood fill blobs')
    plt.figure()
    plt.imshow(img_blobs_xy)
    plt.title('xy blobs')
    plt.figure()
    plt.imshow(img_blobs_dif)
    plt.title('differences')
    