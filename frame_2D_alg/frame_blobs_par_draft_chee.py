'''
Draft of blob-parallel version of frame_blobs
'''


from class_cluster import ClusterStructure, NoneType
from utils import (
    pairwise,
    imread, )
import multiprocessing as mp
from frame_blobs_ma import comp_pixel
from time import time
import numpy as np
from utils import *
from multiprocessing.pool import ThreadPool
from matplotlib import pyplot as plt

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback


class CDert(ClusterStructure):
    # Derts
    i = int
    g = int
    dy = int
    dx = int
    # other data
    sign = NoneType
    x_coord = int
    y_coord = int
    blob = list
    blob_ids = list
    blob_id_min = int
    fopen = bool

    
class CBlob(ClusterStructure):
    # Derts
    I = int
    G = int
    Dy = int
    Dx = int
    S = int
    # other data
    box = list
    sign = NoneType
    dert_coord_ = set  # let derts' id be their coords
    root_dert__ = object
    adj_blobs = list
    fopen = bool
    dert = object
    
def generate_blobs(dert_input,y,x):
    '''
    generate dert & blob based on given dert and their location
    '''
    
    # dert instance from their class
    dert = CDert(i=dert_input[0], g=dert_input[1] - ave,dy=dert_input[2], dx=dert_input[3],
                 x_coord = x, y_coord=y,sign=dert_input[1] - ave > 0, fopen=1)
    
    # blob instance from their class
    blob = CBlob(I=dert_input[0], G=dert_input[1] - ave,Dy=dert_input[2], Dx=dert_input[3],
                 dert_coord_ = [[y, x]], sign=dert_input[1] - ave > 0, dert = dert)
    
    # update blob params into dert
    dert.blob = blob
    dert.blob_ids.append(blob.id)
    dert.blob_id_min = blob.id
    
    return [blob,dert]
       
   
def check_rims(blob,blob_rims):
    '''
    check connectivity of blob's rim derts and update each dert's ids
    '''
    dert = blob.dert # get dert from blob
     
    if blob.sign: # for + sign, check 8 ortho+diag rims
        for blob_rim in blob_rims: 
            if blob_rim: # not empty rim
                _dert = blob_rim.dert
            
                # check same sign
                if dert.sign == _dert.sign:
                        
                        dert.blob_ids.append(blob_rim.id) # add same sign rim blob's id
                        dert.blob_ids += _dert.blob_ids # chain the rim dert' blob' rim dert's blobs ids
                        dert.blob_ids = list(set(dert.blob_ids)) # remove duplicated ids
                        
    else: # for - sign, check 4 ortho rims
        for i,blob_rim in enumerate(blob_rims): 
            if blob_rim and i%2: # not empty rim and ortho rim
                _dert = blob_rim.dert
            
                # check same sign
                if dert.sign == _dert.sign:

                        dert.blob_ids.append(blob_rim.id) # add same sign rim blob's id
                        dert.blob_ids += _dert.blob_ids # chain the rim dert' blob' rim dert's blobs ids
                        dert.blob_ids = list(set(dert.blob_ids)) # remove duplicated ids
    
                
    # min of ids
    dert.blob_id_min = min(dert.blob_ids) 
    return [blob,dert]

# there could be a better way to replace this function with parallel process ,need to think about it
def get_rim_blob(blob_,height,width):
    '''
    generate rims' blobs
    '''
    
    # convert blobdert_ to 2D array
    blob__ = [blob_[i:i+width] for i in range(0, len(blob_), width)]
    
    blob_rims_ = []
    for y in range(height):
        for x in range(width):
            # topleft
            if y-1>=0 and x-1>=0: blob_topleft = blob__[y-1][x-1]
            else: blob_topleft = []
            # top
            if y-1>=0: blob_top = blob__[y-1][x]   
            else: blob_top = []
            # topright
            if y-1>=0 and x+1<=width-1: blob_topright = blob__[y-1][x+1]
            else: blob_topright = []            
            # right
            if x+1<=width-1: blob_right = blob__[y][x+1] 
            else: blob_right = []
            # botright
            if y+1<=height-1 and x+1<=width-1: blob_botright = blob__[y+1][x+1]
            else: blob_botright = []    
            # bot
            if y+1<=height-1: blob_bot = blob__[y+1][x] 
            else: blob_bot = []
            # botleft
            if y+1<=height-1 and x-1>=0: blob_botleft = blob__[y+1][x-1]
            else: blob_botleft = []             
            # left
            if x-1>=0: blob_left = blob__[y][x-1]
            else: blob_left = []             
            
            blob_rims_.append([blob_topleft,blob_top,blob_topright,blob_right,
                                 blob_botright,blob_bot,blob_botleft,blob_left])

    return blob_rims_


def one_cycle_operation(pool,blob_,height,width):
    '''
    1 cycle operation, including getting the updated rim blobs and update the rims
    '''
    
    # get each dert rims (8 rims)
    blob_rims_ = get_rim_blob(blob_,height,width)
    # (parallel process) get updated id in each dert
    blob_,dert_ = zip(*pool.starmap(check_rims, zip(blob_,blob_rims_)))
    
    # get map of min id
    id_map_ = [dert.blob_id_min for dert in dert_]
    id_map__ = [id_map_[i:i+width] for i in range(0, len(id_map_), width)]
    id_map_np__ = np.array(id_map__)
    
    return blob_,id_map_np__


def frame_blobs_parallel(dert__):
    '''
    Draft of parallel blobs forming process, consists
    '''

    pool = ThreadPool(mp.cpu_count()) # initialize pool of threads

    height, width = dert__[0].shape # height and width of image
    
    # generate all x and y coordinates
    dert_coord = [[y,x] for x in range(width) for y in range(height)]    
    y_,x_ = zip(*[[y,x] for y,x in dert_coord])

    # get each non class instance dert from coordinates
    dert_ = [dert__[:,y,x] for y,x in dert_coord] 
    
    # (parallel process) generate instance of derts and blobs from their class
    blob_,dert_ = zip(*pool.starmap(generate_blobs, zip(dert_,y_,x_)))
    
    cycle_count = 0 # count of cycle
    id_map_np_prior___ = np.zeros((height,width)) # prior id_map, to check when to stop iteration 
    f_cycle = 1 # flag to continue cycle, 0 = stop, 1 = continue
    
    ## 1st cycle ##
    blob_,id_map_np__ =  one_cycle_operation(pool, blob_,height,width)
    
    # save output image
    cv2.imwrite("./images/parallel/id_cycle_0.png", ((np.fliplr(np.rot90(np.array(id_map_np__),3))*255)/(width*height)).astype('uint8'))
        
    
    while f_cycle:
        
        print("Running cycle "+str(cycle_count+1))
        
        id_map_np_prior___ = id_map_np__ # update prior id map
        
        ## consecutive cycles ##
        blob_,id_map_np__ =  one_cycle_operation(pool, blob_,height,width)
        
        # check whether there is any change in id
        dif = id_map_np__ - id_map_np_prior___
        
        # if there is no change in ids, stop the iteration
        if (np.sum(dif) == 0):
            f_cycle = 0
        
        # save image
        cv2.imwrite("./images/parallel/id_cycle_" + str(cycle_count+1) + ".png", ((np.fliplr(np.rot90(np.array(id_map_np__),3))*255)/(width*height)).astype('uint8'))
        
        # increase interation count
        cycle_count +=1
        
    print("total cycle= "+str(cycle_count))

    # close pool of threads
    pool.close()
    pool.join()
    
    return np.fliplr(np.rot90(np.array(id_map_np__)))


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])

    start_time = time()

    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation

    id_map__ = frame_blobs_parallel(dert__)
    
    end_time = time() - start_time
    print("time elapsed = "+str(end_time))
