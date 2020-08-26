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

ave = 30  # filter or hyper-parameter, set as a guess, latter adjusted by feedback

    
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
    dert_coord__check = set
    root_dert__ = object
    adj_blobs = list
    fopen = bool

    
def generate_blobs(dert,y,x):
    '''
    generate blob based on given dert and their location
    '''
    
    blob = CBlob(I=dert[0], G=dert[1] - ave,Dy=dert[2], Dx=dert[3],
                 dert_coord_ = [[y, x]], sign=dert[1] - ave > 0)
    
    return blob
       



def frame_blobs_parallel(dert__):
    '''
    grow blob.dert__ by merge_blobs, where remaining vs merged blobs are prior in y(x.
    merge after extension: merged blobs may overlap with remaining blob, check and remove redundant derts
    '''

    height, width = dert__[0].shape # height and width of image
    
    # generate all x and y coordinates
    dert_coord = [[y,x] for x in range(width) for y in range(height)]
    
    # initialize pool of threads
    pool = ThreadPool(mp.cpu_count())
    
    # get derts and their x ,y location
    dert_ = [dert__[:,y,x] for y,x in dert_coord]
    y_ = [y for y,x in dert_coord]
    x_ = [x for y,x in dert_coord]
    
    ## Non Parallel method to generate blobs
    start_time = time()
    blob_ = []
    for dert,y,x in zip(dert_,y_,x_):
        blob_.append(generate_blobs(dert,y,x))
    id_map_ = [blob.id for blob in blob_ ]
    end_time = time() - start_time
    print("Non parallel time: "+str(end_time))

    ## Parallel method to generate blobs
    start_time = time()
    blob_ = pool.starmap(generate_blobs, zip(dert_,y_,x_))
    id_map_ = [blob.id for blob in blob_ ]
    end_time = time() - start_time
    print("Parallel time: "+str(end_time))

    # get 2D blob and id map
    blob__ = [blob_[i:i+width] for i in range(0, len(blob_), width)]
    id_map__ = [id_map_[i:i+width] for i in range(0, len(id_map_), width)]
    sign_map__ = [[None]*width for x in range(height)]

    ## Non Parallel method to merge blobs
    loop = 1
    while loop:
        
        sum_f = 0
        for y,x in dert_coord: 
            f_not_match = merge_blobs(blob__,id_map__,sign_map__,height,width,y,x)
            sum_f += f_not_match
            
        if sum_f == 0:
            loop = 0
        print('Number of blob merging operation = '+str(sum_f))      
     
     ## Non Parallel method to merge blobs 
     ## work in progress
#    loop = 1
#    while loop:   
#        sum_f = 0
#        r=[pool.apply_async(merge_blobs(blob__,dert_coord,id_map,height,width,i)) for i in range(len(dert_coord))]
##        sum_f += f_not_match
##        if sum_f == 0:
##                loop = 0
#        print('Number of blob merging operation = '+str(sum_f))   

    pool.close()
    pool.join()
    
    return blob__, id_map__, sign_map__



def merge_blobs(blob__,id_map__,sign_map__,height,width,x,y):
    
    f_not_match = 0
    
    blob = blob__[y][x]
    
    dert_coord_ = []
    if blob:
        sign_map__[y][x] = blob.sign
        for ite, [x,y] in enumerate(blob.dert_coord_):
        

            # determine neighbors' coordinates, 4 for -, 8 for +
            if blob.sign:   # include diagonals
                adj_dert_coords = [(y - 1, x - 1), (y - 1, x),
                                   (y - 1, x + 1), (y, x + 1),
                                   (y + 1, x + 1), (y + 1, x),
                                   (y + 1, x - 1), (y, x - 1)]
            else:
                adj_dert_coords = [(y - 1, x), (y, x + 1),
                                   (y + 1, x), (y, x - 1)]
        
        
            # search through neighboring derts
            for [y2, x2] in adj_dert_coords:
              
                    # check if image boundary is reached
                    if (y2 < 0 or y2 >= height or
                        x2 < 0 or x2 >= width):
                        blob.fopen = True
                        
                    # check if same-signed 
                    elif blob__[y2][x2]:
                        if blob.sign ==  blob__[y2][x2].sign and blob.id != blob__[y2][x2].id:
                            
                            
                            f_not_match = 1
                            
                            sign_map__[y2][x2] = blob.sign
                                                        
                            adj_blob = blob__[y2][x2]
                            
                            id_map__[y2][x2] = blob.id  # add blob ID to each dert
                            
                            dert_coord_.append([y2, x2])  # add dert coordinate to blob
                            blob.I += adj_blob.I
                            blob.G += adj_blob.G
                            blob.Dy += adj_blob.Dy
                            blob.Dx += adj_blob.Dx
                            blob.S += 1
                            
                            # remove the merged blob
                            blob__[y2][x2] = [] 
        
                    # else assign adjacents
                    else:
                        # TODO: assign adjacents
                        pass
                        
                
        blob.dert_coord_ += dert_coord_
        # terminate blob
        y_coords, x_coords = zip(*blob.dert_coord_)
        y0, yn = minmax(y_coords)
        x0, xn = minmax(x_coords)
        blob.box = (
            y0, yn + 1,  # y0, yn
            x0, xn + 1,  # x0, xn
                )
    
    return f_not_match


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default='./images//raccoon_eye.jpeg')
    arguments = vars(argument_parser.parse_args())
    image = imread(arguments['image'])

    start_time = time()

    dert__ = comp_pixel(image)  # 2x2 cross-comparison / cross-correlation

    blob__, id_map__,sign_map__= frame_blobs_parallel(dert__)
    

    from matplotlib import pyplot as plt
    plt.figure();plt.imshow(np.fliplr(np.rot90(np.array(sign_map__),3)))
    
    end_time = time() - start_time
    print(end_time)
