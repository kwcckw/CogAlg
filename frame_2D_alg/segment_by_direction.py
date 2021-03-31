from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
#from intra_blob import accum_blob_Dert
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import*

flip_ave = 10
ave_dir_val_dert = 1
ave_dir_val_blob = 1


def segment_by_direction(blob, verbose=False):


    flip_eval_blob(blob) # initial flip eval

    dert__ = list(blob.dert__)
    mask__ = blob.mask__
    
    g__  = dert__[3]
    dy__ = dert__[1]
    dx__ = dert__[2]
    dx__[dx__==0] = 1  # solve 0 zero division issue
    dir__ = abs(g__)*(dy__/dx__) # please suggest a better name
    
    dert__.append(dir__) # add Dir as additional param to blob
    
    # segment by direction
    sub_blob_, idmap, adj_pairs = flood_fill(dert__, dir__, verbose=False, mask__=mask__, blob_cls=CBlob, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)
    
    for sub_blob in sub_blob_:
        if sub_blob.Dir >ave_dir_val_blob: # dir blob
          
            flip_eval_blob(sub_blob) # flip eval sub blobs?
            slice_blob(sub_blob)     # slice dir sub blob
            
        else: # weak dir blob 
            # merge the weak dir blobs. Append them to a list or merge them?
            #merge_blob_recursive(sub_blob)
            pass
            
# draft, should be a better way to do this
def merge_blob_recursive(sub_blob):
    # sub_blob.adj_blobs = [ [[adj_blob1, pose1],[adj_blob2, pose2]], A, G, M, Ga]
    for adj_blob, pose in sub_blob.adj_blobs[0]:
        if (adj_blob.Dir <= ave_dir_val_blob) and not (adj_blob.fmerge): # potential merging blob
            if sub_blob not in adj_blob.merge_blob_ and adj_blob not in sub_blob.merge_blob_:
                adj_blob.fmerge = 1
                sub_blob.merge_blob_.append(adj_blob)
                
                merge_blob_recursive(adj_blob)
                
                for merge_blob in adj_blob.merge_blob_:
                    if merge_blob not in sub_blob.merge_blob_:
                        sub_blob.merge_blob_.append(merge_blob)



def flip_eval_blob(blob):

    # L_bias (Lx / Ly) * G_bias (Gy / Gx), blob.box = [y0,yn,x0,xn], ddirection: preferential comp over low G
    horizontal_bias = (blob.box[3] - blob.box[2]) / (blob.box[1] - blob.box[0])  \
                    * (abs(blob.Dy) / abs(blob.Dx))

    if horizontal_bias > 1 and (blob.G * blob.Ma * horizontal_bias > flip_ave / 10):
        blob.fflip = 1  # rotate 90 degrees for scanning in vertical direction
        # swap blob Dy and Dx:
        Dy=blob.Dy; blob.Dy = blob.Dx; blob.Dx = Dy
        # rotate dert__:
        blob.dert__ = tuple([np.rot90(dert) for dert in blob.dert__])
        blob.mask__ = np.rot90(blob.mask__)
        # swap dert dys and dxs:
        blob.dert__ = list(blob.dert__)  # convert to list since param in tuple is immutable
        blob.dert__[1], blob.dert__[2] = \
        blob.dert__[2], blob.dert__[1]
        
        
def accum_dir_blob_Dert(blob, dert__, y, x):
    blob.I += dert__[0][y, x]
    blob.Dy += dert__[1][y, x]
    blob.Dx += dert__[2][y, x]
    blob.G += dert__[3][y, x]
    blob.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork

        blob.Dyy += dert__[5][y, x]
        blob.Dyx += dert__[6][y, x]
        blob.Dxy += dert__[7][y, x]
        blob.Dxx += dert__[8][y, x]
        blob.Ga += dert__[9][y, x]
        blob.Ma += dert__[10][y, x]
        blob.Dir += dert__[11][y,x]

