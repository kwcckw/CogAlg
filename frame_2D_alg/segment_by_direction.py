from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
#from intra_blob import accum_blob_Dert
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import slice_blob

flip_ave = 10
ave_dir_val = 500
ave_M = -500 # i think we should use negative ave M? Since high G blob should have high negative M value

def segment_by_direction(blob, verbose=False):  # draft

    dert__ = list(blob.dert__)
    mask__ = blob.mask__
    merged_blob_ = []
    weak_dir_blob_ = []
    dy__ = dert__[1]; dx__ = dert__[2]
    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:

    dir_blob_, idmap, adj_pairs = flood_fill(dert__, dy__>dx__, verbose=False, mask__=mask__, blob_cls=CBlob, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    for dir_blob in dir_blob_:
        if not dir_blob.fmerged:
            merge_blobs_recursive(dir_blob, merged_blob_, weak_dir_blob_, fmerged=0)

    return merged_blob_, weak_dir_blob_  # merged blobs may or may not be sliced


def merge_blobs_recursive(blob, merged_blob_, weak_dir_blob_, fmerged):

    if blob.Dx == 0: blob.Dx = 1 # solve zero division
    if abs(blob.G * ( blob.Dy / (blob.Dx))) > ave_dir_val:  # direction strength eval
        if (blob.M > ave_M) and (blob.box[1]-blob.box[0]>1): # y size >1, otherwise pointless since we cannot form derP
            blob.fsliced = 1
            merged_blob_.append(slice_blob(blob))  # slice across directional sub-blob
        else:
            merged_blob_.append(blob)  # returned blob is not sliced
            
    elif fmerged: # dir blob is merged previously but still weak, pack them to weak_dir_blob_
        weak_dir_blob_.append(blob) 
        
    else: # merge weak dir blob
        merge_adjacents_recursive(blob, blob.adj_blobs)  # merge dert__ and accumulate params in blob
        merge_blobs_recursive(blob, merged_blob_, weak_dir_blob_, fmerged=1) # eval direction again after merging
           
        
def merge_adjacents_recursive(blob, adj_blobs):

    for adj_blob, pose in blob.adj_blobs[0]:  # sub_blob.adj_blobs = [ [[adj_blob1, pose1],[adj_blob2, pose2]], A, G, M, Ga]
        if not adj_blob.fmerged:  # potential merging blob
            if adj_blob.Dx == 0 : adj_blob.Dx = 1 # solve zero division
            if abs(adj_blob.G * ( adj_blob.Dy / (adj_blob.Dx))) <= ave_dir_val:

                if adj_blob not in blob.merged_blob_:  # and adj_blob not in merged_blob_: it can't be, one pass? This is to prevent we check again the merged blob in main loop
                    adj_blob.fmerged = 1
                    blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
                    merge_adjacents_recursive(blob, adj_blob.adj_blobs)


def merge_blobs(blob, adj_blob):
    # merge adj_blob into blob

    # accumulate params
    # do we need to merge those params such as rdn, rng Ls and sub_layers?
    blob.accumulate(I=adj_blob.I, Dy=adj_blob.Dy, Dx=adj_blob.Dx, G=adj_blob.G, M=adj_blob.M,
                    Dyy=adj_blob.Dyy, Dyx=adj_blob.Dyx, Dxy=adj_blob.Dxy, Dxx=adj_blob.Dxx,
                    Ga=adj_blob.Ga, Ma=adj_blob.Ma, A=adj_blob.A)
    
    blob.merged_blob_ += adj_blob.merged_blob_
    
    # y0, yn, x0, xn for common box between blob and adj blob
    y0 = min([blob.box[0],adj_blob.box[0]])
    yn = max([blob.box[1],adj_blob.box[1]])
    x0 = min([blob.box[2],adj_blob.box[2]])
    xn = max([blob.box[3],adj_blob.box[3]])

    # offset of x0 and y0 from common box
    y0_offset = blob.box[0]-y0
    x0_offset = blob.box[2]-x0
    adj_y0_offset = adj_blob.box[0]-y0
    adj_x0_offset = adj_blob.box[2]-x0

    # create extended mask from common box
    extended_mask__ = np.ones((yn-y0,xn-x0)).astype('bool')
    for y in range(blob.mask__.shape[0]): # blob mask
        for x in range(blob.mask__.shape[1]):
            if not blob.mask__[y,x]: # replace when mask = false
                extended_mask__[y+y0_offset,x+x0_offset] = blob.mask__[y,x]
    for y in range(adj_blob.mask__.shape[0]): # adj_blob mask
        for x in range(adj_blob.mask__.shape[1]):
            if not adj_blob.mask__[y,x]: # replace when mask = false
                extended_mask__[y+adj_y0_offset,x+adj_x0_offset] = adj_blob.mask__[y,x]
    
    # create extended derts from common box
    extended_dert__ = [np.zeros((yn-y0,xn-x0)) for _ in range(len(blob.dert__))]
    for i in range(len(blob.dert__)):
        for y in range(blob.dert__[i].shape[0]): # blob derts
            for x in range(blob.dert__[i].shape[1]):
                if not blob.mask__[y,x]: # replace when mask = false
                    extended_dert__[i][y+y0_offset,x+x0_offset] = blob.dert__[i][y,x]
        for y in range(adj_blob.dert__[i].shape[0]): # adj_blob derts
            for x in range(adj_blob.dert__[i].shape[1]):
                if not adj_blob.mask__[y,x]: # replace when mask = false
                    extended_dert__[i][y+adj_y0_offset,x+adj_x0_offset] = adj_blob.dert__[i][y,x]
    
    # update dert, mask and box
    blob.dert__ = extended_dert__
    blob.mask__ = extended_mask__
    blob.box = [y0,yn,x0,xn]
 
    return blob

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