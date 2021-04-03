from collections import deque
import sys
import numpy as np
from class_cluster import ClusterStructure, NoneType
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import slice_blob

flip_ave = 10
ave_dir_val = 1000
ave_M = -500 # i think we should use negative ave M? Since high G blob should have high negative M value

def segment_by_direction(blob, verbose=False):  # draft

    dert__ = list(blob.dert__)
    mask__ = blob.mask__
    dy__ = dert__[1]; dx__ = dert__[2]
    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:

    dir_blob_, idmap, adj_pairs = flood_fill(dert__, dy__>dx__, verbose=False, mask__=mask__, blob_cls=CBlob, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    blob.dir_blobs = merge_blobs_recursive(dir_blob_,fweak=0)

    for blob in blob.dir_blobs:
        if (blob.Dert.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
            blob.fsliced = True
            slice_blob(blob,verbose)  # slice and comp_slice_ across directional sub-blob

def merge_blobs_recursive(dir_blob_,fweak):

    merged_blob_ = []
    new_weak_merged_blob_ = []

    for blob in dir_blob_:
        
        if fweak:
            # need to check further here, looks like there is empty adjacent blob, which should not be happened
            if blob.adj_blobs[0]: # blob is merged but weak, force it to merge into adjacent blobs, and stop when merged onto strong direction blob
                blob.fmerged = 0 # reset for merging again
                merged_blob = merge_adjacents_recursive(blob, blob.adj_blobs, fmerge=1, adj_list_=[])
            
        elif not blob.fmerged:  # blob evaluation is done in merge_adjacents
            merged_blob = merge_adjacents_recursive(blob, blob.adj_blobs, fmerge=0, adj_list_=[])  # returned blob should always be merged

            rD = merged_blob.Dert.Dy / merged_blob.Dert.Dx if merged_blob.Dert.Dx else 2 * merged_blob.Dert.Dy
            if abs(merged_blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
                new_weak_merged_blob_.append(merged_blob)
            else:
                merged_blob_.append(merged_blob)
                
    # old weak_merged_blobs are not recycled, they should always be merged in merge_adjacents:
    if new_weak_merged_blob_:
        merged_blob_ = merge_blobs_recursive(new_weak_merged_blob_,fweak=1)  # eval direction again after merging

    return merged_blob_  # merged blobs may or may not be sliced


def merge_adjacents_recursive(blob, adj_blobs, fmerge, adj_list_):

    rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
    if abs(blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
        _fweak = 1
    else: _fweak = 0

    for adj_blob, pose in adj_blobs[0]:  # sub_blob.adj_blobs = [ [[adj_blob1, pose1],[adj_blob2, pose2]], A, G, M, Ga]
        
        rD = adj_blob.Dert.Dy / adj_blob.Dert.Dx if adj_blob.Dert.Dx else 2*adj_blob.Dert.Dy
        
        if fmerge and not blob.fmerged: # force merging of weak blobs until all weak blobs are merged into strong blob
            if abs(adj_blob.Dert.G * rD) >= ave_dir_val: # merge weak blob to strong blob
                blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
                blob.fmerged=True
                break
            elif (adj_blob is not blob) and (adj_blob not in adj_list_):
                adj_list_.append(adj_blob) # adjacent list contains all the checked adjacent blobs
                merge_adjacents_recursive(blob, adj_blob.adj_blobs,fmerge=1, adj_list_=adj_list_) # continue search adj'adj blobs until strongblob is found, but this method is expensive
    
        elif not adj_blob.fmerged:  # potential merging blob
            if abs(adj_blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
                fweak = 1
            else: fweak = 0

            if _fweak or fweak:  # if either blob or adj_blob are weak, they should be merged
                adj_blob.fmerged = True
                blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
                merge_adjacents_recursive(blob, adj_blob.adj_blobs,fmerge=0, adj_list_=[])
        
            # we cannot remove adjacent blob if we want to remerge those weak blobs to strong blob
            # blob.adj_blobs[0] = []  # remove adj blobs after merging
            blob.fmerged = True  # always true after checking all adjacents

    return blob

def merge_blobs(blob, adj_blob):  # merge adj_blob into blob

    # accumulate blob Dert
    blob.accumulate(**{param:getattr(adj_blob.Dert, param) for param in adj_blob.Dert.numeric_params})

    # y0, yn, x0, xn for combined blob and adj blob box
    y0 = min([blob.box[0],adj_blob.box[0]])
    yn = max([blob.box[1],adj_blob.box[1]])
    x0 = min([blob.box[2],adj_blob.box[2]])
    xn = max([blob.box[3],adj_blob.box[3]])
    # offsets from combined box
    y0_offset = blob.box[0]-y0
    x0_offset = blob.box[2]-x0
    adj_y0_offset = adj_blob.box[0]-y0
    adj_x0_offset = adj_blob.box[2]-x0

    # create extended mask from combined box
    extended_mask__ = np.ones((yn-y0,xn-x0)).astype('bool')
    for y in range(blob.mask__.shape[0]): # blob mask
        for x in range(blob.mask__.shape[1]):
            if not blob.mask__[y,x]: # replace when mask = false
                extended_mask__[y+y0_offset,x+x0_offset] = blob.mask__[y,x]
    for y in range(adj_blob.mask__.shape[0]): # adj_blob mask
        for x in range(adj_blob.mask__.shape[1]):
            if not adj_blob.mask__[y,x]: # replace when mask = false
                extended_mask__[y+adj_y0_offset,x+adj_x0_offset] = adj_blob.mask__[y,x]

    # create extended derts from combined box
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
    blob.Dert.I += dert__[0][y, x]
    blob.Dert.Dy += dert__[1][y, x]
    blob.Dert.Dx += dert__[2][y, x]
    blob.Dert.G += dert__[3][y, x]
    blob.Dert.M += dert__[4][y, x]

    if len(dert__) > 5:  # past comp_a fork

        blob.Dert.Dyy += dert__[5][y, x]
        blob.Dert.Dyx += dert__[6][y, x]
        blob.Dert.Dxy += dert__[7][y, x]
        blob.Dert.Dxx += dert__[8][y, x]
        blob.Dert.Ga += dert__[9][y, x]
        blob.Dert.Ma += dert__[10][y, x]