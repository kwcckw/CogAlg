'''
- Segment input blob into dir_blobs by primary direction of kernel gradient: dy>dx
- Merge weakly directional dir_blobs, with dir_val < cost of comp_slice_
- Evaluate merged blobs for comp_slice_: if blob.M > ave_M
'''

import numpy as np
from frame_blobs import CBlob, flood_fill, assign_adjacents
from comp_slice_ import slice_blob
import cv2

flip_ave = 10
ave_dir_val = 50
ave_M = -500  # high negative ave M for high G blobs

def segment_by_direction(iblob, verbose=False):

    dert__ = list(iblob.dert__)
    mask__ = iblob.mask__
    dy__ = dert__[1]; dx__ = dert__[2]

    # segment blob into primarily vertical and horizontal sub blobs according to the direction of kernel-level gradient:
    dir_blob_, idmap, adj_pairs = \
        flood_fill(dert__, abs(dy__) > abs(dx__), verbose=False, mask__=mask__, blob_cls=CBlob, fseg=True, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    for i, blob in enumerate(dir_blob_):
        
        blob = merge_adjacents_recursive(blob, blob.adj_blobs)
        if blob: 
            if (blob.Dert.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
                blob.fsliced = True
                slice_blob(blob,verbose)  # slice and comp_slice_ across directional sub-blob
            iblob.dir_blobs.append(blob)

        # To visualizate merging process
        cv2.namedWindow('gray=weak, white=strong', cv2.WINDOW_NORMAL)
        img_blobs   = mask__.copy().astype('float')*0
        # draw weak blobs
        for dir_blob in dir_blob_:
            fweak = directionality_eval(dir_blob) # direction eval on the blob
            y0,yn,x0,xn = dir_blob.box
            if fweak:
                img_blobs[y0:yn,x0:xn] += ((~dir_blob.mask__) * 90) .astype('uint8')
            else:
                img_blobs[y0:yn,x0:xn] += ((~dir_blob.mask__) * 255) .astype('uint8')
        # draw strong blobs
        img_mask = np.ones_like(mask__).astype('bool')
        for dir_blob in iblob.dir_blobs:
            fweak = directionality_eval(dir_blob) # direction eval on the blob
            y0,yn,x0,xn = dir_blob.box
            if ~fweak:
                img_mask[y0:yn,x0:xn] = np.logical_and(img_mask[y0:yn,x0:xn], dir_blob.mask__)
        img_blobs += ((~img_mask).astype('float')*255)
        img_blobs[img_blobs>255] = 255
        cv2.imshow('gray=weak, white=strong',img_blobs.astype('uint8'))
        cv2.resizeWindow('gray=weak, white=strong', 640, 480)
        cv2.waitKey(50)          
    cv2.destroyAllWindows()

def merge_adjacents_recursive(iblob, adj_blobs):

    blob = iblob
    # remove current blob reference in adj' adj blobs, since adj blob are assigned bilaterally
    if blob in adj_blobs[0]: adj_blobs[0].remove(blob)

    _fweak = directionality_eval(blob) # direction eval on the input blob
    adj_blob_list_ = [[],[]]
    for (adj_blob,pose) in zip(*adj_blobs):  # adj_blobs = [ [adj_blob1,adj_blob2], [pose1,pose2] ]
        
        fweak = directionality_eval(adj_blob) # direction eval on the adjacent blob
        if fweak:  # adj blob is weak, merge adj blob to blob, blob could be weak or strong
            blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params 
            if pose != 1: # if adjacent is not internal
                for i,adj_adj_blob in enumerate(adj_blob.adj_blobs[0]):
                    if adj_adj_blob not in adj_blob_list_[0]:
                        adj_blob_list_[0].append(adj_blob.adj_blobs[0][i]) # add adj adj_blobs to search list if they are merged
                        adj_blob_list_[1].append(adj_blob.adj_blobs[1][i])    
                
        elif _fweak: # blob is weak but adj blob is strong, merge blob to adj blob
            blob = merge_blobs(adj_blob, blob)  # merge dert__ and accumulate params

        _fweak = directionality_eval(blob) # direction eval again on the merged blob 

    # if merged blob is still weak，  continue searching and merging with the merged blob's adj blobs
    # else they will stop merging adjacent blob
    if _fweak and adj_blob_list_[0]:
        blob = merge_adjacents_recursive(blob, adj_blob_list_) 

    if blob is iblob: # return only if current blob is strong blob, else the blob is merged to other blob, no need to return     
        return blob


def directionality_eval(blob):
    
    rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
    if abs(blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
        fweak = 1
    else: fweak = 0
    
    return fweak
    
    
def merge_blobs(blob, adj_blob):  # merge adj_blob into blob

    '''
    added AND operation for the merging process, still need to be optimized further
    '''
    
    # 0 = overlap between blob and adj_blob
    # 1 = overlap and blob is in adj blob
    # 2 = overlap and adj blob is in blob
    floc = 0
    
    # accumulate blob Dert
    blob.accumulate(**{param:getattr(adj_blob.Dert, param) for param in adj_blob.Dert.numeric_params})
    
    
    _y0, _yn, _x0, _xn = blob.box
    y0, yn, x0, xn = adj_blob.box
   
    if (_y0<=y0) and (_yn>=yn) and (_x0<=x0) and (_xn>=xn): # adj blob is inside blob
        floc = 2 
        cy0, cyn, cx0, cxn =  blob.box # y0, yn, x0, xn for combined blob is blob box
    elif (y0<=_y0) and (yn>=_yn) and (x0<=_x0) and (xn>=_xn): # blob is inside adj blob
        floc = 1
        cy0, cyn, cx0, cxn =  adj_blob.box # y0, yn, x0, xn for combined blob is adj blob box   
    else:
        # y0, yn, x0, xn for combined blob and adj blob box
        cy0 = min([blob.box[0],adj_blob.box[0]])
        cyn = max([blob.box[1],adj_blob.box[1]])
        cx0 = min([blob.box[2],adj_blob.box[2]])
        cxn = max([blob.box[3],adj_blob.box[3]])
        
    # offsets from combined box
    y0_offset = blob.box[0]-cy0
    x0_offset = blob.box[2]-cx0
    adj_y0_offset = adj_blob.box[0]-cy0
    adj_x0_offset = adj_blob.box[2]-cx0


    if floc == 1:  # blob is inside adj blob
        # extended mask is adj blob's mask, AND extended mask with blob mask
        extended_mask__ = adj_blob.mask__
        extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)] = \
        np.logical_and(blob.mask__, extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)])
        # if blob is inside adj blob, blob derts should be already in adj blob derts
        extended_dert__ = adj_blob.dert__
    
    elif floc == 2: # adj blob is inside blob
        # extended mask is blob's mask, AND extended mask with adj blob mask
        extended_mask__ = blob.mask__
        extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = \
        np.logical_and(adj_blob.mask__, extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)])
        # if adj blob is inside blob, adj blob derts should be already in blob derts
        extended_dert__ = blob.dert__
        
    else:
        # create extended mask from combined box
        extended_mask__ = np.ones((cyn-cy0,cxn-cx0)).astype('bool')
        # AND extended mask with blob mask
        extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)] = \
        np.logical_and(blob.mask__, extended_mask__[y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)])
        
        extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = \
        np.logical_and(adj_blob.mask__, extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)])
        
        # AND extended mask with adj blob mask
        extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = \
        np.logical_and(adj_blob.mask__, extended_mask__[adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)])
        
        # create extended derts from combined box
        extended_dert__ = [np.zeros((cyn-cy0,cxn-cx0)) for _ in range(len(blob.dert__))]
        for i in range(len(blob.dert__)):
            extended_dert__[i][y0_offset:y0_offset+(_yn-_y0), x0_offset:x0_offset+(_xn-_x0)] = blob.dert__[i]
            extended_dert__[i][adj_y0_offset:adj_y0_offset+(yn-y0), adj_x0_offset:adj_x0_offset+(xn-x0)] = adj_blob.dert__[i]
            

    # update dert, mask and box
    blob.dert__ = extended_dert__
    blob.mask__ = extended_mask__
    blob.box = [cy0,cyn,cx0,cxn]
                
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
