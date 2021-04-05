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
        flood_fill(dert__, abs(dy__) > abs(dx__), verbose=False, mask__=mask__, blob_cls=CBlob, f8dir=True, accum_func=accum_dir_blob_Dert)
    assign_adjacents(adj_pairs, CBlob)

    # temporary, for visualization purpose
    id_list_ = []
    for dir_blob in dir_blob_:
        id_list_.append(dir_blob.id)
    max_id = max(id_list_)
    min_id = min(id_list_)
    divisor = max_id-min_id
    if max_id-min_id == 0:
        divisor = 0.01;    

    for i, blob in enumerate(dir_blob_):
        blob = merge_adjacents_recursive(blob, blob.adj_blobs)

        if (blob.Dert.M > ave_M) and (blob.box[1]-blob.box[0]>1):  # y size >1, else we can't form derP
            blob.fsliced = True
            slice_blob(blob,verbose)  # slice and comp_slice_ across directional sub-blob

        iblob.dir_blobs.append(blob)

        # temporary, for visualization purpose
        cv2.namedWindow('merging process', cv2.WINDOW_NORMAL)
        img_id = mask__.copy().astype('uint8')*0
        for dir_blob in dir_blob_:
            if isinstance(dir_blob.merged_blob,CBlob):
                dir_blob = dir_blob.merged_blob
            for y in range(dir_blob.mask__.shape[0]): 
                for x in range(dir_blob.mask__.shape[1]):
                    if not dir_blob.mask__[y,x]: # replace when mask = false
                        
                        ind = np.argwhere(np.array(id_list_)==dir_blob.id)
                        if len(ind):
                            img_id[dir_blob.box[0]+y,dir_blob.box[2]+x] = 255-(((dir_blob.id-min_id)/(divisor))*255)
        cv2.imshow('merging process',img_id)
        cv2.resizeWindow('merging process', 640, 480)
        cv2.waitKey(100)            
    cv2.destroyAllWindows()

def merge_adjacents_recursive(iblob, adj_blobs):

    # remove current blob reference in adj' adj blobs, since adj blob are assigned bilaterally
    if iblob in adj_blobs: adj_blobs.remove(iblob)
    blob = iblob
    _fweak = directionality_eval(blob) # direction eval on the input blob
    
    for i,adj_blob in enumerate(adj_blobs[0]):  # adj_blobs = [ [adj_blob1,adj_blob2], [pose1,pose2] ]
        
        if not isinstance(adj_blob.merged_blob,CBlob):  # potential merging blob
            fweak = directionality_eval(adj_blob) # direction eval on the adjacent blob
            # always merge the weak adj blob to blob first
            if fweak:  # adj blob is weak, merge adj blob to blob
                blob = merge_blobs(blob, adj_blob)  # merge dert__ and accumulate params
            elif _fweak: # blob is weak, merge blob to adj blob
                blob = merge_blobs(adj_blob, blob)  # merge dert__ and accumulate params
      
        else: # adjacent blob is merged to other blob, merge current blob to it
            blob = merge_blobs(adj_blob.merged_blob,blob)  # merge dert__ and accumulate params

        _fweak = directionality_eval(blob) # direction eval again on the merged blob
        # if merged blob is still weak and is not the input blob, continue searching and merging with the merged blob's adj blobs
        if _fweak and blob is not iblob:
            merge_adjacents_recursive(blob, blob.adj_blobs) 
            
    return blob


def directionality_eval(blob):
    
    rD = blob.Dert.Dy / blob.Dert.Dx if blob.Dert.Dx else 2 * blob.Dert.Dy
    if abs(blob.Dert.G * rD) < ave_dir_val:  # direction strength eval
        fweak = 1
    else: fweak = 0
    
    return fweak
    
    
def merge_blobs(blob, adj_blob):  # merge adj_blob into blob

    '''
    temporary changes to speed up the merging process, work in progress
    right now the pixel by pixel assignment is extremely slow
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
        # extended mask is adj blob's mask, update blob mask to adj blob mask
        extended_mask__ = adj_blob.mask__
        for y in range(blob.mask__.shape[0]): # blob mask
            for x in range(blob.mask__.shape[1]):
                if not blob.mask__[y,x]: # replace when mask = false
                    extended_mask__[y+y0_offset,x+x0_offset] = blob.mask__[y,x]    
        # extended derts are adj blob's derts, update blob derts to adj blob derts
        extended_dert__ = adj_blob.dert__
        for i in range(len(blob.dert__)):
            for y in range(blob.dert__[i].shape[0]): # blob derts
                for x in range(blob.dert__[i].shape[1]):
                    if not blob.mask__[y,x]: # replace when mask = false
                        extended_dert__[i][y+y0_offset,x+x0_offset] = blob.dert__[i][y,x]
        
    elif floc == 2: # adj blob is inside blob
        # extended mask is blob's mask, update adj blob mask to blob mask
        extended_mask__ = blob.mask__
        for y in range(adj_blob.mask__.shape[0]): # adj_blob mask
            for x in range(adj_blob.mask__.shape[1]):
                if not adj_blob.mask__[y,x]: # replace when mask = false
                    extended_mask__[y+adj_y0_offset,x+adj_x0_offset] = adj_blob.mask__[y,x]
        # extended derts are blob's derts, update adj blob derts to blob derts
        extended_dert__ = blob.dert__
        for i in range(len(adj_blob.dert__)):
            for y in range(adj_blob.dert__[i].shape[0]): # adj_blob derts
                for x in range(adj_blob.dert__[i].shape[1]):
                    if not adj_blob.mask__[y,x]: # replace when mask = false
                        extended_dert__[i][y+adj_y0_offset,x+adj_x0_offset] = adj_blob.dert__[i][y,x]
    else:
        # create extended mask from combined box
        extended_mask__ = np.ones((cyn-cy0,cxn-cx0)).astype('bool')
        for y in range(blob.mask__.shape[0]): # blob mask
            for x in range(blob.mask__.shape[1]):
                if not blob.mask__[y,x]: # replace when mask = false
                    extended_mask__[y+y0_offset,x+x0_offset] = blob.mask__[y,x]
        for y in range(adj_blob.mask__.shape[0]): # adj_blob mask
            for x in range(adj_blob.mask__.shape[1]):
                if not adj_blob.mask__[y,x]: # replace when mask = false
                    extended_mask__[y+adj_y0_offset,x+adj_x0_offset] = adj_blob.mask__[y,x]
    
        # create extended derts from combined box
        extended_dert__ = [np.zeros((cyn-cy0,cxn-cx0)) for _ in range(len(blob.dert__))]
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
    blob.box = [cy0,cyn,cx0,cxn]
    adj_blob.merged_blob = blob # update merged blob reference

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
