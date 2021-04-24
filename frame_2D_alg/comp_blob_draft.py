'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType
import numpy as np
import cv2

class CderBlob(ClusterStructure):

    blob = object  # core node
    _blob = object # adj blobs
    neg_mB = int
    distance = int
    mB = int
    dB = int
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int

class CBblob(ClusterStructure):

    derBlob = object
    derBlob_ = list
    blob_ = list

ave_mB = 200
ave_rM = .5  # average relative match per input magnitude, at rl=1 or .5?


def cross_comp_blobs(frame):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    blob_ = frame.blob_
    checked_ids_ = []
    derBlobs = [[], []] # index 0 = list of DerBlob, index 1 = list of derBlobs

    for blob in blob_:  # no checking ids, all blobs form unique derBlobs
        DerBlob = CderBlob()
        derBlob_ = []  # DerBlob (including omni_mB), derBlob_

        for adj_blob in blob.adj_blobs[0]:
            if adj_blob.id not in checked_ids_ and not blob.derBlob_:
                # comp between blob, adj_blob
                derBlob = comp_blob(blob, adj_blob)
                accum_DerBlob(DerBlob, derBlob)
                derBlob_.append(derBlob)

        derBlobs[0].append(DerBlob)
        derBlobs[1].append(derBlob_)

    bblob_ = form_bblob_(blob_, derBlobs)  # form blobs of blobs

    visualize_cluster_(bblob_, frame)

    return bblob_

def accum_DerBlob(_derBlob, derBlob):
    # accumulate derBlob
    _derBlob.accumulate(**{param:getattr(derBlob, param) for param in _derBlob.numeric_params})

def comp_blob(_blob, blob):
    '''
    cross compare _blob and blob
    '''
    (_I, _Dy, _Dx, _G, _M, _Dyy, _Dyx, _Dxy, _Dxx, _Ga, _Ma, _Mdx, _Ddx), _A = _blob.Dert.unpack(), _blob.A
    (I, Dy, Dx, G, M, Dyy, Dyx, Dxy, Dxx, Ga, Ma, Mdx, Ddx), A = blob.Dert.unpack(), blob.A

    dI = _I - I  # d is always signed
    mI = min(_I, I)
    dA = _A - A
    mA = min(_A, A)
    dG = _G - G
    mG = min(_G, G)
    dM = _M - M
    mM = min(_M, M)

    mB = mI + mA + mG + mM
    # distance and deviation is computed in form_bblob_: - ave_mB * ave_rM ** (1 + dist / np.sqrt(blob.A))  # average blob match projected at current distance
    dB = dI + dA + dG + dM

    derBlob  = CderBlob(blob=_blob, _blob = blob, mB=mB, dB=dB) # blob is core node, _blob is adjacent blob here

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob


def form_bblob_(blob_, derBlobs):
    '''
    form blob of blobs as a cluster of blobs with positive adjacent derBlob_s, formed by comparing adj_blobs
    '''
    checked_ids = []
    bblob_ = []
    neg_mB = 0

    for blob, DerBlob, derBlob_ in zip(blob_, derBlobs[0], derBlobs[1]): 
        if blob.id not in checked_ids:
            checked_ids.append(blob.id)
            
            bblob = CBblob(derBlob=CderBlob())   # init bblob with previous unconnected blob
            neg_mB += DerBlob.mB                 # accumulate neg_mB
            
            form_bblob_recursive(bblob_, bblob, blob, DerBlob, blob.adj_blobs[0],  blob_, derBlobs, neg_mB, checked_ids)
            bblob_.append(bblob) # append bblob after scan through adjacents recursively
        
    return(bblob_)

# need further optimization on the number of parsing variables
def form_bblob_recursive(bblob_, bblob, blob, DerBlob, adj_blob_,  blob_, derBlobs, neg_mB, checked_ids):
    
    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids:
            checked_ids.append(adj_blob.id)
            
            # tentative, adj_derBlob_ seems not usable here
            adj_DerBlob, adj_derBlob_ = derBlobs[0][blob_.index(adj_blob)] ,derBlobs[1][blob_.index(adj_blob)] # get DerBlob and derblobs of adjacent blob
            
            if (DerBlob.mB>0) == (adj_DerBlob.mB>0):      # check positive and same mB sign
                neg_mB += adj_DerBlob.mB                  # increase neg_mB cost
                derBlob_ = derBlobs[1][blob_.index(blob)] # derBlob_ of current blob
                bblob.blob_.append(blob)                  # pack blob into bblob
                for derBlob in derBlob_: accum_bblob(bblob, derBlob) # accumulate blob's derBlobs into bblob
                
                form_bblob_recursive(bblob_, bblob, adj_blob, adj_DerBlob, adj_blob.adj_blobs[0], blob_, derBlobs, neg_mB, checked_ids) 
             
            elif (blob.Dert.M + neg_mB > ave_mB): # different sign, check ave_mB to search adjacency
                neg_mB += adj_DerBlob.mB # increase neg_mB cost
                form_bblob_recursive(bblob_, bblob, blob, DerBlob, adj_blob.adj_blobs[0], blob_, derBlobs, neg_mB, checked_ids)
            

'''
def form_bblob_recursive(bblob, _blob, blob, checked_ids, compared_blobs):
    
    # As distinct from form_PP_, clustered blobs don't have to be directly adjacent, checking through blob.adj_blobs should be recursive
    
    _Adj_mB = sum([derBlob.mB for derBlob in _blob.derBlob_])
    Adj_mB = sum([derBlob.mB for derBlob in blob.derBlob_])

    if (_Adj_mB>0) and (Adj_mB>0):  # both local-center-blob clusters must be positive
        checked_ids.append(blob.id)

        for derBlob in blob.derBlob_:
            accum_bblob(bblob,derBlob)  # accum same sign node blob into bblob

        compared_blob_ = compared_blobs[1][compared_blobs[0].index(blob)]
        for compared_blob in compared_blob_: # depth first - check potential blob from adjacent cluster of blobs
            if compared_blob.derBlob_ and compared_blob.id not in checked_ids:  # potential element blob
                form_bblob_recursive(bblob, blob, compared_blob, checked_ids, compared_blobs)
'''

def accum_bblob(bblob, derBlob):

    # for debug purpose, on the overflow issue
    print('bblob id = ' +str(bblob.id))
    print('derBlob id = ' +str(derBlob.id))
    print('bblob params : ')
    print(bblob.derBlob)
    print('derBlob params : ')
    print(derBlob)
    print('------------------------------')
    # accumulate derBlob
    bblob.derBlob.accumulate(**{param:getattr(derBlob, param) for param in bblob.derBlob.numeric_params})
    bblob.derBlob_.append(derBlob)

'''
    cross-comp among sub_blobs in nested sub_layers:
    _sub_layer = bblob.sub_layer[0]
    for sub_layer in bblob.sub_layer[1:]:
        for _sub_blob in _sub_layer:
            for sub_blob in sub_layer:
                comp_blob(_sub_blob, sub_blob)
        merge(_sub_layer, sub_layer)  # only for sub-blobs not combined into new bblobs by cross-comp
'''

# draft, to visualize bblob
def visualize_cluster_(bblob_, frame):

    colour_list = []  # list of colours:
    colour_list.append([200, 130, 1])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    colour_list.append([192, 192, 192])  # silver

    # initialization
    ysize, xsize = frame.dert__[0].shape
    blob_img = np.zeros((ysize, xsize,3)).astype('uint8')
    cluster_img = np.zeros((ysize, xsize,3)).astype('uint8')
    img_separator = np.zeros((ysize,3,3)).astype('uint8')

    blob_colour_index = 1
    cluster_colour_index = 1

    cv2.namedWindow('blobs & clusters', cv2.WINDOW_NORMAL)

    for bblob in bblob_:
        if bblob and bblob.derBlob_:
            
            blob = bblob.derBlob_[0].blob
            
            # create mask
            blob_mask = np.zeros_like(frame.dert__[0]).astype('uint8')
            cluster_mask = np.zeros_like(frame.dert__[0]).astype('uint8')

            cy0, cyn, cx0, cxn = blob.box
            # insert index of each blob
            blob_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ * blob_colour_index).astype('uint8')
            blob_colour_index += 1
            # insert index of blob in cluster
            cluster_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ *cluster_colour_index).astype('uint8')

            for derBlob in bblob.derBlob_:
                cy0, cyn, cx0, cxn = derBlob._blob.box
                # insert index of each adjacent blob
                blob_mask[cy0:cyn,cx0:cxn] += (~derBlob._blob.mask__ * blob_colour_index).astype('uint8')
                blob_colour_index += 1
                # insert index of each adjacent blob in cluster
                cluster_mask[cy0:cyn,cx0:cxn] += (~derBlob._blob.mask__ * cluster_colour_index).astype('uint8')

            # increase colour index
            cluster_colour_index += 1

            # insert colour of blobs
            for i in range(1,blob_colour_index):
                blob_img[np.where(blob_mask == i)] = colour_list[i % 10]

            # insert colour of clusters
            for i in range(1,cluster_colour_index):
                cluster_img[np.where(cluster_mask == i)] = colour_list[i % 10]

            # combine images for visualization
            img_concat = np.concatenate((blob_img, img_separator,
                                        cluster_img, img_separator), axis=1)

            # plot cluster of blob
            cv2.imshow('blobs & clusters',img_concat)
            cv2.resizeWindow('blobs & clusters', 1920, 720)
            cv2.waitKey(100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



'''
# draft, to visualize the cluster of blobs, next is to visualize the merged clusters
def visualize_cluster_(blob_, derBlobs, frame):

    colour_list = []  # list of colours:
    colour_list.append([200, 130, 1])  # blue
    colour_list.append([75, 25, 230])  # red
    colour_list.append([25, 255, 255])  # yellow
    colour_list.append([75, 180, 60])  # green
    colour_list.append([212, 190, 250])  # pink
    colour_list.append([240, 250, 70])  # cyan
    colour_list.append([48, 130, 245])  # orange
    colour_list.append([180, 30, 145])  # purple
    colour_list.append([40, 110, 175])  # brown
    colour_list.append([192, 192, 192])  # silver

    # initialization
    ysize, xsize = frame.dert__[0].shape
    blob_img = np.zeros((ysize, xsize,3)).astype('uint8')
    cluster_img = np.zeros((ysize, xsize,3)).astype('uint8')
    img_separator = np.zeros((ysize,3,3)).astype('uint8')

    blob_colour_index = 1
    cluster_colour_index = 1

    cv2.namedWindow('blobs & clusters', cv2.WINDOW_NORMAL)

    for blob, (DerBlob, derBlob_) in zip(blob_, derBlobs):
        if derBlob_:
            # create mask
            blob_mask = np.zeros_like(frame.dert__[0]).astype('uint8')
            cluster_mask = np.zeros_like(frame.dert__[0]).astype('uint8')

            cy0, cyn, cx0, cxn = blob.box
            # insert index of each blob
            blob_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ * blob_colour_index).astype('uint8')
            blob_colour_index += 1
            # insert index of blob in cluster
            cluster_mask[cy0:cyn,cx0:cxn] += (~blob.mask__ *cluster_colour_index).astype('uint8')

            for derBlob in derBlob_:
                cy0, cyn, cx0, cxn = derBlob._blob.box
                # insert index of each adjacent blob
                blob_mask[cy0:cyn,cx0:cxn] += (~derBlob._blob.mask__ * blob_colour_index).astype('uint8')
                blob_colour_index += 1
                # insert index of each adjacent blob in cluster
                cluster_mask[cy0:cyn,cx0:cxn] += (~derBlob._blob.mask__ * cluster_colour_index).astype('uint8')

            # increase colour index
            cluster_colour_index += 1

            # insert colour of blobs
            for i in range(1,blob_colour_index):
                blob_img[np.where(blob_mask == i)] = colour_list[i % 10]

            # insert colour of clusters
            for i in range(1,cluster_colour_index):
                cluster_img[np.where(cluster_mask == i)] = colour_list[i % 10]

            # combine images for visualization
            img_concat = np.concatenate((blob_img, img_separator,
                                        cluster_img, img_separator), axis=1)

            # plot cluster of blob
            cv2.imshow('blobs & clusters',img_concat)
            cv2.resizeWindow('blobs & clusters', 1920, 720)
            cv2.waitKey(100)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

def comp_blob_recursive(blob, adj_blob_, checked_ids_, neg_mB, neg_dist, compared_blob_, compared_blobs):
    '''
    Not used.  called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''
    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_ and not adj_blob.derBlob_: # adj blob not checked and is not the head node
            checked_ids_.append(adj_blob.id)

            dist = neg_dist + np.sqrt(blob.A) / 2 + np.sqrt(adj_blob.A)    # approximated euclidean distance to adj_adj_blobs
            derBlob = comp_blob(blob, adj_blob)  # cross compare blob and adjacent blob, distance should be computed in form_bblobs

            if derBlob.mB>0: # positive mB, stop the searching with prior blob and start with current adj blob
                neg_dist = 0; neg_mB = 0; adj_compared_blob_ = []  # compared to current blob, reinitialized for each new blob
                comp_blob_recursive(adj_blob, adj_blob.adj_blobs[0], checked_ids_, neg_mB, neg_dist, adj_compared_blob_, compared_blobs)
                compared_blobs[0].append(adj_blob)
                compared_blobs[1].append(adj_compared_blob_)  # nested list

            else: # negative mB, continue searching
                blob.derBlob_.append(derBlob)
                neg_dist += dist
                neg_mB += derBlob.mB  # mB accumulated over comparison scope

                # is there any backward match here?
                if blob.Dert.M + neg_mB > ave_mB:  # extend search to adjacents of adjacent, depth-first
                    comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, neg_mB, neg_dist, compared_blob_, compared_blobs)

                else:  # compared adj blobs are potential bblob elements
                    for adj_adj_blob in adj_blob.adj_blobs[0]:
                        if adj_adj_blob not in compared_blob_:
                            compared_blob_.append(adj_adj_blob)  # potential bblob element

