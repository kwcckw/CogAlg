'''
Cross-compare blobs with incremental adjacency, within a frame
'''

from class_cluster import ClusterStructure, NoneType

class CderBlob(ClusterStructure):

    _blob = object # not necessary if we are gonna use blob.adj_blobs to get adjacent derBlob
    blob = object
    dI = int
    mI = int
    dA = int
    mA = int
    dG = int
    mG = int
    dM = int
    mM = int
    mB = int
    dB = int
    sub_derBlob = list  # why sub, this should already be packed in sub_blobs? Yes, this is meant for form derBlob and bblob recursively with the sub blobs, i guess now is not needed?

class CBblob(ClusterStructure):

    derBlob = object
    derBlob_ = list


ave_M = 100
ave_mP = 100
ave_dP = 100


def cross_comp_blobs(blob_):
    '''
    root function of comp_blob: cross compare blobs with their adjacent blobs in frame.blob_, including sub_layers
    '''
    
    
    for blob in blob_:   
        checked_ids_ = [blob.id]; net_M = 0  # checked ids per blob: may be checked from multiple root blobs
        comp_blob_recursive(blob, blob.adj_blobs[0], checked_ids_, net_M)  # comp between blob, blob.adj_blobs

    bblob_ = form_bblob_(blob_)

    print(str(len(bblob_)+" cluster of blobs are formed."))
    return bblob_


def form_bblob_(blob_):
    '''
    form bblob with same sign derBlob
    '''
       
    checked_ids = []
    bblob_ = []
    
    for blob in blob_:
        if blob.derBlob_ and blob.id not in checked_ids: # blob is head node of cluster
            checked_ids.append(blob.id)

            bblob = CBblob(derBlob=CderBlob()) # init new bblob
            for derBlob in blob.derBlob_:
                accum_bblob(bblob,derBlob)        # accum derBlobs into bblob
        
            for pot_blob in blob.pot_blob_:    # depth first - check potential blob from adjacent cluster of blobs
                if pot_blob.derBlob_ and pot_blob.id not in checked_ids: # potential blob is head node of cluster and not checked before
                    form_bblob_recursive(bblob, blob, pot_blob, checked_ids)
    
            bblob_.append(bblob) # pack bblob after checking through all adjacents
    
    
    '''
    bblob_ = []
    checked_ids = []

    for derBlob in derBlob__[1:]:
        if derBlob.id not in checked_ids: # current derBlob is not checked before
            checked_ids.append(derBlob.id)

            bblob = CBblob(derBlob=CderBlob()) # init new bblob
            accum_bblob(bblob,derBlob)         # accum derBlob into bblob

            for adj_blob in derBlob.blob.adj_blobs[0]:
                if adj_blob.derBlob.id not in checked_ids: # adj blob's derBlob is not checked before
                    form_bblob_recursive(bblob, derBlob, adj_blob.derBlob, checked_ids) # recursively add blob to bblob via the adjacency

            bblob_.append(bblob) # pack bblob after checking through all adjacents
    '''
    
    return(bblob_)


def form_bblob_recursive(bblob, _blob, blob, checked_ids):
    '''
    As distinct from form_PP_, consecutive blobs don't have to be adjacent, that needs to be checked through blob.adj_blobs
    '''
    _sum_mB = sum([derBlob.mB for derBlob in _blob.derBlob_])
    sum_mB = sum([derBlob.mB for derBlob in blob.derBlob_])
    
    if (_sum_mB>0) == (sum_mB>0):    # same sign check for adjacent derBlob
        checked_ids.append(blob.id)
        
        for derBlob in blob.derBlob_:
            accum_bblob(bblob,derBlob)         # accum same sign node blob into bblob

        for pot_blob in blob.pot_blob_: # depth first - check potential blob from adjacent cluster of blobs
            if pot_blob.derBlob_ and pot_blob.id not in checked_ids: # potential blob is head node of cluster and not checked before
                form_bblob_recursive(bblob, blob, pot_blob, checked_ids)


def accum_bblob(bblob, derBlob):

    # accumulate derBlob
    bblob.derBlob.accumulate(**{param:getattr(derBlob, param) for param in bblob.derBlob.numeric_params})
    # add derBlob to list
    bblob.derBlob_.append(derBlob)



def comp_blob_recursive(blob, adj_blob_, checked_ids_, net_M):
    '''
    called by cross_comp_blob to recursively compare blob to adj_blobs in incremental layers of adjacency
    '''

    for adj_blob in adj_blob_:
        if adj_blob.id not in checked_ids_:
            checked_ids_.append(adj_blob.id)
            
            derBlob = comp_blob(blob,adj_blob)  # cross compare blob and adjacent blob
            net_M += adj_blob.Dert.M            # cost of extending the cross comparison
            
            # blob is a node, storing multiple derBlobs
            # derBlob.blob is always = current blob, while _blob could be any of the adjacent blobs
            # blob.derBlob_ is a cluster of blobs, where blob is the head node
            # if blob.derBlob is empty, blob is a forking blob of from the main node
            blob.derBlob_.append(derBlob)       
            
            # if crit, expand the cluster of blobs
            if blob.Dert.M - net_M > ave_M:  # if crit: search adjacents of adjacent, depth-first
                comp_blob_recursive(blob, adj_blob.adj_blobs[0], checked_ids_, net_M)

            else: # crit not met, the adj blobs are the potential blob from adjacent cluster
                for adj_adj_blob in adj_blob.adj_blobs[0]:
                    if adj_adj_blob not in blob.pot_blob_:
                        blob.pot_blob_.append(adj_adj_blob) # potential adjacent blob from adjacent cluster of blobs
                        
                

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
    dB = dI + dA + dG + dM

    # form derBlob regardless
    derBlob  = CderBlob(_blob=_blob, blob=blob, dI=dI, mI=mI, dA=dA, mA=mA, dG=dG, mG=mG, dM=dM, mM=mM, mB=mB, dB=dB)

    if _blob.fsliced and blob.fsliced:
        pass

    return derBlob

'''
    cross-comp among sub_blobs in nested sub_layers:

    _sub_layer = bblob.sub_layer[0]
    for sub_layer in bblob.sub_layer[1:]:
        for _sub_blob in _sub_layer:
            for sub_blob in sub_layer:
                comp_blob(_sub_blob, sub_blob)

        merge(_sub_layer, sub_layer)  # only for sub-blobs not combined into new bblobs by cross-comp
'''