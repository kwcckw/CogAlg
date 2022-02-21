'''
    Intra_blob recursively evaluates each blob for three forks of extended internal cross-comparison and sub-clustering:
    -
    - comp_range: incremental range cross-comp in low-variation flat areas of +v--vg: the trigger is positive deviation of negated -vg,
    - comp_angle: angle cross-comp in high-variation edge areas of positive deviation of gradient, forming gradient of angle,
    - comp_slice_ forms roughly edge-orthogonal Ps, their stacks evaluated for rotation, comp_d, and comp_slice
    -
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png
'''

import numpy as np
from frame_blobs import assign_adjacents, flood_fill, CBlob
from intra_comp import comp_r, comp_a
from draw_frame_blobs import visualize_blobs
from itertools import zip_longest
from comp_slice_ import *
from segment_by_direction import segment_by_direction

# filters, All *= rdn:
ave = 50   # cost / dert: of cross_comp + blob formation, same as in frame blobs, use rcoef and acoef if different
aveB = 50  # cost / blob: fixed syntactic overhead
pcoef = 2  # ave_comp_slice / ave: relative cost of p fork;  no ave_ga = .78, ave_ma = 2: no indep eval
# --------------------------------------------------------------------------------------------------------------
# functions:

def intra_blob_root(root_blob, render, verbose):  # initial range+ | angle cross-comp per frame blobs' blob.dert__

    blob_ = root_blob.sublayers[0]  # deep_frame remains root_blob, extended in-place
    deep_blob_i_ = []  # indices of blobs with added sublayers
    deep_blobs = []
    spliced_layers = []  # to extend root_blob sublayers

    for i, blob in enumerate(blob_):  # print('Processing blob number ' + str(bcount))

        blob.root_dert__= root_blob.dert__
        blob.prior_forks=['g']  # called from frame_blobs
        if render and blob.A < 100: render = False  # don't render small blobs
        blob_height = blob.box[1] - blob.box[0];  blob_width = blob.box[3] - blob.box[2]

        if blob_height > 3 and blob_width > 3:  # min blob dimensions
            '''
            Blob M: -|+ predictive value, positive in +M blobs and lent to contrast value of adjacent -M blobs. 
            -M "edge" blobs are valuable as contrast: their negative value cancels positive value of adjacent "flat" +M blobs.
            '''
            deep_blob = intra_forking(blob,spliced_layers, render, verbose)
            if deep_blob:
                deep_blob_i_.append(i)
                deep_blobs.append(deep_blob)

    if verbose: print_deep_blob_forking(deep_blobs); print("\rFinished intra_blob")

    return spliced_layers

                
# merge intra forking and cluster_sub_eval section, but in unpacked form
def intra_forking(blob, spliced_layers, render, verbose):

    if blob.M > 0:
        if blob.M > aveB * blob.rdn:
            blob.fBa = 0; blob.rng += 1; blob.rdn += 1
            ext_dert__, ext_mask__ = extend_dert(blob)  # dert__+= 1: cross-comp in larger kernels
            # comp_r in 4x4:
            new_dert__, new_mask__ = comp_r(ext_dert__, ave*blob.rdn, blob.rng, ext_mask__)
            sign__ = new_dert__[3] > 0  # m__ = ave - g
            if verbose: print('\na fork\n'); blob.prior_forks.extend('r')

    elif -blob.M > aveB * blob.rdn:  # replace with borrow_M if known; root fork: frame_blobs or comp_r
        blob.fBa = 1; blob.rdn += 1
        ext_dert__, ext_mask__ = extend_dert(blob)  # dert__+= 1: cross-comp in larger kernels
        # comp_a in 2x2:
        new_dert__, new_mask__ = comp_a(ext_dert__, ext_mask__)  # compute abs ma, no - ave_ma in comp_a
        sign__ = (-new_dert__[3] * new_dert__[9]) > ave * pcoef  # -m * ma: val comp_slice_
        if verbose: print('\na fork\n'); blob.prior_forks.extend('a')


    if "new_dert__" in locals() and new_mask__.shape[0] > 2 and new_mask__.shape[1] > 2 and False in new_mask__:  # min size in y and x, least one dert in dert__

        # form sub_blobs (cluster_sub_eval):
        new_mask__.fill(False)
        AveB = aveB * blob.rdn
    
        sub_blobs, idmap, adj_pairs = flood_fill(new_dert__, sign__, verbose=False, mask__=new_mask__, blob_cls=CBlob)
        assign_adjacents(adj_pairs, CBlob)
    
        if render: visualize_blobs(idmap, sub_blobs, winname=f"Deep blobs (froot_Ba = {blob.fBa}, froot_Ba = {blob.prior_forks[-1] == 'a'})")
        blob.Ls = len(sub_blobs)  # for visibility
        blob.sublayers = [sub_blobs]  # 1st layer of sub_blobs
    
        for sub_blob in sub_blobs:  # evaluate comp_r or comp_a per sub_blob
            if render and sub_blob.A < 100: render = False
            sub_blob.prior_forks = blob.prior_forks.copy()  # increments forking sequence: m->r, g->a, a->p
            if sub_blob.mask__.shape[0] > 2 and sub_blob.mask__.shape[1] > 2 and False in sub_blob.mask__:  # min L in y and x, >=1 dert in dert__
    
                if blob.fBa:  # p fork:
                    if -sub_blob.M * sub_blob.Ma > AveB * pcoef:
                        if verbose: print('\nslice_blob fork\n'); sub_blob.prior_forks.extend('p')
                        segment_by_direction(sub_blob, verbose=True)
    
                else:  # a or r fork:
                    ''' G = blob.G  # Gr, Grr...
                    adj_M = blob.adj_blobs[3]  # adj_M is incomplete, computed within current dert_only, use root blobs instead:
                    adjacent valuable blobs of any sign are tracked from frame_blobs to form borrow_M?
                    track adjacency of sub_blobs: wrong sub-type but right macro-type: flat blobs of greater range?
                    G indicates or dert__ extend per blob G?
                    borrow_M = min(G, adj_M / 2): usually not available, use average
                    '''
                    
                    blob.sublayers += intra_forking(sub_blob, [], render, verbose)
        
        spliced_layers = [spliced_layers + sublayers for spliced_layers, sublayers in
                                  zip_longest(spliced_layers, blob.sublayers, fillvalue=[])]
                          
    return spliced_layers


def extend_dert(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob.box  # extend dert box:
    rY, rX = blob.root_dert__[0].shape  # higher dert size

    # determine pad size
    y0e = max(0, y0 - 1)
    yne = min(rY, yn + 1)
    x0e = max(0, x0 - 1)
    xne = min(rX, xn + 1)  # e is for extended

    # take ext_dert__ from part of root_dert__
    ext_dert__ = []
    for dert in blob.root_dert__:
        if type(dert) == list:  # tuple of 2 for day, dax - (Dyy, Dyx) or (Dxy, Dxx)
            ext_dert__.append(dert[0][y0e:yne, x0e:xne])
            ext_dert__.append(dert[1][y0e:yne, x0e:xne])
        else:
            ext_dert__.append(dert[y0e:yne, x0e:xne])
    ext_dert__ = tuple(ext_dert__)  # change list to tuple

    # extended mask__
    ext_mask__ = np.pad(blob.mask__,
                        ((y0 - y0e, yne - yn),
                         (x0 - x0e, xne - xn)),
                        constant_values=True, mode='constant')

    return ext_dert__, ext_mask__

def print_deep_blob_forking(deep_layers):

    def check_deep_blob(deep_layer,i):
        for deep_blob_layer in deep_layer:
            if isinstance(deep_blob_layer,list):
                check_deep_blob(deep_blob_layer,i)
            else:
                print('blob num = '+str(i)+', forking = '+'->'.join(deep_blob_layer.prior_forks))

    for i, deep_layer in enumerate(deep_layers):
        if len(deep_layer)>0:
            check_deep_blob(deep_layer,i)
