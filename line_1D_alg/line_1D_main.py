

# add ColAlg folder to system path
import sys
from os.path import dirname, join, abspath

from numpy import int16, int32
sys.path.insert(0, abspath(join(dirname("CogAlg"), '..')))
import cv2
# import argparse
import pickle
from time import time
from matplotlib import pyplot as plt
from itertools import zip_longest
from frame_2D_alg.class_cluster import ClusterStructure, NoneType, comp_param
from line_PPs import *
from line_Ps_olp_draft import *

init_y = 0
halt_y = 999999


def cross_comp_(input__, feedback_H, level, fPd, frecursive):
    '''
    cross comp input recursively
    '''

    output__ = []    # current level output
    Y = len(input__) # Y: frame height

    for y in range(init_y, min(halt_y, Y)):  # y is index of new row pixel_, we only need one row, use init_y=0, halt_y=Y for full frame
        X = len(input__[y])  # X: frame width          
        if X>1: # non empty row's elements (pixels, Ps or Pps)
            element_ = input__[y]                              # get current line input
            compute_feedback(element_, feedback_H, level, fPd) # compute hierarchy feedback       
            dert_ = search_(element_, level, fPd)              # search & compare input elements, then compute derivatives
            pattern_ = form_pattern_(dert_, level, fPd)        # form patterns with same sign derivatives
            output__.append(pattern_)
            
    # max level = 5 now, due to haven't add any deeper level computation evaluation
    if frecursive and level<5: # next level computation
        cross_comp_(output__, feedback_H, level+1, fPd, frecursive) # current level output is next level input

# main search function
def search_(element_, level, fPd):
    
    if level == 0 : # line_Ps
        dert_ = search_pixel_(element_) # import from line_Ps_olp_draft
    elif level == 1: # line_PPs
        dert_ = search_P_(element_, fPd)   # import from line_PPs
    else: # > level 1        
        dert_ = []
        search_Pp_(element_, dert_, fPd, level)
                             
    return dert_


# search for level 2 - search rval_Pp to get rval_Pp__ per param
def search_Pp_(rval_Pp__, iPp_, fPd, level):
    
    if level == 1:
        if len(rval_Pp__)>1: # at least 2 elements to search with step=2     
            return search_pattern_(rval_Pp__, fPd)
        else:
            return [] # preseve index
    else:  
        Pp_ = []
        for rval_Pp_ in rval_Pp__: 
            Pp_.append(search_Pp_(rval_Pp_, Pp_, fPd, level-1))
        iPp_.append(Pp_)


# main form pattern function
def form_pattern_(dert_, level, fPd):

    if level==0:  # input is dert  
         # form m-sign patterns, rootP=None:
        Pm_ = form_P_root(None, dert_, rdn=1, rng=1, fPd=fPd)  # import from line_Ps_olp_draft
        pattern_ = Pm_
    
    elif level == 1: # input is pdert
        Pdert__, dert1_, dert2_,  = dert_ 
        rval_Pp__, Ppm__ = form_Pp_root(Pdert__, dert1_, dert2_, fPd) # import from line_PPs
        pattern_ = rval_Pp__

    else: # input is higher level pdert , > level 1
        pattern_ = []
        form_PPp_root(dert_, pattern_, level, fPd)
    
    return pattern_

# form pattern for level 2 and above
def form_PPp_root(dert_, ipattern_, level, fPd):
    if level == 1: # input is pdert
        if dert_:
            Pdert__, dert1_, dert2_  = dert_
            rval_Pp__, Ppm__ = form_Pp_root(Pdert__, dert1_, dert2_, fPd)
            return rval_Pp__
        else:
            return [] # preserve index

    # input is higher level pdert , > level 1
    elif dert_: # non empty dert
        pattern_ = []
        for pdert__ in dert_[0]: # dert_[0] due to added bracket during search function
            pattern_.append(form_PPp_root(pdert__, pattern_, level-1, fPd))
        ipattern_.append(pattern_)


def compute_feedback(element_, feedback_H, level, fPd):
    '''
    just a simple draft , far from complete
    '''
    
    fbM = fbL = 0
    '''
    for element in element_:
        fbM += element.M; fbL += element.L
        fbM += element.M; fbL += element.L
        if abs(fbM) > ave_Dave:
            if abs(fbM / fbL) > ave_dave:
                fbM = fbL = 0
                pass  # eventually feedback: line_patterns' cross_comp(frame_of_pixels_, ave + fbM / fbL)
                # also terminate Fspan: same-filter frame_ with summed params, re-init at all 0s

        element.I /= element.L; element.D /= element.L; element.M /= element.L  # immediate normalization to a mean
    '''
    return fbM, fbL

def update_aves(fbM, fbL):
    '''
    update aves here
    '''
    pass

        
        
if __name__ == "__main__":


    start_time = time()
    

    # recursive version
    image = cv2.imread('.//raccoon.jpg', 0).astype(int)  # manual load pix-mapped image
    feedback_H = []
    cross_comp_(image, feedback_H, level=0, fPd=0, frecursive=1)
    end_time = time() - start_time
    
    
    print(end_time)