"""
Visualize output of line_patterns
"""

import cv2
import argparse
from time import time
from utils import *
from itertools import zip_longest
from line_PPs_draft import comp_P, form_PPm
from line_patterns import cross_comp



if __name__ == "__main__":
    # Parse argument (image)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image',
                                 help='path to image file',
                                 default='.//raccoon.jpg')
    argument_parser.add_argument('-p', '--output_path',
                                 help='path to output folder',
                                 default='./images/line_patterns/')
    arguments = vars(argument_parser.parse_args())
    # Read image
    image = cv2.imread(arguments['image'], 0).astype(int)  # load pix-mapped image
    assert image is not None, "No image in the path"
    image = image.astype(int)

    start_time = time()
    fline_PPs = 1
    # Main

    # initialize image
    img_ini = np.zeros_like(image).astype('uint8')    
    img_p = img_ini.copy()
    img_d = img_ini.copy()
    img_m = img_ini.copy()

    img_mS = img_ini.copy()
    img_mI = img_ini.copy()
    img_mD = img_ini.copy()
    img_mM = img_ini.copy()
    
    # processing
    frame_of_patterns_ = cross_comp(image)
    if fline_PPs:  # debug line_PPs
        frame_of_dert_P_ = []
        frame_of_PPm_ = []
        for y, P_ in enumerate(frame_of_patterns_):
            dert_P_ = comp_P(P_[0])
            PPm_ = form_PPm(dert_P_)
            
            frame_of_dert_P_.append(dert_P_)
            frame_of_PPm_.append(PPm_)    
    
    # drawing section
    for y, P_ in enumerate(frame_of_patterns_): # loop each y line
        x = 0 # x location start with 0 in each line
        
        for P_num, P in enumerate(P_[0]): # loop each pattern
            
            colour_S = 0 # negative sign, colour = black (0)
            if P[0]: 
                colour_S = 255 # if positive sign, colour = white (255)    
            colour_I = P[2] # I
            colour_D = P[3] # D
            colour_M = P[4] # M
            
            for dert_num, (p,d,m) in enumerate(P[5]): # loop each dert
                if d is None: # set None to 0
                    d = 0    
                    
                img_p[y,x] = p # dert's p
                img_d[y,x] = d # dert's d
                img_m[y,x] = m # dert's m
                img_mS[y,x] = colour_S # 1st layer mP's sign
                img_mI[y,x] = colour_I # 1st layer mP's I
                img_mD[y,x] = colour_D # 1st layer mP's D
                img_mM[y,x] = colour_M # 1st layer mP's M
                
                x+=1
    
    # save images to disk
    cv2.imwrite(arguments['output_path'] + 'p.jpg',  img_p)
    cv2.imwrite(arguments['output_path'] + 'd.jpg',  img_d)
    cv2.imwrite(arguments['output_path'] + 'm.jpg',  img_m)
    cv2.imwrite(arguments['output_path'] + 'mS.jpg',  img_mS)
    cv2.imwrite(arguments['output_path'] + 'mI.jpg',  img_mI)
    cv2.imwrite(arguments['output_path'] + 'mD.jpg',  img_mD)
    cv2.imwrite(arguments['output_path'] + 'mM.jpg',  img_mM)
    

    end_time = time() - start_time
    print(end_time)
