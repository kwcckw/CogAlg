"""
For testing intra_comp operations and 3 layers of intra_comp's forks
Visualize each comp's output with image output
"""

import frame_blobs
from intra_comp import *
from utils import imread, imwrite
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Constants

# Input:
IMAGE_PATH = "./images/raccoon.jpg"

# Outputs:
OUTPUT_PATH = "./visualization/images/"

# -----------------------------------------------------------------------------
# Functions


# add colour
def add_colour(img_comp,size_y,size_x):
    img_colour = np.zeros((3,size_y,size_x))
    img_colour[2] = img_comp
    img_colour[2][img_colour[2]<255] = 0
    img_colour[2][img_colour[2]>0] = 205
    img_colour[1] = img_comp
    img_colour[1][img_colour[1]==255] = 0
    img_colour[1][img_colour[1]>0] = 255
    img_colour = np.rollaxis(img_colour,0,3).astype('uint8')

    return img_colour


# draw dert_, and gdert_
def draw_gray(img_out,g_):
    # loop in each row
    for y in range(g_.shape[0]):
        # loop across each dert
        for x in range(g_.shape[1]):   
            # assign g to image 
            img_out[y,x] = g_[y,x]
    # return uint8
    return img_out.astype('uint8')
            
# draw adert
def draw_gray_a(img_out,g_):
    # loop in each row
    for y in range(g_.shape[0]):
        # loop across each dert
        for x in range(g_.shape[1]):   
            # assign g to image 
            img_out[y,x] = g_[y,x]
    
    # convert to degree
    img_out = img_out * 180/np.pi
    # scale 0 to 180 degree into 0 to 255
    img_out = (img_out /180 )*255
    # return uint8
    return img_out.astype('uint8')

# draw rdert
def draw_gray_rng(img_out,g_,rng):
    # loop in each row, skip 1st row
    for y in range(g_.shape[0]):
        # loop across each dertm skip 1st dert
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+1+rng,(x*rng)+1:(x*rng)+1+rng] = g_[y,x]
    # return uint8
    return img_out.astype('uint8')


# draw adert with previous root fork = rdert
def draw_gray_rng_a(img_out,g_,rng):
    # loop in each row, skip 1st row
    for y in range(g_.shape[0]):
        # loop across each dertm skip 1st dert
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+3,(x*rng)+1:(x*rng)+3] = g_[y,x]
    
   # convert to degree
    img_out = img_out * 180/np.pi
    # scale 0 to 180 degree into 0 to 255
    img_out = (img_out /180 )*255
    # return uint8
    return img_out.astype('uint8')


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    # Initial comp:
    print('Reading image...')
    image = imread(IMAGE_PATH)
    print('Done!')
    
    ## root layer #############################################################
    
    print('Doing first comp...')
    dert_tem_ = frame_blobs.comp_pixel(image)
    
    # temporary add extra m channel to make the code run-able since right now comp pixel doesn't have m
    dert_ = np.zeros((5,dert_tem_.shape[1],dert_tem_.shape[2]))
    dert_[0:4] = dert_tem_[:]
    print('Done!')
    
    ## 1st layer ##############################################################
    
    print('Processing first layer comps...')
    ## 2 forks from comp_pixel ##
    # if +G：
    # comp_a (comp_ag)
    adert_ag = comp_a(dert_,fga = 0)
    # if -G：
    # comp_r (comp_rng_p)
    rdert_rng_p = comp_r(dert_,fig = 0,root_fcr = 0)
    print('Done!')
    
    ## 2nd layer ##############################################################
      
    print('Processing second layer comps...') 
    ## 2 forks from comp_ag ##
    # if +Ga：
    # comp_a (comp_aga)
    adert_aga = comp_a(adert_ag,fga = 1)
    # if -Ga：
    # comp_g (comp_g)
    gdert_g = comp_g(adert_ag)
    
    ## 2 forks from comp_rng_p ##
    # if +Gr:
    # comp_a(comp_agr)
    adert_agr = comp_a(rdert_rng_p,fga = 0)
    # if -Gr:
    # comp_r (comp_rrp)
    rdert_rrp = comp_r(rdert_rng_p,fig = 0,root_fcr = 1)
    print('Done!')
    
    ## 3rd layer ##############################################################
    
    print('Processing third layer comps...')
    ## 2 forks from comp_aga ##
    # if +Gaga：
    # comp_a (comp_aga_ga)
    adert_aga_ga = comp_a(adert_aga,fga = 1)
    # if -Gaga：
    # comp_g (comp_ga)
    gdert_ga = comp_g(adert_aga)
    
    ## 2 forks from comp_g ##
    # if +Gg:
    # comp_a (comp_agg)
    adert_agg = comp_a(gdert_g,fga = 0)
    # if -Gg:
    # comp_r (comp_rng_g)
    rdert_rng_g = comp_r(gdert_g,fig = 1,root_fcr = 0)
    
    ## 2 forks from comp_agr ##
    # if +Gagr：
    # comp_a (comp_aga_gr)
    adert_aga_gr = comp_a(adert_agr,fga = 1)
    # if -Gagr：
    # comp_g (comp_gr)
    gdert_gr = comp_g(adert_agr)
    
    ## 2 forks from comp_rrp ##
    # if +Grr：
    # comp_a (comp_a_grr)
    adert_a_grr = comp_a(rdert_rrp,fga = 0)
    # if -Grr：
    # comp_r (comp_rrrp)
    rdert_rrrp = comp_r(gdert_g,fig = 0,root_fcr = 1)
    print('Done!')
    
    ###########################################################################
    
    print('Drawing each comps...')
    
    # size of image
    size_y = image.shape[0]
    size_x = image.shape[1]
    
    # initialize each image for visualization
    img_comp_pixel = np.zeros((size_y,size_x))  
    # 1st layer
    img_adert_ag = np.zeros((size_y,size_x))
    img_rdert_rng_p = np.zeros((size_y,size_x)) 
    # 2nd layer
    img_adert_aga = np.zeros((size_y,size_x))
    img_gdert_g = np.zeros((size_y,size_x))
    img_adert_agr = np.zeros((size_y,size_x))
    img_rdert_rrp = np.zeros((size_y,size_x))
    # 3rd layer
    img_adert_aga_ga = np.zeros((size_y,size_x))
    img_gdert_ga = np.zeros((size_y,size_x))
    img_adert_agg = np.zeros((size_y,size_x))
    img_rdert_rng_g = np.zeros((size_y,size_x))
    img_adert_aga_gr = np.zeros((size_y,size_x))
    img_gdert_gr = np.zeros((size_y,size_x))
    img_adert_a_grr = np.zeros((size_y,size_x))
    img_rdert_rrrp  = np.zeros((size_y,size_x))
    
    # draw each dert 
    img_comp_pixel = draw_gray(img_comp_pixel,dert_[1])
    # 1st layer
    img_adert_ag = draw_gray_a(img_adert_ag,adert_ag[5])
    img_rdert_rng_p = draw_gray_rng(img_rdert_rng_p,rdert_rng_p[1],rng=2)
    # 2nd layer
    img_adert_aga = draw_gray_a(img_adert_aga,adert_aga[5])
    img_gdert_g = draw_gray(img_gdert_g,gdert_g[1])
    img_adert_agr = draw_gray_rng_a(img_adert_agr,adert_agr[5],rng=2)
    img_rdert_rrp = draw_gray_rng(img_rdert_rrp,rdert_rrp[1],rng=4)
    # 3rd layer
    img_adert_aga_ga = draw_gray_a(img_adert_aga_ga,adert_aga_ga[5])
    img_gdert_ga = draw_gray(img_gdert_ga,gdert_ga[1])
    img_adert_agg = draw_gray_a(img_adert_agg,adert_agg[5])
    img_rdert_rng_g = draw_gray_rng(img_rdert_rng_g,rdert_rng_g[1],rng=2)
    img_adert_aga_gr = draw_gray_rng_a(img_adert_aga_gr,adert_aga_gr[5],rng=2)
    img_gdert_gr = draw_gray_rng(img_gdert_gr,gdert_gr[1],rng=2)
    img_adert_a_grr = draw_gray_rng_a(img_adert_a_grr,adert_a_grr[5],rng=4)
    img_rdert_rrrp  = draw_gray_rng(img_rdert_rrrp,rdert_rrrp[1],rng=8)

    # save to disk
    cv2.imwrite('./images/image_comp_pixel.png',img_comp_pixel)
    cv2.imwrite('./images/image_adert_ag.png',img_adert_ag)
    cv2.imwrite('./images/image_rdert_rng_p.png',img_rdert_rng_p)
    cv2.imwrite('./images/image_adert_aga.png',img_adert_aga)
    cv2.imwrite('./images/image_gdert_g.png',img_gdert_g)
    cv2.imwrite('./images/image_adert_agr.png',img_adert_agr)
    cv2.imwrite('./images/image_rdert_rrp.png',img_rdert_rrp)
    cv2.imwrite('./images/image_adert_aga_ga.png',img_adert_aga_ga)
    cv2.imwrite('./images/image_gdert_ga.png',img_gdert_ga)
    cv2.imwrite('./images/image_adert_agg.png',img_adert_agg)
    cv2.imwrite('./images/image_rdert_rng_g.png',img_rdert_rng_g)
    cv2.imwrite('./images/image_adert_aga_gr.png',img_adert_aga_gr)
    cv2.imwrite('./images/image_gdert_gr.png',img_gdert_gr)
    cv2.imwrite('./images/image_adert_a_grr.png',img_adert_a_grr)
    cv2.imwrite('./images/image_rdert_rrrp.png',img_rdert_rrrp)
    
    print('Done!')
    print('Terminating...')
    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    
      
    
    
    
#    # size of image
#    size_y = image.shape[0]
#    size_x = image.shape[1]
#    
#    
#    from matplotlib import pyplot as plt
#    
#    plt.figure()
#    plt.imshow(dert_)
#    
#    
#    
#    # initialize each image for visualization
#    img_comp_pixel = np.zeros((size_y,size_x))
#    img_adert_ag = np.zeros((size_y,size_x))
#    img_adert_aga = np.zeros((size_y,size_x))
#    img_gdert_g = np.zeros((size_y,size_x))
#
#    
#    # loop in each row
#    for y in range(size_y):
#        # loop across each dert
#        for x in range(size_x):
#            
#            try:
#                    
#                # derts' g
#                g = dert_[1][y,x] - ave
#                
#                # +g fork
#                if g>=0:
#                
#                    # add value to +g
#                    img_comp_pixel[y,x] = 255;
#                    
#                    # adert's ga
#                    ga = adert_ag[1][y,x] - ave_adert_ag
#                
#                    # +ga fork
#                    if ga>=0:
#                        
#                        # add value to +ga 
#                        img_adert_ag[y,x] = 255
#                    
#                        # adert's gaga
#                        gaga = adert_aga[1][y,x] - ave_adert_aga
#                        
#        
#                        # +gaga fork
#                        if gaga>=0:
#                            
#                            # add value to +gaga
#                            img_adert_aga[y,x] = 255
#                            
#                            # comp_aga_ga
#                        
#                        # -gaga fork
#                        else:
#                    
#                            # add value to -gaga
#                            img_adert_aga[y,x] = 128
#                            
#                            # comp_ga
#                            
#                    # -ga fork
#                    else:
#                        
#                        # add value to -ga
#                        img_adert_ag[y,x] = 128
#                        
#                        # adert's gaga
#                        gg = gdert_g[1][y,x] - ave_gdert_g
#                        
#                        # +gg fork
#                        if gg>=0:
#                         
#                            # add value to +gg
#                            img_gdert_g[y,x] = 255
#                            
#                            # comp_agg
#                            
#                        # -gg fork
#                        else:
#                            
#                            # add value to -gg
#                            img_gdert_g[y,x] = 128
#                       
#                            # comp_rng_g
#        
#                # -g fork
#                else:
#                    
#                    # add value to -g
#                    img_comp_pixel[y,x] = 128;
#        
#        
#                    # comp_rng_p
#                    # comp_agr
#                    # comp_rrp
#                    
#                    
#            except:
#                pass
#      
#    
#    
#    
#    print('Done!')
#    
#    
#    print('Saving images to disk...')
#    
#    
#    # add colour
#    # where red = +g, green = -g
#    img_colour_comp_pixel = add_colour(img_comp_pixel,size_y,size_x)
#    img_colour_adert_ag = add_colour(img_adert_ag,size_y,size_x)
#    img_colour_adert_aga = add_colour(img_adert_aga,size_y,size_x)
#    img_colour_gdert_g = add_colour(img_gdert_g,size_y,size_x)
#    
#    # save to disk
#    cv2.imwrite('./images/image_colour_comp_pixel.png',img_colour_comp_pixel)                         
#    cv2.imwrite('./images/image_colour_adert_ag.png',img_colour_adert_ag)                         
#    cv2.imwrite('./images/image_colour_adert_aga.png',img_colour_adert_aga)                         
#    cv2.imwrite('./images/image_colour_gdert_g.png',img_colour_gdert_g)                         
#    cv2.imwrite('./images/image_comp_pixel.png',img_comp_pixel)
#    cv2.imwrite('./images/image_adert_ag.png',img_adert_ag)
#    cv2.imwrite('./images/image_adert_aga.png',img_adert_aga)
#    cv2.imwrite('./images/image_gdert_g.png',img_gdert_g)
#    
#    print('Done!')
#    print('Terminating...')
    
    