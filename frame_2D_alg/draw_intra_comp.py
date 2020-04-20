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
# Input:
IMAGE_PATH = "./images/raccoon.jpg"
# Outputs:
OUTPUT_PATH = "./visualization/images/"

# -----------------------------------------------------------------------------
# Functions

def draw_g(img_out, g_):

    for y in range(g_.shape[0]):  # loop rows, skip 1st row
        for x in range(g_.shape[1]):  # loop columns
            img_out[y,x] = g_[y,x]

    return img_out.astype('uint8')

def draw_ga(img_out, g_):

    for y in range(g_.shape[0]):
        for x in range(g_.shape[1]):
            img_out[y,x] = g_[y,x]

    img_out = img_out * 180/np.pi  # convert to degrees
    img_out = (img_out /180 )*255  # scale 0 to 180 degree into 0 to 255

    return img_out.astype('uint8')

def draw_gr(img_out, g_, rng):

    for y in range(g_.shape[0]):
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+1+rng,(x*rng)+1:(x*rng)+1+rng] = g_[y,x]

    return img_out.astype('uint8')

def draw_gar(img_out, g_, rng):

    for y in range(g_.shape[0]):
        for x in range(g_.shape[1]):
            # project central dert to surrounding rim derts
            img_out[(y*rng)+1:(y*rng)+3,(x*rng)+1:(x*rng)+3] = g_[y,x]

    img_out = img_out * 180/np.pi  # convert to degrees
    img_out = (img_out /180 )*255  # scale 0 to 180 degree into 0 to 255

    return img_out.astype('uint8')

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":

    print('Reading image...')
    image = imread(IMAGE_PATH)

    dert_ = frame_blobs.comp_pixel(image)

    print('Processing first layer comps...')

    ga_dert_ = comp_a(dert_, fga = 0)  # if +G
    gr_dert_ = comp_r(dert_, fig = 0, root_fcr = 0)  # if -G

    print('Processing second layer comps...')
    # comp_a ->
    gaga_dert_ = comp_a(ga_dert_, fga = 1)  # if +Ga
    gg_dert_   = comp_g(ga_dert_)           # if -Ga
    # comp_r ->
    gagr_dert_ = comp_a(gr_dert_, fga = 0)                # if +Gr
    grr_dert_  = comp_r(gr_dert_, fig = 0, root_fcr = 1)  # if -Gr

    print('Processing third layer comps...')
    # comp_aga ->
    ga_gaga_dert_ = comp_a(gaga_dert_, fga = 1) # if +Gaga
    g_ga_dert_    = comp_g(gaga_dert_)          # if -Gaga
    # comp_g ->
    ga_gg_dert_ = comp_a(gg_dert_, fga = 0)                # if +Gg
    g_rg_dert_  = comp_r(gg_dert_, fig = 1, root_fcr = 0)  # if -Gg
    # comp_agr ->
    ga_gagr_dert_ = comp_a(gagr_dert_, fga = 1)  # if +Gagr
    g_gr_dert_    = comp_g(gagr_dert_)           # if -Gagr
    # comp_rr ->
    ga_grr_dert_ = comp_a(grr_dert_, fga = 0)                # if +Grr
    g_rrr_dert_  = comp_r(grr_dert_, fig = 0, root_fcr = 1)  # if -Grr：

    print('Drawing forks...')
    ini_ = np.zeros((image.shape[0], image.shape[1]))  # initialize image y, x

    # 0th layer
    g_ = draw_g(ini_, dert_[1])
    # 1st layer
    ga_ = draw_ga(ini_, ga_dert_[5])
    gr_ = draw_gr(ini_, gr_dert_[1], rng=2)
    # 2nd layer
    gaga_ = draw_ga(ini_, gaga_dert_[5])
    gg_ =   draw_g(ini_,  gg_dert_[1])
    gagr_ = draw_gar(ini_, gagr_dert_[5], rng=2)
    grr_ =  draw_gr(ini_,  grr_dert_[1], rng=4)
    # 3rd layer
    ga_gaga_ = draw_ga(ini_, ga_gaga_dert_[5])
    g_ga_    = draw_g(ini_,  g_ga_dert_[1])
    ga_gg_   = draw_ga(ini_, ga_gg_dert_[5])
    g_rg_    = draw_gr(ini_, g_rg_dert_[1], rng=2)
    ga_gagr_ = draw_gar(ini_, ga_gagr_dert_[5], rng=2)
    g_gr_    = draw_gr(ini_,  g_gr_dert_[1], rng=2)
    ga_grr_  = draw_gar(ini_, ga_grr_dert_[5], rng=4)
    g_rrr_   = draw_gr(ini_,  g_rrr_dert_[1], rng=8)

    # save to disk
    cv2.imwrite('./images/intra_comp/0_g.png', g_)
    cv2.imwrite('./images/intra_comp/1_ga.png', ga_)
    cv2.imwrite('./images/intra_comp/2_gr.png', gr_)
    cv2.imwrite('./images/intra_comp/3_gaga.png', gaga_)
    cv2.imwrite('./images/intra_comp/4_gg.png', gg_)
    cv2.imwrite('./images/intra_comp/5_gagr.png', gagr_)
    cv2.imwrite('./images/intra_comp/6_grr.png', grr_)
    cv2.imwrite('./images/intra_comp/7_ga_gaga.png', ga_gaga_)
    cv2.imwrite('./images/intra_comp/8_g_ga.png', g_ga_)
    cv2.imwrite('./images/intra_comp/9_ga_gg.png', ga_gg_)
    cv2.imwrite('./images/intra_comp/10_g_rg.png', g_rg_)
    cv2.imwrite('./images/intra_comp/11_ga_gagr.png', ga_gagr_)
    cv2.imwrite('./images/intra_comp/12_g_gr.png', g_gr_)
    cv2.imwrite('./images/intra_comp/13_ga_grr.png', ga_grr_)
    cv2.imwrite('./images/intra_comp/14_g_rrr.png', g_rrr_)

    print('Terminating...')


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

''' 
    dertm_ = np.zeros((5, dert_.shape[1], dert_.shape[2])) # add extra m channel 
    dert_[0:4] = dert_[:]
    
    g = np.zeros((size_y,size_x))
    # 1st layer
    ga = np.zeros((size_y,size_x))
    gr = np.zeros((size_y,size_x))
    # 2nd layer
    gaga = np.zeros((size_y,size_x))
    gg = np.zeros((size_y,size_x))
    gagr = np.zeros((size_y,size_x))
    grr = np.zeros((size_y,size_x))
    # 3rd layer
    ga_gaga = np.zeros((size_y,size_x))
    g_ga = np.zeros((size_y,size_x))
    ga_gg = np.zeros((size_y,size_x))
    g_rg = np.zeros((size_y,size_x))
    ga_gagr = np.zeros((size_y,size_x))
    g_gr = np.zeros((size_y,size_x))
    ga_grr = np.zeros((size_y,size_x))
    g_rrr  = np.zeros((size_y,size_x))

    # draw each dert
    img_g = draw_g(ini_, dert_[1])
    # 1st layer
    img_ga = draw_ga(ga, ga_dert_[5])
    img_gr = draw_gr(gr, gr_dert_[1], rng=2)
    # 2nd layer
    img_gaga = draw_ga(gaga, gaga_dert_[5])
    img_gg =   draw_g(gg, gaga_dert_[1])
    img_gagr = draw_gar(gagr, gagr_dert_[5], rng=2)
    img_grr =  draw_gr(grr, gagr_dert_[1], rng=4)
    # 3rd layer
    img_ga_gaga = draw_ga(ga_gaga, gaga_dert_[5])
    img_g_ga    = draw_g(g_ga, gaga_dert_[1])
    img_ga_gg   = draw_ga(ga_gg, gg_dert_[5])
    img_g_rg    = draw_gr(g_rg, gg_dert_[1], rng=2)
    img_ga_gagr = draw_gar(ga_gagr, gagr_dert_[5], rng=2)
    img_g_gr    = draw_gr(g_gr, gagr_dert_[1], rng=2)
    img_ga_grr  = draw_gar(ga_grr, grr_dert_[5], rng=4)
    img_g_rrr   = draw_gr(g_rrr, grr_dert_[1], rng=8)

'''
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
#    # initialize each image for visualization
#    img_comp_pixel = np.zeros((size_y,size_x))
#    img_adert_ag = np.zeros((size_y,size_x))
#    img_adert_aga = np.zeros((size_y,size_x))
#    img_gdert_g = np.zeros((size_y,size_x))
#
#    # loop in each row
#    for y in range(size_y):
#        # loop across each dert
#        for x in range(size_x):
#
#            try:
#                # derts' g
#                g = dert_[1][y,x] - ave
#                # +g fork
#                if g>=0:
#
#                    # add value to +g
#                    img_comp_pixel[y,x] = 255;
#                    # adert's ga
#                    ga = adert_ag[1][y,x] - ave_adert_ag
#
#                    # +ga fork
#                    if ga>=0:
#
#                        # add value to +ga
#                        img_adert_ag[y,x] = 255
#                        # adert's gaga
#                        gaga = adert_aga[1][y,x] - ave_adert_aga
#
#                        # +gaga fork
#                        if gaga>=0:
#
#                            # add value to +gaga
#                            img_adert_aga[y,x] = 255
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
#                            # comp_agg
#
#                        # -gg fork
#                        else:
#
#                            # add value to -gg
#                            img_gdert_g[y,x] = 128
#                            # comp_rng_g
#
#                # -g fork
#                else:
#
#                    # add value to -g
#                    img_comp_pixel[y,x] = 128;
#
#                    # comp_rng_p
#                    # comp_agr
#                    # comp_rrp
#
#            except:
#                pass
#
#    print('Done!')
#    print('Saving images to disk...')
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