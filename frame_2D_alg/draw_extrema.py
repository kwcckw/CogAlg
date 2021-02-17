"""
Visualize estimated extrema: e_= m_ - g_, over incremental comp_r
"""

from alternative_versions.comp_pixel_versions import comp_pixel_m
from intra_comp import *
from alternative_versions.LUT import Y_COEFFS, X_COEFFS

import cv2
import argparse
import numpy as np

# -----------------------------------------------------------------------------
# Input:
IMAGE_PATH = "./images/raccoon.jpg"
# Outputs:
OUTPUT_PATH = "./images/intra_comp0/"

# -----------------------------------------------------------------------------
# Functions

def shift_img(img,rng):
    '''
    shift image based on the rng directions
    '''

    minimum_input_size = (rng*2)+1 # minimum input size based on rng
    output_size_y = img.shape[0] - (rng*2) # expected output size after shifting
    output_size_x = img.shape[1] - (rng*2) # expected output size after shifting

    total_shift_direction = rng*8 # total shifting direction based on rng

    # initialization
    img_shift_ = []
    x = -rng
    y = -rng
    sstep = 1
    # sstep = rng+1 # skip version

    # get shifted images if output size >0
    if output_size_y>0 and output_size_x>0:


        img_center__ = img[rng:-(rng):sstep,rng:-(rng):sstep]
        # img_center__ = img[rng:-(rng):rng+1,rng:-(rng):rng+1] # skip version


        for i in range(total_shift_direction):

            # get images in shifted direction
            if (x<=0 and y<=0) :
                if y == -rng:
                    img_shift = img[:y*2:sstep, rng+x:(x*2)-(rng+x):sstep]
                elif x == -rng:
                    img_shift = img[rng+y:(y*2)-(rng+y):sstep,:x*2:sstep]
            elif x>0 and y<=0:
                if x == rng:
                    img_shift = img[rng+y:(y*2)-(rng+y):sstep, rng+x::sstep]
                else:
                    img_shift = img[rng+y:(y*2)-(rng+y):sstep, rng+x:x-rng:sstep]
            elif x<=0 and y>0:
                if y == rng:
                    img_shift = img[rng+y::sstep, rng+x:(x*2)-(rng+x):sstep]
                else:
                    img_shift = img[rng+y:y-rng:sstep, rng+x:(x*2)-(rng+x):sstep]
            elif x>0 and y>0:
                if x == rng and y == rng:
                    img_shift = img[rng+y::sstep, rng+x::sstep]
                elif x == rng:
                    img_shift = img[rng+y:y-rng:sstep, rng+x::sstep]
                elif y == rng:
                    img_shift = img[rng+y::sstep, rng+x:x-rng:sstep]


            # update x and y shifting value
            if x == -rng and y>-rng:
                y-=1
            elif x < rng and y < rng:
                x+=1
            elif x >= rng and y < rng:
                y+=1
            elif y >= rng and x >-rng:
                x-=1

            img_shift_.append(img_shift)

    return img_center__, img_shift_


def comp_rng(dert__, ave, root_fia, rng, mask__=None):

    print('rng = '+str(rng))
    '''
    Cross-comparison of input param (dert[0]) over rng passed from intra_blob.
    This fork is selective for blobs with below-average gradient,
    where input intensity didn't vary much in shorter-range cross-comparison.
    Such input is predictable enough for selective sampling: skipping current
    rim derts as kernel-central derts in following comparison kernels.
    Skipping forms increasingly sparse output dert__ for greater-range cross-comp, hence
    rng (distance between centers of compared derts) increases as 2^n, with n starting at 0:
    rng = 1: 3x3 kernel,
    rng = 2: 5x5 kernel,
    rng = 4: 9x9 kernel,
    ...
    Due to skipping, configuration of input derts in next-rng kernel will always be 3x3, see:
    https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_comp_diagrams.png
    '''
    i__ = dert__[0]  # i is pixel intensity

    '''
    sparse aligned i__center and i__rim arrays:
    rotate in first call only: same orientation as from frame_blobs?
    '''

    i__center, i__directional_ = shift_img(i__,rng)

    ''' 
    unmask all derts in kernels with only one masked dert (can be set to any number of masked derts), 
    to avoid extreme blob shrinking and loss of info in other derts of partially masked kernels
    unmasked derts were computed due to extend_dert() in intra_blob   
    '''
    if mask__ is not None:
        majority_mask__, mask__directional_ = shift_img(mask__,rng)
        majority_mask__ = majority_mask__.astype(int)

        for mask__directional in mask__directional_:
            majority_mask__ += mask__directional.astype(int)

    else:
        majority_mask__ = None  # returned at the end of function

    if root_fia:  # initialize derivatives:
        dy__ = np.zeros_like(i__center)  # sparse to align with i__center
        dx__ = np.zeros_like(dy__)
        m__ = np.zeros_like(dy__)

    else:  # root fork is comp_r, accumulate derivatives:
        dy__ = dert__[1][rng:-(rng):1, rng:-(rng):1].copy()  # sparse to align with i__center
        dx__ = dert__[2][rng:-(rng):1, rng:-(rng):1].copy()
        m__  = dert__[4][rng:-(rng):1, rng:-(rng):1].copy()

    # compare diametrically opposed pairs of rim pixels:

    i_diag_dif_y_ = []
    i_diag_dif_x_ = []

    # different x and y coeffs for different rng
    Y_COEFF = Y_COEFFS[rng]
    X_COEFF = Y_COEFFS[rng]

    # get difference of each diametrically opposed pairs * coeffs
    for i, i__directional in enumerate(i__directional_[:int(len(i__directional_)/2)]):

        # index of diametrically opposed image
        diag_index = int(len(i__directional_)/2)+i

        # differences * coeffs
        i_diag_dif_y_.append( (i__directional - i__directional_[diag_index]) * Y_COEFF [i] )
        i_diag_dif_x_.append( (i__directional - i__directional_[diag_index]) * X_COEFF [i] )

    # sum of differences
    dy__ = sum(i_diag_dif_y_)
    dx__ = sum(i_diag_dif_x_)


    g__ = np.hypot(dy__, dx__) - ave  # gradient, recomputed at each comp_r
    g__abs = np.hypot(dy__, dx__)

    '''
    inverse match = SAD, direction-invariant and more precise measure of variation than g
    (all diagonal derivatives can be imported from prior 2x2 comp)
    ave SAD = ave g * 1.41:
    '''

    m__abs = m__.copy()
    m__dif = []

    # get directional differences of m
    for i__directional in i__directional_:
        m__dif.append(abs(i__center - i__directional))

    # compute m
    m__ += int(ave * 1.41) - sum(m__dif)
    m__abs += sum(m__dif)

    # g multiplcation ratio in computing e__
    gratio = (2/len(i__directional_) ) * 3

    # e = m - (g_ratio * g)
    e__ = m__abs - (gratio*g__abs)

    # invert to enable dark contour and bright edges
    m__ = np.max(m__) - m__

    # scale negative values to positive
    if np.min(g__)<0:
        g__ += abs(np.min(g__))

    # scale negative values to positive
    if np.min(e__)<0:
        e__ += abs(np.min(e__))

    return (i__center, dy__, dx__, g__, m__, e__), majority_mask__


def draw_g(img_out, g_):
    endy = min(img_out.shape[0], g_.shape[0])
    endx = min(img_out.shape[1], g_.shape[1])
    img_out[:endy, :endx] = (g_[:endy, :endx] * 255) / g_.max()  # scale to max=255, less than max / 255 is 0

    return img_out

def draw_gr(img_out, g_):

    img_out[:] = cv2.resize((g_[:] * 255) / g_.max(),  # normalize g to uint
                            (img_out.shape[1], img_out.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    return img_out

def imread(filename, raise_if_not_read=True):
    "Read an image in grayscale, return array."
    try:
        return cv2.imread(filename, 0).astype(int)
    except AttributeError:
        if raise_if_not_read:
            raise SystemError('image is not read')
        else:
            print('Warning: image is not read')
            return None

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--image', help='path to image file', default=IMAGE_PATH)
    argument_parser.add_argument('-o', '--output', help='path to output folder', default=OUTPUT_PATH)
    arguments = argument_parser.parse_args()

    print('Reading image...')
    image = imread(arguments.image)

    dert_ = comp_pixel_m(image)

    ave = 50

    print('Processing first layer comps...')
    # comp_p ->
    gr_dert_1, _ = comp_rng(dert_, ave, root_fia = 0, rng= 1)
    gr_dert_2, _ = comp_rng(dert_, ave, root_fia = 0, rng= 2)
    gr_dert_3, _ = comp_rng(dert_, ave, root_fia = 0, rng= 3)
    gr_dert_4, _ = comp_rng(dert_, ave, root_fia = 0, rng= 4)
    gr_dert_5, _ = comp_rng(dert_, ave, root_fia = 0, rng= 5)
    gr_dert_6, _ = comp_rng(dert_, ave, root_fia = 0, rng= 6)
    gr_dert_7, _ = comp_rng(dert_, ave, root_fia = 0, rng= 7)
    gr_dert_8, _ = comp_rng(dert_, ave, root_fia = 0, rng= 8)
    gr_dert_9, _ = comp_rng(dert_, ave, root_fia = 0, rng= 9)
    gr_dert_10, _ = comp_rng(dert_, ave, root_fia = 0, rng= 10)




    print('Drawing forks...')
    ini_ = np.zeros((image.shape[0], image.shape[1]), 'uint8')  # initialize image y, x

    # 0th layer
    g_ = draw_g(ini_.copy(), dert_[3])
    m_ = draw_g(ini_.copy(), dert_[6])
    # 1st layer

    # rng = 1
    gr_1 = draw_gr(ini_.copy(), gr_dert_1[3])
    mr_1 = draw_gr(ini_.copy(), gr_dert_1[4])
    re_1 = draw_gr(ini_.copy(),  gr_dert_1[5])
    # rng = 2
    gr_2 = draw_gr(ini_.copy(), gr_dert_2[3])
    mr_2 = draw_gr(ini_.copy(), gr_dert_2[4])
    re_2 = draw_gr(ini_.copy(),  gr_dert_2[5])
    # rng = 3
    gr_3 = draw_gr(ini_.copy(), gr_dert_3[3])
    mr_3 = draw_gr(ini_.copy(), gr_dert_3[4])
    re_3 = draw_gr(ini_.copy(),  gr_dert_3[5])
    # rng = 4
    gr_4 = draw_gr(ini_.copy(), gr_dert_4[3])
    mr_4 = draw_gr(ini_.copy(), gr_dert_4[4])
    re_4 = draw_gr(ini_.copy(),  gr_dert_4[5])
    # rng = 5
    gr_5 = draw_gr(ini_.copy(), gr_dert_5[3])
    mr_5 = draw_gr(ini_.copy(), gr_dert_5[4])
    re_5 = draw_gr(ini_.copy(),  gr_dert_5[5])
    # rng = 6
    gr_6 = draw_gr(ini_.copy(), gr_dert_6[3])
    mr_6 = draw_gr(ini_.copy(), gr_dert_6[4])
    re_6 = draw_gr(ini_.copy(),  gr_dert_6[5])
    # rng = 7
    gr_7 = draw_gr(ini_.copy(), gr_dert_7[3])
    mr_7 = draw_gr(ini_.copy(), gr_dert_7[4])
    re_7 = draw_gr(ini_.copy(),  gr_dert_7[5])
    # rng = 8
    gr_8 = draw_gr(ini_.copy(), gr_dert_8[3])
    mr_8 = draw_gr(ini_.copy(), gr_dert_8[4])
    re_8 = draw_gr(ini_.copy(),  gr_dert_8[5])
    # rng = 9
    gr_9 = draw_gr(ini_.copy(), gr_dert_9[3])
    mr_9 = draw_gr(ini_.copy(), gr_dert_9[4])
    re_9 = draw_gr(ini_.copy(),  gr_dert_9[5])
    # rng = 10
    gr_10 = draw_gr(ini_.copy(), gr_dert_10[3])
    mr_10 = draw_gr(ini_.copy(), gr_dert_10[4])
    re_10 = draw_gr(ini_.copy(),  gr_dert_10[5])


    # save to disk
    cv2.imwrite(arguments.output + '0_g.jpg',  g_)
    cv2.imwrite(arguments.output + '1_m.jpg',  m_)

    cv2.imwrite(arguments.output + 'rng1_gr.jpg',  gr_1)
    cv2.imwrite(arguments.output + 'rng1_mr.jpg',  mr_1)
    cv2.imwrite(arguments.output + 'rgn1_re.jpg',  re_1)

    cv2.imwrite(arguments.output + 'rng2_gr.jpg',  gr_2)
    cv2.imwrite(arguments.output + 'rng2_mr.jpg',  mr_2)
    cv2.imwrite(arguments.output + 'rgn2_re.jpg',  re_2)

    cv2.imwrite(arguments.output + 'rng3_gr.jpg',  gr_3)
    cv2.imwrite(arguments.output + 'rng3_mr.jpg',  mr_3)
    cv2.imwrite(arguments.output + 'rgn3_re.jpg',  re_3)

    cv2.imwrite(arguments.output + 'rng4_gr.jpg',  gr_4)
    cv2.imwrite(arguments.output + 'rng4_mr.jpg',  mr_4)
    cv2.imwrite(arguments.output + 'rgn4_re.jpg',  re_4)

    cv2.imwrite(arguments.output + 'rng5_gr.jpg',  gr_5)
    cv2.imwrite(arguments.output + 'rng5_mr.jpg',  mr_5)
    cv2.imwrite(arguments.output + 'rgn5_re.jpg',  re_5)

    cv2.imwrite(arguments.output + 'rng6_gr.jpg',  gr_6)
    cv2.imwrite(arguments.output + 'rng6_mr.jpg',  mr_6)
    cv2.imwrite(arguments.output + 'rgn6_re.jpg',  re_6)

    cv2.imwrite(arguments.output + 'rng7_gr.jpg',  gr_7)
    cv2.imwrite(arguments.output + 'rng7_mr.jpg',  mr_7)
    cv2.imwrite(arguments.output + 'rgn7_re.jpg',  re_7)

    cv2.imwrite(arguments.output + 'rng8_gr.jpg',  gr_8)
    cv2.imwrite(arguments.output + 'rng8_mr.jpg',  mr_8)
    cv2.imwrite(arguments.output + 'rgn8_re.jpg',  re_8)

    cv2.imwrite(arguments.output + 'rng9_gr.jpg',  gr_9)
    cv2.imwrite(arguments.output + 'rng9_mr.jpg',  mr_9)
    cv2.imwrite(arguments.output + 'rgn9_re.jpg',  re_9)

    cv2.imwrite(arguments.output + 'rng10_gr.jpg',  gr_10)
    cv2.imwrite(arguments.output + 'rng10_mr.jpg',  mr_10)
    cv2.imwrite(arguments.output + 'rgn10_re.jpg',  re_10)

    print('Done...')


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