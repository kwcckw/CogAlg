import numpy.ma as ma
import numpy as np
'''
comp_pixel (lateral, vertical, diagonal) forms dert, queued in dert__: tuples of pixel + derivatives, over the whole frame.
Coefs scale down pixel dy and dx contribution to kernel g in proportion to the ratio of that pixel distance and angle 
to ortho-pixel distance and angle. This is a proximity-ordered search, comparing ortho-pixels first, thus their coef = 1.  

This is a more precise equivalent to Sobel operator, but works in reverse, the latter sets diagonal pixel coef = 1, and scales 
contribution of other pixels up, in proportion to the same ratios (relative contribution of each rim pixel to g in Sobel is 
similar but lower resolution). This forms integer coefs, vs our fractional coefs, which makes computation a lot faster. 
We will probably switch to integer coefs for speed, and are open to using Scharr operator in the future.

kwidth = 3: input-centered, low resolution kernel: frame | blob shrink by 2 pixels per row,
kwidth = 2: co-centered, grid shift, 1-pixel row shrink, no deriv overlap, 1/4 chance of boundary pixel in kernel?
kwidth = 2: quadrant g = ((dx + dy) * .705 + d_diag) / 2, no i res decrement, ders co-location, + orthogonal quadrant for full rep?
'''
# Constants:
MAX_G = 255  # 721.2489168102785 without normalization

G_NORMALIZER_3x3 = 255.9 / (255 * 2 ** 0.5 * 2)
G_NORMALIZER_2x2 = 255.9 / (255 * 2 ** 0.5)

# Coefficients and translating slices sequence for 3x3 window comparison
XCOEF = np.array([-0.5, 0, 0.5, 1, 0.5, 0, -0.5,-1])/4
YCOEF = np.array([-0.5, -1, -0.5, 0, 0.5, 1, 0.5,0])/4

def comp_pixel(image):  # 3x3 and 2x2 pixel cross-correlation within image

    gdert__ = comp_2x2(image)  # cross-compare four adjacent pixels diagonally
    rdert__ = comp_3x3(image)  # compare each pixel to 8 rim pixels

    return gdert__, rdert__


def comp_2x2(image):
    
    # initialize empty array
    dx__ = np.zeros((image.shape[0]-1,image.shape[1]-1,))
    dy__ = np.zeros((image.shape[0]-1,image.shape[1]-1,))
    p__ = np.zeros((image.shape[0]-1,image.shape[1]-1,))

    # slide across each pixel, compute dy,dx and reconstruct central pixel
    for y in range(image.shape[0]-1):
        for x in range(image.shape[1]-1):
                
            # 2x2 kernel, given 4 pixels below:
            #  p1 p2 
            #  p3 p4
            # dx  = ((p2-p3) + (p4-p1)) /2 (get mean of diagonal difference)
            # dy  = ((p4-p1) + (p3-p2)) /2 (get mean of diagonal difference)
            # p__ = (p1+p2+p3+p4) /4
            
            dx__[y,x] = ((image[y,x+1] - image[y+1,x]) + \
                         (image[y+1,x+1] - image[y,x]))/2
                 
            dy__[y,x] = ((image[y+1,x+1] - image[y,x]) + \
                            (image[y+1,x] - image[y,x+1]))/2     
            
            p__[y,x] = ((image[y,x] + image[y+1,x]) + \
                       (image[y,x+1] + image[y+1,x+1]))/4
        
    g__ = np.hypot(dy__, dx__)  # compute gradients per kernel, converted to 0-255 range
    
    return ma.stack((p__, g__, dy__, dx__))


def comp_3x3(image):
    
    # initialization
    dy__ = np.zeros((image.shape[0]-2,image.shape[1]-2))
    dx__ = np.zeros((image.shape[0]-2,image.shape[1]-2))
    
    # start in 2nd pixels and end at 2nd last pixels
    # 1st and last pixels doesn't have sufficient 8 surrounding pixels
    for y in range(image.shape[0]-2):
        for x in range(image.shape[1]-2):
            
            # 3x3 kernel, given 9 pixels below:
            #  p1 p2 p3
            #  p8 c1 p4
            #  p7 p6 p5
            # directional differences, d = p - c
            # dx = sum(8 directional differences * x coefficients) 
            # dy = sum(8 directional differences * y coefficients) 
            # p__ = central pixel , which is c1
            
            
            # get difference in 8 directions = directional pixels - center pixel
            d_top_left     = (image[y][x]     - image[y+1][x+1])
            d_top          = (image[y][x+1]   - image[y+1][x+1])
            d_top_right    = (image[y][x+2]   - image[y+1][x+1])
            d_right        = (image[y+1][x+2] - image[y+1][x+1])
            d_bottom_right = (image[y+2][x+2] - image[y+1][x+1])
            d_bottom       = (image[y+2][x+1] - image[y+1][x+1])
            d_bottom_left  = (image[y+2][x]   - image[y+1][x+1])
            d_left         = (image[y+1][x]   - image[y+1][x+1])
                   
                           
            dy__[y,x] =   (d_top_left     * YCOEF[0]) +\
                          (d_top          * YCOEF[1]) +\
                          (d_top_right    * YCOEF[2]) +\
                          (d_right        * YCOEF[3]) +\
                          (d_bottom_right * YCOEF[4]) +\
                          (d_bottom       * YCOEF[5]) +\
                          (d_bottom_left  * YCOEF[6]) +\
                          (d_left         * YCOEF[7])
                             
            dx__[y,x] =   (d_top_left     * XCOEF[0]) +\
                          (d_top          * XCOEF[1]) +\
                          (d_top_right    * XCOEF[2]) +\
                          (d_right        * XCOEF[3]) +\
                          (d_bottom_right * XCOEF[4]) +\
                          (d_bottom       * XCOEF[5]) +\
                          (d_bottom_left  * XCOEF[6]) +\
                          (d_left         * XCOEF[7])
    
    p__ = image[1:-1, 1:-1]     # central pixels
    g__ = np.hypot(dy__, dx__)  # compute gradients per kernel, converted to 0-255 range

    return ma.stack((p__, g__, dy__, dx__))