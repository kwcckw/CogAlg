'''
test shift_img function based on different rng
'''

import numpy as np

rng = 4

if rng == 1:
    img = np.array([[11,12,13],
                    [18,00,14],
                    [17,16,15]])

elif rng ==2:
     img = np.array([[11,12,13,14,15],
                     [26,00,00,00,16],
                     [25,00,00,00,17],
                     [24,00,00,00,18],
                     [23,22,21,20,19]])   
    
elif rng ==3: 
    img = np.array([[11,12,13,14,15,16,17],
                    [34,00,00,00,00,00,18],
                    [33,00,00,00,00,00,19],
                    [32,00,00,00,00,00,20],
                    [31,00,00,00,00,00,21],
                    [30,00,00,00,00,00,22],
                    [29,28,27,26,25,24,23]])
    
    
elif rng ==4:    
    img = np.array([[11,12,13,14,15,16,17,18,19],
                    [42,00,00,00,00,00,00,00,20],
                    [41,00,00,00,00,00,00,00,21],
                    [40,00,00,00,00,00,00,00,22],
                    [39,00,00,00,00,00,00,00,23],
                    [38,00,00,00,00,00,00,00,24],
                    [37,00,00,00,00,00,00,00,25],
                    [36,00,00,00,00,00,00,00,26],
                    [35,34,33,32,31,30,29,28,27]])


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
    
    # get shifted images if output size >0
    if output_size_y>0 and output_size_x>0:
    
        for i in range(total_shift_direction):
           
            # get images in shifted direction    
            if (x<=0 and y<=0) :
                if y == -rng:    
                    img_shift = img[:y*2, rng+x:(x*2)-(rng+x)]
                elif x == -rng:
                    img_shift = img[rng+y:(y*2)-(rng+y),:x*2]          
            elif x>0 and y<=0:
                if x == rng:
                    img_shift = img[rng+y:(y*2)-(rng+y), rng+x:]         
                else:
                    img_shift = img[rng+y:(y*2)-(rng+y), rng+x:x-rng]
            elif x<=0 and y>0:  
                if y == rng:
                    img_shift = img[rng+y:, rng+x:(x*2)-(rng+x)]       
                else:
                    img_shift = img[rng+y:y-rng, rng+x:(x*2)-(rng+x)]  
            elif x>0 and y>0:  
                if x == rng and y == rng:
                    img_shift = img[rng+y:, rng+x:]
                elif x == rng:
                    img_shift = img[rng+y:y-rng, rng+x:]
                elif y == rng:
                    img_shift = img[rng+y:, rng+x:x-rng]
        
        
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

    return img_shift_

img_shift_ = shift_img(img,rng)
print(img_shift_)

