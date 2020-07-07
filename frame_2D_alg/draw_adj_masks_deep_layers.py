from utils import *
from frame_blobs import *
import argparse

# Main #

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon_head.jpg')
arguments = vars(argument_parser.parse_args())
image = imread(arguments['image'])

frame = image_to_blobs(image)

print('processing intra blobs..')

intra = 1
if intra:  # Tentative call to intra_blob, omit for testing frame_blobs:

    from intra_blob import *

    deep_frame = frame, frame
    bcount = 0
    deep_blob_i_ = []
    deep_layers = []
    layer_count = 0

    for blob in frame['blob__']:
        bcount += 1
        # print('Processing blob number ' + str(bcount))
        # blob.update({'fcr': 0, 'fig': 0, 'rdn': 0, 'rng': 1, 'ls': 0, 'sub_layers': []})

        if blob['sign']:
            if blob['Dert']['G'] > aveB and blob['Dert']['S'] > 20 and blob['dert__'].shape[1] > 3 and blob['dert__'].shape[2] > 3:
                blob = update_dert(blob)

                deep_layers.append(intra_blob(blob, rdn=1, rng=.0, fig=0, fcr=0))  # +G blob' dert__' comp_g
                layer_count += 1

        elif -blob['Dert']['G'] > aveB and blob['Dert']['S'] > 6 and blob['dert__'].shape[1] > 3 and blob['dert__'].shape[2] > 3:

            blob = update_dert(blob)

            deep_layers.append(intra_blob(blob, rdn=1, rng=1, fig=0, fcr=1))  # -G blob' dert__' comp_r in 3x3 kernels
            layer_count += 1

        if len(deep_layers) > 0:
            if len(deep_layers[layer_count - 1]) > 2:
                deep_blob_i_.append(bcount)  # indices of blobs with deep layers

# Draw blobs --------------------------------------------------------------

print('drawing blobs..')

for root_blob_count, blob__ in enumerate(deep_layers):
   
    if len(blob__)>0:
    
        for layer_count, blob_ in enumerate(blob__):
    
            for blob_count,blob in enumerate(blob_):
    
                # initialize image with 3 channels (colour image)
                img_blob_ = np.zeros((blob['root_dert__'].shape[1], blob['root_dert__'].shape[2], 3))
                img_blob_box = img_blob_.copy()
            
                # check if there are adjacent blobs and there are unmasked values
                if blob['adj_blob_'] and False in blob['dert__'][0].mask:
                    dert__mask = ~blob['dert__'][0].mask  # get inverted mask value (we need plot mask = false)
                    dert__mask = dert__mask * 255  # set intensity of colour
            
                    # draw blobs into image
                    # current blob - whilte colour
                    img_blob_[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3], 0] += dert__mask
                    img_blob_[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3], 1] += dert__mask
                    img_blob_[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3], 2] += dert__mask
                    img_blob_box[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3], 0] += dert__mask
                    img_blob_box[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3], 1] += dert__mask
                    img_blob_box[blob['box'][0]:blob['box'][1], blob['box'][2]:blob['box'][3], 2] += dert__mask
            
                    # draw bounding box
                    cv2.rectangle(img_blob_box, (blob['box'][2], blob['box'][0]),
                                  (blob['box'][3], blob['box'][1]),
                                  color=(255, 255, 255), thickness=1)
            
                    for j, adj_blob in enumerate(blob['adj_blob_'][0]):
            
                        # check if there are unmasked values
                        if False in adj_blob['dert__'][0].mask:
                            adj_dert__mask = ~adj_blob['dert__'][0].mask  # get inverted mask value (we need plot mask = false)
                            adj_dert__mask = adj_dert__mask * 255  # set intensity of colour
            
                            if blob['adj_blob_'][1][j] == 1:  # external blob, colour = green
                                # draw blobs into image
                                img_blob_[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3], 1] += adj_dert__mask
                                img_blob_box[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3], 1] += adj_dert__mask
            
                                # draw bounding box
                                cv2.rectangle(img_blob_box, (adj_blob['box'][2], adj_blob['box'][0]),
                                              (adj_blob['box'][3], adj_blob['box'][1]),
                                              color=(0, 155, 0), thickness=1)
            
                            elif blob['adj_blob_'][1][j] == 0:  # internal blob, colour = red
                                # draw blobs into image
                                img_blob_[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3], 2] += adj_dert__mask
                                img_blob_box[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3], 2] += adj_dert__mask
            
                                # draw bounding box
                                cv2.rectangle(img_blob_box, (adj_blob['box'][2], adj_blob['box'][0]),
                                              (adj_blob['box'][3], adj_blob['box'][1]),
                                              color=(0, 0, 155), thickness=1)
                            else:  # open， colour = blue
                                # draw blobs into image
                                img_blob_[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3], 0] += adj_dert__mask
                                img_blob_box[adj_blob['box'][0]:adj_blob['box'][1], adj_blob['box'][2]:adj_blob['box'][3], 0] += adj_dert__mask
            
                                # draw bounding box
                                cv2.rectangle(img_blob_box, (adj_blob['box'][2], adj_blob['box'][0]),
                                              (adj_blob['box'][3], adj_blob['box'][1]),
                                              color=(155, 0, 0), thickness=1)
            
                        else:
                            break
            
                    cv2.imwrite("./images/adj_blob_masks_deep_layers/mask_adj_blob_" + str(root_blob_count) + '_' + str(layer_count)+'_' + str(blob_count)+".png", img_blob_.astype('uint8'))
                    cv2.imwrite("./images/adj_blob_masks_deep_layers/mask_adj_blob_" + str(root_blob_count) + '_' + str(layer_count)+'_' + str(blob_count)+ "_box.png", img_blob_box.astype('uint8'))

print('Done !!')