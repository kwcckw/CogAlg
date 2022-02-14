
from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
from frame_blobs import *
from frame_bblobs import *

def frame_recursive(image, intra, render, verbose):

    frame = frame_blobs(image, intra, render, verbose)
    frames = frame_bblobs(frame, intra, render, verbose)
     
    return frame_level_root(frames)
    


def frame_level_root(frames):
    
    current_frame = frames[-1]
    
    # recursive function here
    # frames.append(frame_out)
    
    





