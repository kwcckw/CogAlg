from class_cluster import ClusterStructure, NoneType, comp_param, Cdert
from frame_blobs import *
from frame_bblobs import *


def frame_recursive(image, intra, render, verbose):
    
    frame = derts2blobs(image, intra, render, verbose, use_c=False)
    frame_bblobs(frame, intra, render, verbose)

    return frame_level_root(frame)


def frame_level_root(frame):
    current_frame = frame[-1]

    # recursive function here
    # frames.append(frame_out)


