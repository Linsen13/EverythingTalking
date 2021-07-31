import os
import sys
import cv2


class VideoWriter(object):
    def __init__(self, path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.path = path
        self.out = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

    def write_frame(self, frame):
        self.out.write(frame)

def landmark3d_center(landmark3d):
    center_x = (landmark3d[3*0]+landmark3d[3*6])/2
    center_y = (landmark3d[3*0+1]+landmark3d[3*6+1])/2
    center_z = (landmark3d[3*0+2]+landmark3d[3*6+2])/2
    return center_x,center_y,center_z



