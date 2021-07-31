import os
import sys
import cv2
import pickle
from skimage import io
import numpy as np
import torch
import argparse
from utils import VideoWriter, landmark3d_center
from preprocess_dataset.extract_register_landmark3d_video import TemplateAlign3d
from points import image_points
from warper import BezierWarper
from bezier import MotionExtractor, MotionAnimator


parser = argparse.ArgumentParser()
parser.add_argument('--test_video', type=str, default='./test/result_.mp4', help='test directory')
parser.add_argument('--test_fps', type=float, default='29.97', help='generated video fps')
parser.add_argument('--test_ldmk', type=str, default='./test/yrb_256.txt', help='test landmark file path')
parser.add_argument('--test_image', type=str, default='./test/speaker.png', help='test face image file path')

# load input options
opts = parser.parse_args()

# get audio feature, 3d landmark (aligned, affine mat) 
image_path = opts.test_image
ldmk_path = opts.test_ldmk

# set video metadata: resolution, fps, video duration, video frame number
face = cv2.imread(image_path)
height, width = face.shape[0], face.shape[1]
warp_video_path = opts.test_video 
warp_video_writer = VideoWriter(warp_video_path, width, height, opts.test_fps)

# load pca
with open('data/pcaw.pickle', 'rb') as f:
    pca = pickle.load(f)
# debug
pca_comp = np.array(pca.components_, copy=True)
pca_comp[:,::1] = pca_comp[:,::1] * 0.4

# read landmark list
ldmk3d_dict = {}
template_align = TemplateAlign3d()
f=open(ldmk_path, 'r')
lines = f.readlines()
for line in lines:
    info = line.strip().split()
    # frame_idx = int(info[-1].split('.')[0])
    frame_idx = int(info[-1].split('.')[0])
    ldmk3d = np.array([float(it) for it in info[:-1]]).reshape((-1, 3))
    ldmk3d, _ = template_align.align(ldmk3d)
    ldmk3d_dict[frame_idx] = ldmk3d.reshape((68*3,))
frame_num = len(ldmk3d_dict.keys())

# shapes of mouth and eyes
name = opts.test_image.split('/')[-1].split('.')[0]
mouth_shape = image_points[name]['mouth_shape']
left_eye_shape = image_points[name]['left_eye_shape']
right_eye_shape = image_points[name]['right_eye_shape']

# mouth and eyes motion extractor and animator
motion_extractor = MotionExtractor()
eye_motion_extractor = MotionExtractor()

mouth_animator = MotionAnimator()
mouth_animator.register(mouth_shape)
bezier_warper = BezierWarper(image_path, mouth_animator.bpoints, mouth_animator.rotate, mouth_animator.trans, 0.8, 1.2)

left_eye_animator = MotionAnimator()
left_eye_animator.register(left_eye_shape)
leye_bezier_warper = BezierWarper(image_path, left_eye_animator.bpoints, left_eye_animator.rotate, left_eye_animator.trans, 0.9, 2)

right_eye_animator = MotionAnimator()
right_eye_animator.register(right_eye_shape)
reye_bezier_warper = BezierWarper(image_path, right_eye_animator.bpoints, right_eye_animator.rotate, right_eye_animator.trans, 0.9, 2)

# main test code
with torch.no_grad():
    # loop to generate video frame by frame
    # for frame_idx in range(frame_num):
    for frame_idx in [28,209,233,368,460]:     # yrb_256
        ldmk3d_pred = ldmk3d_dict[frame_idx][48*3:68*3]
        center_x, center_y,center_z = landmark3d_center(ldmk3d_pred)
        ldmk3d_pred[0::3] = ldmk3d_pred[0::3] - center_x  # 0,3,... x
        ldmk3d_pred[1::3] = ldmk3d_pred[1::3] - center_y  # 1,4,... y
        ldmk3d_pred[2::3] = ldmk3d_pred[2::3] - center_z  # 2,5,... z
        # pca debug
        ldmk3d_pred_new = pca.transform(ldmk3d_pred.reshape(1,-1)).reshape(-1)
        #ldmk3d_pred_new = pca.inverse_transform(ldmk3d_pred_new)
        ldmk3d_pred_new = np.dot(ldmk3d_pred_new, np.sqrt(pca.explained_variance_[:, np.newaxis]) * pca_comp) + pca.mean_
        # mouth movement
        mouth_motion = motion_extractor.extract(ldmk3d_pred_new, cate='mouth')
        warp_image = bezier_warper(mouth_motion)
        # eye movement
        leye_motion = eye_motion_extractor.extract(ldmk3d_dict[frame_idx][36*3:42*3], cate='eye')
        warp_image = leye_bezier_warper(leye_motion, warp_image)
        reye_motion = eye_motion_extractor.extract(ldmk3d_dict[frame_idx][42*3:48*3], cate='eye')
        warp_image = reye_bezier_warper(reye_motion, warp_image)
        warp_video_writer.write_frame(warp_image)
        sys.stdout.write('\rWrite frames for video: %05d/%05d'%(frame_idx+1, frame_num))
    print()








