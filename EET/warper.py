import os
import sys
import cv2
import time
import numpy as np
from bezier import bezier_curve
from scipy.interpolate import interp1d,interp2d


def scale_location(bcurves, point):
    ''' get scale factor and location time(0<=t<=2*n_times-1) by given point related to bezier curves
    @ bcurves: points of bezier curves, inner_up and inner_down
    @ point: given point
    '''
    point = point[0]
    n_times = bcurves['inner_up'].shape[0]
    bcurve_points = np.concatenate((bcurves['inner_up'],bcurves['inner_down']), axis=0)
    len_points = np.linalg.norm(bcurve_points[:,:2], axis=1)
    sin_points = bcurve_points[:,1]/len_points
    cos_points = bcurve_points[:,0]/len_points
    len_point = np.linalg.norm(point[:2])
    sin_point = point[1]/len_point
    cos_point = point[0]/len_point
    point_idx = np.argmin(np.abs(sin_points-sin_point)+np.abs(cos_points-cos_point))
    scale = len_point/len_points[point_idx]
    time = point_idx
    return scale, time


def aff_points(points, rotate, trans):
    ''' affine point/points by rotate and trans x'=R(x-t)
    @ points: points need to affine
    @ rotate: rotation matrix
    @ trans: translation vector
    '''
    aff_points = np.matmul(rotate, points.T-trans).T
    return aff_points


def invaff_points(points, rotate, trans):
    ''' inverse affine point/points by rotate and trans x=inv(R)x'+t
    @ points: points need to inverse affine
    @ rotate: rotation matrix
    @ trans: translation vector
    '''
    invaff_points = (np.matmul(rotate.T, points.T)+trans).T
    return invaff_points


def get_point(bcurve, scale, time):
    ''' return point location on bcurve*scale at location(time)
    @ bcurve: points on bezier curve
    @ scale: scale factor for input bezier curve
    @ time: location on input bezier curve
    '''
    n_times = bcurve['inner_up'].shape[0]
    if time < n_times:
        point = bcurve['inner_up'][time:time+1] * scale
    else:
        point = bcurve['inner_down'][time-n_times:time-n_times+1] * scale
    return point


class InsideBcurves(object):
    def __init__(self, bcurves):
        ''' judge if the given point is inside bezier curves
        @ bcurves: points of bezier curves, inner_up and inner_down
        '''
        self.bcurves = bcurves
        self.f_inner_up = interp1d(bcurves['inner_up'][:,0], bcurves['inner_up'][:,1], axis=0) 
        self.f_inner_down = interp1d(bcurves['inner_down'][:,0], bcurves['inner_down'][:,1], axis=0)

    def __call__(self, point, scale):
        ''' judge if the given point is inside bezier curves
        @ point: given point
        @ scale: scale factor for bezier curves
        '''
        point = point[0]/scale
        if point[0] <= self.bcurves['inner_up'][0][0] or point[0] >= self.bcurves['inner_up'][-1][0]:
            return False
        if point[0] <= self.bcurves['inner_down'][0][0] or point[0] >= self.bcurves['inner_down'][-1][0]:
            return False
        if point[1] > self.f_inner_up(point[0]) and point[1] < self.f_inner_down(point[0]):
            return True
        return False


class Scale2Bcurves(object):
    def __init__(self, scale_min, scale_max, src_bpoints, motion, fine=100):
        '''
        @ scale_min, scale_max: scale range
        @ src_bpoints: source bezier control points
        @ motion: scalar, vector, matrix
        @ fine: finegrain for sacle, (0.8,1.5)->(80,150)
        '''
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.src_bpoints = src_bpoints
        self.motion = motion
        self.fine = fine
        self.scale2bcurves = {}

    def scale2motion(self, s):
        # @ s: scale, independent var.
        m = {}
        s0 = 1
        for branch_name in self.motion.keys():
            if s < s0:
                m[branch_name] = self.motion[branch_name]
            else:
                m[branch_name] = self.motion[branch_name] + (s-s0)/(self.scale_max-s0) * (1-self.motion[branch_name])
        return m
    
    def motion2bcurves(self, m):
        # @ m: motion, new mouth motion for each control points
        bpoints = {}
        for branch_name in self.src_bpoints.keys():
            bpoints[branch_name] = m[branch_name]*self.src_bpoints[branch_name]
        bcurves = bcurves_from_bpoints(bpoints)
        return bcurves

    def proxy(self, s):
        # @ s: scale, scale factor of bezier curve
        s_idx = int(round(s*self.fine))
        if s_idx in self.scale2bcurves.keys():
            return self.scale2bcurves[s_idx]
        m = self.scale2motion(s)
        bcurves = self.motion2bcurves(m)
        self.scale2bcurves[s_idx] = bcurves
        return bcurves


def bcurves_from_bpoints(bpoints, n_times=200):
    ''' control points to bezier curve points
    @ bpoints: input bezier control points
    @ n_times: sample number of bezier curve
    '''
    bcurves = {}
    for branch_name in bpoints.keys():
        bcurves[branch_name] = bezier_curve(bpoints[branch_name], n_times)
    return bcurves


def warp_map(src_bpoints, tgt_bpoints, motion, rotate, trans, size, scale_min, scale_max):
    ''' get warp map from source and target bezier curve defined by control points
    @ src_bpoints: control points of source bezier curve, affined
    @ tgt_bpoints: control points of target bezier curve, affined
    @ motion: mouth motion on bezier control points, tgt_bpoints/src_bpoints
    @ rotate: rotation matrix for affine, x'=R(x-t)
    @ trans: translation vecto for affine, x'=R(x-t)
    @ size: (width, height), a tuple about image size
    @ scale_min: scalar, <1, define warp area near mouth
    @ scale_max: scalar, >1, define warp area near mouth
    '''
    # get source and target bezier curve points
    src_bcurves = bcurves_from_bpoints(src_bpoints)
    tgt_bcurves = bcurves_from_bpoints(tgt_bpoints)
    # set warp map from source image to target image
    width, height = size[0], size[1]
    # init warp map
    x_range, y_range = np.arange(0,width), np.arange(0,height)
    xx, yy = np.meshgrid(x_range, y_range)
    # warp_map at (x,y) return (x,y)
    warp_map = np.stack((yy,xx),axis=2).astype(np.int32)
    mask = np.ones((height, width), dtype=np.int32)
    m = np.ones((height, width), dtype=np.int32)
    # loop to set warped coordinates
    inside_bcurves_src = InsideBcurves(src_bcurves)
    inside_bcurves_tgt = InsideBcurves(tgt_bcurves)
    scale2bcurves = Scale2Bcurves(scale_min, scale_max, src_bpoints, motion)
    for y in range(height):
        for x in range(width):
            point = aff_points(np.array([[x,y,0]]), rotate, trans)  # affine points on image plane
            if inside_bcurves_src(point, scale_max) and not inside_bcurves_tgt(point, scale_min):
                mask[y, x] = 0
                m[y, x] = 0
    for y in range(height):
        for x in range(width):
            point = aff_points(np.array([[x,y,0]]), rotate, trans)  # affine points on image plane
            if inside_bcurves_src(point, scale_max) and not inside_bcurves_src(point, scale_min):
                # need to warp/map
                scale, time = scale_location(src_bcurves, point)
                tgt_bcurves_new = scale2bcurves.proxy(scale) 
                tgt_point = invaff_points(get_point(tgt_bcurves_new, scale, time), rotate, trans).astype(np.int32)
                ty, tx = tgt_point[0,1], tgt_point[0,0]
                ty, tx = max(min(ty, 255), 0), max(min(tx, 255), 0)
                warp_map[ty,tx,0], warp_map[ty,tx,1] = y, x  # set target pixel location
                mask[ty,tx] = 1
    return warp_map, mask


def warp_image(src_image, warp_map, mask):
    ''' warp input image by warp_map
    @ src_image: input image
    @ warp_map: warp_map, used to warp input image
    '''
    assert src_image.shape[:2] == warp_map.shape[:2], 'src_image and warp_map should have same width and height'
    height, width = src_image.shape[0], src_image.shape[1]
    # init target image
    tgt_image = np.zeros((height, width, 3), dtype=np.uint8)
    # color for miss pixel
    color = np.array((19,15,152), dtype=np.uint8)      # speaker
    for y in range(height):
        for x in range(width):
            src_pixel_y, src_pixel_x = warp_map[y,x][0], warp_map[y,x][1]
            src_pixel_y, src_pixel_x = min(max(src_pixel_y,0), 255), min(max(src_pixel_x,0),255)
            if mask[y, x] == 0:
                tgt_image[y, x] = color
            else:
                tgt_image[y, x] = src_image[src_pixel_y,src_pixel_x]
    return tgt_image


class BezierWarper(object):
    def __init__(self, src_image_path, src_control_points, rotate, trans, scale_min=0.8, scale_max=1.2):
        # set self attributes
        self.src_image_path = src_image_path
        self.src_control_points = src_control_points
        self.rotate = rotate
        self.trans = trans
        self.scale_min = scale_min
        self.scale_max = scale_max
        # read data and metadata from src_image_path
        self.src_image = cv2.imread(src_image_path)
        self.size = (self.src_image.shape[1], self.src_image.shape[0])

    def get_flow_mask(self, motion):
        tgt_control_points = {}
        for branch_name in self.src_control_points.keys():
            tgt_control_points[branch_name] = self.src_control_points[branch_name]*motion[branch_name]
        warp, mask = warp_map(self.src_control_points, tgt_control_points, motion, self.rotate, self.trans, self.size, self.scale_min, self.scale_max)
        return warp, mask

    def __call__(self, motion, src_image=None):
        tgt_control_points = {}
        for branch_name in self.src_control_points.keys():
            tgt_control_points[branch_name] = self.src_control_points[branch_name]*motion[branch_name]
        warp, mask = warp_map(self.src_control_points, tgt_control_points, motion, self.rotate, self.trans, self.size, self.scale_min, self.scale_max)
        if src_image is None:
            tgt_image = warp_image(self.src_image, warp, mask)
        else:
            tgt_image = warp_image(src_image, warp, mask)
        return tgt_image

