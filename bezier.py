import os
import sys
import cv2
import pickle
import numpy as np
from scipy.special import comb


class BezierCurveFit(object):
    def __init__(self, num_control_points, n_times=100):
        ''' fit a Bezier curve from given data
        @num_control_points: number of control points, num_control_points-1: order of bezier curve
        @n_times: number of averagely sampled points on $t\in (0,1)$
        '''
        self.num_control_points = num_control_points
        self.n_times = n_times

    @staticmethod
    def bernstein_poly(i, n, t):
        ''' return coefficient of control point in Bezier curve formula
        @i, n: used for Combination Number Formula C(n,i)
        @t: sampled value of $t\in (0,1)$
        '''
        return comb(n,i) * (t**(n-i)) * ((1-t)**i)
    
    def ployline(self, data):
        ''' construct a polyline from given data points
        @data: a matrix of data points, rows: data number, cols: dimension, order: up -> down
        '''
        self.data = data
        # dat_num:number of data points, data_num-1: number of lines/segments
        self.line_num = data.shape[0]-1
        self.line_len = np.linalg.norm(data[1:]-data[:-1], axis=1)
        length = np.sum(self.line_len)
        self.line_len = self.line_len/length.reshape(-1)
        self.line_cumlen = np.cumsum(self.line_len, axis=0)
        self.line_cumlen = np.append([0], self.line_cumlen, axis=0)

    def polyline_point(self, t):
        ''' return point corrdinate on polyline referred by t
        @t: time parameter, $t\in (0,1)$, scale or vector
        '''
        assert np.min(t)>-0.000001 and np.max(t) <1.000001, 't should be in [0,1]'
        # t lies on line line_p-1,  $line_p-1 \in {0,1,...,line_num-1}$
        line_p = np.searchsorted(self.line_cumlen, t)
        line_p[0], line_p[-1] = 1, self.line_num 
        ratio = ((t-self.line_cumlen[line_p-1])/self.line_len[line_p-1]).reshape(-1,1)
        points = ratio * self.data[line_p] + (1-ratio) * self.data[line_p-1]
        return points

    def fit(self, data):
        ''' fit a Bezier curve from given data
        @data: a matrix of data points, rows: data number, cols: dimension, order: up -> down
        '''
        # get line length of polylines constructed by data points
        self.ployline(data)
        # get target points corresponding to control points
        self.T = np.linspace(0,1,self.n_times)
        target_points = self.polyline_point(self.T)
        coefficients = np.array([[self.bernstein_poly(i, self.num_control_points-1, t)\
                for i in range(self.num_control_points)] for t in self.T])
        control_points = np.matmul(np.linalg.pinv(coefficients), target_points)
        return control_points


def bezier_curve(control_points, n_times=1000):
    ''' get x and y values for bezier curve defined by control points
    @ control_points: n x 2 matrix, control points of bezier curve
    @ n_times: number of sampled points on bezier curve
    '''
    num_points = control_points.shape[0]
    T = np.linspace(0,1,n_times)
    coefficients = np.array([[BezierCurveFit.bernstein_poly(i, num_points-1, t)\
            for i in range(num_points)] for t in T])
    curve_points = np.matmul(coefficients, control_points)
    return curve_points


def branch2_from_mouth_landmark(mouth_ldmk):
    # get 2 lip curves (inner_up, inner_down)
    mouth_ldmk = mouth_ldmk.reshape(-1,3)
    # frontalize
    center = (mouth_ldmk[48-48]+mouth_ldmk[54-48])/2
    rotate = MotionAnimator.rotate_u2v((mouth_ldmk[54-48]-center).reshape(3,-1), np.array([[np.linalg.norm(mouth_ldmk[54-48]-mouth_ldmk[48-48])/2],[0],[0]]))
    mouth_ldmk = np.matmul(rotate, (mouth_ldmk - center).T).T
    # get branches
    inner_up_points = mouth_ldmk[60-48:64+1-48]
    inner_down_points = np.concatenate((mouth_ldmk[64-48:67+1-48], mouth_ldmk[60-48:60+1-48]))
    return inner_up_points, inner_down_points


def branch2_from_eye_landmark(eye_ldmk):
    # get 2 eyelid curves (inner_up, inner_down)
    eye_ldmk = eye_ldmk.reshape(-1,3)
    # frontalize
    center = (eye_ldmk[36-36]+eye_ldmk[39-36])/2
    rotate = MotionAnimator.rotate_u2v((eye_ldmk[39-36]-center).reshape(3,-1), np.array([[np.linalg.norm(eye_ldmk[39-36]-eye_ldmk[36-36])/2],[0],[0]]))
    eye_ldmk = np.matmul(rotate, (eye_ldmk - center).T).T
    # get branches
    inner_up_points = eye_ldmk[36-36:39+1-36]
    inner_down_points = np.concatenate((eye_ldmk[39-36:41+1-36], eye_ldmk[36-36:36+1-36]))
    return inner_up_points, inner_down_points

class MotionExtractor(object):
    def __init__(self, refer_bpoints_path='./bezier_control_points/inout_5-7.pickle'):
        # read and load referred bezier control points 
        with open(refer_bpoints_path, 'rb') as f:
            refer_bpoints = pickle.load(f)
        self.refer_bpoints = {}
        self.refer_bpoints['inner_up'] = refer_bpoints['inner_up']
        self.refer_bpoints['inner_down'] = refer_bpoints['inner_down']
        # initialize bezier cureve fit
        self.bcurve_fit_in = BezierCurveFit(5)
        # max scale of x,y
        self.x_max = 0.85
        self.y_max = 0.85

    def extract(self, ldmk, cate):
        # get branches of landmarks
        if cate == 'mouth':
            inner_up_points, inner_down_points = branch2_from_mouth_landmark(ldmk)
        elif cate == 'eye':
            inner_up_points, inner_down_points = branch2_from_eye_landmark(ldmk)
        else:
            print('cate error')
            exit(0)
        # fit curves
        inner_up_bpoints = self.bcurve_fit_in.fit(inner_up_points)
        inner_down_bpoints = self.bcurve_fit_in.fit(inner_down_points)
        # get control points / refer control points
        control_points = {}
        control_points['inner_up'] = inner_up_bpoints/self.refer_bpoints['inner_up']
        control_points['inner_down'] = inner_down_bpoints/self.refer_bpoints['inner_down']
        # simple scale, debug
        for branch_name in control_points.keys():
            control_points[branch_name][:, 0] /= self.x_max
            control_points[branch_name][:, 1] /= self.y_max
            control_points[branch_name][:, 2] = 0
        # correct for remove minus values
        for branch_name in ['inner_up', 'inner_down']:
            control_points[branch_name][control_points[branch_name] < 0] = 0
        # correct for div zero
        control_points['inner_up'][0][1:], control_points['inner_up'][-1][1:], control_points['inner_up'][2][0] = 0, 0, 0
        control_points['inner_down'][0][1:], control_points['inner_down'][-1][1:], control_points['inner_down'][2][0] = 0, 0, 0
        return control_points


class MotionAnimator(object):
    def __init__(self):
        # bezier curve fit for mouth branches
        self.bcurve_fit_in = BezierCurveFit(5)

    def frontalize(self, branches):
        # use the follow two points to frontalize the branches
        inner_left = branches['inner_up'][0].reshape(3,-1)
        inner_right = branches['inner_up'][-1].reshape(3,-1)
        size_x = np.linalg.norm(inner_right-inner_left)
        inner_right_front = np.array([[size_x/2],[0],[0]])
        # R(x-t)=x', R:rotate matrix, t: translation vector, x: coords befor frontalization, x': coords after frontalization
        self.trans = (inner_left+inner_right)/2
        self.rotate = self.__class__.rotate_u2v(inner_right-self.trans, inner_right_front)
        self.inv_rotate = self.rotate.T  # inverse of self.rotate is its transpose
        # frontalize all branches
        for branch_key in branches.keys():
            branches[branch_key] = np.matmul(self.rotate, branches[branch_key].T-self.trans).T
        return branches

    @staticmethod
    def rotate_u2v(u,v):
        # rotation matrix from vector u to vector v
        a = np.cross(u.reshape(-1),v.reshape(-1)).reshape(3,-1) # a means axis, rotation axis
        # if u and v have same direction, no need to rotate
        a_size = np.linalg.norm(a)
        if a_size < np.finfo(np.float32).eps:
            return np.identity(3)
        au = a/np.linalg.norm(a)
        u_size, v_size, a_size = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(a)
        sin_theta = a_size/(u_size*v_size)      # theta: rotate angle
        cos_theta = np.dot(u.reshape(-1),v)/(u_size*v_size)
        r = np.zeros((3,3))
        r[0,0] = cos_theta+au[0]**2*(1-cos_theta)
        r[0,1] = au[0]*au[1]*(1-cos_theta)-au[2]*sin_theta
        r[0,2] = au[0]*au[2]*(1-cos_theta)+au[1]*sin_theta
        r[1,0] = au[1]*au[0]*(1-cos_theta)+au[2]*sin_theta
        r[1,1] = cos_theta+au[1]**2*(1-cos_theta)
        r[1,2] = au[1]*au[2]*(1-cos_theta)-au[0]*sin_theta
        r[2,0] = au[2]*au[0]*(1-cos_theta)-au[1]*sin_theta
        r[2,1] = au[2]*au[1]*(1-cos_theta)+au[0]*sin_theta
        r[2,2] = cos_theta+u[2]**2*(1-cos_theta)
        return r

    def register(self, branches):
        # frontalize face and get bezier control points
        self.branches = self.frontalize(branches)
        self.bpoints = dict()
        for branch_name in self.branches.keys():
            if 'inner' in branch_name:
                self.bpoints[branch_name] = self.bcurve_fit_in.fit(self.branches[branch_name])
            else:
                sys.exit('key error, unexpected key %s'%(branch_name))

    def animate(self, motion):
        # animate bezier control points by motion
        bpoints = dict()
        for branch_name in self.bpoints.keys():
            bpoints[branch_name] = (np.matmul(self.inv_rotate, (self.bpoints[branch_name]*motion[branch_name]).T) + self.trans).T
            # extract 2D control points
            bpoints[branch_name] = bpoints[branch_name][:,:2]
        return bpoints

