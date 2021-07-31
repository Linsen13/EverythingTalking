import os
import sys
import pickle
import numpy as np
import face_alignment
from skimage import io


# 3d landmark register and alignment
class TemplateAlign3d(object):
    def __init__(self, template_path='./preprocess_dataset/template_68_3d.pickle'):
        with open(template_path, 'rb') as f:
            data=pickle.load(f)
        assert data.shape == (68,3), 'Template landmark shape should be (68,3)'
        self.ones = np.ones((68,1), dtype=np.float32)
        self.target_ldmk = np.concatenate((data, self.ones), 1)

    def align(self, source_ldmk):
        assert source_ldmk.shape == (68,3), 'Source landmark shape should be (68,3)'
        self.source_ldmk = np.concatenate((source_ldmk, self.ones), 1)
        tran = np.matmul(self.target_ldmk.T, np.linalg.pinv(self.source_ldmk.T))
        esti = np.matmul(self.source_ldmk, tran.T)[:,:3]
        return esti, tran

