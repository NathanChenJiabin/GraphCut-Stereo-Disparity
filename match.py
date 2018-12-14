import numpy as np
import maxflow
from PIL import Image
from matplotlib import pyplot as plt
from pylab import *
import cv2


class Energy(object):
    pass


class Parameters:
    def __init__(self, L1, L2, denominator, edgeThresh, lambda1, lambda2, K, maxIter):
        self.maxIter = maxIter
        self.K = K
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        self.edgeThresh = edgeThresh
        self.L2 = L2
        self.L1 = L1
        self.denominator = denominator


class CONST(object):
    val = int(2 ** 31 - 1)

    def __setattr__(self, *_):
        pass


# Main class for Kolmogorov-Zabih algorithm

class Match:

    def __init__(self, imgL, imgR, color=False):
        # originalHeightL = imgL.shape[0]
        height = min(imgL.shape[0], imgR.shape[0])
        self.imSizeL = (height, imgL.shape[1])  # left image dimensions
        self.imSizeR = (height, imgR.shape[1])  # right image dimensions

        if color:
            self.imColorL = imgL
            self.imColorR = imgR
        else:
            self.imgL = imgL
            self.imgR = imgR

        self.dispMin = self.dispMax = 0  # range of disparities
        self.currentEnergy = 0  # current energy
        self.params = None  # set of parameters
        self.disparityL = np.zeros(shape=self.imSizeL, dtype=np.int64)  # disparity map
        self.vars0 = np.zeros(shape=self.imSizeL, dtype=np.int8)  # Variables before alpha expansion
        self.varsA = np.zeros(shape=self.imSizeL, dtype=np.int8)  # Variables after alpha expansion
        self.OCCLUDED = CONST()  # Special value of disparity meaning occlusion

        def setParameters(params):
            self.params = params

        def run():
            pass

        def initSubPixel():
            pass

        # Data penalty functions

        def data_penalty_gray(coordL, coordR):
            pass

        def data_penalty_color(coordL, coordR):
            pass

        # Smoothness penalty functions
        def smoothness_penalty_gray(coordP, coordNp, d):
            pass

        def smoothness_penalty_color(coordP, coordNp, d):
            pass

        # Kolmogorov-Zabih algorithm
        def data_occlusion_penalty(coordL, coordR):
            pass

        def smoothness_penalty(coordP, coordNp, d):
            pass

        def ComputeEnergy():
            pass

        def ExpansionMove(alpha):
            pass

        # Graph construction
        def build_nodes(energy, coordP, alpha):
            pass

        def build_smoothness(energy, coordP, coordNp, alpha):
            pass

        def build_uniqueness(energy, coordP, alpha):
            pass

        def update_disparity(energy, alpha):
            pass

        def kolmogorov_zabih():
            pass