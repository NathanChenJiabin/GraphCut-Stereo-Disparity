import numpy as np
import maxflow
from PIL import Image
from matplotlib import pyplot as plt
from pylab import *
import energy
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
            self.imgL = imgL
            self.imgR = imgR
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

    def SetDispRange(self, dMin, dMax):
        """
         Specify disparity range
         :param dMin: int
         :param dMax: int
         :return: void
        """
        self.dispMin = dMin
        self.dispMax = dMax
        self.disparityL = np.ones_like(self.disparityL, dtype=np.int64) * self.OCCLUDED.val

    def setParameters(self, params):
        self.params = params

    def run(self):
        """
        Main algorithm: run a series of alpha-expansions
        :return: void
        """
        dispSize = self.dispMax - self.dispMin + 1
        permutation = np.random.permutation(dispSize)  # random permutation

        self.currentEnergy = self.ComputeEnergy()
        print("Initial energy : " + str(self.currentEnergy))

        done = np.full(dispSize, False)
        nDone = dispSize  # number of False in done

        step = 0
        for iteration in range(self.params.maxIter):
            if nDone <= 0:
                break

            for idx in range(dispSize):
                label = permutation[idx]

                if done[label]:
                    continue

                step += 1
                if self.ExpansionMove(self.dispMin + label):
                    done = np.full(dispSize, False)
                    nDone = dispSize

                done[label] = True
                nDone -= 1

            print("Energy :" + str(self.currentEnergy))

        print(str(step / dispSize) + "iterations...")

    def initSubPixel(self):
        pass

    # Data penalty functions

    def data_penalty_gray(self, coordL, coordR):
        pass

    def data_penalty_color(self, coordL, coordR):
        pass

    # Smoothness penalty functions
    def smoothness_penalty_gray(self, coordP, coordNp, d):
        pass

    def smoothness_penalty_color(self, coordP, coordNp, d):
        pass

    # Kolmogorov-Zabih algorithm
    def data_occlusion_penalty(self, coordL, coordR):
        pass

    def smoothness_penalty(self, coordP, coordNp, d):
        pass

    def ComputeEnergy(self):
        return 0

    def ExpansionMove(self, alpha):
        """
        Compute the minimum a-expansion configuration
        :param alpha: int
        :return: bool, whether the move is different from identity
        """
        nb = self.imSizeL[0] * self.imSizeL[1]
        e = energy.Energy(2 * nb, 12 * nb)

        # Build Graph
        for index, x in np.ndenumerate(self.imgL):
            self.build_nodes(e, index, alpha)

        # TODO: smooth term

        for index, x in np.ndenumerate(self.imgL):
            self.build_uniqueness(e, index, alpha)

        oldEnergy = self.currentEnergy
        # Max-flow, give the lowest-energy expansion move
        self.currentEnergy = e.minimize()

        # lower energy, accept the expansion move
        if self.currentEnergy < oldEnergy:
            self.update_disparity(e, alpha)
            assert (self.ComputeEnergy() == self.currentEnergy)
            return True

        return False

    # Graph construction
    def build_nodes(self, energy, coordP, alpha):
        pass

    def build_smoothness(self, energy, coordP, coordNp, alpha):
        pass

    def build_uniqueness(self, energy, coordP, alpha):
        pass

    def update_disparity(self, energy, alpha):
        pass

    def kolmogorov_zabih(self):
        # verify parameters

        # print parameters

        self.run()
        return
