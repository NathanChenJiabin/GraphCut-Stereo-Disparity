import energy
import numpy as np
import preprocessing
import cv2
np.seterr(over='ignore')


# Define global variables
VAR_ALPHA = -1
VAR_ABSENT = -2
NEIGHBORS = [(0, -1), (1, 0)]
CUTOFF = 30
MAX_DENOM = int(2 ** 4)


def set_fractions(params, K, lambda1, lambda2):
    minError = float(2 ** 30)
    for i in range(1, MAX_DENOM + 1):
        e = 0
        numK = 0
        num1 = 0
        num2 = 0
        if K > 0:
            numK = int(i * K + .5)
            e += abs(numK / (i * K) - 1.0)

        if lambda1 > 0:
            num1 = int(i * lambda1 + .5)
            e += abs(num1 / (i * lambda1) - 1.0)

        if lambda2 > 0:
            num2 = int(i * lambda2 + .5)
            e += abs(num2 / (i * lambda2) - 1.0)

        if e < minError:
            minError = e
            params.denominator = i
            params.K = numK
            params.lambda1 = num1
            params.lambda2 = num2
    return params


def fix_parameters(mch, params, K, lambda_, lambda1, lambda2):
    """

    :param mch: Match object
    :param params:
    :param K:
    :param lambda_:
    :param lambda1:
    :param lambda2:
    :return:
    """
    if K < 0:
        mch.setParameters(params)
        K = mch.getK()
    if lambda_ < 0:
        lambda_ = K / 5
    if lambda1 < 0:
        lambda1 = 3 * lambda_
    if lambda2 < 0:
        lambda2 = lambda_
    params = set_fractions(params, K, lambda1, lambda2)
    mch.setParameters(params)
    print("Fix parameters finished...")
    return mch


def coord_add(coordP, coordQ):
    """

    :param coordP:
    :param coordQ:
    :return:
    """
    res = (coordP[0] + coordQ[0], coordP[1] + coordQ[1])
    return res


def inRect(coordP, coordR):
    """
    Is p inside rectangle r?
    :param coordP:
    :param coordR:
    :return:
    """
    return 0 <= coordP[0] and 0 <= coordP[1] and coordP[0] < coordR[0] and coordP[1] < coordR[1]


def dist_interval(v, mi, ma):
    """
    Distance from v to interval [min,max]
    """
    if v < mi:
        return mi - v
    if v > ma:
        return v - ma
    return 0


class Parameters:
    def __init__(self, is_L2, denominator, edgeThresh, lambda1, lambda2, K, maxIter):
        self.maxIter = maxIter
        self.K = K
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        self.edgeThresh = edgeThresh
        self.L2 = is_L2
        self.denominator = denominator


class CONST(object):
    val = int(2 ** 31 - 1)

    def __setattr__(self, *_):
        pass


# Main class for Kolmogorov-Zabih algorithm

class Match:

    def __init__(self, imgLeft, imgRight, color):
        """
        Constructor
        :param imgLeft: left input image
        :param imgRight: right input image
        :param color: is input image color ?
        """
        self.originalHeightL = imgLeft.shape[0]
        height = min(imgLeft.shape[0], imgRight.shape[0])
        self.imSizeL = (height, imgLeft.shape[1])  # left image dimensions
        self.imSizeR = (height, imgRight.shape[1])  # right image dimensions
        self.posIter = np.ones(shape=self.imSizeL)
        self.color = color
        if color:
            self.imgL = self.imgR = None
            self.imgLMin = self.imgRMin = None
            self.imgLMax = self.imgRMax = None
            self.imgColorL = imgLeft
            self.imgColorR = imgRight
            self.imgColorLMin = self.imgColorLMax = None
            self.imgColorRMin = self.imgColorRMax = None
        else:
            self.imgColorL = self.imgColorR = None
            self.imgColorLMin = self.imgColorMin = None
            self.imgColorLMax = self.imgColorRMax = None
            self.imgL = imgLeft
            self.imgR = imgRight
            self.imgLMin = self.imgLMax = self.imgRMin = self.imgRMax = None

        self.dispMin = self.dispMax = 0  # range of disparities
        self.currentEnergy = 0  # current energy
        self.params = None  # set of parameters
        self.disparityL = np.zeros(shape=self.imSizeL, dtype=np.int64)  # disparity map
        self.vars0 = np.zeros(shape=self.imSizeL, dtype=np.int64)  # Variables before alpha expansion
        self.varsA = np.zeros(shape=self.imSizeL, dtype=np.int64)  # Variables after alpha expansion
        self.OCCLUDED = CONST()  # Special value of disparity meaning occlusion

    def getDisparity(self):
        return self.disparityL

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
        self.initSubPixel()
        return

    def getK(self):
        """heuristic"""
        dispsize = self.dispMax - self.dispMin + 1
        k = int((dispsize + 2) / 4)
        if k < 3:
            k = 3
        array = np.ones(k)
        sum_val = 0
        num = 0
        xmin = max(0, -self.dispMin)  # 0 <= x, x + dispMin
        xmax = min(self.imSizeL[1], self.imSizeR[1] - self.dispMax)  # x < wl, x + dispMax < wr

        for y in range(min(self.imSizeL[0], self.imSizeR[0])):
            for x in range(xmin, xmax):
                # compute k'th smallest value among data_penalty(p, p+d) for all d
                i = 0
                for d in range(self.dispMin, self.dispMax + 1):
                    delta = self.data_penalty_color((y, x), (y, x + d)) if self.color else self.data_penalty_gray(
                        (y, x), (y, x + d))
                    if i < k:
                        array[i] = delta
                        i += 1
                    else:
                        for j in range(k):
                            if delta < array[j]:
                                tmp = delta
                                delta = array[j]
                                array[j] = tmp
                sum_val += np.max(array)
                num += 1
        assert (num != 0 and sum_val != 0)
        K = 1.*float(sum_val)/num
        print("Computing statistics: K(data_penalty noise) = " + str(K))
        return K

    def saveDisparity(self, filename):
        """
        save scaled disparity map as 8-bit color image (gray between 64 and 255)
        """
        im = np.zeros(shape=(self.originalHeightL, self.imSizeL[1], 3), dtype=np.uint8)

        im[:, :, 0] = 0
        im[:, :, 1] = 255
        im[:, :, 2] = 255

        dispSize = self.dispMax - self.dispMin
        iterP = np.zeros(self.imSizeL)
        for idx, _ in np.ndenumerate(iterP):
            d = self.disparityL[idx]
            if d != self.OCCLUDED.val:
                if dispSize == 0:
                    c = 255
                else:
                    c = 255 * (d - self.dispMin) / dispSize

                im[idx[0], idx[1], 0] = c
                im[idx[0], idx[1], 1] = c
                im[idx[0], idx[1], 2] = c

        np.save("./results/dispMap1.npy", self.disparityL)
        np.save("./results/dispImg1.npy", im)
        cv2.imwrite(filename, im)
        print("Save disparity map successfully !")
        return

    def run(self):
        """
        Main algorithm: run a series of alpha-expansions
        :return: void
        """
        dispSize = self.dispMax - self.dispMin + 1

        self.currentEnergy = self.ComputeEnergy()
        print("Initial energy : " + str(self.currentEnergy))

        done = np.full(dispSize, False)
        nDone = dispSize  # number of False in done

        step = 0
        for iteration in range(self.params.maxIter):
            if nDone <= 0:
                break

            permutation = np.random.permutation(dispSize)  # random permutation
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
        """
        Preprocessing for faster Birchfield-Tomasi distance computation
        :return: void
        """
        if (self.imgL is not None) and (self.imgLMin is None):
            self.imgLMin = np.zeros(shape=self.imSizeL)
            self.imgLMax = np.zeros(shape=self.imSizeL)
            self.imgRMin = np.zeros(shape=self.imSizeR)
            self.imgRMin = np.zeros(shape=self.imSizeR)

            self.imgLMin, self.imgLMax = preprocessing.SubPixel(self.imgL, self.imgLMin.shape)
            self.imgRMin, self.imgRMax = preprocessing.SubPixel(self.imgR, self.imgRMin.shape)

        if (self.imgColorL is not None) and (self.imgColorLMin is None):
            self.imgColorLMin = np.zeros(shape=(self.imSizeL[0], self.imSizeL[1], 3))
            self.imgColorLMax = np.zeros(shape=(self.imSizeL[0], self.imSizeL[1], 3))
            self.imgColorRMin = np.zeros(shape=(self.imSizeR[0], self.imSizeR[1], 3))
            self.imgColorRMin = np.zeros(shape=(self.imSizeR[0], self.imSizeR[1], 3))

            self.imgColorLMin, self.imgColorLMax = preprocessing.SubPixelColor(self.imgColorL, self.imgColorLMin.shape)
            self.imgColorRMin, self.imgColorRMax = preprocessing.SubPixelColor(self.imgColorR, self.imgColorRMin.shape)

        return

    # Data penalty functions
    def data_penalty_gray(self, coordL, coordR):
        """
        Birchfield-Tomasi gray distance between pixels p and q
        """
        Ip = self.imgL[coordL]
        Iq = self.imgR[coordR]

        IpMin = self.imgLMin[coordL]
        IqMin = self.imgRMin[coordR]

        IpMax = self.imgLMax[coordL]
        IqMax = self.imgRMax[coordR]

        dp = dist_interval(Ip, IqMin, IqMax)
        dq = dist_interval(Iq, IpMin, IpMax)
        d = min(dp, dq)

        if d > CUTOFF:
            d = CUTOFF
        if self.params.L2:
            d = d * d

        return d

    def data_penalty_color(self, coordL, coordR):
        """
        Birchfield-Tomasi color distance between pixels p and q
        """
        dSum = 0
        for i in range(3):
            Ip = self.imgColorL[coordL[0], coordL[1], i]
            Iq = self.imgColorR[coordR[0], coordR[1], i]

            IpMin = self.imgColorLMin[coordL[0], coordL[1], i]
            IqMin = self.imgColorRMin[coordR[0], coordR[1], i]

            IpMax = self.imgColorLMax[coordL[0], coordL[1], i]
            IqMax = self.imgColorRMax[coordR[0], coordR[1], i]

            dp = dist_interval(Ip, IqMin, IqMax)
            dq = dist_interval(Iq, IpMin, IpMax)
            d = min(dp, dq)

            if d > CUTOFF:
                d = CUTOFF
            if self.params.L2:
                d = d * d
            dSum += d

        return dSum

    # Smoothness penalty functions
    def smoothness_penalty_gray(self, coordP1, coordP2, disp):
        """
        Smoothness penalty between assignments (p1,p1+disp) and (p2,p2+disp).
        """
        # |I1(p1)-I1(p2)| and |I2(p1+disp)-I2(p2+disp)|

        dl = abs(self.imgL[coordP1] - self.imgL[coordP2])
        dr = abs(self.imgR[coordP1[0], coordP1[1] + disp] - self.imgR[coordP2[0], coordP2[1] + disp])

        return self.params.lambda1 if (
                dl < self.params.edgeThresh and dr < self.params.edgeThresh) else self.params.lambda2

    def smoothness_penalty_color(self, coordP1, coordP2, disp):
        dMax = 0
        for i in range(3):
            d = abs(self.imgColorL[coordP1[0], coordP1[1], i] - self.imgColorL[coordP2[0], coordP2[1], i])
            if dMax < d:
                dMax = d
            d = abs(self.imgColorR[coordP1[0], coordP1[1] + disp, i] - self.imgColorR[coordP2[0], coordP2[1] + disp, i])
            if dMax < d:
                dMax = d

        return self.params.lambda1 if dMax < self.params.edgeThresh else self.params.lambda2

    # Kolmogorov-Zabih algorithm
    def data_occlusion_penalty(self, coordL, coordR):
        """
        Compute the data+occlusion penalty (D(a)-K)
        :param coordL:
        :param coordR:
        :return:  int
        """
        if self.color:
            D = self.data_penalty_color(coordL, coordR)
        else:
            D = self.data_penalty_gray(coordL, coordR)

        return self.params.denominator * D - self.params.K

    def smoothness_penalty(self, coordP1, coordP2, d):
        """
        Compute the smoothness penalty of assignments (p1,p1+d) and (p2,p2+d)
        :param coordP1:
        :param coordP2:
        :param d:
        :return: int
        """
        if self.color:
            return self.smoothness_penalty_color(coordP1, coordP2, d)
        else:
            return self.smoothness_penalty_gray(coordP1, coordP2, d)

    def ComputeEnergy(self):
        """
        Compute current energy, we use this function only for sanity check
        :return: current energy
        """
        E = 0
        for index, _ in np.ndenumerate(self.posIter):
            d1 = self.disparityL[index]

            if d1 != self.OCCLUDED.val:
                E += self.data_occlusion_penalty(index, (index[0], index[1] + d1))

            for neighbor in NEIGHBORS:
                coordP2 = coord_add(index, neighbor)
                if inRect(coordP2, self.imSizeL):
                    d2 = self.disparityL[coordP2]
                    if d1 == d2:  # smoothness satisfied
                        continue
                    if d1 != self.OCCLUDED.val and inRect((coordP2[0], coordP2[1] + d1), self.imSizeR):
                        E += self.smoothness_penalty(index, coordP2, d1)
                    if d2 != self.OCCLUDED.val and inRect((index[0], index[1] + d2), self.imSizeR):
                        E += self.smoothness_penalty(index, coordP2, d2)

        return E

    def ExpansionMove(self, alpha):
        """
        Compute the minimum a-expansion configuration
        :param alpha: int
        :return: bool, whether the move is different from identity
        """
        nb = self.imSizeL[0] * self.imSizeL[1]
        e = energy.Energy(2 * nb, 12 * nb)

        # Build Graph
        # data and occlusion term
        for index, _ in np.ndenumerate(self.posIter):
            self.build_nodes(e, index, alpha)

        # smooth term
        for index, _ in np.ndenumerate(self.posIter):
            for neighbor in NEIGHBORS:
                coordP2 = coord_add(index, neighbor)
                if inRect(coordP2, self.imSizeL):
                    self.build_smoothness(e, index, coordP2, alpha)

        # uniqueness term
        for index, _ in np.ndenumerate(self.posIter):
            self.build_uniqueness(e, index, alpha)

        oldEnergy = self.currentEnergy
        # Max-flow, give the lowest-energy expansion move
        self.currentEnergy = e.minimize()

        # lower energy, accept the expansion move
        if self.currentEnergy < oldEnergy:
            self.update_disparity(e, alpha)
            # assert (self.ComputeEnergy() == self.currentEnergy)
            return True
        else:
            self.currentEnergy = oldEnergy

        return False

    # Graph construction
    def build_nodes(self, ener, coordP, alpha):
        """
        Build nodes in graph representing data+occlusion penalty for pixel p.
        For assignments in A^0: SOURCE means active, SINK means inactive.
        For assignments in A^{alpha}: SOURCE means inactive, SINK means active.
        :param ener:
        :param coordP: index in array
        :param alpha:
        :return: void
        """
        d = self.disparityL[coordP]
        coordQ = (coordP[0], coordP[1] + d)
        if alpha == d:
            # active assignment (p,p+a) in A^a will remain active
            self.vars0[coordP] = VAR_ALPHA
            self.varsA[coordP] = VAR_ALPHA
            ener.add_constant(self.data_occlusion_penalty(coordP, coordQ))
            return

        if d != self.OCCLUDED.val:
            self.vars0[coordP] = ener.add_variable(self.data_occlusion_penalty(coordP, coordQ), 0)
        else:
            self.vars0[coordP] = VAR_ABSENT

        coordQ = (coordP[0], coordP[1] + alpha)
        if inRect(coordQ, self.imSizeR):
            # (p,p+a) in A^a can become active
            self.varsA[coordP] = ener.add_variable(0, self.data_occlusion_penalty(coordP, coordQ))
        else:
            self.varsA[coordP] = VAR_ABSENT

        return

    def build_smoothness(self, ener, coordP, coordQ, alpha):
        """
        Build smoothness term for neighbor pixels p1 and p2 with disparity a.
        :param ener:
        :param coordP:
        :param coordQ:
        :param alpha:
        :return:
        """
        d1 = self.disparityL[coordP]
        a1 = self.varsA[coordP]
        o1 = self.vars0[coordP]

        d2 = self.disparityL[coordQ]
        a2 = self.varsA[coordQ]
        o2 = self.vars0[coordQ]

        # disparity a
        if a1 != VAR_ABSENT and a2 != VAR_ABSENT:
            delta = self.smoothness_penalty(coordP, coordQ, alpha)
            if a1 != VAR_ALPHA:
                # (p1,p1+a) is variable
                if a2 != VAR_ALPHA:
                    # Penalize different activity
                    # print("must existe: "+str(a2))
                    ener.add_term2(a1, a2, 0, delta, delta, 0)
                else:
                    # Penalize (p1,p1+a) inactive
                    ener.add_term1(a1, delta, 0)
            elif a2 != VAR_ALPHA:
                # (p1,p1+a) active, (p2,p2+a) variable
                ener.add_term1(a2, delta, 0)  # Penalize(p2, p2 + a) inactive

        # disparity d==nd!=a
        if d1 == d2 and o1 >= 0 and o2 >= 0:
            assert (d1 != alpha and d1 != self.OCCLUDED.val)
            delta = self.smoothness_penalty(coordP, coordQ, d1)
            ener.add_term2(o1, o2, 0, delta, delta, 0)  # // Penalize different activity

        # disparity d1, a!=d1!=d2, (p2,p2+d1) inactive neighbor assignment
        if d1 != d2 and o1 >= 0 and inRect((coordQ[0], coordQ[1] + d1), self.imSizeR):
            ener.add_term1(o1, self.smoothness_penalty(coordP, coordQ, d1), 0)

        # disparity d2, a!=d2!=d1, (p1,p1+d2) inactive neighbor assignment
        if d2 != d1 and o2 >= 0 and inRect((coordP[0], coordP[1] + d2), self.imSizeR):
            ener.add_term1(o2, self.smoothness_penalty(coordP, coordQ, d2), 0)

        return

    def build_uniqueness(self, ener, coordP, alpha):
        """
        Build edges in graph enforcing uniqueness at pixels p and p+d:
        - Prevent (p,p+d) and (p,p+a) from being both active
        - Prevent (p,p+d) and (p+d-alpha,p+d) from being both active.
        :param ener:
        :param coordP:
        :param alpha:
        :return: void
        """
        o = self.vars0[coordP]
        if o < 0:
            return

        #  Enforce unique image of p
        a = self.varsA[coordP]
        if a != VAR_ABSENT:
            ener.forbid01(o, a)

        # Enforce unique antecedent of p+d
        d = self.disparityL[coordP]
        assert (d != self.OCCLUDED.val)
        coordP = (coordP[0], coordP[1] + d - alpha)
        if inRect(coordP, self.imSizeL):
            a = self.varsA[coordP]
            assert (a >= 0)  # not active because of current uniqueness
            ener.forbid01(o, a)

        return

    def update_disparity(self, ener, alpha):
        """
        Update the disparity map according to min cut of energy
        :param ener:
        :param alpha:
        :return: void
        """
        for index, o in np.ndenumerate(self.vars0):
            if o >= 0 and ener.get_var(o) == 1:
                self.disparityL[index] = self.OCCLUDED.val

        for index, a in np.ndenumerate(self.varsA):
            if a >= 0 and ener.get_var(a) == 1:
                # New disparity
                self.disparityL[index] = alpha

        return

    def kolmogorov_zabih(self):
        # verify parameters

        # print parameters

        self.run()
        return self.disparityL
