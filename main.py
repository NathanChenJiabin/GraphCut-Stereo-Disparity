import match
import cv2

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


# main programme
def main(file1, file2, color, dispMin, dispMax):
    # load two images
    imgLeft = cv2.imread(file1)
    imgRight = cv2.imread(file2)

    # Default parameters
    K = -1
    lambda_ = -1
    lambda1 = -1
    lambda2 = -1
    params = match.Parameters(is_L2=False,
                              denominator=1,
                              edgeThresh=8,
                              lambda1=lambda1,
                              lambda2=lambda2,
                              K=K,
                              maxIter=4)

    # create match instance
    m = match.Match(imgLeft, imgRight, color)

    m.SetDispRange(dispMin, dispMax)
    m = fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    m.kolmogorov_zabih()
    m.saveDisparity("./results/disparity2.jpg")


if __name__ == '__main__':
    filename_left = "./images/imgL2.png"
    filename_right = "./images/imgR2.png"
    is_color = True
    disMin = -16
    disMax = 16
    main(filename_left, filename_right, is_color, disMin, disMax)
