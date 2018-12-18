import match
import cv2


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
    params = match.Parameters(is_L2=True,
                              denominator=1,
                              edgeThresh=8,
                              lambda1=lambda1,
                              lambda2=lambda2,
                              K=K,
                              maxIter=4)

    # create match instance
    m = match.Match(imgLeft, imgRight, color)

    m.SetDispRange(dispMin, dispMax)
    m = match.fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    m.kolmogorov_zabih()
    m.saveDisparity("./results/disparity1.jpg")


if __name__ == '__main__':
    filename_left = "./images/imgR1.png"
    filename_right = "./images/imgL1.png"
    is_color = True
    disMin = -16
    disMax = 16
    main(filename_left, filename_right, is_color, disMin, disMax)
