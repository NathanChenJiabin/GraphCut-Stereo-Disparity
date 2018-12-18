import cv2
import geometry
import numpy as np
import match


def main(mi, ma):
    print("Reading images...")
    left = cv2.imread('./images/perra_7.jpg')
    right = cv2.imread('./images/perra_8.jpg')

    # left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    # right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    print("Compute keypoints matching...")
    akaze = cv2.AKAZE_create()

    kpts1, desc1 = akaze.detectAndCompute(left, None)
    kpts2, desc2 = akaze.detectAndCompute(right, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, True)

    matches = matcher.match(desc1, desc2)

    sortedmatches = sorted(matches, key=lambda x: x.distance)

    nb_matches = 300
    good_matches = sortedmatches[:nb_matches]

    obj = []
    scene = []

    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj.append(kpts1[good_matches[i].queryIdx].pt)
        scene.append(kpts2[good_matches[i].trainIdx].pt)

    F, mask = cv2.findFundamentalMat(np.array(obj), np.array(scene), cv2.FM_RANSAC)

    correct_kpts1 = []
    correct_kpts2 = []
    correct_matches = []

    for i in range(len(mask)):
        if mask[i, 0] > 0:
            correct_kpts1.append(obj[i])
            correct_kpts2.append(scene[i])
            correct_matches.append(good_matches[i])

    print(len(correct_kpts1))
    res = np.empty((max(left.shape[0], right.shape[0]), left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(left, kpts1, right, kpts2, correct_matches, res)
    # cv2.imwrite("./results/details/keypoints.jpg", res)
    cv2.imshow("key points matching", res)
    cv2.waitKey(0)

    print("Computing rectification...")

    # """
    # We have implemented our rectification method based on papar
    # Physically-valid view synthesis by image interpolation, Charles R. Dyer Steven M. Seitz.
    #
    # In this case, we can obtain the matrix of rotation, translation and scale to de-rectify, but
    # this algorithm has a very poor performance compared to OpenCV built-in rectification method.
    # Furthermore, this method is not good enough to calculate the disparity, so we comment the few following lines
    # and we use OpenCV.
    # """

    # grey1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    # grey2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # dst1, dst2, theta1, theta2, s, T1, T2 = geometry.rectify(np.array(obj), np.array(scene), grey1, grey2)
    # cv2.imshow("rectified_left", dst1)
    # cv2.imshow("rectified_right", dst2)
    # cv2.waitKey(0)

    # """
    # The following few lines use OpenCV to rectify two images
    # """
    correct_kpts1 = np.array(correct_kpts1)
    correct_kpts1 = correct_kpts1.reshape((correct_kpts1.shape[0] * 2, 1))
    correct_kpts2 = np.array(correct_kpts2)
    correct_kpts2 = correct_kpts2.reshape((correct_kpts2.shape[0] * 2, 1))
    shape = (left.shape[1], left.shape[0])

    rectBool, H1, H2 = cv2.stereoRectifyUncalibrated(correct_kpts1, correct_kpts2, F, shape, threshold=1)
    R1 = cv2.warpPerspective(left, H1, shape)
    R2 = cv2.warpPerspective(right, H2, shape)
    cv2.imshow("rectified_left", R1)
    cv2.imshow("rectified_right", R2)
    cv2.waitKey(0)

    print("Computing disparity...")

    # """
    # We found that the quality of disparity map obtained by OpenCV StereoSGBM is depend
    # strongly the choice of parameters. So we implement the method based on paper:
    # Kolmogorov and zabih’sgraph cuts stereo matching algorithm, Pauline Tan Vladimir Kolmogorov, Pascal Monasse.
    # It suffice to set a good disparity range [Min, Max].
    # Attention: with this python version implementation, this method is very slow, so to quickly have a result,
    # we force here the images used can't be larger than 200*200
    # """

    R1 = cv2.resize(R1, dsize=None, fx=0.25, fy=0.25)
    R2 = cv2.resize(R2, dsize=None, fx=0.25, fy=0.25)
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
    is_color = True if R1.shape[-1] == 3 else False
    m = match.Match(R1, R2, is_color)

    m.SetDispRange(mi, ma)
    m = match.fix_parameters(m, params, K, lambda_, lambda1, lambda2)
    disparity = m.kolmogorov_zabih()

    np.save("./results/dispDog.npy", disparity)

    # disparity = np.load("./results/dispDog.npy")
    # cv2.imwrite("./r1dog.png", R1)
    # cv2.imwrite("./r2dog.png", R2)

    print("Computing interpolation...")
    ir = geometry.interpolate(1.5, R1, R2, disparity)
    cv2.imshow("interpolated view", ir)
    cv2.waitKey(0)
    cv2.imwrite("./results/dog.png", ir)
    print("Save successfully interpolated view !")


if __name__ == '__main__':
    disMin = -11
    disMax = 11
    main(disMin, disMax)
