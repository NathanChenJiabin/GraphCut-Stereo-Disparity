import numpy as np

# Preprocessing for Birchfield-Tomasi


def SubPixel(grayImg, shape):
    """
    generate ImMin and ImMax from grayImg (gray version)
    :param shape:
    :param grayImg: numpy  array, gray image
    :return: numpy array imMin and imMax filled
    """
    ymax, xmax = shape
    imgrayMin = np.zeros(shape)
    imgrayMax = np.zeros(shape)

    for y in range(ymax):
        for x in range(xmax):
            I = int(grayImg[y, x])
            I1 = (int(grayImg[y, x - 1]) + I) / 2 if x > 0 else I
            I2 = (int(grayImg[y, x + 1]) + I) / 2 if x + 1 < xmax else I
            I3 = (int(grayImg[y - 1, x]) + I) / 2 if y > 0 else I
            I4 = (int(grayImg[y + 1, x]) + I) / 2 if y + 1 < ymax else I

            IMin = min([I, I1, I2, I3, I4])
            IMax = max([I, I1, I2, I3, I4])

            imgrayMin[y, x] = IMin
            imgrayMax[y, x] = IMax

    return imgrayMin, imgrayMax


def SubPixelColor(imgColor, shape):
    """
    generate ImMin and ImMax from imgColor (color version)
    :param imgColor:
    :param shape:
    :return: numpy array imMin and imMax filled
    """
    ymax, xmax, channels = shape
    imgMin = np.zeros(shape)
    imgMax = np.zeros(shape)

    for y in range(ymax):
        for x in range(xmax):
            for i in range(channels):
                I = int(imgColor[y, x, i])
                I1 = (int(imgColor[y, x - 1, i]) + I) / 2 if x > 0 else I
                I2 = (int(imgColor[y, x + 1, i]) + I) / 2 if x + 1 < xmax else I
                I3 = (int(imgColor[y - 1, x, i]) + I) / 2 if y > 0 else I
                I4 = (int(imgColor[y + 1, x, i]) + I) / 2 if y + 1 < ymax else I

                IMin = min([I, I1, I2, I3, I4])
                IMax = max([I, I1, I2, I3, I4])

                imgMin[y, x, i] = IMin
                imgMax[y, x, i] = IMax

    return imgMin, imgMax

