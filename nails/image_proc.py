import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import imutils


def upright( msk, is_left = True):
    orientation = 0
    rc, msk1 = cv2.threshold(msk, 0.5, 1, cv2.THRESH_BINARY)
    hsum = np.sum(msk1, axis=0)
    x_pos = [0] * 4
    is_vertical = True
    region_count = 0
    in_nail = False
    for i in range(len(hsum)):
        if (not in_nail) and (hsum[i] > 0):
            x1 = i
            in_nail = True
            region_count += 1
        elif in_nail and (hsum[i] == 0):
            if ( i - x1 ) > 20:
                x_pos[region_count - 1] = int( ( x1 + i) / 2 )
            else:
                region_count -= 1
            in_nail = False
        #
    if region_count < 4:
        is_vertical = False
        msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)
        msk1 = cv2.rotate(msk1, cv2.ROTATE_90_CLOCKWISE)
        hsum = np.sum(msk1, axis=0)
        region_count = 0
        in_nail = False
        for i in range(len(hsum)):
            if (not in_nail) and (hsum[i] > 0):
                x1 = i
                in_nail = True
                region_count += 1
            elif in_nail and (hsum[i] == 0):
                x_pos[region_count - 1] = int((x1 + i) / 2)
                in_nail = False
        #now vertical
    y_pos = [0] * 4
    for i in [0, 1, 2, 3]:
        j = 0
        while msk1[j,x_pos[i]] < 1.0:
            j += 1;
        y_pos[i] = j;
    if ( y_pos[0] < y_pos[1] ) or ( y_pos[2] > y_pos[3]):
        msk = cv2.rotate(msk, cv2.ROTATE_180)
        if is_vertical:
            orientation = 2
        else:
            orientation = 1
    else:
        if is_vertical:
            orientation = 0
        else:
            orientation = 3
    return orientation, msk


def smooth( img ):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    if len(contours) > 1:
        max_area = 0
        max_index = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > max_area):
                max_area = area
                max_index = i
        contour = contours[max_index]
    hull = cv2.convexHull(contour, False)
    mask = np.zeros( img.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, [hull], 0, 255, -1, 8)
    return mask

def clip_finger_mask( idx, r, img, doWideClip = True ):
    msk1 = np.zeros(img.shape[:2], dtype="uint8")
    msk1[r['masks'][:, :, idx]] = 255
    msk1 = smooth(msk1)
    nail_clipped = clip(msk1)
    return nail_clipped


def pad( mask, shaped_pad=False):
    Y, X = mask.shape
    if shaped_pad:
        crp = np.zeros([Y+2, X+2], dtype="uint8")
        crp[1:Y+1, 1:X+1] = mask
    else:
        if Y > X:
            diff = int((Y - X) / 2)
            crp = np.zeros([Y, Y], dtype="uint8")
            crp[0:Y, diff:diff + X] = mask
        elif X > Y:
            diff = int((X - Y) / 2)
            crp = np.zeros([X, X], dtype="uint8")
            crp[diff:diff + Y, 0:X] = mask
    return crp


def clip( msk1, doWideClip = False ):
    hsum = np.sum(msk1, axis=0)
    vsum = np.sum(msk1, axis=1)
    x1=0; y1=0; x2=0; y2=0
    in_nail = False
    for i in range( len(hsum) ):
        if ( not in_nail ) and ( hsum[i] > 0):
            x1 = i
            in_nail = True
        elif in_nail and (hsum[i] == 0):
            x2 = i
            break
    if x2 == 0:
        x2 = len(hsum) -1
    in_nail = False
    for i in range( len(vsum) ):
        if ( not in_nail ) and ( vsum[i] > 0):
            y1 = i
            in_nail = True
        elif in_nail and (vsum[i] == 0):
            y2 = i
            break
    if y2 == 0:
        y2 = len(vsum) -1
    rc, msk = cv2.threshold( msk1, 0.5, 255, cv2.THRESH_BINARY)
    # y1 = y1 -1
    # if y1 < 0:
    #     y1 = 0
    # x1 = x1 - 1
    # if x1 < 0:
    #     x1 = 0
    # r, c = msk.shape
    # x2 = x2 + 1
    # if x2 >= c:
    #     x2 = c-1
    # y2 = y2 + 1
    # if y2 >= r:
    #     y2 = r-1
    cropped_nail = msk[ y1:y2, x1:x2]
    crp = cropped_nail.copy()
    if doWideClip:
        crp = pad( crp )
    return crp

def clip1( msk1, doWideClip = False ):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 1:
        contours = contours[0]
    else:
        max_area = 0
        max_index = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > max_area):
                max_area = area
                max_index = i
        contours = contours[max_index]


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    #
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def get_orientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return angle


def nail_upright( img ):
    contours, _ = cv2.findContours( img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 1:
        contours = contours[0]
    else:
        max_area = 0
        max_index = 0
        for i in range( len(contours) ):
            area = cv2.contourArea( contours[i] )
            if ( area > max_area):
                max_area = area
                max_index = i
        contours = contours[max_index]
    # _ = cv2.drawContours( img, [contours], 0, (0, 255, 0), 2)

    img1 = pad(img)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img1 = imutils.rotate(img1, agl)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print("angle: ", agl)
    img1 = clip(img1)
    return img1


###
# Store the center of the object
    #cntr = (int(mean[0, 0]), int(mean[0, 1]))
    #
    #rc = cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    #p1 = (
    #cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    #p2 = (
    #cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    #drawAxis(img, cntr, p1, (0, 255, 0), 1)
    #drawAxis(img, cntr, p2, (255, 255, 0), 5)
