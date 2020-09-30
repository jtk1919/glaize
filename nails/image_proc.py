import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi


def clip_finger_mask( idx, r, img ):
    msk1 = np.zeros(img.shape[:2], dtype="uint8")
    msk1[r['masks'][:, :, idx]] = 255
    #m1 = cv2.fastNlMeansDenoising(msk1)
    nail_clipped = clip(msk1)
    return nail_clipped

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

def clip( msk1 ):
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
    in_nail = False
    for i in range( len(vsum) ):
        if ( not in_nail ) and ( vsum[i] > 0):
            y1 = i
            in_nail = True
        elif in_nail and (vsum[i] == 0):
            y2 = i
            break
    rc, msk = cv2.threshold( msk1, 0.5, 255, cv2.THRESH_BINARY)
    cropped_nail = msk[ y1-2:y2+2, x1-2:x2+2 ]
    return cropped_nail


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
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    #
    rc = cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    ##drawAxis(img, cntr, p1, (0, 255, 0), 1)
    ##drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return angle