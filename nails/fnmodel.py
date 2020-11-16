import cv2
import os, glob
import numpy as np
import image_proc as ip
import configparser
import imutils

CONFIG_FILE = "C:\\Apps\\glaize\\glaize_config.txt"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
DATA_DIR = config['DEFAULT']['data_dir']

TEST_DIR = DATA_DIR + 'testref\\'
MODEL_DIR = TEST_DIR

COMBI_FINGERS = [ 'Left fingers combi 1', 'Left fingers combi 2 and 3',
                  'Left fingers combi 2 and 3', 'Left fingers combi 4',
                  'Left fingers combi 5 and 7', 'Left fingers combi 6',
                  'Left fingers combi 5 and 7',  'Left fingers combi 8 and 9',
                  'Left fingers combi 8 and 9',  'Left fingers combi 10',
                  'Left fingers combi 11', 'Left fingers combi 12',
                  'Left fingers combi 13 and 14', 'Left fingers combi 13 and 14' ]

COMBI_THUMBS = [ 'Thumb combi 1 and 2', 'Thumb combi 1 and 2',
                 'Thumb combi 3 4 and 5', 'Thumb combi 3 4 and 5',
                 'Thumb combi 3 4 and 5', 'Thumb combi 6 7 and 8',
                 'Thumb combi 6 7 and 8', 'Thumb combi 6 7 and 8',
                 'Thumb combi 9 10 11 12 and 13', 'Thumb combi 9 10 11 12 and 13',
                 'Thumb combi 9 10 11 12 and 13', 'Thumb combi 9 10 11 12 and 13',
                 'Thumb combi 9 10 11 12 and 13', 'Thumb combi 14']


files = [   "D:\\data\\flattened\\flat14.png",
            "D:\\data\\flattened\\flat13.png",
            "D:\\data\\flattened\\flat12.png",
            "D:\\data\\flattened\\flat11.png",
            "D:\\data\\flattened\\flat10.png",
            "D:\\data\\flattened\\flat09.png",
            "D:\\data\\flattened\\flat08.png",
            "D:\\data\\flattened\\flat07.png",
            "D:\\data\\flattened\\flat06.png",
            "D:\\data\\flattened\\flat05.png",
            "D:\\data\\flattened\\flat04.png",
            "D:\\data\\flattened\\flat03.png",
            "D:\\data\\flattened\\flat02.png",
            "D:\\data\\flattened\\flat01.png"
          ]

combi = [   [ None, None,  None, None, None],    [ None, None,  None, None, None],
            [ None, None,  None, None, None],    [ None, None,  None, None, None],
            [ None, None,  None, None, None],    [ None, None,  None, None, None],
            [ None, None,  None, None, None],    [ None, None,  None, None, None],
            [ None, None,  None, None, None],    [ None, None,  None, None, None],
            [ None, None,  None, None, None],    [ None, None,  None, None, None],
            [ None, None,  None, None, None],    [ None, None,  None, None, None]
        ]

turn_angles = [ [ -2, -3, -2, -3, 0],
                [ 3 ],
                [ -2, -1, -2, -7, 0 ],
                [ 7, 3, 0, -1],
                [],
                [ -2],
                [ 1 ],
                [ 0, 2, -2 ],
                [-1],
                [7],
                [1],
                [4],
                [ -4, -4],
                [3]
              ]


def scale_nail( fimg, f):
    img = cv2.imread(f)
    h, w = img.shape[:2]
    r, c = fimg.shape[:2]
    height = h
    wd = c * h / r
    width = int(np.round(wd))
    print( "  size r,c: ", height, " - ", width )
    img = cv2.resize( fimg, ( width, height), interpolation = cv2.INTER_AREA )
    return img


def scale_combi_fingers( cid ):
    fdir = MODEL_DIR + COMBI_FINGERS[cid] + '\\'
    for i in [0, 1, 2, 3] :
        print( "combi - finger: {}, {}".format( cid, i))
        fin = glob.glob( fdir + "*_l{}.png".format( i ))
        assert( len(fin) == 1)
        img = scale_nail( combi[cid][i], fin[0] )
        wf = MODEL_DIR + COMBI_FINGERS[cid] + '\\' + "f3d_{}.png".format( i )
        cv2.imwrite( wf, img)
    tdir = MODEL_DIR + COMBI_THUMBS[cid] + '\\'
    fin = glob.glob( tdir + "*_l4.png" )
    assert (len(fin) == 1)
    img = scale_nail(combi[cid][4], fin[0])
    wf = MODEL_DIR + COMBI_THUMBS[cid] + '\\' + "f3d_4.png"
    cv2.imwrite( wf, img )


print( "-------", 0)
fc = 0
f = files[fc]
cc = 13
img = cv2.imread(f)
#cv2.imshow( "image", img); cv2.waitKey(0)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
for xi, i in c_loc:
    mask = np.zeros( gray1.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
    mask = ip.clip(mask)
    img1 = ip.pad(mask)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    angl = turn_angles[fc][i]
    print(i, " - ", angl )
    img1 = imutils.rotate_bound(img1, angl)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = ip.clip(img1)
    combi[cc][i] = img1.copy()
    ##cv2.imshow("cfin{}".format(i), img1)
    ##cv2.waitKey(0)
scale_combi_fingers( cc )

print( "-------", 1)
fc = 1
f = files[fc]
cc = 12
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
for xi, i in c_loc:
    if i in [0, 1, 2, 3]:
        combi[cc][i] = combi[13][i]
    elif i == 4 :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][0]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][i] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
scale_combi_fingers( cc )


print( "-------", 2)
fc = 2
f = files[fc]
cc = 11
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
for xi, i in c_loc:
    mask = np.zeros( gray1.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
    mask = ip.clip(mask)
    img1 = ip.pad(mask)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    angl = turn_angles[fc][i]
    print(i, " - ", angl )
    img1 = imutils.rotate_bound(img1, angl)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = ip.clip(img1)
    combi[cc][i] = img1.copy()
    ##cv2.imshow("cfin{}".format(i), img1)
    ##cv2.waitKey(0)
scale_combi_fingers( cc )


print( "-------", 3)
fc = 3
f = files[fc]
cc = 10
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
for xi, i in c_loc:
    mask = np.zeros( gray1.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
    mask = ip.clip(mask)
    img1 = ip.pad(mask)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    angl = turn_angles[fc][i]
    print(i, " - ", angl )
    img1 = imutils.rotate_bound(img1, angl)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = ip.clip(img1)
    combi[cc][i] = img1.copy()
    ##cv2.imshow("cfin{}".format(i), img1)
    ##cv2.waitKey(0)
combi[cc][4] = combi[12][4]
scale_combi_fingers( cc )


print( "-------", 4)
fc = 4
cc = 9
combi[cc][0] = combi[11][0]
combi[cc][1] = combi[12][1]
combi[cc][2] = combi[10][2]
combi[cc][3] = combi[12][3]
combi[cc][4] = combi[10][4]
scale_combi_fingers( cc )


print( "-------", 5)
fc = 5
f = files[fc]
cc = 8
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
for xi, i in c_loc:
    if i == 0 :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][1] = img1.copy()
        #cv2.imshow("cfin{}".format(i), img1)
        #cv2.waitKey(0)
combi[cc][0] = combi[10][0]
combi[cc][2] = combi[9][2]
combi[cc][3] = combi[10][3]
combi[cc][4] = combi[10][4]
scale_combi_fingers( cc )


print( "-------", 6)
fc = 6
f = files[fc]
cc = 7
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
for xi, i in c_loc:
    if i == 0 :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][4] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
combi[cc][0] = combi[8][0]
combi[cc][1] = combi[8][1]
combi[cc][2] = combi[8][2]
combi[cc][3] = combi[8][3]
scale_combi_fingers( cc )


print( "-------", 7)
fc = 7
f = files[fc]
cc = 6
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
pr = [ 0, 2, 3 ]
for xi, i in c_loc:
    mask = np.zeros( gray1.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
    mask = ip.clip(mask)
    img1 = ip.pad(mask)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    angl = turn_angles[fc][i]
    print(i, " - ", angl )
    img1 = imutils.rotate_bound(img1, angl)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = ip.clip(img1)
    combi[cc][ pr[i] ] = img1.copy()
    ##cv2.imshow("cfin{}".format(i), img1)
    ##cv2.waitKey(0)
combi[cc][1] = combi[10][1]
combi[cc][4] = combi[7][4]
scale_combi_fingers( cc )


print( "-------", 8)
fc = 8
f = files[fc]
cc = 5
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
for xi, i in c_loc:
    if i in [0] :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][1] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
combi[cc][0] = combi[6][0]
combi[cc][2] = combi[6][2]
combi[cc][3] = combi[6][3]
combi[cc][4] = combi[6][4]
scale_combi_fingers( cc )



print( "-------", 9 )
fc = 9
f = files[fc]
cc = 4
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
for xi, i in c_loc:
    if i in [0] :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][4] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
combi[cc][0] = combi[6][0]
combi[cc][1] = combi[6][1]
combi[cc][2] = combi[6][2]
combi[cc][3] = combi[6][3]
scale_combi_fingers( cc )


print( "-------", 10)
fc = 10
f = files[fc]
cc = 3
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
for xi, i in c_loc:
    if i in [0] :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][0] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
combi[cc][1] = combi[5][1]
combi[cc][2] = combi[5][2]
combi[cc][3] = combi[5][3]
combi[cc][4] = combi[4][4]
scale_combi_fingers( cc )


print( "-------", 11)
fc = 11
f = files[fc]
cc = 2
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
for xi, i in c_loc:
    if i in [0] :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][3] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
combi[cc][0] = combi[5][0]
combi[cc][1] = combi[5][1]
combi[cc][2] = combi[5][2]
combi[cc][4] = combi[4][4]
scale_combi_fingers( cc )



print( "-------", 12)
fc = 12
f = files[fc]
cc = 1
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
p = [2, 4]
for xi, i in c_loc:
    mask = np.zeros( gray1.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
    mask = ip.clip(mask)
    img1 = ip.pad(mask)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    angl = turn_angles[fc][i]
    print(i, " - ", angl )
    img1 = imutils.rotate_bound(img1, angl)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = ip.clip(img1)
    combi[cc][ p[i] ] = img1.copy()
    ##cv2.imshow("cfin{}".format(i), img1)
    ##cv2.waitKey(0)
combi[cc][0] = combi[2][0]
combi[cc][1] = combi[2][1]
combi[cc][3] = combi[3][2]
scale_combi_fingers( cc )


print( "-------", 13)
fc = 13
f = files[fc]
cc = 0
img = cv2.imread(f)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
_, gray1 = cv2.threshold( gray, 0.5, 255, cv2.THRESH_BINARY )
gray1 = (255 - gray1)
contours, _ = cv2.findContours( gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
lenc = len(contours)
c_loc = []
j = 0
for i in range( lenc ):
    xmin = min( contours[i][:, 0, 0] )
    if xmin > 0:
        c_loc.append( [ xmin, j] )
        j += 1
    c_loc.sort( key=lambda x: x[0] )
#
for xi, i in c_loc:
    if i in [0] :
        mask = np.zeros( gray1.shape[:2], np.uint8)
        mask = cv2.drawContours(mask, contours, i, 255, -1, 8)
        mask = ip.clip(mask)
        img1 = ip.pad(mask)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        angl = turn_angles[fc][i]
        print(i, " - ", angl )
        img1 = imutils.rotate_bound(img1, angl)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = ip.clip(img1)
        combi[cc][1] = img1.copy()
        ##cv2.imshow("cfin{}".format(i), img1)
        ##cv2.waitKey(0)
combi[cc][0] = combi[1][0]
combi[cc][2] = combi[1][2]
combi[cc][3] = combi[1][3]
combi[cc][4] = combi[1][4]
scale_combi_fingers( cc )