import cv2
import numpy as np
import configparser
import argparse
import csv
import os
import imutils
import image_proc as ip


CC_LEN_PX = 2400

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

CONFIG_FILE = "C:\\Apps\\glaize\\glaize_config.txt"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
DATA_DIR = config['DEFAULT']['data_dir']

NAIL3D_DIR = DATA_DIR + "testref\\"
FIN_DIR = DATA_DIR + "test\\"


parser = argparse.ArgumentParser()
#parser.add_argument('--image', type=str, default='D:\\data\\results\\csv\\Left fingers 1.csv',
#parser.add_argument('--image', type=str, default='D:\\data\\results\\csv\\Left fingers A.csv',
#parser.add_argument('--image', type=str, default='D:\\data\\results\\csv\\Left fingers B.csv',
#parser.add_argument('--image', type=str, default='D:\\data\\results\\csv\\Left fingers E.csv',
parser.add_argument('--image', type=str, default='D:\\data\\results\\csv\\Left fingers F.csv',
                        help='name of the image parameters csv file')
opt = parser.parse_args()


def getCombiMasks( combi ):
    combi_fin = [ None, None, None, None, None ]
    fdir = NAIL3D_DIR + COMBI_FINGERS[combi]
    csvf = fdir + "\\IMG.csv"
    cc_len = 0
    cfile = open( csvf, "r")
    creader = csv.reader( cfile )
    i = 0
    for row in creader:
        i += 1
        if i == 11 :
            cc_len = int(row[0])
            cc_len
    cfile.close()
    scale = CC_LEN_PX / cc_len
    for i in [0, 1, 2, 3 ] :
        f = fdir + "\\f3d_{}.png".format(i)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r, c = img.shape[:2]
        r = round( scale * r )
        c = round( scale * c )
        combi_fin[i] = cv2.resize( img, (c, r), interpolation=cv2.INTER_AREA)
    fdir = NAIL3D_DIR + COMBI_THUMBS[combi]
    csvf = fdir + "\\IMG.csv"
    cc_len = 0
    cfile = open(csvf, "r")
    creader = csv.reader(cfile)
    i = 0
    for row in creader:
        i += 1
        if i == 5:
            cc_len = int(row[0])
            cc_len
    cfile.close()
    scale = CC_LEN_PX / cc_len
    f = fdir + "\\f3d_4.png"
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, c = img.shape[:2]
    r = round(scale * r)
    c = round(scale * c)
    combi_fin[4] = cv2.resize(img, (c, r), interpolation=cv2.INTER_AREA)
    #
    return combi_fin


## "D:\\data\\test\\Left fingers 1.csv",
## "D:\\data\\results\\csv\\Left fingers 1.csv",
def getHandNails( csvf ):
    fin = [ None, None, None, None, None ]
    turn_angles = [ 0, 0, 0, 0, 0]
    cc_len = [ 1, 1]
    cfile = open( csvf, "r")
    creader = csv.reader( cfile )
    fdir = FIN_DIR
    fn = csvf.split('\\')[-1]
    fn = fn.split('.csv')[0]
    i = 0
    for row in creader:
        i += 1
        if i == 12:
            turn_angles = [-int(x) for x in row]
            print( "angles: ", turn_angles)
        elif i == 13:
            cc_len = [int(x) for x in row ]
            print("cc length pixels: ", cc_len)
    cfile.close()
    scale = CC_LEN_PX / cc_len[0]
    for i in [0, 1, 2, 3 ] :
        f = fdir + fn + '_l{}.png'.format(i)
        img = cv2.imread(f)
        img = imutils.rotate_bound(img, turn_angles[i] )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = ip.clip(img)
        r, c = img.shape[:2]
        r = round(scale * r)
        c = round(scale * c)
        fin[i] = cv2.resize( img, (c, r), interpolation=cv2.INTER_AREA)
    # thumb
    scale = CC_LEN_PX / cc_len[1]
    f = fdir + fn + '_l4.png'
    img = cv2.imread(f)
    img = imutils.rotate_bound(img, turn_angles[4])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ip.clip(img)
    r, c = img.shape[:2]
    r = round(scale * r)
    c = round(scale * c)
    fin[4] = cv2.resize(img, (c, r), interpolation=cv2.INTER_AREA)
    #
    return fin


## "D:\\data\\results\\csv\\Left fingers 1.csv"
csvf = opt.image

cid = int( os.popen('type "D:\\data\\results\\csv\\temp.txt"' ).read() )

combi = getCombiMasks(cid)
#for i in range( len(combi) ):
#    cv2.imshow( "fin{}".format(i), combi[i])
#cv2.waitKey(0)

fingr = getHandNails( csvf )
## turned angles and symmetry
H = 1
for i in range( len(fingr) ):
    fin = fingr[i]
    cmb = combi[i]
    contours, _ = cv2.findContours(fin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contr = contours[0]
    h, w = fingr[i].shape
    r, c = combi[i].shape
    H = h
    xtop = []
    xbtm = []
    for p in contr:
        x, y = p[0]
        if y < 7:
            xtop.append(x)
        elif y > (h - 8):
            xbtm.append(x)
     #
    midx = round((min(xtop) + max(xtop) + min(xbtm) + max(xbtm) ) / 4 )
    new_width = w
    if (midx + 1) - (w - midx - 1) > 4:    ## 20
        new_width = 2 * (midx + 1)
    elif (w - midx - 1) - (midx + 1) > 3:
        new_width = 2 * (w - midx - 1)
    #
    change = (new_width / w  - 1 ) * 100
    if (change > 10):
        midx = round(( min(xbtm) + max(xbtm)) / 2)
    if 2 * midx > c:
        midx = int(w/2)
    half_w = max( midx , w - midx )   # 30
    X = c
    Y = max( h, r )
    mid_X = int( X/2)
    canvasf = np.zeros( (Y, X), np.uint8)
    canvasc = canvasf.copy()
    new_width = w
    if (midx +1) - (w - midx -1) > 8:
        print( "Left Half")
        crop = np.zeros( (h, midx+1), np.uint8);
        crop[ 0:h, 0:midx+1] = fingr[i][ 0:h, 0:midx+1]  # 40
        mirror = cv2.flip( crop, 1)
        canvasf[ Y-h:Y, mid_X-midx-1:mid_X]  = crop
        canvasf[ Y-h:Y, mid_X:mid_X+midx+1] = mirror
        new_width = 2 * (midx + 1)
    elif (w - midx -1) - (midx + 1) > 8:
        print("Right Half")
        crop = np.zeros((h, w - midx -1), np.uint8);
        crop[0:h, 0:w - midx -1] = fingr[i][0:h, midx:w-1]
        mirror = cv2.flip(crop, 1)
        canvasf[ Y-h:Y, mid_X - w + midx +1 :mid_X ] = mirror    #50
        canvasf[ Y-h:Y, mid_X:mid_X+w -midx -1] = crop
        new_width = 2 * (w - midx - 1)
    else:
        print("Straight")
        w2 = round( w/2)
        canvasf[ Y-h:Y, mid_X - w2: mid_X - w2 + w ] = fingr[i]
    #
    contours, _ = cv2.findContours(canvasf, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    hull = cv2.convexHull(contour, False)
    mask = np.zeros(canvasf.shape[:2], np.uint8)
    mask = cv2.drawContours(mask, [hull], 0, 255, -1, 8)
    #
    contours, _ = cv2.findContours(combi[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    hull = cv2.convexHull(contour, False)
    mask1 = np.zeros(combi[i].shape[:2], np.uint8)
    mask1 = cv2.drawContours(mask1, [hull], 0, 255, -1, 8)
    if ( Y > r):
        mask2 = np.zeros((Y,X), np.uint8)
        mask2[Y-r:Y, 0:X] = mask1
        mask1 = mask2.copy()
    # composition  -----------------------------------------
    ref_btm = mask1.copy()
    fin_top_h = 1
    if h > r:
        fin_top_h =  round( h * 0.22)
    else:
        fin_top_h = r - h + round( h * 0.22)
    ref_btm[ 0:fin_top_h, 0:X ] = np.zeros( ( fin_top_h, X ), np.uint8 )
    comp = (ref_btm + mask).clip(0, 255).astype("uint8")
    contours, _ = cv2.findContours( comp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    hull = cv2.convexHull(contour, False)
    comp1 = np.zeros( comp.shape[:2], np.uint8 )
    comp1 = cv2.drawContours(comp1, [hull], 0, 255, -1, 8)
    #
    diff = cv2.absdiff( mask, mask1 )
    print(w, new_width, new_width/w )
    cv2.imshow("combi{}".format(i), combi[i])
    cv2.imshow("fin{}".format(i), fingr[i])
    cv2.imshow("sm_fin", mask)
    cv2.imshow("sm_ref", mask1)
    cv2.imshow("comp", comp1)
    cv2.waitKey(0)
#

