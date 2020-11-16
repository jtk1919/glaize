import os
import sys
import numpy as np
import argparse
import glob
import csv
import configparser
from scipy.spatial import distance


CONFIG_FILE = "C:\\Apps\\glaize\\glaize_config.txt"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
DATA_DIR = config['DEFAULT']['data_dir']

IMAGES_DIR = DATA_DIR + 'ref_nails\\'
LEFT_THUMB_DIR = DATA_DIR + 'ref_nails\\'
RESULTS_DIR = DATA_DIR + 'results\ref_fingers\\'
TEST_DIR = DATA_DIR + 'testref\\'
MODEL_DATA_DIR = TEST_DIR
CSV_DIR = DATA_DIR + "results\\csv\\"

REF_FINGERS = [ 'Left fingers combi 1', 'Left fingers combi 2 and 3',
                'Left fingers combi 4', 'Left fingers combi 5 and 7',
                'Left fingers combi 6', 'Left fingers combi 8 and 9',
                'Left fingers combi 10','Left fingers combi 11',
                'Left fingers combi 12', 'Left fingers combi 13 and 14']

REF_THUMBS = [ 'Thumb combi 1 and 2', 'Thumb combi 3 4 and 5',
               'Thumb combi 6 7 and 8', 'Thumb combi 9 10 11 12 and 13',
               'Thumb combi 14']

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

NUM_COMBI = len(COMBI_FINGERS)

SIZE_CHART = [ [ 11, 11, 11, 11, 10, 10, 10, 9,	9,	7,	9,	7,	6,	6 ],
               [ 9, 7, 7, 7, 6,	7, 6, 5, 5,	3, 6, 4, 3,	3 ],
               [ 6, 6, 6, 5, 5, 5, 5, 4, 4,	4, 4, 3, 2,	2 ],
               [ 7, 7, 7, 6, 6, 6, 6, 5, 5, 3, 5, 4, 3, 3 ],
               [ 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0 ] ]

NOT_LIVE = True

csvf = TEST_DIR + "rec.csv"
csverr = TEST_DIR + "err.csv"

NUM_CROSS_SEC = [ 20, 24, 24, 24, 30 ]


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='test', help='name of image')
opt = parser.parse_args()


def get_fnames_from_path(fpath):
    path, fname = os.path.split(fpath)
    fn, ext = fname.split("1.")
    write_path = path.replace( 'ref_nails', 'testref') + '\\'
    path = path + '\\'
    return fn, path, write_path


def getFingerCross(fn):
    with open(fn, "r") as csvf:
        reader = csv.reader(csvf)
        _ = next(reader)
        fid = -1
        while fid < 5:
            row = next(reader)
            fid = np.int(row[0])
            lcross = np.array(row[2:-1], dtype=float)
            row = next(reader)
            rcross = np.array(row[2:-1], dtype=float)
            yield fid, lcross, rcross


def getThumbCross(fn):
    with open(fn, "r") as csvf:
        reader = csv.reader(csvf)
        _ = next(reader)
        fid = -1
        row = next(reader)
        fid = np.int(row[0])
        lcross = np.array(row[2:-1], dtype=float)
        row = next(reader)
        rcross = np.array(row[2:-1], dtype=float)
        yield fid, lcross, rcross


class Ref_Nail_Model:
    def __init__(self, nfile):
        self.nfile=nfile
        self.data = getFingerCross(nfile)
        self.half_cross_model = []
        self.full_cross_model = []
        for i in range(4):
            vlen = NUM_CROSS_SEC[i]
            lcross = np.zeros( vlen, dtype=float)
            rcross = np.zeros( vlen, dtype=float)
            fid, lft, rht = next( self.data )
            if len(lft) < vlen:
                lcross[:len(lft)] = lft
                rcross[:len(rht)] = rht
            else:
                lcross = lft[:vlen]
                rcross = rht[:vlen]
            self.half_cross_model.append( np.concatenate( [ lcross, rcross ] ) )
            self.full_cross_model.append( np.add( lcross, rcross) )


class Thumb_Model:
    def __init__(self, tfile):
        self.tfile=tfile
        self.data = getThumbCross(tfile)
        self.half_cross_model = []
        self.full_cross_model = []
        vlen = NUM_CROSS_SEC[4]
        lcross = np.zeros( vlen, dtype=float)
        rcross = np.zeros( vlen, dtype=float)
        fid, lft, rht = next( self.data )
        if len(lft) < vlen:
            lcross[:len(lft)] = lft
            rcross[:len(rht)] = rht
        else:
            lcross = lft[:vlen]
            rcross = rht[:vlen]
        self.half_cross_model = np.concatenate( [ lcross, rcross ] )
        self.full_cross_model = np.add( lcross, rcross)


class Combi_Hand_Model:
    def __init__( self, combi_id):
        self.half_cross_models = []
        self.full_cross_models = []
        combi_dir = TEST_DIR + COMBI_FINGERS[combi_id] + "\\"
        files = glob.glob( combi_dir + "*.csv")
        fn = files[0]
        nmodel = Ref_Nail_Model(fn)
        for i in range(4):
            self.half_cross_models.append( (nmodel.half_cross_model[i] * 10).astype('int32').tolist() )
            self.full_cross_models.append( (nmodel.full_cross_model[i] * 10).astype('int32').tolist() )
        combi_dir = TEST_DIR + COMBI_THUMBS[combi_id] + "\\"
        files = glob.glob(combi_dir + "*.csv")
        fn = files[0]
        tmodel = Thumb_Model(fn)
        self.half_cross_models.append((tmodel.half_cross_model * 10).astype('int32').tolist())
        self.full_cross_models.append((tmodel.full_cross_model * 10).astype('int32').tolist())
        #
    def geometric_distance_to(self, nail_vec, nail_id, half_cross=False):
        vm = self.full_cross_models[nail_id]
        if half_cross:
            vm = self.half_cross_models[nail_id]
        dist = distance.euclidean( nail_vec, vm )
        return dist
        #
    def get_full_cross_combivec(self, fin_only=True):
        vec = np.concatenate( self.full_cross_models[0:5], axis = 0 )
        if fin_only:
            vec = np.concatenate(self.full_cross_models[0:4], axis=0)
        return vec


class Combi_Model_Set:
    def __init__(self, cimbi_data_dir):
        self.combi_models = []
        for i in range( len(COMBI_FINGERS) ):
            mdl = Combi_Hand_Model( i )
            self.combi_models.append( mdl )
        #
    def geometric_hand_distance(self, hand_model, fin_only=True ):
        hv = hand_model.get_full_cross_combivec( fin_only)
        dist_vec = []
        for i in range(NUM_COMBI):
            cv = self.combi_models[i].get_full_cross_combivec( fin_only )
            dist = distance.euclidean( hv, cv)
            dist_vec.append([i, dist])
            if NOT_LIVE:
                print( "Full-cross Combi distance from hand to combi {} is {}".format(i+1, dist) )
        dist_vec.sort(key=lambda x: x[1])
        if NOT_LIVE:
            print( "Fingers classified to combi {} with distance {}".format( dist_vec[0][0]+1, dist_vec[0][1]) )
        return dist_vec[:3]


class Ref_Thumb_Models:
    def __init__(self):
        self.models = []
        len( REF_THUMBS )
        for i in range( len(REF_THUMBS) ):
            dir = MODEL_DATA_DIR + REF_THUMBS[i] + "\\"
            files = glob.glob( dir + "*_lthumb.csv")
            f = files[0]
            self.models.append( Nail_Model( f ) )


class Hand_Model:
    def __init__(self, file):
        self.file = file
        self.gen = getFingerCross(file)
        self.half_cross_model = []
        self.full_cross_model = []
        for i in range(5):
            vlen = NUM_CROSS_SEC[i]
            lcross = np.zeros( vlen, dtype=float)
            rcross = np.zeros( vlen, dtype=float)
            fid, lft, rht = next( self.gen )
            if len(lft) < vlen:
                lcross[:len(lft)] = lft
                rcross[:len(rht)] = rht
            else:
                lcross = lft[:vlen]
                rcross = rht[:vlen]
            self.half_cross_model.append( ( np.concatenate( [ lcross, rcross ] * 10).astype('int32').tolist() ) )
            self.full_cross_model.append( (np.add( lcross, rcross) * 10).astype('int32').tolist() )
        #
    def get_full_cross_combivec(self, fin_only=True):
        vec = np.concatenate(hm.full_cross_model[0:5], axis = 0 )
        if fin_only:
            vec = np.concatenate(hm.full_cross_model[0:4], axis=0)
        return vec


cm_set = Combi_Model_Set( "D:\\data\\testref\\" )

flist = [ "D:\\data\\test_fake_nail\\Left fingers 1.csv",
          'D:\\data\\test_fake_nail\\Left fingers A.csv',
          'D:\\data\\test_fake_nail\\Left fingers B.csv',
          'D:\\data\\test_fake_nail\\Left fingers D.csv',
          'D:\\data\\test_fake_nail\\Left fingers E.csv',
          'D:\\data\\test_fake_nail\\Left fingers F.csv'
          ]

# backed up code

if opt.image in ['ALL', 'all']:
    flist = glob.glob( CSV_DIR + "*.csv")
    for f in flist:
        hm = Hand_Model(f)
        combi = cm_set.geometric_hand_distance(hm, False)
        print( combi )
        print(f)
        print()
elif opt.image in ['TEST', 'test']:
    for f in flist:
        hm = Hand_Model(f)
        combi = cm_set.geometric_hand_distance(hm, False)
        print( combi )
        print(f)
        print()
else:
    NOT_LIVE = False
    f = opt.image
    hm = Hand_Model(f)
    combi = cm_set.geometric_hand_distance(hm, False)
    print( combi[0][0] )


