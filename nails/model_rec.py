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

csvf = TEST_DIR + "rec.csv"
csverr = TEST_DIR + "err.csv"

NUM_CROSS_SEC = [ 16, 20, 20, 20, 24 ]


parser = argparse.ArgumentParser()
parser.add_argument('--image_name', type=str, default='test_image1.jpg', help='name of image')
parser.add_argument('--save_images', type=str, default=False, help='whether or not to save images (default False)')
opt = parser.parse_args()


def get_fnames_from_path(fpath):
    path, fname = os.path.split(fpath)
    fn, ext = fname.split("1.")
    write_path = path.replace( 'ref_nails', 'testref') + '\\'
    path = path + '\\'
    return fn, path, write_path


def getFingerCross( fn):
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
        self.data = getFingerCross(tfile)
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
            self.half_cross_models.append( (nmodel.half_cross_model[i] * 10).astype('int32') )
            self.full_cross_models.append( (nmodel.full_cross_model[i] * 10).astype('int32') )
        combi_dir = TEST_DIR + COMBI_THUMBS[combi_id] + "\\"
        files = glob.glob(combi_dir + "*.csv")
        fn = files[0]
        tmodel = Thumb_Model(fn)
        self.half_cross_models.append((tmodel.half_cross_model[i] * 10).astype('int32'))
        self.full_cross_models.append((tmodel.full_cross_model[i] * 10).astype('int32'))
        #
    def geometric_distance_to(self, nail_vec, nail_id, half_cross=False):
        vm = self.full_cross_models[nail_id]
        if half_cross:
            vm = self.half_cross_models[nail_id]
        dist = distance.euclidean( nail_vec, vm )
        return dist


class Combi_Model_Set:
    def __init__(self, cimbi_data_dir):
        self.combi_models = []
        for i in len(REF_FINGERS):
            mdl = Combi_Hand_Model( i )
            self.combi_models.append( mdl )




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
            self.half_cross_model.append( np.concatenate( [ lcross, rcross ] ) )
            self.full_cross_model.append( np.add( lcross, rcross) )


th = Ref_Thumb_Models()

f = "D:\\data\\results\\csv\\fingers 1.csv"
hm = Hand_Model(f)
vf = (hm.full_cross_model[4] * 10).astype('int32')  ## thumbs
vh = (hm.half_cross_model[4] * 10).astype('int32')  ## thumbs

min_dist = 9999
min_idx = 0
for i in range( len(th.models) ):
    vm = ( th.models[i].full_cross_model * 10 ).astype('int32')
    dist = distance.euclidean( vf, vm)
    #d1 = distance.euclidean( vh, vm1)
    print( 'Model {} - Euclidean distance: {:07.2f} full cross'.format( i, dist) )
    if ( dist < min_dist ):
        min_dist = dist
        min_idx = i

print( "The thumb is classified to size id {} in {}".format( min_idx, REF_THUMBS[min_idx] ))


