import io
import os
import csv
import glob
import csv
import numpy as np

import configparser

CONFIG_FILE = "C:\\Apps\\glaize\\glaize_config.txt"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
DATA_DIR = config['DEFAULT']['data_dir']
CROSS_SEC_DIR = DATA_DIR + "results\\csv\\"


MODEL_NUM_CROSS_SECTIONS = [ 16, 20, 20, 20, 24 ]
MODEL_NUM_FINGERS = len(MODEL_NUM_CROSS_SECTIONS)


def getFingerCross( fn ):
    with open( fn, "r" ) as csvf:
        reader = csv.reader(csvf)
        _ = next(reader)
        row = next(reader)
        fid = np.int(row[0])
        lcross = np.array( row[2:-1], dtype=float)
        row = next(reader)
        rcross = np.array(row[2:-1], dtype=float)
        yield fid, lcross, rcross


files = glob.glob( CROSS_SEC_DIR + "*.csv")
for f in files:
    for fid, lcross, rcross in getFingerCross():
        cross = lcross + rcross
        veclen = MODEL_NUM_CROSS_SECTIONS[fid]
        vec = []
        if len(cross) < veclen:
            vec = np.zeros( MODEL_NUM_CROSS_SECTIONS[ np.int(fid)] )
            vec[ :len(cross)] = cross
        else:
            vec = cross[:veclen]


