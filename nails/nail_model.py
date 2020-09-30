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


MODEL_NUM_CROSS_SECTIONS = [16, 20, 20, 20, 24]
MODEL_NUM_FINGERS = len(MODEL_NUM_CROSS_SECTIONS)


class FullCrossModel:
    def __init__(self):
        self.isModelBuilt = False
        self.num_samples = 0
        self.full_cross_samples = []
        self.mean = np.zeros(np.sum(MODEL_NUM_CROSS_SECTIONS))
        self.covariance = []
        self.VI = []
    #
    def addSample(self, crossFile):
        sample_vec = []
        for i in range(MODEL_NUM_FINGERS):
            fid, lcross, rcross = next(self.getFingerCross(crossFile))
            cross = lcross + rcross
            veclen = MODEL_NUM_CROSS_SECTIONS[i]
            vec = []
            if len(cross) < veclen:
                vec = np.zeros(MODEL_NUM_CROSS_SECTIONS[i])
                vec[:len(cross)] = cross
            else:
                vec = cross[:veclen]
            sample_vec.extend(vec)
            print( i, len(vec))
        self.getFingerCross(crossFile).close()
        self.num_samples += 1
        self.full_cross_samples.append( sample_vec )
    #
    def buildModel(self):
        self.mean = np.mean( self.full_cross_samples, axis=0)
        self.covariance = np.cov ( self.full_cross_samples, rowvar=False )
        self.VI = np.linalg.inv(self.covariance)
        self.isModelBuilt = True
        self.full_cross_samples.clear()
    #
    def getFingerCross(self, fn):
        with open(fn, "r") as csvf:
            reader = csv.reader(csvf)
            _ = next(reader)
            row = next(reader)
            fid = np.int(row[0])
            lcross = np.array(row[2:-1], dtype=float)
            row = next(reader)
            rcross = np.array(row[2:-1], dtype=float)
            yield fid, lcross, rcross


model = FullCrossModel()
files = glob.glob(CROSS_SEC_DIR + "*.csv")
for f in files:
    model.addSample(f)
    if model.num_samples >= 10:
        break

