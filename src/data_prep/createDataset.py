#!/usr/bin/env python
import os
import sys
#import datasetBuilderColor
#import datasetBuilderColor2
from joblib import Parallel, delayed

import subprocess

def runPrograms(path):
    print('Running the params')
    cmd = 'python /home/akshay/pytorchEnv/targetless_calibration/src/data_prep/datasetBuilderColor.py '+path
    Process1 = subprocess.Popen(cmd, shell=True)
    Process1.wait()

    cmd = 'python /home/akshay/pytorchEnv/targetless_calibration/src/data_prep/datasetBuilderColor2.py '+path
    Process2 = subprocess.Popen(cmd, shell=True)
    Process2.wait()


def main():
    path = sys.argv[1]
    testSubDirs = os.listdir(path)
    
    for idx in range(0,len(testSubDirs)):
        dir = '_'.join(testSubDirs[idx].split('_')[:-1])
        testSubDirs[idx] = os.path.join(path, dir)
    
    testSubDirs = list(dict.fromkeys(testSubDirs))

    Parallel(n_jobs=-1, verbose=1, backend='multiprocessing')(
        map(delayed(runPrograms),testSubDirs)
    )





if __name__ == "__main__":
    main()