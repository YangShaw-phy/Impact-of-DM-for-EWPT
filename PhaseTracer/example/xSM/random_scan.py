import os,sys,signal
import re,shutil
import subprocess
import numpy as np
import time

#n_total = 100000
n_total = 1000

np.random.seed(1)
ms = np.random.uniform(10,110,n_total)
lambda_s = np.random.uniform(0.01,0.3,n_total)
lambda_hs = np.random.uniform(0.1,0.5,n_total)

file_name = "random_scan_results"

fo = open("random_scan_results.txt", "w")
for ii in range(n_total):
    cmd = "./../../bin/scan_xSM " + str(ms[ii]) + " " +str(lambda_s[ii]) + " " + str(lambda_hs[ii]) 
    print cmd
    os.system(cmd)
    output = open("output.txt").readline()
    fo.write( output )
fo.close()


