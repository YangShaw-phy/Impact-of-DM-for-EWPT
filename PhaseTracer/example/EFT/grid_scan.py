import os,sys,signal
import re,shutil
import subprocess
import numpy as np
import time


def scan(cmd, file_name):
  fo = open(file_name+".txt", "w")
  for ii in np.linspace(2,4,50):
    dof = str(int(10**ii))
    print dof
    os.system(cmd+" "+dof+" > output.txt")
    output = open("output.txt").readline()
    fo.write( dof + " " +output )
  fo.close()


##################### gauge dependence #################


scan("./../../bin/run_EFT", "scan_result")

