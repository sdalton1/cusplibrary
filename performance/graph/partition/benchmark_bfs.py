#!/usr/bin/env python 
import os,csv,subprocess
import numpy as np
import matplotlib.pyplot as plt

from pylab import savefig
from scipy.io import mmread

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

linewidth = 2

binary_filename = './bfs'         # command used to run the tests

mats = []
path = os.environ['HOME'] + '/laplacian_matrices/'

structured_mats = [
                   ('2D_5pt','structured_2d_5pt.mtx'),
                   ('2D_9pt','structured_2d_9pt.mtx'),
                   ('3D_7pt','structured_3d_7pt.mtx'),
                   ('3D_27pt','structured_3d_27pt.mtx')
                  ]
unstructured_mats = [
                     ('fe_tooth', 'fe_tooth.mtx'),
                     ('fe_rotor', 'fe_rotor.mtx'),
                     ('598a', '598a.mtx'),
                     ('fe_ocean', 'fe_ocean.mtx'),
                     ('144', '144.mtx'),
                     ('wave','wave.mtx'),
                     ('m14b', 'm14b.mtx'),
                     ('auto', 'auto.mtx')
                     ]

structured_mats = [ mat + (path,) for mat in structured_mats]
unstructured_mats = [ mat + (path,) for mat in unstructured_mats]

# assemble suite of matrices
trials = unstructured_mats

def run_tests(num_iterations):

    # run benchmark for each file
    for matrix,filename,path in trials:
        matrix_filename = path + filename
        A = mmread(matrix_filename);

        # execute the benchmark on this file
        for source in np.random.randint(0, A.shape[0], size=num_iterations) :
            # setup the command to execute
            # proc = subprocess.Popen([binary_filename, matrix_filename, str(source)], stdout=subprocess.PIPE, shell=True)
            proc = subprocess.Popen([binary_filename, matrix_filename, str(source)], stdout=subprocess.PIPE)
            (out, err) = proc.communicate()
            print out
            # bfs_data = out.split("\n")[2].split(",")
            # print bfs_data

run_tests(1)
