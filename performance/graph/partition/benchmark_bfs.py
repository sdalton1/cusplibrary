#!/usr/bin/env python 
import os,csv,subprocess
import numpy as np
import matplotlib.pyplot as plt

from pylab import savefig
from scipy.io import mmread, mmwrite

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

def crop_figure(fullname):
    os.system('pdfcrop ' + fullname + ' ' + fullname)

def save_figure(plt, name, ext='.pdf'):
    fullname = name + ext
    plt.savefig('./' + fullname)
    crop_figure(fullname)

def run_tests(num_iterations):

    # run benchmark for each file
    for matrix,filename,path in trials:
        matrix_filename = path + filename
        A = mmread(matrix_filename)

        bfs_matrix = np.zeros((num_iterations,2))

        # execute the benchmark on this file
        for index,source in enumerate(np.random.randint(0, A.shape[0], size=num_iterations)) :
            # setup the command to execute
            # proc = subprocess.Popen([binary_filename, matrix_filename, str(source)], stdout=subprocess.PIPE, shell=True)
            proc = subprocess.Popen([binary_filename, matrix_filename, str(source)], stdout=subprocess.PIPE)
            (out, err) = proc.communicate()
            bfs_data = out.split("\n")[2].split(",")
            bfs_time = bfs_data[0].split(":")[1].split()[0]
            bfs_levels = bfs_data[1].split(":")[1].split()[0]

            bfs_matrix[index,0] = bfs_time
            bfs_matrix[index,1] = bfs_levels

        # outfile_name = matrix + "_bfs.mtx"
        # mmwrite(outfile_name, bfs_matrix)
        # print "Wrote data file: {}".format(outfile_name)

        plt.figure()
        ax = plt.subplot(111)
        ax.plot(bfs_matrix[:,1], bfs_matrix[:,0], 'o')
        ax.set_xlabel("Number of Levels")
        ax.set_ylabel("Time (ms)")
        ax.set_title(matrix.replace("_","\\_"))
        # save_figure(plt, "bfs_levels_perf_{}".format(matrix))

    plt.show()

if __name__=="__main__":

    run_tests(10)
