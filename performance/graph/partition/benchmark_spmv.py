#!/usr/bin/env python 
import os,csv
import numpy as np
import matplotlib.pyplot as plt

from pylab import savefig
from scipy.io import mmread

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

linewidth = 2

binary_filename = '../../spmv/spmv'         # command used to run the tests
output_file = 'benchmark_output.log'        # file where results are stored

mats = []
path = '~/laplacian_matrices/'

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
#trials = structured_mats + unstructured_mats
trials = unstructured_mats

def run_tests(value_type):
    # remove previous result (if present)
    open(output_file,'w').close()

    # run benchmark for each file
    for matrix,filename,path in trials:
        matrix_filename = path + filename

        # setup the command to execute
        cmd = binary_filename 
        cmd += ' ' + matrix_filename                  # e.g. pwtk.mtx
        cmd += ' --value_type=' + value_type          # e.g. float or double

        # execute the benchmark on this file
        os.system(cmd)

    # process output_file
    matrices = {}
    results = {}
    kernels = set()
    #
    fid = open(output_file)
    for line in fid.readlines():
        tokens = dict( [tuple(part.split('=')) for part in line.split()] )

        if 'file' in tokens:
            file = os.path.split(tokens['file'])[1]
            matrices[file] = tokens
            results[file] = {}
        else:
            kernel = tokens['kernel']
            results[file][kernel] = tokens
            kernels.add(tokens['kernel'])

    ## put CPU results before GPU results
    #kernels = ['csr_serial'] + sorted(kernels - set(['csr_serial']))
    kernels = sorted(kernels)

    # write out CSV formatted results
    def write_csv(field):
        fid = open('bench_' + value_type + '_' + field + '.csv','w')
        writer = csv.writer(fid)
        writer.writerow(['matrix','file','rows','cols','nonzeros'] + kernels)

        for (matrix,file,path) in trials:
            line = [matrix, file, matrices[file]['rows'], matrices[file]['cols'], matrices[file]['nonzeros']]

            matrix_results = results[file]
            for kernel in kernels:
                if kernel in matrix_results:
                    line.append( matrix_results[kernel][field] )
                else:
                    line.append(' ')
            writer.writerow( line )
        fid.close()

    write_csv('gflops') #GFLOP/s
    write_csv('gbytes') #GBytes/s

def process_data_files(value_type):

    # process output_file
    matrices = {}
    results = {}
    kernels = set()
    #
    fid = open(output_file)
    for line in fid.readlines():
        tokens = dict( [tuple(part.split('=')) for part in line.split()] )

        if 'file' in tokens:
            file = os.path.split(tokens['file'])[1]
            matrices[file] = tokens
            results[file] = {}
        else:
            kernel = tokens['kernel']
            results[file][kernel] = tokens
            kernels.add(tokens['kernel'])

    field = 'gflops'
    fid = open('bench_' + value_type + '_' + field + '.csv','r')

    for (matrix,file,path) in trials:
        output_filename = file.split('/')[-1].split('.')[0];

        matrix_results = results[file]
        block_data = np.zeros(5)
        for index,block_size in enumerate(2**np.arange(1,6)):
            block_data[index] = matrix_results['csr_block('+str(block_size)+')']['gflops']

        single_data = float(matrix_results['csr_vector']['gflops'])

        plt.figure()
        plt.semilogx(2**np.arange(1,6), np.ones(5)*single_data, '--k', basex=2, label='Single SpMV')
        plt.semilogx(2**np.arange(1,6), block_data, '-o', basex=2, label='Block SpMV')
        plt.xlabel('Number of columns')
        plt.ylabel('GFLOPs')
        plt.legend(loc='best')

        pdfname = output_filename + '_gflops.pdf'
        savefig(pdfname)
        os.system('pdfcrop ' + pdfname + ' ' + pdfname) # whitespace crop
        os.system('pdfcrop --margins \'0 -6 0 0\' ' + pdfname + ' ' + pdfname) # top tickmark crop
        plt.show()


run_tests('float')
# run_tests('double')

process_data_files('float')
# process_data_files('double')
