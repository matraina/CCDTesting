#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
-------------------

*By: Michelangelo Traina to study skipper CCD data

-------------------
'''
##############################################################################                             
# Input values from command line

import sys

#input FITS file
arg1 = sys.argv[1] + '.fits'
#output FITS file with skipper images
arg2 = sys.argv[2] + '.fits'

import json
with open('config.json') as config_file:
    config = json.load(config_file)
default_directory_structure = config['raw_processed_header_reports_dir_structure']
workingdirectory = config['working_directory']

if default_directory_structure:
    arg1 = 'raw/' + arg1
    arg2 = 'processed/' + arg2

##############################################################################                             
# Get Numpy, and relevant Scipy

import numpy as np
import scipy.fftpack
from scipy import stats
from scipy.stats import norm
from scipy import signal

##############################################################################
# Set up matplotlib and use a nicer set of plot parameters

import matplotlib.pyplot as plt
plt.rc('text',usetex = True)
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

##############################################################################
# Get the FITS package from astropy

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

##############################################################################
# Get processing modules

import functions
from functions import reverse, analysisregion
import reconstruction
import chargeloss
import calibrationdc
import latekreport

##############################################################################
# Specify path (can be out of the main tree)

import os
os.chdir(workingdirectory)

##############################################################################                             
# Open the data image

image_file = get_pkg_data_filename(arg1)

##############################################################################
# Use `astropy.io.fits.info()` to display the structure of the file

fits.info(image_file)

##############################################################################      
# read n. of pixels in x and y axes and number of skips in the image
# the image has nrows and a total of nallcolumns equivalent to ncolumns*nskips

hdr = fits.getheader(image_file,0)
nallcolumns = hdr['NAXIS1'] # n of pixels in the x axis, include the skips  
nrows = hdr['NAXIS2'] # n of pixels in the y axis, i.e. n of rows 
nskips = hdr['NDCMS']  # n of skips
ncolumns = int(nallcolumns/nskips) # n of columns in the image
ampl = hdr['AMPL']
exposuretime = hdr['MEXP']
rdoutime = hdr['MREAD']
if __name__ == '__main__': print('N. rows columns skips ',nrows,ncolumns,nskips)

##############################################################################    
# if leach: image is fixed in reconstruction module
# image reconstruction

image_data,skipper_image_start,skipper_image_end,skipper_averages,skipper_diff, skipper_diff_01,skipper_avg0, skipper_std = reconstruction.reconstructSkipperImage(image_file,arg2)

##############################################################################
#HEREON ONLY SKIPPER IMGS PROCESSING #########################################
##############################################################################
if __name__ == '__main__':
    ##############################################################################
    #ESTIMATE NOISE AT SKIPS: 1, 10, 100 . . . 1000 ##############################
    ##############################################################################
    
    if nskips < 10: naverages = 0
    elif nskips < 100: naverages = 1
    else:
        index=1
        while index <= nskips/100:
            naverages = index+1; index+=1
    startskipfitpar = functions.sigmaFinder(skipper_image_start, debug=False) #ampss, muss, stdss, stduncss
    #if startskipfitpar[1] < 1E+3: imageIsGood *= False; print('Pedestal value is too small: LEACH might have failed.')
    ampmanyskip, mumanyskip, stdmanyskip, stduncmanyskip = [],[],[],[]
    for k in range(naverages): amp, mu, std, stdunc = functions.sigmaFinder(skipper_averages[:,:,k], debug=False); ampmanyskip.append(amp); mumanyskip.append(mu); stdmanyskip.append(std); stduncmanyskip.append(stdunc)

    ##############################################################################
    #FIRST LAST SKIP CHARGE LOSS CHECK: KCL AND SKEW##############################
    ##############################################################################

    diff_image_core = functions.selectImageRegion(skipper_diff,'no_borders')
    PCDDstudyparameters = chargeloss.firstLastSkipPCDDCheck(diff_image_core, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, muPCDD, stdPCDD
    kclsignificance = PCDDstudyparameters[2]/PCDDstudyparameters[3]
    if abs(kclsignificance) > 3: imageIsGood *= False; print('Kcl value flags probable charge loss')

    ##############################################################################
    #ADU TO e- CALIBRATION AND DARK CURRENT ESTIMATES#############################
    ##############################################################################

    parametersDCfit, reducedchisquared, offset = calibrationdc.calibrationDC(skipper_avg0, stdmanyskip[-1], reverse, debug=False)
    calibrationconstant = parametersDCfit[0][5]; calibratedsigma = stdmanyskip[-1]/calibrationconstant
    skipper_avg_cal = -int(reverse)*(skipper_avg0 - offset)/calibrationconstant
    darkcurrentestimateAC = calibrationdc.anticlusteringDarkCurrent(functions.selectImageRegion(skipper_avg_cal,analysisregion), calibratedsigma, debug=False)

    ##############################################################################
    #LATEK REPORTS ###############################################################
    ##############################################################################

    latekreport.produceReport(startskipfitpar, mumanyskip, stdmanyskip, stduncmanyskip, PCDDstudyparameters, offset, reducedchisquared, darkcurrentestimateAC, *parametersDCfit)
