#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (CENPA, University of Washington and LPNHE, Sorbonne Universite) to study skipper CCD data
Executable devoted to guide the clocks/bias parameters tweaking process in skipper CCD and achieve optimal performance using both register amplifiers (L and U) (or different CCDs in modules).

-------------------
'''

import warnings
warnings.filterwarnings("ignore")#to ignore numba warnings. Periodic check required

##############################################################################                             
# Input values from command line

import sys
from numba import jit


#input FITS file
arg1 = sys.argv[1] + '.fits'
#output FITS file with skipper images
arg2 = sys.argv[2] + '.fits'

##############################################################################
# Input values from configuration file. Setting analysis logic
import json
with open('config.json') as config_file:
    config = json.load(config_file)
test = config['test']
workingdirectory = config['working_directory']
default_directory_structure = config['raw_processed_header_reports_dir_structure']
iskipstart = config['skip_start']
iskipend = config['skip_end']
fixLeachReco = config['fix_leach_reconstruction']
reverse = config['reverse']
reversign = 1
if reverse: reversign = -1
row_pedestal_subtract = config['subtract_pedestal_row_by_row']
applymask = config['apply_mask']
if applymask: mask_fits_file = config['mask_file']
chargethreshold = config['profiles_charge_threshold']
registersize = config['ccd_active_register_size']
prescan = config['prescan']
overscan = config['overscan']
analysisregion = config['analysis_region']
kclthreshold = config['kcl_threshold']
calibrationguess = config['calibration_constant_guess']
printheader = config['print_header']
printreport = config['print_report']
if printreport:
    reportHeader = config['tweaking_analysis'][-1]['report'][-1]['header']
    reportImage = config['tweaking_analysis'][-1]['report'][-1]['image']
    reportPCD = config['tweaking_analysis'][-1]['report'][-1]['pcds']
    reportChargeLoss = config['tweaking_analysis'][-1]['report'][-1]['chargeloss']
    reportCalibrationDarkcurrent = config['tweaking_analysis'][-1]['report'][-1]['calibration_darkcurrent']
    reportColumnChargeProfile = config['tweaking_analysis'][-1]['report'][-1]['column_charge_profile']
    reportRowChargeProfile = config['tweaking_analysis'][-1]['report'][-1]['row_charge_profile']
    reportFFTskips = config['tweaking_analysis'][-1]['report'][-1]['fft_skips']
    reportFFTrow = config['tweaking_analysis'][-1]['report'][-1]['fft_row']

if test != 'tweaking':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for two amps tweaking analysis. Test selected in configuration file is different from 'tweaking': do you want to perform tweaking analysis?\nPlease answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with tweaking analysis')

import time
start = time.perf_counter()

if default_directory_structure:
    arg1 = 'raw/' + arg1
    arg2 = 'processed/' + arg2

##############################################################################                             
# Get Numpy and Matplotlib

import numpy as np

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

import m_functions
from m_functions import make_colorbar_with_padding, gauss, factorial, convolutionGaussianPoisson, round_sig_2
import m_reconstruction
import m_chargeloss
import m_calibrationdc

##############################################################################
# Specify path (can be out of the main tree)

import os
os.chdir(workingdirectory)

##############################################################################
#SINGLE IMAGE TWEAKING ANALYSIS ##############################################
##############################################################################
    
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
print('N. rows columns skips ',nrows,ncolumns,nskips)

##############################################################################
# if leach: image is fixed in reconstruction module
# image reconstruction

image_data_L,image_data_U,skipper_image_start_L,skipper_image_start_U,skipper_image_end_L,skipper_image_end_U,skipper_averages_L,skipper_averages_U,skipper_diff_L,skipper_diff_U,skipper_diff_01_L,skipper_diff_01_U,skipper_avg0_L,skipper_avg0_U,skipper_std_L,skipper_std_U = m_reconstruction.reconstructTwoAmpSkipperImages(image_file,arg2,flip_U_img=True)

#pedestal subtraction
if row_pedestal_subtract:
    skipper_image_start_L = m_reconstruction.subtractPedestalRowByRow(skipper_image_start_L)[0]
    skipper_image_start_U = m_reconstruction.subtractPedestalRowByRow(skipper_image_start_U)[0]
    skipper_avg0_L = m_reconstruction.subtractPedestalRowByRow(skipper_avg0_L)[0]
    skipper_avg0_U = m_reconstruction.subtractPedestalRowByRow(skipper_avg0_U)[0]

#apply mask
if applymask:
    mask = fits.getdata(mask_fits_file, ext=0)
    mask_L = m_reconstruction.getMask(mask,'L')
    mask_U = m_reconstruction.getMask(mask,'U')
    skipper_image_start_L = m_reconstruction.applyMask(skipper_image_start_L, mask_L)
    skipper_image_start_U = m_reconstruction.applyMask(skipper_image_start_U, mask_U)
    skipper_avg0_L = m_reconstruction.applyMask(skipper_avg0_L, mask_L)
    skipper_avg0_U = m_reconstruction.applyMask(skipper_avg0_U, mask_U)

##############################################################################
#ESTIMATE NOISE AT SKIPS: 1, 10, 100 . . . 1000 ##############################
##############################################################################

startskipfitpar_L = m_functions.sigmaFinder(skipper_image_start_L, fwhm_est=False, debug=False) #ampss muss, stdss, stduncss
if reportPCD or reportCalibrationDarkcurrent:
    if nskips < 10: naverages = 0
    elif nskips < 100: naverages = 1; numberskips=[10]
    else:
        numberskips=[10]; index=1
        while index <= nskips/100:
            numberskips.append(index*100)
            naverages = index+1; index+=1
    ampmanyskip_L, mumanyskip_L, stdmanyskip_L, stduncmanyskip_L = [],[],[],[]
    for k in range(naverages):
        amp, mu, std, munc, stdunc = m_functions.sigmaFinder(skipper_averages_L[:,:,k], fwhm_est=False, debug=False)
        ampmanyskip_L.append(amp)
        mumanyskip_L.append(mu)
        stdmanyskip_L.append(std)
        stduncmanyskip_L.append(stdunc)

startskipfitpar_U = m_functions.sigmaFinder(skipper_image_start_U, fwhm_est=False, debug=False) #ampss muss, stdss, stduncss
if reportPCD or reportCalibrationDarkcurrent:
    ampmanyskip_U, mumanyskip_U, stdmanyskip_U, stduncmanyskip_U = [],[],[],[]
    for k in range(naverages):
        amp, mu, std, munc, stdunc = m_functions.sigmaFinder(skipper_averages_U[:,:,k], fwhm_est=False, debug=False)
        ampmanyskip_U.append(amp)
        mumanyskip_U.append(mu)
        stdmanyskip_U.append(std)
        stduncmanyskip_U.append(stdunc)
    
##############################################################################
#FIRST LAST SKIP CHARGE LOSS CHECK: KCL AND SKEW##############################
##############################################################################
    
if reportChargeLoss and nskips!=1:
    diff_image_core_01_L,diff_image_core_L = m_functions.selectImageRegion(skipper_diff_01_L,'exposed_pixels_exclude_first_row'),m_functions.selectImageRegion(skipper_diff_L,'exposed_pixels_exclude_first_row')
    diff_image_core_01_U,diff_image_core_U = m_functions.selectImageRegion(skipper_diff_01_U,'exposed_pixels_exclude_first_row'),m_functions.selectImageRegion(skipper_diff_U,'exposed_pixels_exclude_first_row')
    PCDDstudyparameters01_L = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core_01_L, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD
    PCDDstudyparameters01_U = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core_01_U, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD
    PCDDstudyparameters_L = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core_L, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD
    PCDDstudyparameters_U = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core_U, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD
    kclsignificance01_L,kclsignificance_L = PCDDstudyparameters01_L[2]/PCDDstudyparameters01_L[3],PCDDstudyparameters_L[2]/PCDDstudyparameters_L[3]
    kclsignificance01_U,kclsignificance_U = PCDDstudyparameters01_U[2]/PCDDstudyparameters01_U[3],PCDDstudyparameters_U[2]/PCDDstudyparameters_U[3]
    if abs(kclsignificance01_L) > 3 or abs(kclsignificance_L) > 3: print('Kcl value flags probable charge loss for amp L')
    if abs(kclsignificance01_U) > 3 or abs(kclsignificance_U) > 3: print('Kcl value flags probable charge loss for amp U')
    
##############################################################################
#ADU TO e- CALIBRATION AND DARK CURRENT ESTIMATES#############################
##############################################################################

if reportCalibrationDarkcurrent and nskips!=1:
    parametersDCfit_L, reducedchisquared_L, offset_L, nbins_plot_L = m_calibrationdc.calibrationDC(skipper_avg0_L, stdmanyskip_L[-1], reverse, debug=False)
    calibrationconstant_L = parametersDCfit_L[0][5]; calibratedsigma_L = stdmanyskip_L[-1]/calibrationconstant_L
    skipper_avg_cal_L = reversign*(skipper_avg0_L - offset_L)/calibrationconstant_L
    darkcurrentestimateAC_L = m_calibrationdc.anticlusteringDarkCurrent(m_functions.selectImageRegion(skipper_avg_cal_L,analysisregion), calibratedsigma_L, debug=False)
    parametersDCfit_U, reducedchisquared_U, offset_U, nbins_plot_U = m_calibrationdc.calibrationDC(skipper_avg0_U, stdmanyskip_U[-1], reverse, debug=False)
    calibrationconstant_U = parametersDCfit_U[0][5]; calibratedsigma_U = stdmanyskip_U[-1]/calibrationconstant_U
    skipper_avg_cal_U = reversign*(skipper_avg0_U - offset_U)/calibrationconstant_U
    darkcurrentestimateAC_U = m_calibrationdc.anticlusteringDarkCurrent(m_functions.selectImageRegion(skipper_avg_cal_U,analysisregion), calibratedsigma_U, debug=False)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# LATEK REPORT ###############################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

if not printreport or not (reportHeader or reportImage or reportPCD or reportChargeLoss or reportCalibrationDarkcurrent or reportFFTrow or reportFFTskips):
    print('No information to be reported. Exiting'); sys.exit()

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report on 2-Amp Parameter Tweaking'))
doc.preamble.append(Command('author', 'DAMIC-M'))
doc.preamble.append(NoEscape(r'\usepackage{tocloft}'))
doc.preamble.append(NoEscape(r'\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}'))
doc.preamble.append(NoEscape(r'\usepackage{hyperref}'))
doc.preamble.append(NoEscape(r'\usepackage{bookmark}'))
doc.append(NoEscape(r'\maketitle'))
doc.append(NoEscape(r'\tableofcontents'))
doc.append(NoEscape(r'\thispagestyle{empty}'))
doc.append(NewPage())

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# SINGLE IMAGE LATEK REPORT ##################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#############################################
#Print acqusition parameters value in report#
#############################################
if reportHeader:
    if printheader and default_directory_structure: fileheader = open(workingdirectory + 'header/' + sys.argv[2] + '.txt', 'r')
    elif printheader and not default_directory_structure: fileheader = open(workingdirectory + sys.argv[2] + '.txt', 'r')
    if printheader: lines = fileheader.readlines()
    else: lines = repr(fits.getheader(image_file, 0)).splitlines()
    with doc.create(Section('Image Acquisition Parameters')):
        with doc.create(Description()) as desc:
            for line in lines[0:100]:
                if line.split()[0]!='COMMENT': desc.add_item(line,'')
                #desc.add_item(line.split()[0].replace('=','')+'='+line.split()[-1],'')
                if line.split()[0]=='MREAD': break
doc.append(NewPage())

#############################################
###############Image section#################
#############################################

ampss_L, muss_L, stdss_L, muncss_L, stduncss_L = startskipfitpar_L #ss: start skip
ampss_U, muss_U, stdss_U, muncss_U, stduncss_U = startskipfitpar_U #ss: start skip

if reportImage == True: reportImage='cluster'

if reportImage=='full':
        
    with doc.create(Section('Images')):
        
        if nskips!=1:
            plotrange = [0,np.size(skipper_image_start_L,0),0,np.size(skipper_image_start_L,1)]
            width = 8
            fig=plt.figure(figsize=(1.2*width,width))
            from matplotlib import gridspec
            
            gs = gridspec.GridSpec(6,1)
            #gs.update(wspace=0.025, hspace=0.05)
            
            ax1=fig.add_subplot(gs[0,0])
            plt.imshow(skipper_image_start_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Start skip")
            plt.ylabel("row")
            cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
            cb1 = plt.colorbar(cax=cax1)
                
            ax2=fig.add_subplot(gs[1,0])
            plt.imshow(skipper_image_end_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("End skip")
            plt.ylabel("row")
            cax2=make_colorbar_with_padding(ax2) # add a colorbar within its own axis the same size as the image plot
            cb2 = plt.colorbar(cax=cax2)
            
            ax3=fig.add_subplot(gs[2,0])
            plt.imshow(skipper_avg0_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Average")
            plt.ylabel("row")
            cax3=make_colorbar_with_padding(ax3) # add a colorbar within its own axis the same size as the image plot
            cb3 = plt.colorbar(cax=cax3)
            
            ax4=fig.add_subplot(gs[3,0])
            plt.imshow(skipper_std_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Standard deviation")
            plt.ylabel("row")
            cax4=make_colorbar_with_padding(ax4) # add a colorbar within its own axis the same size as the image plot
            cb4 = plt.colorbar(cax=cax4)
            
            ax5=fig.add_subplot(gs[4,0])
            plt.imshow(skipper_diff_01_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("First-second skip difference")
            plt.ylabel("row")
            cax5=make_colorbar_with_padding(ax5) # add a colorbar within its own axis the same size as the image plot
            cb5 = plt.colorbar(cax=cax5)
            
            ax6=fig.add_subplot(gs[5,0])
            plt.imshow(skipper_diff_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))#,extent=(,570,10,0))
            plt.title("Second-end skip difference")
            plt.ylabel("row")
            plt.xlabel("column")
            cax6=make_colorbar_with_padding(ax6) # add a colorbar within its own axis the same size as the image plot
            cb6 = plt.colorbar(cax=cax6)
            
            fig.tight_layout()
            #fig.subplots_adjust(bottom=None,top=None,right=None,hspace = None)

        else:
            #halfrangey = np.size(skipper_image_start,0)/2; halfrangex = np.size(skipper_image_start,1)/2
            plotrange = [0,np.size(skipper_image_start_L,0),0,np.size(skipper_image_start_L,1)]
            fig=plt.figure(figsize=(8,8))
                
            ax1=fig.add_subplot(111)
            plt.imshow(skipper_image_start_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Start skip")
            plt.ylabel("row")
            cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
            cb1 = plt.colorbar(cax=cax1)

        fig.tight_layout(pad=.001)
            
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Exposed pixels region for various images (L-side).')
        plt.clf()
        doc.append(NewPage())
    
        if nskips!=1:
            plotrange = [0,np.size(skipper_image_start_U,0),0,np.size(skipper_image_start_U,1)]
            width = 8
            fig=plt.figure(figsize=(1.2*width,width))
            from matplotlib import gridspec
            
            gs = gridspec.GridSpec(6,1)
            #gs.update(wspace=0.025, hspace=0.05)
            
            ax1=fig.add_subplot(gs[0,0])
            plt.imshow(skipper_image_start_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Start skip")
            plt.ylabel("row")
            cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
            cb1 = plt.colorbar(cax=cax1)
                
            ax2=fig.add_subplot(gs[1,0])
            plt.imshow(skipper_image_end_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("End skip")
            plt.ylabel("row")
            cax2=make_colorbar_with_padding(ax2) # add a colorbar within its own axis the same size as the image plot
            cb2 = plt.colorbar(cax=cax2)
            
            ax3=fig.add_subplot(gs[2,0])
            plt.imshow(skipper_avg0_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Average")
            plt.ylabel("row")
            cax3=make_colorbar_with_padding(ax3) # add a colorbar within its own axis the same size as the image plot
            cb3 = plt.colorbar(cax=cax3)
            
            ax4=fig.add_subplot(gs[3,0])
            plt.imshow(skipper_std_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Standard deviation")
            plt.ylabel("row")
            cax4=make_colorbar_with_padding(ax4) # add a colorbar within its own axis the same size as the image plot
            cb4 = plt.colorbar(cax=cax4)
            
            ax5=fig.add_subplot(gs[4,0])
            plt.imshow(skipper_diff_01_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("First-second skip difference")
            plt.ylabel("row")
            cax5=make_colorbar_with_padding(ax5) # add a colorbar within its own axis the same size as the image plot
            cb5 = plt.colorbar(cax=cax5)
            
            ax6=fig.add_subplot(gs[5,0])
            plt.imshow(skipper_diff_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))#,extent=(,570,10,0))
            plt.title("Second-end skip difference")
            plt.ylabel("row")
            plt.xlabel("column")
            cax6=make_colorbar_with_padding(ax6) # add a colorbar within its own axis the same size as the image plot
            cb6 = plt.colorbar(cax=cax6)
            
            fig.tight_layout()
            #fig.subplots_adjust(bottom=None,top=None,right=None,hspace = None)

        else:
            #halfrangey = np.size(skipper_image_start,0)/2; halfrangex = np.size(skipper_image_start,1)/2
            plotrange = [0,np.size(skipper_image_start_U,0),0,np.size(skipper_image_start_U,1)]
            fig=plt.figure(figsize=(8,8))
                
            ax1=fig.add_subplot(111)
            plt.imshow(skipper_image_start_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Start skip")
            plt.ylabel("row")
            cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
            cb1 = plt.colorbar(cax=cax1)

        fig.tight_layout(pad=.001)
                
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Exposed pixels region for various images (U-side).')
        plt.clf()
        doc.append(NewPage())

if reportImage=='cluster':

    centeredsstoplot = reversign*(skipper_image_start_L - muss_L)
    clustercandidates = m_reconstruction.findChargedPixelNoBorder(centeredsstoplot,stdss_L)
    isChargedCrown = True; coor = np.size(centeredsstoplot,0)//2, np.size(centeredsstoplot,1)//2
    for coor in clustercandidates:
        isChargedCrown = m_reconstruction.chargedCrown(coor,centeredsstoplot,stdss_L)
        if (isChargedCrown):
            #print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
            break
    if not isChargedCrown: coor = np.size(centeredsstoplot,0)//2, np.size(centeredsstoplot,1)//2
        
    if nskips!=1:
    
        halfrangey = 5; halfrangex = 40
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],10-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],80-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
    
        with doc.create(Section('Images')):
    
            fig=plt.figure(figsize=(10,11))
            
            ax1=fig.add_subplot(611)
            plt.imshow(skipper_image_start_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Start skip")
            plt.ylabel("row")
            cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
            cb1 = plt.colorbar(cax=cax1)
            
            fig.subplots_adjust(right=0.9)
            
            ax2=fig.add_subplot(612)
            plt.imshow(skipper_image_end_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("End skip")
            plt.ylabel("row")
            cax2=make_colorbar_with_padding(ax2) # add a colorbar within its own axis the same size as the image plot
            cb2 = plt.colorbar(cax=cax2)
        
            ax3=fig.add_subplot(613)
            plt.imshow(skipper_avg0_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Average")
            plt.ylabel("row")
            cax3=make_colorbar_with_padding(ax3) # add a colorbar within its own axis the same size as the image plot
            cb3 = plt.colorbar(cax=cax3)
            
            ax4=fig.add_subplot(614)
            plt.imshow(skipper_std_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Standard deviation")
            plt.ylabel("row")
            cax4=make_colorbar_with_padding(ax4) # add a colorbar within its own axis the same size as the image plot
            cb4 = plt.colorbar(cax=cax4)
            
            ax5=fig.add_subplot(615)
            plt.imshow(skipper_diff_01_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("First-second skip difference")
            plt.ylabel("row")
            cax5=make_colorbar_with_padding(ax5) # add a colorbar within its own axis the same size as the image plot
            cb5 = plt.colorbar(cax=cax5)
            
            ax6=fig.add_subplot(616)
            plt.imshow(skipper_diff_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))#,extent=(,570,10,0))
            plt.title("Second-end skip difference")
            plt.ylabel("row")
            plt.xlabel("column")
            cax6=make_colorbar_with_padding(ax6) # add a colorbar within its own axis the same size as the image plot
            cb6 = plt.colorbar(cax=cax6)

    else:
        with doc.create(Section('Images')):
            halfrangey = 40; halfrangex = 40
            if coor[0] > halfrangey: deltay = halfrangey,halfrangey
            else: deltay = coor[0],80-coor[0]
            if coor[1] > halfrangex: deltax = halfrangex,halfrangex
            else: deltax = coor[1],80-coor[1]
            plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
            plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
            fig=plt.figure(figsize=(8,8))
            
            ax1=fig.add_subplot(111)
            plt.imshow(skipper_image_start_L[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
            plt.title("Start skip")
            plt.ylabel("row")
            cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
            cb1 = plt.colorbar(cax=cax1)

        fig.tight_layout(pad=.001)
            
    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.9\linewidth'))
        plot.add_caption('Exposed pixels region for various images (L-side).')
    plt.clf()
    doc.append(NewPage())
    
    centeredsstoplot = reversign*(skipper_image_start_U - muss_U)
    clustercandidates = m_reconstruction.findChargedPixelNoBorder(centeredsstoplot,stdss_U)
    isChargedCrown = True; coor = np.size(centeredsstoplot,0)//2, np.size(centeredsstoplot,1)//2
    for coor in clustercandidates:
        isChargedCrown = m_reconstruction.chargedCrown(coor,centeredsstoplot,stdss_U)
        if (isChargedCrown):
            #print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
            break
    if not isChargedCrown: coor = np.size(centeredsstoplot,0)//2, np.size(centeredsstoplot,1)//2
    
    if nskips!=1:
    
        halfrangey = 5; halfrangex = 40
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],10-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],80-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
    
        fig=plt.figure(figsize=(10,11))
        
        ax1=fig.add_subplot(611)
        plt.imshow(skipper_image_start_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title("Start skip")
        plt.ylabel("row")
        cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
        cb1 = plt.colorbar(cax=cax1)
        
        fig.subplots_adjust(right=0.9)
        
        ax2=fig.add_subplot(612)
        plt.imshow(skipper_image_end_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title("End skip")
        plt.ylabel("row")
        cax2=make_colorbar_with_padding(ax2) # add a colorbar within its own axis the same size as the image plot
        cb2 = plt.colorbar(cax=cax2)
        
        ax3=fig.add_subplot(613)
        plt.imshow(skipper_avg0_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title("Average")
        plt.ylabel("row")
        cax3=make_colorbar_with_padding(ax3) # add a colorbar within its own axis the same size as the image plot
        cb3 = plt.colorbar(cax=cax3)
        
        ax4=fig.add_subplot(614)
        plt.imshow(skipper_std_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title("Standard deviation")
        plt.ylabel("row")
        cax4=make_colorbar_with_padding(ax4) # add a colorbar within its own axis the same size as the image plot
        cb4 = plt.colorbar(cax=cax4)
        
        ax5=fig.add_subplot(615)
        plt.imshow(skipper_diff_01_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title("First-second skip difference")
        plt.ylabel("row")
        cax5=make_colorbar_with_padding(ax5) # add a colorbar within its own axis the same size as the image plot
        cb5 = plt.colorbar(cax=cax5)
        
        ax6=fig.add_subplot(616)
        plt.imshow(skipper_diff_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))#,extent=(,570,10,0))
        plt.title("Second-end skip difference")
        plt.ylabel("row")
        plt.xlabel("column")
        cax6=make_colorbar_with_padding(ax6) # add a colorbar within its own axis the same size as the image plot
        cb6 = plt.colorbar(cax=cax6)

    else:
        halfrangey = 40; halfrangex = 40
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],80-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],80-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
        fig=plt.figure(figsize=(8,8))
        
        ax1=fig.add_subplot(111)
        plt.imshow(skipper_image_start_U[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title("Start skip")
        plt.ylabel("row")
        cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
        cb1 = plt.colorbar(cax=cax1)

        fig.tight_layout(pad=.001)
            
    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.9\linewidth'))
        plot.add_caption('Exposed pixels region for various images (U-side).')
    plt.clf()
    doc.append(NewPage())

#############################################
#Pixel charge distribution and noise section#
#############################################

if reportPCD:
    with doc.create(Section('Pixel Charge Distributions and Noise')):
        import m_functions
        
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
        
        skipper_image_start_region_L = m_functions.selectImageRegion(skipper_image_start_L,analysisregion)
        if applymask: skipper_image_start_ravel_L = skipper_image_start_region_L.compressed()
        else: skipper_image_start_ravel_L = skipper_image_start_region_L.ravel()
        #instead of removing 0-entries from histogram use numpy mask to avoid discrepancies between gaussian and plotted PCD skipper_image0ravel
        #skipper_image = [s for s in skipper_image_start_ravel if s != 0]
        if reverse: skipper_image_unsaturated_L = np.ma.masked_equal(skipper_image_start_ravel_L, 0.0, copy=False)
        else: skipper_image_unsaturated_L = skipper_image_start_ravel_L
        skipper_imagehist_L, binedges = np.histogram(skipper_image_unsaturated_L, bins = 800, density=False)
        ampss_L = skipper_imagehist_L[np.argmax(skipper_imagehist_L)]
        axs[0].hist(skipper_image_start_ravel_L, 800, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='start skip pixel charge distribution')
        bincenters = np.arange(muss_L - 3*stdss_L, muss_L + 3*stdss_L + 6*stdss_L/100, 6*stdss_L/100) #last term in upper bound to get ~sym drawing
        axs[0].plot(bincenters, gauss(bincenters,ampss_L,muss_L,stdss_L), label='gaussian fit curve', linewidth=1, color='red')
        axs[0].legend( prop={'size': 16})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        try: axs[0].set_title('Start skip PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdss_L,4)) + ' ADU. Est. noise: ' + str(round(stdss_L/calibrationconstant_L,4)) + ' $e^{-}$')
        except: axs[0].set_title('Start skip PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdss_L,4)) + ' ADU')

        
        if nskips!=1:
            try:
                calibrationconstant_L
                guessCC = False
                if calibrationconstant_L<=1 or calibrationconstant_L>1.5*calibrationguess:
                    calibrationconstant_L = calibrationguess
                    guessCC = True
                    reportCalibrationDarkcurrent = False
                    print('WARNING: inaccurate calibration constant. Using guess value. Will not report on calibration')
            except:
                calibrationconstant_L = calibrationguess
                guessCC = True
                print('WARNING: calibration constant not defined for ADU/e- noise conversion. Using guess value')
            averageimageoffset_L,averageimagestd_L = m_functions.sigmaFinder(skipper_avg0_L, fwhm_est=True, debug=False)[1:3]
            skipper_avg0_region_L = m_functions.selectImageRegion(skipper_avg0_L,analysisregion)
            if applymask: avg_image_0ravel_L = skipper_avg0_region_L.compressed()
            else: avg_image_0ravel_L = skipper_avg0_region_L.ravel()
            if reverse:
                avg_image_unsaturated_L = np.ma.masked_equal(avg_image_0ravel_L, 0.0, copy=False)
                avg_image_unsaturated_L = [s for s in avg_image_unsaturated_L if averageimageoffset_L - 5*calibrationconstant_L < s < averageimageoffset_L + calibrationconstant_L]
                rangeadhoc_L =  (averageimageoffset_L - 5*calibrationconstant_L, averageimageoffset_L + calibrationconstant_L)
            else:
                avg_image_unsaturated_L = avg_image_0ravel_L
                avg_image_unsaturated_L = [s for s in avg_image_unsaturated_L if averageimageoffset_L - calibrationconstant_L < s < 5*averageimageoffset_L + calibrationconstant_L]
                rangeadhoc_L =  (averageimageoffset_L - calibrationconstant_L, averageimageoffset_L + 5*calibrationconstant_L)
            if len(avg_image_unsaturated_L) < 50:
                avg_image_unsaturated_L = [s for s in np.ma.masked_equal(avg_image_0ravel_L, 0.0, copy=False) if - 20*calibrationconstant_L < s - averageimageoffset_L < 20*calibrationconstant_L]
                rangeadhoc_L =  (averageimageoffset_L - 20*calibrationconstant_L, averageimageoffset_L + 20*calibrationconstant_L)
            avg_image_hist_L, binedges = np.histogram([s for s in avg_image_0ravel_L if s != 0], range=rangeadhoc_L, bins = 200, density=False)
            ampls_L = avg_image_hist_L[np.argmax(avg_image_hist_L)]
            bincenters = np.arange(averageimageoffset_L - 3*stdmanyskip_L[-1], averageimageoffset_L + 3*stdmanyskip_L[-1] + 6*stdmanyskip_L[-1]/200, 6*stdmanyskip_L[-1]/200)
            if abs(rangeadhoc_L[1]-rangeadhoc_L[0]) < max(bincenters) - min(bincenters):
                rangeadhoc_L = (min(bincenters),max(bincenters))
                avg_image_hist_L, binedges = np.histogram([s for s in avg_image_0ravel_L if s != 0], range=rangeadhoc_L, bins = 200, density=False)
                ampls_L = avg_image_hist_L[np.argmax(avg_image_hist_L)]
            
            if not guessCC:
                axs[1].hist(avg_image_0ravel_L, 200, rangeadhoc_L, density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')
                if abs(calibrationconstant_L-calibrationguess)/3 < 1: axs[1].plot(bincenters, gauss(bincenters,ampls_L,averageimageoffset_L,averageimagestd_L), label='gaussian fit curve', linewidth=1, color='red')
            else:
                axs[1].hist(avg_image_0ravel_L, 200, range=(min(avg_image_0ravel_L),0.002*max(avg_image_0ravel_L)), density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')
            
            axs[1].legend( prop={'size': 16})
            axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[1].get_yticklabels(), visible=True)
            axs[1].set_title('Average image PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip_L[-1],4)) + ' ADU. Est. noise: ' + str(round(stdmanyskip_L[-1]/calibrationconstant_L,4)) + ' $e^{-}$')
            if guessCC: axs[1].set_title('Average image PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip_L[-1],4)) + ' ADU')
        
        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Start skip and avg image pixel charge distributions computed on '+analysisregion+' image region (L-side).')
        plt.clf()
        doc.append(NewPage())
        
        def r(ns):
            return stdss_L/np.sqrt(ns)
        fig, axs = plt.subplots(1, 1, figsize=(8,6), sharey=True, tight_layout=True)
        #numberSkips = [10,100,200,300,400,500,600,700,800,900,1000]
        ns = np.arange(1,nskips,1)
        #resolution = plt.plot(1,stdss,'ro',numberSkips[0:len(stdmanyskip)],stdmanyskip,'ro',ns,r(ns),'k-')
        if nskips!=1: resolution_L = plt.errorbar(numberskips[0:len(stdmanyskip_L)],stdmanyskip_L,stduncmanyskip_L,xerr=None,ecolor='red',marker='o',fmt='.', mfc='red', mec='red', ms=4, label='measured resolution in ADU')
        else: resolution_L = plt.errorbar([],[])
        resolution_L += plt.errorbar(1,stdss_L,stduncss_L,xerr=None,ecolor='red',marker='o',fmt='.',mfc='red', mec='red', ms=4)
        resolution_L = plt.plot(ns,r(ns),'k--',label='expected $1/\sqrt(N_{skip})$ trend based on first skip sigma')
        plt.legend()
        plt.ylabel('resolution [ADU]')
        plt.xlabel('number of skips')
        plt.xscale('log')
        plt.yscale('log')
        ax.axis([0.9, nskips*1.1, 0.1, 100])
        ax.loglog()
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.title('Resolution trend')
            
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Resolution trend computed on '+analysisregion+' image region, as function of average image skip number (L-side).')
        plt.clf()
        doc.append(NewPage())

        #U-side
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
        
        skipper_image_start_region_U = m_functions.selectImageRegion(skipper_image_start_U,analysisregion)
        if applymask: skipper_image_start_ravel_U = skipper_image_start_region_U.compressed()
        else: skipper_image_start_ravel_U = skipper_image_start_region_U.ravel()
        #instead of removing 0-entries from histogram use numpy mask to avoid discrepancies between gaussian and plotted PCD skipper_image0ravel
        #skipper_image = [s for s in skipper_image_start_ravel if s != 0]
        if reverse: skipper_image_unsaturated_U = np.ma.masked_equal(skipper_image_start_ravel_U, 0.0, copy=False)
        else: skipper_image_unsaturated_U = skipper_image_start_ravel_U
        skipper_imagehist_U, binedges = np.histogram(skipper_image_unsaturated_U, bins = 800, density=False)
        ampss_U = skipper_imagehist_U[np.argmax(skipper_imagehist_U)]
        axs[0].hist(skipper_image_start_ravel_U, 800, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='start skip pixel charge distribution')
        bincenters = np.arange(muss_U - 3*stdss_U, muss_U + 3*stdss_U + 6*stdss_U/100, 6*stdss_U/100) #last term in upper bound to get ~sym drawing
        axs[0].plot(bincenters, gauss(bincenters,ampss_U,muss_U,stdss_U), label='gaussian fit curve', linewidth=1, color='red')
        axs[0].legend(prop={'size':16})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        try: axs[0].set_title('Start skip PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdss_U,4)) + ' ADU. Est. noise: ' + str(round(stdss_U/calibrationconstant_U,4)) + ' $e^{-}$')
        except: axs[0].set_title('Start skip PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdss_U,4)) + ' ADU')

        
        if nskips!=1:
            try:
                calibrationconstant_U
                guessCC = False
                if calibrationconstant_U<=1 or calibrationconstant_U>1.5*calibrationguess:
                    calibrationconstant_U = calibrationguess
                    guessCC = True
                    reportCalibrationDarkcurrent = False
                    print('WARNING: inaccurate calibration constant. Using guess value. Will not report on calibration')
            except:
                calibrationconstant_U = calibrationguess
                guessCC = True
                print('WARNING: calibration constant not defined for ADU/e- noise conversion. Using guess value')
            averageimageoffset_U,averageimagestd_U = m_functions.sigmaFinder(skipper_avg0_U, fwhm_est=True, debug=False)[1:3]
            skipper_avg0_region_U = m_functions.selectImageRegion(skipper_avg0_U,analysisregion)
            if applymask: avg_image_0ravel_U = skipper_avg0_region_U.compressed()
            else: avg_image_0ravel_U = skipper_avg0_region_U.ravel()
            if reverse:
                avg_image_unsaturated_U = np.ma.masked_equal(avg_image_0ravel_U, 0.0, copy=False)
                avg_image_unsaturated_U = [s for s in avg_image_unsaturated_U if averageimageoffset_U - 5*calibrationconstant_U < s < averageimageoffset_U + calibrationconstant_U]
                rangeadhoc_U =  (averageimageoffset_U - 5*calibrationconstant_U, averageimageoffset_U + calibrationconstant_U)
            else:
                avg_image_unsaturated_U = avg_image_0ravel_U
                avg_image_unsaturated_U = [s for s in avg_image_unsaturated_U if averageimageoffset_U - calibrationconstant_U < s < 5*averageimageoffset_U + calibrationconstant_U]
                rangeadhoc_U =  (averageimageoffset_U - calibrationconstant_U, averageimageoffset_U + 5*calibrationconstant_U)
            if len(avg_image_unsaturated_U) < 50:
                avg_image_unsaturated_U = [s for s in np.ma.masked_equal(avg_image_0ravel_U, 0.0, copy=False) if - 20*calibrationconstant_U < s - averageimageoffset_U < 20*calibrationconstant_U]
                rangeadhoc_U =  (averageimageoffset_U - 20*calibrationconstant_U, averageimageoffset_U + 20*calibrationconstant_U)
            avg_image_hist_U, binedges = np.histogram([s for s in avg_image_0ravel_U if s != 0], range=rangeadhoc_U, bins = 200, density=False)
            ampls_U = avg_image_hist_U[np.argmax(avg_image_hist_U)]
            bincenters = np.arange(averageimageoffset_U - 3*stdmanyskip_U[-1], averageimageoffset_U + 3*stdmanyskip_U[-1] + 6*stdmanyskip_U[-1]/200, 6*stdmanyskip_U[-1]/200)
            if abs(rangeadhoc_U[1]-rangeadhoc_U[0]) < max(bincenters) - min(bincenters):
                rangeadhoc_U = (min(bincenters),max(bincenters))
                avg_image_hist_U, binedges = np.histogram([s for s in avg_image_0ravel_U if s != 0], range=rangeadhoc_U, bins = 200, density=False)
                ampls_U = avg_image_hist_U[np.argmax(avg_image_hist_U)]
            if not guessCC:
                axs[1].hist(avg_image_0ravel_U, 200, rangeadhoc_U, density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')
                if abs(calibrationconstant_U-calibrationguess)/3 < 1: axs[1].plot(bincenters, gauss(bincenters,ampls_U,averageimageoffset_U,averageimagestd_U), label='gaussian fit curve', linewidth=1, color='red')
            else:
                axs[1].hist(avg_image_0ravel_U, 200, range=(min(avg_image_0ravel_U),0.002*max(avg_image_0ravel_U)), density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')

            axs[1].legend(prop={'size':16})
            axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[1].get_yticklabels(), visible=True)
            axs[1].set_title('Average image PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip_U[-1],4)) + ' ADU. Est. noise: ' + str(round(stdmanyskip_U[-1]/calibrationconstant_U,4)) + ' $e^{-}$')
            if guessCC: axs[1].set_title('Average image PCD in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip_U[-1],4)) + ' ADU')
        
        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Start skip and avg image pixel charge distributions computed on '+analysisregion+' image region (U-side).')
        plt.clf()
        doc.append(NewPage())
        
        def r(ns):
            return stdss_U/np.sqrt(ns)
        fig, axs = plt.subplots(1, 1, figsize=(8,6), sharey=True, tight_layout=True)
        #numberSkips = [10,100,200,300,400,500,600,700,800,900,1000]
        ns = np.arange(1,nskips,1)
        #resolution = plt.plot(1,stdss,'ro',numberSkips[0:len(stdmanyskip)],stdmanyskip,'ro',ns,r(ns),'k-')
        if nskips!=1: resolution_U = plt.errorbar(numberskips[0:len(stdmanyskip_U)],stdmanyskip_U,stduncmanyskip_U,xerr=None,ecolor='red',marker='o',fmt='.', mfc='red', mec='red', ms=4, label='measured resolution in ADU')
        else: resolution_U = plt.errorbar([],[])
        resolution_U += plt.errorbar(1,stdss_U,stduncss_U,xerr=None,ecolor='red',marker='o',fmt='.', mfc='red', mec='red', ms=4)
        resolution_U = plt.plot(ns,r(ns),'k--',label='expected $1/\sqrt(N_{skip})$ trend based on first skip sigma')
        plt.legend( prop={'size': 14})
        plt.ylabel('resolution [ADU]')
        plt.xlabel('number of skips')
        plt.xscale('log')
        plt.yscale('log')
        ax.axis([0.9, nskips*1.1, 0.1, 100])
        ax.loglog()
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.title('Resolution trend')
            
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Resolution trend computed on '+analysisregion+' image region, as function of average image skip number (U-side).')
        plt.clf()
        doc.append(NewPage())



#############################################
#Charge loss indicators and skewness section#
#############################################
if reportChargeLoss and nskips!=1:
    with doc.create(Section('Charge-loss')):
        skewnessPCDD01_L, skewnessPCDDuncertainty01_L, kclPCDD01_L, kclPCDDuncertainty01_L, ampPCDD01_L, muPCDD01_L, stdPCDD01_L = PCDDstudyparameters01_L
        skewnessPCDD_L, skewnessPCDDuncertainty_L, kclPCDD_L, kclPCDDuncertainty_L, ampPCDD_L, muPCDD_L, stdPCDD_L = PCDDstudyparameters_L
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
        skipperdiffcoreravelled01_L = diff_image_core_01_L.ravel()
        skipperdiffcoreravelled_L = diff_image_core_L.ravel()
        
        skipperdiffcoreravelledinrange01_L = [s for s in skipperdiffcoreravelled01_L if s > muPCDD01_L - 3*stdPCDD01_L and s < muPCDD01_L + 3*stdPCDD01_L and s != 0]
        numbins = int(max(skipperdiffcoreravelledinrange01_L) - min(skipperdiffcoreravelledinrange01_L))
        skipperdiffcoreravelledinrangehist01_L, binedges = np.histogram(skipperdiffcoreravelledinrange01_L, numbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        pguess = [ampPCDD01_L,muPCDD01_L,stdPCDD01_L]
        try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist01_L, p0=pguess); PCDDhistfit01_L = gauss(bincenters,*pfit)
        except: pfit = pguess; PCDDhistfit01_L = gauss(bincenters,*pfit)
        axs[0].hist(skipperdiffcoreravelledinrange01_L, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
        axs[0].plot(bincenters, PCDDhistfit01_L, label='gaussian fit curve', linewidth=1, color='red')
        #axs[0].plot(bincenters,skipperdiffcoreravelledinrangehist01, label='pixel charge difference distribution', color='teal')
        axs[0].legend( prop={'size': 16})
        axs[0].set_yscale('linear')
        axs[0].set_ylim(0,1.4*max(PCDDhistfit01_L))
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[0].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
        skipperdiffcoreravelledinrange_L = [s for s in skipperdiffcoreravelled_L if s > muPCDD_L - 3*stdPCDD_L and s < muPCDD_L + 3*stdPCDD_L and s != 0]
        numbins = int(max(skipperdiffcoreravelledinrange_L) - min(skipperdiffcoreravelledinrange_L))
        skipperdiffcoreravelledinrangehist_L, binedges = np.histogram(skipperdiffcoreravelledinrange_L, numbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        pguess = [ampPCDD_L,muPCDD_L,stdPCDD_L]
        try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist_L, p0=pguess); PCDDhistfit_L = gauss(bincenters,*pfit)
        except: pfit = pguess; PCDDhistfit_L = gauss(bincenters,*pfit)
        axs[1].hist(skipperdiffcoreravelledinrange_L, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
        axs[1].plot(bincenters, PCDDhistfit_L, label='gaussian fit curve', linewidth=1, color='red')
        #axs[1].plot(bincenters,skipperdiffcoreravelledinrangehist, label='pixel charge difference distribution', color='teal')
        axs[1].legend( prop={'size': 16})
        axs[1].set_yscale('linear')
        axs[1].set_ylim(0,1.4*max(PCDDhistfit_L))
        axs[1].tick_params(axis='both', which='both', length=10, direction='in')
        axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[1].get_yticklabels(), visible=True)
        axs[1].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Full image pixel charge difference distributions (PCDD) between first and second skip (top) and second and end skip (bottom) (L-side). Entries at 0 (saturation digitizer range) might be masked for analysis purposes.')
        doc.append(NoEscape(r'NOTE: A good gaussian fit of the PCDDs is essential for $S_{k_{cl}}$ to be an effective charge loss classifier'))
        plt.clf()
        doc.append(NewPage())
            
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
        centeredskipperdiffcore01_L = [s for s in skipperdiffcoreravelled01_L-muPCDD01_L if s != -muPCDD01_L]
        axs[0].hist(centeredskipperdiffcore01_L, 600, range = (-20*stdPCDD01_L,10*stdPCDD01_L), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
        #axs[0].set_ylim(0,1.4*max())
        axs[0].legend(loc='upper left', prop={'size': 17})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[0].set_title('$k_{cl}~=~$' + str(round(kclPCDD01_L,4)) + '$\pm$'+ str(round(kclPCDDuncertainty01_L,4)) + ', $S_{k_{cl}}~=~$' + str(round(kclPCDD01_L/kclPCDDuncertainty01_L,4)) + ', skewness = ' + str(round(skewnessPCDD01_L,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty01_L,4)))

        centeredskipperdiffcore_L = [s for s in skipperdiffcoreravelled_L-muPCDD_L if s != -muPCDD_L]
        axs[1].hist(centeredskipperdiffcore_L, 600, range = (-20*stdPCDD_L,10*stdPCDD_L), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
        #axs[1].set_ylim(0,1.4*max())
        axs[1].legend(loc='upper left', prop={'size': 17})
        axs[1].tick_params(axis='both', which='both', length=10, direction='in')
        axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[1].set_title('$k_{cl}~=~$' + str(round(kclPCDD_L,4)) + '$\pm$'+ str(round(kclPCDDuncertainty_L,4)) + ', $S_{k_{cl}}~=~$' + str(round(kclPCDD_L/kclPCDDuncertainty_L,4)) + ', skewness = ' + str(round(skewnessPCDD_L,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty_L,4)))

        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Pedestal-subtracted full-image PCDDs: first and second skip (top) and second and end skip (bottom) (L-side).')
            from scipy.stats import norm
            doc.append('First-second skip lower tail entries: '+str(len([s for s in centeredskipperdiffcore01_L if s < -kclthreshold*stdPCDD01_L]))+'. First-second skip upper tail entries: '+str(len([s for s in centeredskipperdiffcore01_L if s > kclthreshold*stdPCDD01_L]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffcore01_L)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.\n Second-last skip lower tail entries: '+str(len([s for s in centeredskipperdiffcore_L if s < -kclthreshold*stdPCDD_L]))+'. Second-last skip upper tail entries: '+str(len([s for s in centeredskipperdiffcore_L if s > kclthreshold*stdPCDD_L]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffcore_L)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.')

        plt.clf()
        doc.append(NewPage())
            
        skewnessPCDD01_U, skewnessPCDDuncertainty01_U, kclPCDD01_U, kclPCDDuncertainty01_U, ampPCDD01_U, muPCDD01_U, stdPCDD01_U = PCDDstudyparameters01_U
        skewnessPCDD_U, skewnessPCDDuncertainty_U, kclPCDD_U, kclPCDDuncertainty_U, ampPCDD_U, muPCDD_U, stdPCDD_U = PCDDstudyparameters_U
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
        skipperdiffcoreravelled01_U = diff_image_core_01_U.ravel()
        skipperdiffcoreravelled_U = diff_image_core_U.ravel()
        
        skipperdiffcoreravelledinrange01_U = [s for s in skipperdiffcoreravelled01_U if s > muPCDD01_U - 3*stdPCDD01_U and s < muPCDD01_U + 3*stdPCDD01_U and s != 0]
        numbins = int(max(skipperdiffcoreravelledinrange01_U) - min(skipperdiffcoreravelledinrange01_U))
        skipperdiffcoreravelledinrangehist01_U, binedges = np.histogram(skipperdiffcoreravelledinrange01_U, numbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        pguess = [ampPCDD01_U,muPCDD01_U,stdPCDD01_U]
        try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist01_U, p0=pguess); PCDDhistfit01_U = gauss(bincenters,*pfit)
        except: pfit = pguess; PCDDhistfit01_U = gauss(bincenters,*pfit)
        axs[0].hist(skipperdiffcoreravelledinrange01_U, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
        axs[0].plot(bincenters, PCDDhistfit01_U, label='gaussian fit curve', linewidth=1, color='red')
        #axs[0].plot(bincenters,skipperdiffcoreravelledinrangehist01, label='pixel charge difference distribution', color='teal')
        axs[0].set_ylim(0,1.4*max(PCDDhistfit01_U))
        axs[0].legend(prop={'size': 16})
        axs[0].set_yscale('linear')
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[0].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
        skipperdiffcoreravelledinrange_U = [s for s in skipperdiffcoreravelled_U if s > muPCDD_U - 3*stdPCDD_U and s < muPCDD_U + 3*stdPCDD_U and s != 0]
        numbins = int(max(skipperdiffcoreravelledinrange_U) - min(skipperdiffcoreravelledinrange_U))
        skipperdiffcoreravelledinrangehist_U, binedges = np.histogram(skipperdiffcoreravelledinrange_U, numbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        pguess = [ampPCDD_U,muPCDD_U,stdPCDD_U]
        try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist_U, p0=pguess); PCDDhistfit_U = gauss(bincenters,*pfit)
        except: pfit = pguess; PCDDhistfit_U = gauss(bincenters,*pfit)
        axs[1].hist(skipperdiffcoreravelledinrange_U, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
        axs[1].plot(bincenters, PCDDhistfit_U, label='gaussian fit curve', linewidth=1, color='red')
        #axs[1].plot(bincenters,skipperdiffcoreravelledinrangehist, label='pixel charge difference distribution', color='teal')
        axs[1].set_ylim(0,1.4*max(PCDDhistfit_U))
        axs[1].legend(prop={'size': 16})
        axs[1].set_yscale('linear')
        axs[1].tick_params(axis='both', which='both', length=10, direction='in')
        axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[1].get_yticklabels(), visible=True)
        axs[1].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Full image pixel charge difference distributions (PCDD) between first and second skip (top) and second and end skip (bottom) (U-side). Entries at 0 (saturation digitizer range) might be masked for analysis purposes.')
        doc.append(NoEscape(r'NOTE: A good gaussian fit of the PCDDs is essential for $S_{k_{cl}}$ to be an effective charge loss classifier'))
        plt.clf()
        doc.append(NewPage())
            
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
        centeredskipperdiffcore01_U = [s for s in skipperdiffcoreravelled01_U-muPCDD01_U if s != -muPCDD01_U]
        axs[0].hist(centeredskipperdiffcore01_U, 600, range = (-20*stdPCDD01_U,10*stdPCDD01_U), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
        axs[0].legend(loc='upper left', prop={'size': 17})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[0].set_title('$k_{cl}~=~$' + str(round(kclPCDD01_U,4)) + '$\pm$'+ str(round(kclPCDDuncertainty01_U,4)) + ', $S_{k_{cl}}~=~$' + str(round(kclPCDD01_U/kclPCDDuncertainty01_U,4)) + ', skewness = ' + str(round(skewnessPCDD01_U,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty01_U,4)))

        centeredskipperdiffcore_U = [s for s in skipperdiffcoreravelled_U-muPCDD_U if s != -muPCDD_U]
        axs[1].hist(centeredskipperdiffcore_U, 600, range = (-20*stdPCDD_U,10*stdPCDD_U), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
        axs[1].legend(loc='upper left', prop={'size': 17})
        axs[1].tick_params(axis='both', which='both', length=10, direction='in')
        axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[1].set_title('$k_{cl}~=~$' + str(round(kclPCDD_U,4)) + '$\pm$'+ str(round(kclPCDDuncertainty_U,4)) + ', $S_{k_{cl}}~=~$' + str(round(kclPCDD_U/kclPCDDuncertainty_U,4)) + ', skewness = ' + str(round(skewnessPCDD_U,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty_U,4)))

        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Pedestal-subtracted full-image PCDDs: first and second skip (top) and second and end skip (bottom) (U-side).')
            from scipy.stats import norm
            doc.append('First-second skip lower tail entries: '+str(len([s for s in centeredskipperdiffcore01_U if s < -kclthreshold*stdPCDD01_U]))+'. First-second skip upper tail entries: '+str(len([s for s in centeredskipperdiffcore01_U if s > kclthreshold*stdPCDD01_U]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffcore01_U)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.\n Second-last skip lower tail entries: '+str(len([s for s in centeredskipperdiffcore_U if s < -kclthreshold*stdPCDD_U]))+'. Second-last skip upper tail entries: '+str(len([s for s in centeredskipperdiffcore_U if s > kclthreshold*stdPCDD_U]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffcore_U)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.')
        plt.clf()
        doc.append(NewPage())

#############################################
##Calibrated image and Dark Current section##
#############################################

if reportCalibrationDarkcurrent and nskips!=1:
    #transform to masked array with no mask, if using no mask
    if not applymask: skipper_avg_cal_L = np.ma.masked_array(skipper_avg_cal_L, mask=None)
    #after masking select region of interest
    if analysisregion == 'arbitrary': skipper_avg_cal_L = m_functions.selectImageRegion(skipper_avg_cal_L,analysisregion)
    skipperavgcalibratedravel_L = skipper_avg_cal_L.compressed()
    #case for failed calibrations (very large constant not included)
    #if calibrationconstant_L == calibrationguess or calibrationconstant_L <= 1: calibrationconstant_L = calibrationguess #skipperavgcalibratedravel_L = [s for s in skipperavgcalibratedravel_L if s > -10 and  s < 10]
    #nbins factor keeps track of binning for plot consistency
    nbinsfactor = 10
    nbins = nbinsfactor*nbins_plot_L
    #if nbins_plot=0 there is a problem with the image or the mask
    if nbins == 0: nbins=100
    #plot calibrated average image histogram and gauss-poisson fit function
    skipperavgcalibratedravelhist_L, binedges = np.histogram(skipperavgcalibratedravel_L, nbins, density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    #both amplitude [2] and sigma [3] must be rescaled by calibration constant
    dcpar_L = parametersDCfit_L[0][0], parametersDCfit_L[0][1], parametersDCfit_L[0][2]/calibrationconstant_L, parametersDCfit_L[0][3]/calibrationconstant_L
    #same for uncertainty
    try: dcparunc_L = parametersDCfit_L[1][0], parametersDCfit_L[1][1], parametersDCfit_L[1][2]/calibrationconstant_L, parametersDCfit_L[1][3]/calibrationconstant_L, parametersDCfit_L[1][5]
    #if calibration failed
    except: dcparunc_L = 0,0,0,0,0
    #plot histogram
    skipperavgcalibratedravelhistfit_L = convolutionGaussianPoisson(bincenters,*dcpar_L)/nbinsfactor
    plt.hist(skipperavgcalibratedravel_L, nbins, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='avg image calibrated pixel charge distribution')
    #plot fit function
    plt.plot(bincenters, skipperavgcalibratedravelhistfit_L, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(reducedchisquared_L)), color='red')
    #cosmetics
    plt.legend( prop={'size': 17})
    plt.xlabel('pixel value [e$^-$]')
    plt.ylabel('counts')
    plt.yscale("log")
    #x range in electrons
    plt.xlim(-1.5,4)
    #y range, upper limit is 0.5 maximum amplitude (0-electron counts only)
    plt.ylim(0.8, 0.5*parametersDCfit_L[0][2]/calibrationconstant_L*nbinsfactor)
    plt.tick_params(axis='both', which='both', length=10, direction='in')
    plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
    #plt.setp(ax.get_yticklabels(), visible=True)
    try: plt.title('$I_{CF}~=~$' + str(round(dcpar_L[0],6)) + '$\pm$' + str(round_sig_2(dcparunc_L[0])) + ' $e^-$pix$^{-1}$, $I_{AC}~=~$' + str(round(darkcurrentestimateAC_L,6)) + ' $e^-$pix$^{-1}$')
    except: plt.title('$I_{CF}~=~$' + str(round(dcpar_L[0],6)) + '$\pm$' + str(dcparunc_L[0]) + ' $e^-$pix$^{-1}$, $I_{AC}~=~$' + str(round(darkcurrentestimateAC_L,6)) + ' $e^-$pix$^{-1}$')
        
    with doc.create(Section('Dark Current')):
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            if analysisregion == 'arbitrary': plot.add_caption('Calibrated pixel charge distribution (L-side). Dark current values computed with convolution fit (on arbitrary image region) and anticlustering (on '+analysisregion+' image region).')
            else: plot.add_caption('Calibrated pixel charge distribution (L-side). Dark current values computed with convolution fit (on full image region) and anticlustering (on '+analysisregion+' image region).')
        calibrationline = 'Calibration constant is: '+str(round(calibrationconstant_L,4))+'±'+str(round_sig_2(dcparunc_L[4]))+' ADU per electron.'
        doc.append(calibrationline)
        plt.clf()
        doc.append(NewPage())
   
    #transform to masked array with no mask, if using no mask
    if not applymask: skipper_avg_cal_U = np.ma.masked_array(skipper_avg_cal_U, mask=None)
    #after masking select region of interest
    if analysisregion == 'arbitrary': skipper_avg_cal_U = m_functions.selectImageRegion(skipper_avg_cal_U,analysisregion)
    skipperavgcalibratedravel_U = skipper_avg_cal_U.compressed()
    #case for failed calibrations (very large constant not included)
    #if calibrationconstant_U == calibrationguess or calibrationconstant_U <= 1: calibrationconstant_U = calibrationguess #skipperavgcalibratedravel_L = [s for s in skipperavgcalibratedravel_L if s > -10 and  s < 10]
    #nbins factor keeps track of binning for plot consistency
    nbinsfactor = 10
    nbins = nbinsfactor*nbins_plot_U
    #if nbins_plot=0 there is a problem with the image or the mask
    if nbins == 0: nbins=100
    #plot calibrated average image histogram and gauss-poisson fit function
    skipperavgcalibratedravelhist_U, binedges = np.histogram(skipperavgcalibratedravel_U, nbins, density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    #both amplitude [2] and sigma [3] must be rescaled by calibration constant
    dcpar_U = parametersDCfit_U[0][0], parametersDCfit_U[0][1], parametersDCfit_U[0][2]/calibrationconstant_U, parametersDCfit_U[0][3]/calibrationconstant_U
    #same for uncertainty
    try: dcparunc_U = parametersDCfit_U[1][0], parametersDCfit_U[1][1], parametersDCfit_U[1][2]/calibrationconstant_U, parametersDCfit_U[1][3]/calibrationconstant_U, parametersDCfit_U[1][5]
    #if calibration failed
    except: dcparunc_L = 0,0,0,0,0
    #plot histogram
    skipperavgcalibratedravelhistfit_U = convolutionGaussianPoisson(bincenters,*dcpar_U)/nbinsfactor
    plt.hist(skipperavgcalibratedravel_U, nbins, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='avg image calibrated pixel charge distribution')
    #plot fit function
    plt.plot(bincenters, skipperavgcalibratedravelhistfit_U, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(reducedchisquared_U)), color='red')
    #cosmetics
    plt.legend( prop={'size': 17})
    plt.xlabel('pixel value [e$^-$]')
    plt.ylabel('counts')
    plt.yscale("log")
    #x range in electrons
    plt.xlim(-1.5,4)
    #y range, upper limit is 0.5 maximum amplitude (0-electron counts only)
    plt.ylim(0.8, 0.5*parametersDCfit_U[0][2]/calibrationconstant_U*nbinsfactor)
    plt.tick_params(axis='both', which='both', length=10, direction='in')
    plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
    #plt.setp(ax.get_yticklabels(), visible=True)
    try: plt.title('$I_{CF}~=~$' + str(round(dcpar_U[0],6)) + '$\pm$' + str(round_sig_2(dcparunc_U[0])) + ' $e^-$pix$^{-1}$, $I_{AC}~=~$' + str(round(darkcurrentestimateAC_U,6)) + ' $e^-$pix$^{-1}$')
    except: plt.title('$I_{CF}~=~$' + str(round(dcpar_U[0],6)) + '$\pm$' + str(dcparunc_U[0]) + ' $e^-$pix$^{-1}$, $I_{AC}~=~$' + str(round(darkcurrentestimateAC_U,6)) + ' $e^-$pix$^{-1}$')
        
    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.9\linewidth'))
        if analysisregion == 'arbitrary': plot.add_caption('Calibrated pixel charge distribution (U-side). Dark current values computed with convolution fit (on arbitrary image region) and anticlustering (on '+analysisregion+' image region).')
        else: plot.add_caption('Calibrated pixel charge distribution (U-side). Dark current values computed with convolution fit (on full image region) and anticlustering (on '+analysisregion+' image region).')
    calibrationline = 'Calibration constant is: '+str(round(calibrationconstant_U,4))+'±'+str(round_sig_2(dcparunc_U[4]))+' ADU per electron.'
    doc.append(calibrationline)
    plt.clf()
    doc.append(NewPage())

if reportColumnChargeProfile and nskips!=1:

    madthreshold = 5

    #column profile plots (only when calibrated)
    from scipy.stats import median_abs_deviation
    skipper_avg_cal_full_L = reversign*(skipper_avg0_L - offset_L)/calibrationconstant_L
    columnprofile_L,do_plot_profile = m_functions.profileCharge(skipper_avg_cal_full_L,'columns',chargethreshold,do_plot=False)
    columns = np.arange(np.size(columnprofile_L))
        
    if do_plot_profile:
        
        cpmad_L = median_abs_deviation(columnprofile_L)
        hotcols_L = []
        for i in range(np.size(columns)):
            if columnprofile_L[i] > madthreshold*cpmad_L: hotcols_L.append(columns[i])
        
        plt.plot(columns,columnprofile_L,'o',color='teal', markersize=4,linestyle=':')
        plt.axhline(y = madthreshold*cpmad_L, color = 'red', linestyle = '-', label='y = '+str(madthreshold)+' MAD')
        plt.xlabel('column number')
        plt.ylabel('counts')
        #plt.yscale('log')
        plt.legend()
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.title('L-side column charge profile')
    
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
        doc.append('L-side column profile with charge threshold set to '+str(chargethreshold)+' electrons. Columns above '+str(madthreshold)+' MAD ('+str(madthreshold*cpmad_L)+' counts) are: '+ str(hotcols_L))
        plt.clf()
        doc.append(NewPage())
    
    skipper_avg_cal_full_U = reversign*(skipper_avg0_U - offset_U)/calibrationconstant_U
    columnprofile_U = m_functions.profileCharge(skipper_avg_cal_full_U,'columns',chargethreshold,do_plot=False)[0]

    if do_plot_profile:
    
        cpmad_U = median_abs_deviation(columnprofile_U)
        hotcols_U = []
        for i in range(np.size(columns)):
            if columnprofile_U[i] > madthreshold*cpmad_U: hotcols_U.append(columns[i])
    
        plt.bar(np.arange(np.size(columnprofile_U)),columnprofile_U,3,color='teal')
        plt.axhline(y = madthreshold*cpmad_U, color = 'red', linestyle = '-', label='y = '+str(madthreshold)+' MAD')
        plt.xlabel('column number')
        plt.ylabel('counts')
        #plt.yscale('log')
        plt.legend()
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.title('U-side column charge profile')
    
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
        doc.append('U-side column profile with charge threshold set to '+str(chargethreshold)+' electrons. Columns above '+str(madthreshold)+' MAD ('+str(madthreshold*cpmad_L)+' counts) are: '+ str(list(hotcols_U)))
        plt.clf()
        doc.append(NewPage())
   
#############################################
#######Fast Fourier Transform section #######
#############################################
if (reportFFTskips or reportFFTrow):
    import m_functions
    nallcolumns = hdr['NAXIS1']
    nrows = hdr['NAXIS2']
    nskips = hdr['NDCMS']
    with doc.create(Section('Fourier Analysis')):

        if reportFFTskips and nskips!=1:
            ncolumns = int(nallcolumns/nskips)//2 # n of columns in the image
            samplet = hdr['MREAD']*0.001/(nrows*nallcolumns//2)
            m_functions.pixelFFT(image_data_L, nrows-1, ncolumns-1, nskips, samplet)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Full image Fast Fourier Transform (first to last skip) (L-side).')
            plt.clf()
            
            ncolumns = int(nallcolumns/nskips)//2 # n of columns in the image
            samplet = hdr['MREAD']*0.001/(nrows*nallcolumns//2)
            m_functions.pixelFFT(image_data_U, nrows-1, ncolumns-1, nskips, samplet)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Full image Fast Fourier Transform (first to last skip) (U-side).')
            plt.clf()
        
        if reportFFTrow:
            ncolumns = int(nallcolumns/nskips)//2 # n of columns in the image
            samplet = hdr['MREAD']*0.001/(nrows*ncolumns)
            if nskips!=1: m_functions.rowFFT(skipper_avg0_L, nrows-1, ncolumns//2-1, samplet)
            else: m_functions.rowFFT(image_data_L, nrows-1, ncolumns//2-1, samplet)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Average image Fast Fourier Transform (all row pixels) (L-side).')
            plt.clf()
            doc.append(NewPage())
            
            ncolumns = int(nallcolumns/nskips)//2 # n of columns in the image
            samplet = hdr['MREAD']*0.001/(nrows*ncolumns)
            if nskips!=1: m_functions.rowFFT(skipper_avg0_U, nrows-1, ncolumns-1, samplet)
            else: m_functions.rowFFT(image_data_U, nrows-1, ncolumns//2-1, samplet)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Average image Fast Fourier Transform (all row pixels) (U-side).')
            plt.clf()
            doc.append(NewPage())

#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure: reportname = 'reports/twoeaking_'+sys.argv[2]
else: reportname = 'twoeaking_'+sys.argv[2]
doc.generate_pdf(reportname, clean_tex=True)

#############################################
#############Attach values in file###########
#############################################

#with open("dc6414.txt", "a") as file_object:
#    file_object.write(str(round(dcpar_L[0],6))+' '+str(round(dcpar_U[0],6))+'\r')
#
#with open("gain6414.txt", "a") as file_object:
#    file_object.write(str(round(calibrationconstant_L,4))+' '+str(round(calibrationconstant_U,4))+'\r')

#############################################
#############END#############################
#############################################
end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')

'''
# create data

L = skipper_image_start_L.flatten()
U = skipper_image_start_U.flatten()

L -= muss_L
U -= muss_U

# Big bins
hist=plt.hist2d(L, U, bins=(100, 100), cmap=plt.cm.jet)
#xcenters = 0.5*(hist[1][:-1]+hist[1][1:])
#ycenters = 0.5*(hist[2][:-1]+hist[2][1:])
#plt.contour(xcenters,ycenters,hist[0], levels=[100,220], colors='teal')
plt.hist2d(L, U, bins=(100, 100), cmap=plt.cm.jet)
#plt.hist2d(L, U, bins=(100, 100), cmap=plt.cm.BuPu)
plt.xlabel('L amplifier pixel value [ADU]')
plt.ylabel('U amplifier pixel value [ADU]')
plt.colorbar()
plt.title('Amplifiers Correlation Map')
plt.show()
'''
