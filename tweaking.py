#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Executable devoted to guide the clocks/bias parameters tweaking process in skipper CCD and achieve optimal performance.

-------------------
'''

##############################################################################                             
# Input values from command line

import sys

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
registersize = config['ccd_active_register_size']
prescan = config['prescan']
overscan = config['overscan']
analysisregion = config['analysis_region']
kclthreshold = config['kcl_threshold']
calibrationguess = config['calibration_constant_guess']
printheader = config['print_header']
multipleimages = config['tweaking_analysis'][-1]['multiple_images'][-1]['use_multiple_images']
reportHeader = config['tweaking_analysis'][-1]['report'][-1]['header']
reportImage = config['tweaking_analysis'][-1]['report'][-1]['image']
reportPCD = config['tweaking_analysis'][-1]['report'][-1]['pcds']
reportChargeLoss = config['tweaking_analysis'][-1]['report'][-1]['chargeloss']
reportCalibrationDarkcurrent = config['tweaking_analysis'][-1]['report'][-1]['calibration_darkcurrent']
reportFFTskips = config['tweaking_analysis'][-1]['report'][-1]['fft_skips']
reportFFTrow = config['tweaking_analysis'][-1]['report'][-1]['fft_row']

if test != 'tweaking':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for one-amp tweaking analysis. Test selected in configuration file is different from 'tweaking': do you want to perform tweaking analysis?\nPlease answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with tweaking analysis')

import time
start = time.perf_counter()

if default_directory_structure:
    arg1 = 'raw/' + arg1
    arg2 = 'processed/' + arg2
    
if multipleimages:
    arg1 = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    arg2 = ''.join([i for i in arg2 if not i.isdigit()]).replace('.fits','')

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

if not multipleimages:
    
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

    image_data,skipper_image_start,skipper_image_end,skipper_averages,skipper_diff,skipper_diff_01,skipper_avg0,skipper_std = m_reconstruction.reconstructSkipperImage(image_file,arg2)
    
    #pedestal subtraction
    if row_pedestal_subtract:
        skipper_image_start = m_reconstruction.subtractPedestalRowByRow(skipper_image_start)[0]
        skipper_avg0 = m_reconstruction.subtractPedestalRowByRow(skipper_avg0)[0]
        
    #apply mask
    if applymask:
        mask = fits.getdata(mask_fits_file, ext=0)
        skipper_image_start = m_reconstruction.applyMask(skipper_image_start, mask)
        skipper_avg0 = m_reconstruction.applyMask(skipper_avg0, mask)

    ##############################################################################
    #ESTIMATE NOISE AT SKIPS: 1, 10, 100 . . . 1000 ##############################
    ##############################################################################

    startskipfitpar = m_functions.sigmaFinder(skipper_image_start, fwhm_est=False, debug=False) #ampss, muss, stdss, stduncss
    if reportPCD or reportCalibrationDarkcurrent:
        if nskips < 10: naverages = 0
        elif nskips < 100: naverages = 1; numberskips=[10]
        else:
            numberskips=[10]; index=1
            while index <= nskips/100:
                numberskips.append(index*100)
                naverages = index+1; index+=1
        ampmanyskip, mumanyskip, stdmanyskip, stduncmanyskip = [],[],[],[]
        for k in range(naverages): amp, mu, std, munc, stdunc = m_functions.sigmaFinder(skipper_averages[:,:,k], fwhm_est=True, debug=False); ampmanyskip.append(amp);mumanyskip.append(mu); stdmanyskip.append(std); stduncmanyskip.append(stdunc)

    ##############################################################################
    #FIRST LAST SKIP CHARGE LOSS CHECK: KCL AND SKEW##############################
    ##############################################################################
        
    if reportChargeLoss and nskips!=1:
        diff_image_core_01,diff_image_core = m_functions.selectImageRegion(skipper_diff_01,'exposed_pixels_exclude_first_row'),m_functions.selectImageRegion(skipper_diff,'exposed_pixels_exclude_first_row')
        PCDDstudyparameters01 = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core_01, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD,
        PCDDstudyparameters = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD
        kclsignificance01,kclsignificance = PCDDstudyparameters01[2]/PCDDstudyparameters01[3],PCDDstudyparameters[2]/PCDDstudyparameters[3]
        if abs(kclsignificance01) > 3 or abs(kclsignificance) > 3: print('Kcl value flags probable charge loss')

    ##############################################################################
    #ADU TO e- CALIBRATION AND DARK CURRENT ESTIMATES#############################
    ##############################################################################

    if reportCalibrationDarkcurrent and nskips!=1:
        parametersDCfit, reducedchisquared, offset = m_calibrationdc.calibrationDC(skipper_avg0, stdmanyskip[-1], reverse, debug=False)
        calibrationconstant = parametersDCfit[0][5]; calibratedsigma = stdmanyskip[-1]/calibrationconstant
        skipper_avg_cal = reversign*(skipper_avg0 - offset)/calibrationconstant
        darkcurrentestimateAC = m_calibrationdc.anticlusteringDarkCurrent(m_functions.selectImageRegion(skipper_avg_cal,analysisregion), calibratedsigma, debug=False)

##############################################################################
#MULTIPLE IMAGES TWEAKING ANALYSIS ###########################################
##############################################################################

if multipleimages:
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for tweaking analysis using multiple images. This code assumes multiple images have the same structure (size and number of skips). If this is not the case code may fail.\nDo you want to continue? Please answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with multiple images tweaking analysis')
    lowerindex = config['tweaking_analysis'][-1]['multiple_images'][-1]['lower_index']
    upperindex = config['tweaking_analysis'][-1]['multiple_images'][-1]['upper_index']
    nameprefix = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    nimages = upperindex - lowerindex + 1
    ##############################################################################
    # Use `astropy.io.fits.info()` to display the structure of the file
    image_file = get_pkg_data_filename(nameprefix+str(lowerindex)+'.fits')
    ##############################################################################
    # Use `astropy.io.fits.info()` to display the structure of the file
    fits.info(image_file)
    ##############################################################################
    # read n. of pixels in x and y axes and number of skips in the first image
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
    # multiple images reconstruction
    ##############################################################################
    image_data_stack,skipper_image_start_stack,skipper_image_end_stack,skipper_averages_stack,skipper_diff_stack,skipper_diff_01_stack,skipper_avg0_stack,skipper_std_stack = m_reconstruction.reconstructMultipleSkipperImages(nameprefix,lowerindex,upperindex)
    
    ##############################################################################
    #ESTIMATE NOISES AT SKIPS: 1, 10, 100 . . . 1000 ##############################
    ##############################################################################
    if reportPCD or reportCalibrationDarkcurrent:
        if nskips < 10: naverages = 0
        elif nskips < 100: naverages = 1; numberskips=[10]
        else:
            numberskips=[10]; index=1
            while index <= nskips/100:
                numberskips.append(index*100)
                naverages = index+1; index+=1
    startskipfitpar_stack = np.zeros((5, nimages), dtype=np.float64) # 5 = output_sigma_finder
    ampmanyskip_stack = np.zeros((naverages, nimages), dtype=np.float64)
    mumanyskip_stack = np.zeros((naverages, nimages), dtype=np.float64)
    stdmanyskip_stack = np.zeros((naverages, nimages), dtype=np.float64)
    muncmanyskip_stack = np.zeros((naverages, nimages), dtype=np.float64)
    stduncmanyskip_stack = np.zeros((naverages, nimages), dtype=np.float64)
    for iimage in range(upperindex-lowerindex+1):
        startskipfitpar_stack[:,iimage] = m_functions.sigmaFinder(skipper_image_start_stack[:,:,iimage], fwhm_est=False, debug=False)
        #ampss, muss, stdss, stduncss
        for k in range(naverages):
                ampmanyskip_stack[k,iimage], mumanyskip_stack[k,iimage], stdmanyskip_stack[k,iimage], muncmanyskip_stack[k,iimage], stduncmanyskip_stack[k,iimage] = m_functions.sigmaFinder(skipper_averages_stack[:,:,k,iimage], fwhm_est=True, debug=False)
        

    ##############################################################################
    #FIRST LAST SKIP CHARGE LOSS CHECK: KCL AND SKEW##############################
    ##############################################################################
    if reportChargeLoss and nskips!=1:
        PCDDstudyparameters01_stack = np.zeros((7,nimages), dtype=np.float64) #7 = pcdd check output
        PCDDstudyparameters_stack = np.zeros((7,nimages), dtype=np.float64) #7 = pcdd check output
        kclsignificance01_stack = np.zeros((nimages), dtype=np.float64)
        kclsignificance_stack = np.zeros((nimages), dtype=np.float64)
        for iimage in range(upperindex-lowerindex+1):
            diff_image_core_01,diff_image_core = m_functions.selectImageRegion(skipper_diff_01_stack[:,:,iimage],'exposed_pixels_exclude_first_row'),m_functions.selectImageRegion(skipper_diff_stack[:,:,iimage],'exposed_pixels_exclude_first_row')
            PCDDstudyparameters01_stack[:,iimage] = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core_01, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD,
            PCDDstudyparameters_stack[:,iimage] = m_chargeloss.firstLastSkipPCDDCheck(diff_image_core, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD
            kclsignificance01_stack[iimage],kclsignificance_stack[iimage] = PCDDstudyparameters01_stack[2,iimage]/PCDDstudyparameters01_stack[3,iimage],PCDDstudyparameters_stack[2,iimage]/PCDDstudyparameters_stack[3,iimage]

    ##############################################################################
    #ADU TO e- CALIBRATION AND DARK CURRENT ESTIMATES#############################
    ##############################################################################

    if reportCalibrationDarkcurrent and nskips!=1:
        parametersDCfit_stack = np.zeros((2,6, nimages), dtype=np.float64) # 6,2 = output_dc_calibration
        reducedchisquared_stack = np.zeros((nimages), dtype=np.float64)
        offset_stack = np.zeros((nimages), dtype=np.float64)
        calibrationconstant_stack = np.zeros((nimages), dtype=np.float64)
        calibratedsigma_stack = np.zeros((nimages), dtype=np.float64)
        skipper_avg_cal_stack = np.zeros((nrows, ncolumns, nimages), dtype=np.float64)
        darkcurrentestimateAC_stack = np.zeros((nimages), dtype=np.float64)
        for iimage in range(nimages):
            parametersDCfit, reducedchisquared_stack[iimage], offset_stack[iimage] = m_calibrationdc.calibrationDC(skipper_avg0_stack[:,:,iimage], stdmanyskip_stack[-1,iimage], reverse, debug=False)
            parametersDCfit_stack[:,:,iimage] = np.array(parametersDCfit)
            calibrationconstant_stack[iimage] = parametersDCfit_stack[0,5,iimage]; calibratedsigma_stack[iimage] = stdmanyskip_stack[-1,iimage]/calibrationconstant_stack[iimage]
            skipper_avg_cal_stack[:,:,iimage] = reversign*(skipper_avg0_stack[:,:,iimage] - offset_stack[iimage])/calibrationconstant_stack[iimage]
            darkcurrentestimateAC_stack[iimage] = m_calibrationdc.anticlusteringDarkCurrent(m_functions.selectImageRegion(skipper_avg_cal_stack[:,:,iimage],analysisregion), calibratedsigma_stack[iimage], debug=False)
    
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

if not (reportHeader or reportImage or reportPCD or reportChargeLoss or reportCalibrationDarkcurrent or reportFFTrow or reportFFTskips):
    print('No information to be reported. Exiting'); sys.exit()

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report on 1-Amp Parameter Tweaking'))
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

if not multipleimages:
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

    ampss, muss, stdss, muncss, stduncss = startskipfitpar #ss: start skip
    if reportImage == True: reportImage='cluster'
    if reportImage=='full':
        
        with doc.create(Section('Images')):
        
            if nskips!=1:
                plotrange = [0,np.size(skipper_image_start,0),0,np.size(skipper_image_start,1)]
                width = 8
                fig=plt.figure(figsize=(1.2*width,width))
                from matplotlib import gridspec
                
                gs = gridspec.GridSpec(6,1)
                #gs.update(wspace=0.025, hspace=0.05)
                
                ax1=fig.add_subplot(gs[0,0])
                plt.imshow(skipper_image_start[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Start skip")
                plt.ylabel("row")
                cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
                cb1 = plt.colorbar(cax=cax1)
                
                ax2=fig.add_subplot(gs[1,0])
                plt.imshow(skipper_image_end[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("End skip")
                plt.ylabel("row")
                cax2=make_colorbar_with_padding(ax2) # add a colorbar within its own axis the same size as the image plot
                cb2 = plt.colorbar(cax=cax2)
                
                ax3=fig.add_subplot(gs[2,0])
                plt.imshow(skipper_avg0[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Average")
                plt.ylabel("row")
                cax3=make_colorbar_with_padding(ax3) # add a colorbar within its own axis the same size as the image plot
                cb3 = plt.colorbar(cax=cax3)
                
                ax4=fig.add_subplot(gs[3,0])
                plt.imshow(skipper_std[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Standard deviation")
                plt.ylabel("row")
                cax4=make_colorbar_with_padding(ax4) # add a colorbar within its own axis the same size as the image plot
                cb4 = plt.colorbar(cax=cax4)
                
                ax5=fig.add_subplot(gs[4,0])
                plt.imshow(skipper_diff_01[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("First-second skip difference")
                plt.ylabel("row")
                cax5=make_colorbar_with_padding(ax5) # add a colorbar within its own axis the same size as the image plot
                cb5 = plt.colorbar(cax=cax5)
                
                ax6=fig.add_subplot(gs[5,0])
                plt.imshow(skipper_diff[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))#,extent=(,570,10,0))
                plt.title("Second-end skip difference")
                plt.ylabel("row")
                plt.xlabel("column")
                cax6=make_colorbar_with_padding(ax6) # add a colorbar within its own axis the same size as the image plot
                cb6 = plt.colorbar(cax=cax6)
                
                fig.tight_layout()
                #fig.subplots_adjust(bottom=None,top=None,right=None,hspace = None)

            else:
                #halfrangey = np.size(skipper_image_start,0)/2; halfrangex = np.size(skipper_image_start,1)/2
                plotrange = [0,np.size(skipper_image_start,0),0,np.size(skipper_image_start,1)]
                fig=plt.figure(figsize=(8,8))
                    
                ax1=fig.add_subplot(111)
                plt.imshow(skipper_image_start[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Start skip")
                plt.ylabel("row")
                cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
                cb1 = plt.colorbar(cax=cax1)

            fig.tight_layout(pad=.001)
            
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Exposed pixels region for various images.')
        plt.clf()
        doc.append(NewPage())

    if reportImage=='cluster':

        centeredsstoplot = reversign*(skipper_image_start - muss)
        clustercandidates = m_reconstruction.findChargedPixelNoBorder(centeredsstoplot,stdss)
        isChargedCrown = True; coor = np.size(centeredsstoplot,0)//2, np.size(centeredsstoplot,1)//2
        for coor in clustercandidates:
            isChargedCrown = m_reconstruction.chargedCrown(coor,centeredsstoplot,stdss)
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
                plt.imshow(skipper_image_start[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Start skip")
                plt.ylabel("row")
                cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
                cb1 = plt.colorbar(cax=cax1)
                
                fig.subplots_adjust(right=0.9)
                
                ax2=fig.add_subplot(612)
                plt.imshow(skipper_image_end[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("End skip")
                plt.ylabel("row")
                cax2=make_colorbar_with_padding(ax2) # add a colorbar within its own axis the same size as the image plot
                cb2 = plt.colorbar(cax=cax2)
            
                ax3=fig.add_subplot(613)
                plt.imshow(skipper_avg0[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Average")
                plt.ylabel("row")
                cax3=make_colorbar_with_padding(ax3) # add a colorbar within its own axis the same size as the image plot
                cb3 = plt.colorbar(cax=cax3)
                
                ax4=fig.add_subplot(614)
                plt.imshow(skipper_std[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Standard deviation")
                plt.ylabel("row")
                cax4=make_colorbar_with_padding(ax4) # add a colorbar within its own axis the same size as the image plot
                cb4 = plt.colorbar(cax=cax4)
                
                ax5=fig.add_subplot(615)
                plt.imshow(skipper_diff_01[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("First-second skip difference")
                plt.ylabel("row")
                cax5=make_colorbar_with_padding(ax5) # add a colorbar within its own axis the same size as the image plot
                cb5 = plt.colorbar(cax=cax5)
                
                ax6=fig.add_subplot(616)
                plt.imshow(skipper_diff[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))#,extent=(,570,10,0))
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
                plt.imshow(skipper_image_start[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
                plt.title("Start skip")
                plt.ylabel("row")
                cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
                cb1 = plt.colorbar(cax=cax1)

            fig.tight_layout(pad=.001)
            
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Exposed pixels region for various images.')
        plt.clf()
        doc.append(NewPage())
                
    #############################################
    #Pixel charge distribution and noise section#
    #############################################
    if reportPCD:
        with doc.create(Section('Pixel Charge Distributions and Noise')):
            import m_functions
            
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
            skipper_image_start_region = m_functions.selectImageRegion(skipper_image_start,analysisregion)
            if applymask: skipper_image_start_ravel = skipper_image_start_region.compressed()
            else: skipper_image_start_ravel = skipper_image_start_region.ravel()
            #instead of removing 0-entries from histogram use numpy mask to avoid discrepancies between gaussian and plotted PCD skipper_image0ravel
            #skipper_image = [s for s in skipper_image_start_ravel if s != 0]
            if reverse: skipper_image_unsaturated = np.ma.masked_equal(skipper_image_start_ravel, 0.0, copy=False)
            else: skipper_image_unsaturated = skipper_image_start_ravel
            skipper_imagehist, binedges = np.histogram(skipper_image_unsaturated, bins = 1200, density=False)
            ampss = skipper_imagehist[np.argmax(skipper_imagehist)]
            axs[0].hist(skipper_image_start_ravel, 1200, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='start skip pixel value distribution')
            bincenters = np.arange(muss - 3*stdss, muss + 3*stdss + 6*stdss/100, 6*stdss/100) #last term in upper bound to get ~sym drawing
            axs[0].plot(bincenters, gauss(bincenters,ampss,muss,stdss), label='gaussian fit curve', linewidth=1, color='red')
            if reverse: axs[0].legend(prop={'size': 14})
            else: axs[0].legend(prop={'size': 14})
            axs[0].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            try: axs[0].set_title('Start skip pixel value distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdss,4)) + ' ADU; estimated noise: ' + str(round(stdss/calibrationconstant,4)) + ' $e^{-}$')
            except: axs[0].set_title('Start skip pixel value distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdss,4)) + ' ADU')

            
            if nskips!=1:
                try: calibrationconstant; guessCC = False
                except: calibrationconstant = calibrationguess; guessCC = True; print('WARNING: calibration constant not defined for ADU/e- noise conversion. Using guess value 10 ADU/e-')
                averageimageoffset,averageimagestd = m_functions.sigmaFinder(skipper_avg0, fwhm_est=True, debug=False)[1:3]
                skipper_avg0_region = m_functions.selectImageRegion(skipper_avg0,analysisregion)
                if applymask: avg_image_0ravel = skipper_avg0_region.compressed()
                else: avg_image_0ravel = skipper_avg0_region.ravel()
                if reverse: avg_image_unsaturated = np.ma.masked_equal(avg_image_0ravel, 0.0, copy=False)
                    avg_image_unsaturated = [s for s in avg_image_unsaturated if averageimageoffset - 5*calibrationconstant < s < averageimageoffset + calibrationconstant]
                    rangeadhoc =  (averageimageoffset - 5*calibrationconstant, averageimageoffset + calibrationconstant)
                else:
                    avg_image_unsaturated = [s for s in avg_image_0ravel if averageimageoffset - calibrationconstant < s < 5*averageimageoffset + calibrationconstant]
                    rangeadhoc =  (averageimageoffset - calibrationconstant, averageimageoffset + 5*calibrationconstant)
                if len(avg_image_unsaturated) < 50:
                    avg_image_unsaturated = [s for s in np.ma.masked_equal(avg_image_0ravel, 0.0, copy=False) if - 20*calibrationconstant < s - averageimageoffset < 20*calibrationconstant]
                    rangeadhoc =  (averageimageoffset - 20*calibrationconstant, averageimageoffset + 20*calibrationconstant)
                avg_image_hist, binedges = np.histogram([s for s in avg_image_0ravel if s != 0], range=rangeadhoc, bins = 200, density=False)
                ampls = avg_image_hist[np.argmax(avg_image_hist)]
                bincenters = np.arange(averageimageoffset - 3*stdmanyskip[-1], averageimageoffset + 3*stdmanyskip[-1] + 6*stdmanyskip[-1]/200, 6*stdmanyskip[-1]/200)
                if abs(rangeadhoc[1]-rangeadhoc[0]) < max(bincenters) - min(bincenters):
                    rangeadhoc = (min(bincenters),max(bincenters))
                    avg_image_hist, binedges = np.histogram([s for s in avg_image_0ravel if s != 0], range=rangeadhoc, bins = 200, density=False)
                    ampls = avg_image_hist[np.argmax(avg_image_hist)]
                #axs[1].plot(bincenters, gauss(bincenters,ampls,averageimageoffset,stdmanyskip[-1]), label='gaussian fit curve', linewidth=1, color='red')
                axs[1].hist(avg_image_0ravel, 200, rangeadhoc, density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel value distribution')
                axs[1].plot(bincenters, gauss(bincenters,ampls,averageimageoffset,averageimagestd), label='gaussian fit curve', linewidth=1, color='red')
                if reverse: axs[1].legend(loc='upper left',prop={'size': 14})
                else: axs[1].legend(loc='upper right',prop={'size': 14})
                axs[1].tick_params(axis='both', which='both', length=10, direction='in')
                axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[1].get_yticklabels(), visible=True)
                axs[1].set_title('Average image pixel value distribution: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip[-1],4)) + ' ADU; estimated noise: ' + str(round(stdmanyskip[-1]/calibrationconstant,4)) + ' $e^{-}$')
                if guessCC: axs[1].set_title('Average image pixel value distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip[-1],4)) + ' ADU')
            
            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Start skip and avg image pixel charge distributions computed on '+analysisregion+' image region.')
            #plt.show()
            plt.clf()
            doc.append(NewPage())
            
            def r(ns):
                return stdss/np.sqrt(ns)
            fig, axs = plt.subplots(1, 1, figsize=(8,6), sharey=True, tight_layout=True)
            #numberSkips = [10,100,200,300,400,500,600,700,800,900,1000]
            ns = np.arange(1,nskips,1)
            #resolution = plt.plot(1,stdss,'ro',numberSkips[0:len(stdmanyskip)],stdmanyskip,'ro',ns,r(ns),'k-')
            if nskips!=1: resolution = plt.errorbar(numberskips[0:len(stdmanyskip)],stdmanyskip,stduncmanyskip,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measured resolution in ADU')
            else: resolution = plt.errorbar([],[])
            resolution += plt.errorbar(1,stdss,stduncss,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4)
            resolution = plt.plot(ns,r(ns),'k--',label='expected $1/\sqrt(N_{skip})$ trend based on first skip sigma')
            plt.legend(loc='upper right',prop={'size': 14})
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
                plot.add_caption('Resolution trend computed on '+analysisregion+' image region, as function of average image skip number.')
            plt.show()
            plt.clf()
            doc.append(NewPage())
            
    #############################################
    #Charge loss indicators and skewness section#
    #############################################
    if reportChargeLoss and nskips!=1:
        with doc.create(Section('Charge-loss')):
            skewnessPCDD01, skewnessPCDDuncertainty01, kclPCDD01, kclPCDDuncertainty01, ampPCDD01, muPCDD01, stdPCDD01 = PCDDstudyparameters01
            skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD = PCDDstudyparameters
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
            skipperdiffcoreravelled01 = diff_image_core_01.ravel()
            skipperdiffcoreravelled = diff_image_core.ravel()
            
            skipperdiffcoreravelledinrange01 = [s for s in skipperdiffcoreravelled01 if s > muPCDD01 - 3*stdPCDD01 and s < muPCDD01 + 3*stdPCDD01 and s != 0]
            numbins = int(max(skipperdiffcoreravelledinrange01) - min(skipperdiffcoreravelledinrange01))
            skipperdiffcoreravelledinrangehist01, binedges = np.histogram(skipperdiffcoreravelledinrange01, numbins, density=False)
            bincenters=(binedges[:-1] + binedges[1:])/2
            pguess = [ampPCDD01,muPCDD01,stdPCDD01]
            try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist01, p0=pguess); PCDDhistfit01 = gauss(bincenters,*pfit)
            except: pfit = pguess; PCDDhistfit01 = gauss(bincenters,*pfit)
            axs[0].plot(bincenters, PCDDhistfit01, label='gaussian fit curve', linewidth=1, color='red')
            axs[0].hist(skipperdiffcoreravelledinrange01, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
            #axs[0].plot(bincenters,skipperdiffcoreravelledinrangehist01, label='pixel charge difference distribution', color='teal')
            axs[0].legend(loc='upper right',prop={'size': 14})
            axs[0].set_yscale('linear')
            axs[0].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            axs[0].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
            skipperdiffcoreravelledinrange = [s for s in skipperdiffcoreravelled if s > muPCDD - 3*stdPCDD and s < muPCDD + 3*stdPCDD and s != 0]
            numbins = int(max(skipperdiffcoreravelledinrange) - min(skipperdiffcoreravelledinrange))
            skipperdiffcoreravelledinrangehist, binedges = np.histogram(skipperdiffcoreravelledinrange, numbins, density=False)
            bincenters=(binedges[:-1] + binedges[1:])/2
            pguess = [ampPCDD,muPCDD,stdPCDD]
            try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist, p0=pguess); PCDDhistfit = gauss(bincenters,*pfit)
            except: pfit = pguess; PCDDhistfit = gauss(bincenters,*pfit)
            axs[1].plot(bincenters, PCDDhistfit, label='gaussian fit curve', linewidth=1, color='red')
            axs[1].hist(skipperdiffcoreravelledinrange, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
                #axs[1].plot(bincenters,skipperdiffcoreravelledinrangehist, label='pixel charge difference distribution', color='teal')
            axs[1].legend(loc='upper right',prop={'size': 14})
            axs[1].set_yscale('linear')
            axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[1].get_yticklabels(), visible=True)
            axs[1].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Full image pixel charge difference distributions (PCDD) between first and second skip (top) and second and end skip (bottom). Entries at 0 (saturation digitizer range) might be masked for analysis purposes.')
            doc.append(NoEscape(r'NOTE: A good gaussian fit of the PCDDs is essential for $S_{k_{cl}}$ to be an effective charge loss classifier'))
            plt.clf()
            doc.append(NewPage())
            
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
            centeredskipperdiffcore01 = [s for s in skipperdiffcoreravelled01-muPCDD01 if s != -muPCDD01]
            axs[0].hist(centeredskipperdiffcore01, 600, range = (-20*stdPCDD01,10*stdPCDD01), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
            axs[0].legend(loc='upper right',prop={'size': 14})
            axs[0].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            axs[0].set_title('$k_{cl}~=~$' + str(round(kclPCDD01,4)) + '$\pm$'+ str(round(kclPCDDuncertainty01,4)) + ', $S_{k_{cl}}~=~$' + str(round(kclPCDD01/kclPCDDuncertainty01,4)) + ', skewness = ' + str(round(skewnessPCDD01,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty01,4)))

            centeredskipperdiffcore = [s for s in skipperdiffcoreravelled-muPCDD if s != -muPCDD]
            axs[1].hist(centeredskipperdiffcore, 600, range = (-20*stdPCDD,10*stdPCDD), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
            axs[1].legend(loc='upper right',prop={'size': 14})
            axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            axs[1].set_title('$k_{cl}~=~$' + str(round(kclPCDD,4)) + '$\pm$'+ str(round(kclPCDDuncertainty,4)) + ', $S_{k_{cl}}~=~$' + str(round(kclPCDD/kclPCDDuncertainty,4)) + ', skewness = ' + str(round(skewnessPCDD,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty,4)))

            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts') #removed per adu, corect elsewher
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Pedestal-subtracted full-image PCDDs: first and second skip (top) and second and end skip (bottom).')
                from scipy.stats import norm
                doc.append('First-second skip lower tail entries: '+str(len([s for s in centeredskipperdiffcore01 if s < -kclthreshold*stdPCDD01]))+'. First-second skip upper tail entries: '+str(len([s for s in centeredskipperdiffcore01 if s > kclthreshold*stdPCDD01]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffcore01)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.\n Second-last skip lower tail entries: '+str(len([s for s in centeredskipperdiffcore if s < -kclthreshold*stdPCDD]))+'. Second-last skip upper tail entries: '+str(len([s for s in centeredskipperdiffcore if s > kclthreshold*stdPCDD]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffcore)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.')

            plt.clf()
            doc.append(NewPage())
            
    #############################################
    ##Calibrated image and Dark Current section##
    #############################################
    if reportCalibrationDarkcurrent and nskips!=1:
        if analysisregion == 'arbitrary': skipper_avg_cal = m_functions.selectImageRegion(skipper_avg_cal,analysisregion)
        skipperavgcalibrated = skipper_avg_cal.ravel()
        try:#if calibration went wrong skipperavgcalibratedravel could be empty because limits are out of range
            if calibrationconstant == calibrationguess: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -10 and  s < 10]
            elif calibrationconstant <= 1:
                skipperavgcalibratedravel =  [s for s in skipperavgcalibrated.ravel() if s > -200 and  s < 200]
                if 0.01 < calibrationconstant <=0.1: skipperavgcalibratedravel =  [s for s in skipperavgcalibrated.ravel() if s > -2000 and  s < 2000]
                elif calibrationconstant <=0.01: skipperavgcalibratedravel =  [s for s in skipperavgcalibrated.ravel() if s > -20000 and  s < 20000]
                nbins=int((max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))/10)
            else:
                skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -2 and  s < 4]
                nbins=20*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
        except: #if so we keep skipperavgcalibratedravel without range
            skipperavgcalibratedravel = skipperavgcalibrated
            nbins=20*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
        if nbins == 0: nbins=100
        skipperavgcalibratedravelhist, binedges = np.histogram(skipperavgcalibratedravel, nbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        npeaksp = 3
        dcpar = parametersDCfit[0][0], npeaksp, parametersDCfit[0][2]/(20/0.5), parametersDCfit[0][3]/calibrationconstant
        if calibrationconstant <= 1: dcpar = parametersDCfit[0][0], npeaksp, parametersDCfit[0][2]/(0.1/0.5), parametersDCfit[0][3]/calibrationconstant
        #dcparunc has one more component (the gain) than dcpar (dcpar is an argument for the calibrated gaussian)
        try: dcparunc = parametersDCfit[1][0], parametersDCfit[1][1], parametersDCfit[1][2]/(20/0.5), parametersDCfit[1][3]/calibrationconstant, parametersDCfit[1][5]
        except: dcparunc = 0,0,0,0,0
        skipperavgcalibratedravelhistfit = convolutionGaussianPoisson(bincenters,*dcpar)
        #plt.plot(bincenters,skipperavgcalibratedravelhist,label='avg img calibrated pixel charge distribution', color='teal')
        plt.hist(skipperavgcalibratedravel, len(bincenters), density = False, histtype='step', linewidth=2, log = False, color = 'teal', label='avg image calibrated pixel charge distribution')
        plt.plot(bincenters, skipperavgcalibratedravelhistfit, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(reducedchisquared)), color='red')
        #plt.hist(skipperavgcalibrated.ravel(), 200, (-1,5), density = False, histtype='step', linewidth=2, log = True, color='teal')
        plt.legend(loc='upper right',prop={'size': 20})
        plt.xlabel('pixel value [e$^-$]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.ylim(0.8)
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        #plt.setp(ax.get_yticklabels(), visible=True)
        try: plt.title('$I_{darkCF}~=~$' + str(round(dcpar[0],6)) + '$\pm$' + str(round_sig_2(dcparunc[0])) + ' $e^-$pix$^{-1}$, $I_{darkAC}~=~$' + str(round(darkcurrentestimateAC,6)) + ' $e^-$pix$^{-1}$')
        except: plt.title('$I_{darkCF}~=~$' + str(round(dcpar[0],6)) + '$\pm$' + str(dcparunc[0]) + ' $e^-$pix$^{-1}$, $I_{darkAC}~=~$' + str(round(darkcurrentestimateAC,6)) + ' $e^-$pix$^{-1}$')
        
        with doc.create(Section('Dark Current')):
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                if analysisregion == 'arbitrary': plot.add_caption('Calibrated pixel charge distribution. Dark current values computed with convolution fit (on arbitrary image region) and anticlustering (on '+analysisregion+' image region)')
                else: plot.add_caption('Calibrated pixel charge distribution. Dark current values computed with convolution fit (on full image region) and anticlustering (on '+analysisregion+' image region)')
            calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+'±'+str(round_sig_2(dcparunc[4]))+' ADU per electron.'
            doc.append(calibrationline)
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
                samplet = hdr['MREAD']*0.001/(nrows*nallcolumns)
                ncolumns = int(nallcolumns/nskips) # n of columns in the image
                m_functions.pixelFFT(image_data, nrows-1, ncolumns-1, nskips, samplet)
                with doc.create(Figure(position='htb!')) as plot:
                    plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                    plot.add_caption('Full image Fast Fourier Transform (first to last skip).')
                plt.clf()
        
            if reportFFTrow:
                samplet = hdr['MREAD']*0.001/(nrows*ncolumns)
                if nskips!=1: m_functions.rowFFT(skipper_avg0, nrows-1, ncolumns-1, samplet)
                else: m_functions.rowFFT(image_data, nrows-1, ncolumns-1, samplet)
                with doc.create(Figure(position='htb!')) as plot:
                    plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                    plot.add_caption('Average image Fast Fourier Transform (all row pixels).')
                plt.clf()
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
# MULTIPLE IMAGES LATEK REPORT ###############################################
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
    
if multipleimages:
    
    scanreport = config['tweaking_analysis'][-1]['multiple_images'][-1]['scan_report']
    if scanreport:
        scanparameters = config['tweaking_analysis'][-1]['multiple_images'][-1]['scan_parameters']
        scanintervals = config['tweaking_analysis'][-1]['multiple_images'][-1]['scan_intervals']
        with doc.create(Section('Information')):
            doc.append(NoEscape(r'This a report on multiple images for a scan on parameters: '+scanparameters+'.'))
            doc.append(NoEscape(r'The acquisition intervals for these parameters are: '+scanintervals+', respectively.'))
            doc.append(NoEscape(r'Interval convention: [lower-value, upper-value, scan-step].'))
            doc.append(NoEscape(r'Each section reports the same plot instance for all the scan images, as an efficient way to compare the Charge-Coupled device performance in different operating conditions.'))
            doc.append(NoEscape(r'\newpage'))
            
            
            
            
    #############################################
    #Pixel charge distribution and noise section#
    #############################################
    if reportPCD:
        with doc.create(Section('Pixel Charge Distributions and Noise')):
            import m_functions
            
            for iimage in range(nimages):
                fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
                
                skipper_image_start_region = m_functions.selectImageRegion(skipper_image_start_stack[:,:,iimage],analysisregion)
                skipper_image_start_ravel = skipper_image_start_region.ravel()
                #instead of removing 0-entries from histogram use numpy mask to avoid discrepancies between gaussian and plotted PCD skipper_image0ravel
                #skipper_image = [s for s in skipper_image_start_ravel if s != 0]
                skipper_image_unsaturated = np.ma.masked_equal(skipper_image_start_ravel, 0.0, copy=False)
                skipper_imagehist, binedges = np.histogram(skipper_image_unsaturated, bins = 800, density=False)
                ampss = skipper_imagehist[np.argmax(skipper_imagehist)]
                axs[0].hist(skipper_image_start_ravel, 800, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='start skip pixel charge distribution')
                bincenters = np.arange(startskipfitpar_stack[1,iimage] - 3*startskipfitpar_stack[2,iimage], startskipfitpar_stack[1,iimage] + 3*startskipfitpar_stack[2,iimage] + 6*startskipfitpar_stack[2,iimage]/100, 6*startskipfitpar_stack[2,iimage]/100) #last term in upper bound to get ~sym drawing
                axs[0].plot(bincenters, gauss(bincenters,ampss,startskipfitpar_stack[1,iimage],startskipfitpar_stack[2,iimage]), label='gaussian fit curve', linewidth=1, color='red')
                if reverse: axs[0].legend(loc='upper left',prop={'size': 14})
                else: axs[0].legend(loc='upper right',prop={'size': 14})
                axs[0].tick_params(axis='both', which='both', length=10, direction='in')
                axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[0].get_yticklabels(), visible=True)
                try: axs[0].set_title('Start skip pixel charge distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(startskipfitpar_stack[2,iimage],4)) + ' ADU; estimated noise: ' + str(round(startskipfitpar_stack[2,iimage]/calibrationguess,4)) + ' $e^{-}$')
                except: axs[0].set_title('Start skip pixel charge distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(startskipfitpar_stack[2,iimage],4)) + ' ADU')
    
                
                if nskips!=1:
                    try: calibrationconstant = calibrationconstant_stack[iimage]; guessCC = False
                    except: calibrationconstant = calibrationguess; guessCC = True; print('WARNING: calibration constant not defined for ADU/e- noise conversion. Using guess value 10 ADU/e-')
                    averageimageoffset = m_functions.sigmaFinder(skipper_avg0_stack[:,:,iimage], fwhm_est=True, debug=False)[1]
                    skipper_avg0_region = m_functions.selectImageRegion(skipper_avg0_stack[:,:,iimage],analysisregion)
                    avg_image_0ravel = skipper_avg0_region.ravel()
                    avg_image_unsaturated = np.ma.masked_equal(avg_image_0ravel, 0.0, copy=False)
                    if reverse:
                        avg_image_unsaturated = [s for s in avg_image_unsaturated if averageimageoffset - 5*calibrationconstant < s < averageimageoffset + calibrationconstant]
                        rangeadhoc =  (averageimageoffset - 5*calibrationconstant, averageimageoffset + calibrationconstant)
                    else:
                        avg_image_unsaturated = [s for s in avg_image_unsaturated if averageimageoffset - calibrationconstant < s < 5*averageimageoffset + calibrationconstant]
                        rangeadhoc =  (averageimageoffset - calibrationconstant, averageimageoffset + 5*calibrationconstant)
                    if len(avg_image_unsaturated) < 50:
                        avg_image_unsaturated = [s for s in np.ma.masked_equal(avg_image_0ravel, 0.0, copy=False) if averageimageoffset - 20*calibrationconstant < s < averageimageoffset + 20*calibrationconstant]
                        rangeadhoc =  (averageimageoffset - 20*calibrationconstant, averageimageoffset + 20*calibrationconstant)
                    avg_image_hist, binedges = np.histogram([s for s in avg_image_0ravel if s != 0], range=rangeadhoc, bins = 200, density=False)
                    ampls = avg_image_hist[np.argmax(avg_image_hist)]
                    bincenters = np.arange(averageimageoffset - 3*stdmanyskip_stack[-1,iimage], averageimageoffset + 3*stdmanyskip_stack[-1,iimage] + 6*stdmanyskip_stack[-1,iimage]/200, 6*stdmanyskip_stack[-1,iimage]/200)
                    axs[1].plot(bincenters, gauss(bincenters,ampls,averageimageoffset,stdmanyskip_stack[-1,iimage]), label='gaussian fit curve', linewidth=1, color='red')
                    axs[1].hist(avg_image_0ravel, 200, rangeadhoc, density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')
                    if reverse: axs[1].legend(loc='upper left',prop={'size': 14})
                    else: axs[1].legend(loc='upper right',prop={'size': 14})
                    axs[1].tick_params(axis='both', which='both', length=10, direction='in')
                    axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
                    plt.setp(axs[1].get_yticklabels(), visible=True)
                    axs[1].set_title('Average image pixel charge distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip_stack[-1,iimage],4)) + ' ADU; estimated noise: ' + str(round(stdmanyskip_stack[-1,iimage]/calibrationconstant,4)) + ' $e^{-}$')
                    if guessCC: axs[1].set_title('Average image pixel charge distribution in '+analysisregion+' image region: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip_stack[-1,iimage],4)) + ' ADU')
                
                    plt.subplots_adjust(hspace=0.5)
                    for ax in axs.flat:
                        ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
                    with doc.create(Figure(position='htb!')) as plot:
                        plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                        plot.add_caption('Start skip and avg image pixel charge distributions computed on '+analysisregion+' image region.')
                    plt.clf()
                    doc.append(NewPage())
                
            
            fig, axs = plt.subplots(nimages, 1, figsize=(8,6), sharey=True, tight_layout=True)
            
            for iimage in range(nimages):
            
                def r(ns):
                    return startskipfitpar_stack[2,iimage]/np.sqrt(ns)
                #numberSkips = [10,100,200,300,400,500,600,700,800,900,1000]
                ns = np.arange(1,nskips,1)
                #resolution = plt.plot(1,stdss,'ro',numberSkips[0:len(stdmanyskip)],stdmanyskip,'ro',ns,r(ns),'k-')
                if nskips!=1: resolution = axs[iimage].errorbar(numberskips[0:len(stdmanyskip_stack[:,iimage])],stdmanyskip_stack[:,iimage],stduncmanyskip_stack[:,iimage],xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measured resolution in ADU')
                else: resolution = axs[iimage].errorbar([],[])
                resolution += axs[iimage].errorbar(1,startskipfitpar_stack[2,iimage],startskipfitpar_stack[4,iimage],xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4)
                resolution = axs[iimage].plot(ns,r(ns),'k--',label='expected $1/\sqrt(N_{skip})$ trend based on start skip sigma')
                axs[iimage].legend(loc='lower left',prop={'size': 8})
                axs[iimage].set_xscale('log')
                axs[iimage].set_yscale('log')
                axs[iimage].axis([0.9, nskips*1.1, 0.1, 100])
                axs[iimage].loglog()
                axs[iimage].tick_params(axis='both', which='both', length=10, direction='in')
                axs[iimage].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[iimage].get_yticklabels(), visible=True)
                axs[iimage].set_title('Resolution trend')
                
                for ax in axs.flat:
                    ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
            
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Resolution trend computed on '+analysisregion+' image region, as function of average image skip number.')
            plt.clf()
        doc.append(NewPage())
    
    #############################################
    #Charge loss indicators and skewness section#
    #############################################
    if reportChargeLoss and nskips!=1:
        with doc.create(Section('Charge-loss')):
        
            fig, axs = plt.subplots(nimages, 2, figsize=(11,11), sharey=True, tight_layout=True)
            
            for iimage in range(nimages):
            
                skewnessPCDD01, skewnessPCDDuncertainty01, kclPCDD01, kclPCDDuncertainty01, ampPCDD01, muPCDD01, stdPCDD01 = PCDDstudyparameters01_stack[:,iimage]
                skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD = PCDDstudyparameters_stack[:,iimage]
            
                skipperdiffcoreravelled01 = m_functions.selectImageRegion(skipper_diff_01_stack[:,:,iimage],'no_borders').ravel()
                skipperdiffcoreravelled = m_functions.selectImageRegion(skipper_diff_stack[:,:,iimage],'no_borders').ravel()
            
                skipperdiffcoreravelledinrange01 = [s for s in skipperdiffcoreravelled01 if s > muPCDD01 - 3*stdPCDD01 and s < muPCDD01 + 3*stdPCDD01 and s != 0]
                numbins = int(max(skipperdiffcoreravelledinrange01) - min(skipperdiffcoreravelledinrange01))
                skipperdiffcoreravelledinrangehist01, binedges = np.histogram(skipperdiffcoreravelledinrange01, numbins, density=False)
                bincenters=(binedges[:-1] + binedges[1:])/2
                pguess = [ampPCDD01,muPCDD01,stdPCDD01]
                try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist01, p0=pguess); PCDDhistfit01 = gauss(bincenters,*pfit)
                except: pfit = pguess; PCDDhistfit01 = gauss(bincenters,*pfit)
                numbins = int(max(skipperdiffcoreravelled01) - min(skipperdiffcoreravelled01))
                skipperdiffcoreravelledhist01, binedges = np.histogram([s for s in skipperdiffcoreravelled01 if s!=0], numbins, density=False)
                bincenters=(binedges[:-1] + binedges[1:])/2
                pfit = [np.mean(skipperdiffcoreravelledhist01[np.argmax(skipperdiffcoreravelledhist01)-2:np.argmax(skipperdiffcoreravelledhist01)+3]),muPCDD01,stdPCDD01]
                PCDDhistfit01 = gauss(bincenters,*pfit)
                axs[iimage,0].plot(bincenters, PCDDhistfit01, label='gaussian fit', linewidth=1, color='red')
                axs[iimage,0].hist([s for s in skipperdiffcoreravelled01 if s!=0], len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='PCDD')
                axs[iimage,0].set_xlim([muPCDD01-10*stdPCDD01,muPCDD01+10*stdPCDD01])
                axs[iimage,0].set_ylim(1,skipperdiffcoreravelledhist01[np.argmax(skipperdiffcoreravelledhist01)]+100)
                #axs[iimage,0].plot(bincenters,skipperdiffcoreravelledinrangehist01, label='PCDD', color='teal')
                axs[iimage,0].legend(loc='lower center',prop={'size': 10})
                axs[iimage,0].set_yscale('log')
                axs[iimage,0].tick_params(axis='both', which='both', length=10, direction='in')
                axs[iimage,0].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[iimage,0].get_yticklabels(), visible=True)
                from scipy.stats import norm
                axs[iimage,0].set_title('$S_{k_{cl}}~=~$' + str(round(kclsignificance01_stack[iimage],4)) + '. Tails obs/exp: ' + str(len([s for s in skipperdiffcoreravelled01 if s < muPCDD01-kclthreshold*stdPCDD01])) + '/' + str( int(round_sig_2( len(skipperdiffcoreravelled01)*norm(loc = 0, scale = 1).cdf(-kclthreshold)))) +'; '+ str(len([s for s in skipperdiffcoreravelled01 if s > muPCDD01+kclthreshold*stdPCDD01])) + '/' + str( int(round_sig_2( len(skipperdiffcoreravelled01)*norm(loc = 0, scale = 1).cdf(-kclthreshold)))) )
                
                skipperdiffcoreravelledinrange = [s for s in skipperdiffcoreravelled if s > muPCDD - 3*stdPCDD and s <  muPCDD + 3*stdPCDD and s != 0]
                numbins = int(max(skipperdiffcoreravelledinrange) - min(skipperdiffcoreravelledinrange))
                skipperdiffcoreravelledinrangehist, binedges = np.histogram(skipperdiffcoreravelledinrange, numbins, density=False)
                bincenters=(binedges[:-1] + binedges[1:])/2
                pguess = [ampPCDD,muPCDD,stdPCDD]
                try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist, p0=pguess);     PCDDhistfit = gauss(bincenters,*pfit)
                except: pfit = pguess; PCDDhistfit = gauss(bincenters,*pfit)
                
                numbins = int(max(skipperdiffcoreravelled) - min(skipperdiffcoreravelled))
                skipperdiffcoreravelledhist, binedges = np.histogram([s for s in skipperdiffcoreravelled if s!=0], numbins, density=False)
                bincenters=(binedges[:-1] + binedges[1:])/2
                pfit = [np.mean(skipperdiffcoreravelledhist[np.argmax(skipperdiffcoreravelledhist)-2:np.argmax(skipperdiffcoreravelledhist)+3]),muPCDD,stdPCDD]
                PCDDhistfit = gauss(bincenters,*pfit)
                axs[iimage,1].plot(bincenters, PCDDhistfit, label='gaussian fit', linewidth=1, color='red')
                axs[iimage,1].hist([s for s in skipperdiffcoreravelled if s!=0], len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='PCDD')
                axs[iimage,1].set_xlim([muPCDD-10*stdPCDD,muPCDD+10*stdPCDD])
                axs[iimage,1].set_ylim(1,skipperdiffcoreravelledhist[np.argmax(skipperdiffcoreravelledhist)]+100)
                #axs[iimage,1].plot(bincenters, PCDDhistfit, label='gaussian fit', linewidth=1, color='red')
                #axs[iimage,1].hist(skipperdiffcoreravelledinrange, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='PCDD')
                #axs[1].plot(bincenters,skipperdiffcoreravelledinrangehist, label='pixel charge difference distribution', color='teal')
                axs[iimage,1].legend(loc='lower center',prop={'size': 10})
                axs[iimage,1].set_yscale('log')
                axs[iimage,1].tick_params(axis='both', which='both', length=10, direction='in')
                axs[iimage,1].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[iimage,1].get_yticklabels(), visible=True)
                axs[iimage,1].set_title('$S_{k_{cl}}~=~$' + str(round(kclsignificance_stack[iimage],4)) + '. Tails obs/exp: ' + str(len([s for s in skipperdiffcoreravelled if s < muPCDD-kclthreshold*stdPCDD])) + '/' + str( int(round_sig_2( len(skipperdiffcoreravelled)*norm(loc = 0, scale = 1).cdf(-kclthreshold)))) +'; '+ str(len([s for s in skipperdiffcoreravelled if s > muPCDD+kclthreshold*stdPCDD])) + '/' + str( int(round_sig_2( len(skipperdiffcoreravelled)*norm(loc = 0, scale = 1).cdf(-kclthreshold)))) )
            
            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Full image pixel charge difference distributions (PCDD) between first and second skip (left) and second and end skip (right). Entries at 0 (saturation digitizer range) might be masked for analysis purposes.')
                doc.append(NoEscape(r'NOTE: A good gaussian fit of the PCDDs is essential for $S_{k_{cl}}$ to be an effective charge loss classifier'))
            plt.clf()
            doc.append(NewPage())
            
            
    
    #############################################
    ##Calibrated image and Dark Current section##
    #############################################
    if reportCalibrationDarkcurrent and nskips!=1:
        with doc.create(Section('Dark Current')):
            for iimage in range(nimages):
                skipperavgcalibrated = skipper_avg_cal_stack[:,:,iimage].ravel()
                try:#if calibration went wrong skipperavgcalibratedravel could be empty because limits are out of range
                    if calibrationconstant == calibrationguess: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -10 and  s < 10]
                    elif calibrationconstant <= 1:
                        skipperavgcalibratedravel =  [s for s in skipperavgcalibrated.ravel() if s > -200 and  s < 200]
                        if 0.01 < calibrationconstant <=0.1: skipperavgcalibratedravel =  [s for s in skipperavgcalibrated.ravel() if s > -2000 and  s < 2000]
                        elif calibrationconstant <= 0.01: skipperavgcalibratedravel =  [s for s in skipperavgcalibrated.ravel() if s > -20000 and  s < 20000]
                        nbins=int((max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))/10)
                    else:
                        skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -2 and  s < 4]
                        nbins=20*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
                except: #if so we keep skipperavgcalibratedravel without range
                    skipperavgcalibratedravel = skipperavgcalibrated
                    nbins=20*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
                if nbins == 0: nbins=100
                skipperavgcalibratedravelhist, binedges = np.histogram(skipperavgcalibratedravel, nbins, density=False)
                bincenters=(binedges[:-1] + binedges[1:])/2
                npeaksp = 3
                dcpar = parametersDCfit_stack[0][0][iimage], npeaksp, parametersDCfit_stack[0][2][iimage]/(20/0.5), parametersDCfit_stack[0][3][iimage]/calibrationconstant
                #dcparunc has one more component (the gain) than dcpar (dcpar is an argument for the calibrated gaussian)
                try: dcparunc = parametersDCfit_stack[1][0][iimage], parametersDCfit_stack[1][1][iimage], parametersDCfit_stack[1][2][iimage]/(20/0.5), parametersDCfit_stack[1][3][iimage]/calibrationconstant, parametersDCfit_stack[1][5][iimage]
                except: dcparunc = 0,0,0,0,0
                skipperavgcalibratedravelhistfit = convolutionGaussianPoisson(bincenters,*dcpar)
                #plt.plot(bincenters,skipperavgcalibratedravelhist,label='avg img calibrated pixel charge distribution', color='teal')
                plt.hist(skipperavgcalibratedravel, len(bincenters), density = False, histtype='step', linewidth=2, log = False, color = 'teal', label='avg image calibrated pixel charge distribution')
                plt.plot(bincenters, skipperavgcalibratedravelhistfit, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(reducedchisquared_stack[iimage])), color='red')
                #plt.hist(skipperavgcalibrated.ravel(), 200, (-1,5), density = False, histtype='step', linewidth=2, log = True, color='teal')
                plt.legend(loc='upper right',prop={'size': 20})
                plt.xlabel('pixel value [e$^-$]')
                plt.ylabel('counts')
                plt.tick_params(axis='both', which='both', length=10, direction='in')
                plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
                #plt.setp(ax.get_yticklabels(), visible=True)
                try: plt.title('$I_{darkCF}~=~$' + str(round(dcpar[0],6)) + '$\pm$' + str(round_sig_2(dcparunc[0])) + ' $e^-$pix$^{-1}$, $I_{darkAC}~=~$' + str(round(darkcurrentestimateAC_stack[iimage],6)) + ' $e^-$pix$^{-1}$')
                except: plt.title('$I_{darkCF}~=~$' + str(round(dcpar[0],6)) + '$\pm$' + str(dcparunc[0]) + ' $e^-$pix$^{-1}$, $I_{darkAC}~=~$' + str(round(darkcurrentestimateAC_stack[iimage],6)) + ' $e^-$pix$^{-1}$')
        
                with doc.create(Figure(position='htb!')) as plot:
                    plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                    plot.add_caption('Calibrated pixel charge distribution. Dark current values computed with convolution fit (on full image) and anticlustering (on '+analysisregion+' image region)')
                calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+'±'+str(round_sig_2(dcparunc[4]))+' ADU per electron.'
                doc.append(calibrationline)
                plt.clf()
                doc.append(NewPage())
    

#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure:
    if not multipleimages: reportname = 'reports/tweaking_'+sys.argv[2]
    if multipleimages: reportname = 'reports/tweaking_'+str(lowerindex)+'_'+str(upperindex)
else:
    if not multipleimages: reportname = 'tweaking_'+sys.argv[2]
    if multipleimages: reportname = 'tweaking_'+str(lowerindex)+'_'+str(upperindex)

doc.generate_pdf(reportname, clean_tex=True)

end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')
    
