#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Executable devoted to monitor the quality of data produced by the CCD.

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
registersize = config['ccd_active_register_size']
prescan = config['prescan']
overscan = config['overscan']
calibrationguess = config['calibration_constant_guess']
printheader = False
multipleimages = config['quality_analysis'][-1]['multiple_images'][-1]['use_multiple_images']
if multipleimages:
    makemask = config['quality_analysis'][-1]['multiple_images'][-1]['produce_mask']
    if makemask:
        imfrhotp = config['quality_analysis'][-1]['multiple_images'][-1]['image_fraction_hot_pixel']
        pfrhotcl = config['quality_analysis'][-1]['multiple_images'][-1]['pixel_fraction_hot_column']
reportHeader = config['quality_analysis'][-1]['report'][-1]['header']
reportImage = config['quality_analysis'][-1]['report'][-1]['image']
reportQuality = config['quality_analysis'][-1]['report'][-1]['quality']
reportQualityLogScale = config['quality_analysis'][-1]['report'][-1]['quality_plots'][-1]['log_scale']

if test != 'quality':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for data quality analysis. Test selected in configuration file is different from 'quality': do you want to perform data quality analysis?\nPlease answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with data quality analysis')

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

from m_functions import sigmaFinder,selectImageRegion, make_colorbar_with_padding, gauss
from m_reconstruction import *
import m_chargeloss
import m_calibrationdc

##############################################################################
# Specify path (can be out of the main tree)

import os
os.chdir(workingdirectory)

##############################################################################
# Import warnings for warning control

import warnings

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Start processing for data quality analysis
# if leach: image is fixed in reconstruction module
##############################################################################
#SINGLE IMAGE ANALYSIS: FIND ROW/COLUMN MEDIANS AND MADS######################
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
    warnings.filterwarnings("error")
    if nskips == 1:
        image_data = getSingleSkipImage(image_file)
        image_data0 = image_data
        try: image_overscan = selectImageRegion(image_data,'overscan')
        except: print('ERROR: Image has no overscan. Cannot estimate median and MAD in overscan. Exiting'); sys.exit()
        overscanmedian = np.median(image_overscan.ravel())
        overscanMAD = np.median(abs(image_overscan.ravel()-overscanmedian))
        rowmedian,rowmad = medianMadRowByRow(image_data)
        colmedian,colmad = medianMadColByCol(image_data)
    else:
        skip_image_stack = getManySkipImageStack(image_file)
        overscanmedian,overscanMAD = [],[]
        for skip in range(nskips):
            try: image_overscan = selectImageRegion(skip_image_stack[:,:,skip],'overscan')
            except: print('ERROR: Image has no overscan. Cannot estimate median and MAD in overscan. Exiting'); sys.exit()
            overscanmedian.append(np.median(image_overscan.ravel()))
            overscanMAD.append(np.median(abs(image_overscan.ravel()-overscanmedian[-1])))
        skipper_average = getAverageSkipperImage(image_file)
        image_data0 = skipper_average
        rowmedian,rowmad = medianMadRowByRow(skipper_average)
        colmedian,colmad = medianMadColByCol(skipper_average)
    warnings.filterwarnings("default")

##############################################################################
#MULTIPLE IMAGES ANALYSIS: FIND BASELINES AND MAKE MASK#######################
##############################################################################
if multipleimages: #this analysis carried out on single skip image by default. If nskips > 1, iskipstart is used for the study
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for data quality analysis using multiple images. This code assumes multiple images have the same structure (size and number of skips). If this is not the case code may fail.\nDo you want to continue? Please answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with multiple images quality analysis')
    lowerindex = config['quality_analysis'][-1]['multiple_images'][-1]['lower_index']
    upperindex = config['quality_analysis'][-1]['multiple_images'][-1]['upper_index']
    nameprefix = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    ##############################################################################
    # Open the data image
    image_file = get_pkg_data_filename(nameprefix+str(lowerindex)+'.fits')
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
    ##############################################################################
    # WARNING: code assumes all images have same structure as first
    if nskips != 1:
        print('WARNING: Stacking iskipstart skip for all images for mask and baseline study')
        print('Following image structure expected: N. rows columns skips ',nrows,ncolumns,nskips)
        single_skip_image_stack = reconstructSkipNImageStack(nameprefix,lowerindex,upperindex)
    else:
        print('Stacking single skip images for mask and baseline study')
        print('Following image structure expected: N. rows columns skips ',nrows,ncolumns,nskips)
        single_skip_image_stack = getManySingleImageStack(nameprefix,lowerindex,upperindex)

    ##############################################################################
    # Find images baselines: overscan median and MAD, row medians and mad ########
    image_data0 = single_skip_image_stack[:,:,0]
    overscanmedian,overscanMAD,rowmedian,rowmad,colmedian,colmad=[],[],[],[],[],[]
    for i in range(upperindex-lowerindex+1):
        rowmedian.append([])
        rowmad.append([])
        colmedian.append([])
        colmad.append([])
    warnings.filterwarnings("error")
    for i in range(upperindex-lowerindex+1):
        image_data = single_skip_image_stack[:,:,i]
        try: image_overscan = selectImageRegion(image_data,'overscan')
        except: print('ERROR: Image has no overscan. Cannot estimate median and MAD in overscan. Exiting'); sys.exit()
        overscanmedian.append(np.median(image_overscan.ravel())); overscanMAD.append(np.median(abs(image_overscan.ravel()-overscanmedian[i])))
        tmprowmed,tmprowmad = medianMadRowByRow(image_data); tmpcolmed, tmpcolmad = medianMadColByCol(image_data)
        rowmedian[i].append(tmprowmed); rowmad[i].append(tmprowmad); colmedian[i].append(tmpcolmed); colmad[i].append(tmpcolmad)
    warnings.filterwarnings("default")
    
    ##############################################################################
    # Make mask using row medians and mads: if pixel outlies in half the images or
    # more, it is hot and masked
    if makemask:
        mask = np.zeros((nrows,ncolumns),dtype=np.float64)
        for i in range(upperindex-lowerindex+1):
            image_data = single_skip_image_stack[:,:,i]
            mask += findOutliers(image_data,rowmedian[i],rowmad[i])
        mask /= (upperindex-lowerindex+1)
        for col in range(ncolumns):
            for row in range(nrows):
                if mask[row,col] >= imfrhotp: mask[row,col] = True
                else: mask[row,col] = False
            if sum(mask[:,col]) >= pfrhotcl*nrows: mask[:,col] = True #if x0% or more than a column's pixels are hot, mask entire column
        hdr_copy = hdr.copy()
        hdu0 = fits.PrimaryHDU(data=mask,header=hdr_copy)
        new_hdul = fits.HDUList([hdu0])
        if default_directory_structure: new_hdul.writeto('processed/MASK_'+str(lowerindex)+'_'+str(upperindex)+'.fits', overwrite=True)
        else: new_hdul.writeto('MASK_'+str(lowerindex)+'_'+str(upperindex)+'.fits', overwrite=True)

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

if not (reportHeader or reportImage or reportQuality): print('No information to be reported. Report will not be produced. Exiting'); sys.exit()

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report on Data Quality'))
doc.preamble.append(Command('author', 'DAMIC-M'))
doc.preamble.append(NoEscape(r'\usepackage{tocloft}'))
doc.preamble.append(NoEscape(r'\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}'))
doc.preamble.append(NoEscape(r'\usepackage{hyperref}'))
doc.preamble.append(NoEscape(r'\usepackage{bookmark}'))
doc.append(NoEscape(r'\maketitle'))
doc.append(NoEscape(r'\tableofcontents'))
doc.append(NoEscape(r'\thispagestyle{empty}'))
doc.append(NewPage())

#############################################
#Print acqusition parameters value in report#
#############################################
if reportHeader:
    if printheader and default_directory_structure: fileheader = open(workingdirectory + 'header/' + sys.argv[2] + '.txt', 'r')
    elif printheader and not default_directory_structure: fileheader = open(workingdirectory + sys.argv[2] + '.txt', 'r')
    if printheader: lines = fileheader.readlines()
    else:
        try: lines = repr(fits.getheader(image_file, 0)).splitlines()
        except: lines = '' #this break will act for multiple images where no single header
    with doc.create(Section('Image Acquisition Parameters')):
        with doc.create(Description()) as desc:
            for line in lines[0:100]:
                if line.split()[0]!='COMMENT': desc.add_item(line,'')
                #desc.add_item(line.split()[0].replace('=','')+'='+line.split()[-1],'')
                if line.split()[0]=='MREAD': break
doc.append(NewPage())
        
##############################################
################Image section#################
##############################################
if reportImage:
    stddev = sigmaFinder(image_data0,False)[2]
    clustercandidates = findChargedPixelNoBorder(image_data0,stddev)
    isChargedCrown = True; coor = np.size(image_data0,0)//2, np.size(image_data0,1)//2
    for coor in clustercandidates:
        isChargedCrown = chargedCrown(coor,image_data0,stddev)
        if (isChargedCrown):
            #print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
            break
    if not isChargedCrown: coor = np.size(image_data0,0)//2, np.size(image_data0,1)//2
    with doc.create(Section('Images')):
        if np.size(image_data0,0) > 80: halfrangey = 40
        else: halfrangey = np.size(image_data0,0)//2
        if np.size(image_data0,1) > 80: halfrangex = 40
        else: halfrangex = np.size(image_data0,1)//2
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],2*halfrangey-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],2*halfrangex-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
        fig=plt.figure(figsize=(8,8))

        ax1=fig.add_subplot(111)
        plt.imshow(image_data0[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        if nskips==1: plt.title('Single skip image')
        else: plt.title('Average skip image')
        plt.ylabel("row")
        cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
        cb1 = plt.colorbar(cax=cax1)

        fig.tight_layout(pad=.001)

    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.99\linewidth'))
        if nskips==1: plot.add_caption('Exposed pixels region for single skip image. In multiple image case, lower index image is shown')
        else: plot.add_caption('Exposed pixels region for average skip image. In multiple image case, lower index image is shown')
    plt.clf()
    doc.append(NewPage())

if reportQuality:
    with doc.create(Section('Data Quality')):
        if nskips == 1:
            #overscanmean,overscansigma=sigmaFinder(image_overscan,False)[1:3]
            fig, axs = plt.subplots(4, 1, figsize=(11,10), sharey=False, tight_layout=True)
            if multipleimages:
                for i in range(upperindex-lowerindex+1):
                    if i == 0:
                        axs[0].plot(np.arange(ncolumns),[y for x in colmedian[i] for y in x], label='img column median',color='teal')
                        axs[1].plot(np.arange(ncolumns),[y for x in colmad[i] for y in x], label='img column MAD',color='teal')
                        axs[2].plot(np.arange(nrows),[y for x in rowmedian[i] for y in x],label='img row median',color='teal')
                        axs[3].plot(np.arange(nrows),[y for x in rowmad[i] for y in x],label='img row MAD',color='teal')
                    else:
                        axs[0].plot(np.arange(ncolumns),[y for x in colmedian[i] for y in x])
                        axs[1].plot(np.arange(ncolumns),[y for x in colmad[i] for y in x])
                        axs[2].plot(np.arange(nrows),[y for x in rowmedian[i] for y in x])
                        axs[3].plot(np.arange(nrows),[y for x in rowmad[i] for y in x])
            else:
                axs[0].plot(np.arange(ncolumns),colmedian, label='img column median',color='teal')
                axs[1].plot(np.arange(ncolumns),colmad, label='img column MAD',color='teal')
                axs[2].plot(np.arange(nrows),rowmedian,label='img row median',color='teal')
                axs[3].plot(np.arange(nrows),rowmad,label='img row MAD',color='teal')
            axs[0].legend(loc='upper right',prop={'size': 14}); axs[1].legend(loc='upper right',prop={'size': 14})
            axs[2].legend(loc='upper right',prop={'size': 14}); axs[3].legend(loc='upper right',prop={'size': 14})
            axs[0].tick_params(axis='both', which='both', length=10, direction='in'); axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[2].tick_params(axis='both', which='both', length=10, direction='in'); axs[3].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both'); axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            axs[2].grid(color='grey', linestyle=':', linewidth=1, which='both'); axs[3].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True); plt.setp(axs[1].get_yticklabels(), visible=True)
            plt.setp(axs[2].get_yticklabels(), visible=True); plt.setp(axs[3].get_yticklabels(), visible=True)
            axs[0].set_title('Image row and column median and MAD'); axs[1].set_title(''); axs[2].set_title(''); axs[3].set_title('')
            axs[0].set(xlabel='column number', ylabel='ADU'); axs[1].set(xlabel='column number', ylabel='ADU')
            axs[2].set(xlabel='row number', ylabel='ADU'); axs[3].set(xlabel='row number', ylabel='ADU')
            if reportQualityLogScale:
                for ax in axs.flat:
                    ax.set_yscale('log')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.99\linewidth'))
                plot.add_caption('Single skip median and median absolute deviation (MAD) as functions of column and row numbers')
            plt.clf()
            doc.append(NewPage())
        #import pdb; pdb.set_trace()
        else:
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            if multipleimages:
                axs[0].plot(np.arange(lowerindex, upperindex+1),overscanmedian, label='overscan median',color='teal')
                axs[1].plot(np.arange(lowerindex, upperindex+1),overscanMAD, label='overscan MAD',color='teal')
                axs[0].set_title('Overscan median and MAD with image number'); axs[1].set_title('')
                axs[0].set(xlabel='image number', ylabel='ADU'); axs[1].set(xlabel='image number', ylabel='ADU')
                from matplotlib.ticker import MaxNLocator
                axs[0].xaxis.set_major_locator(MaxNLocator(integer=True)); axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
                axs[0].legend(loc='upper right',prop={'size': 14}); axs[1].legend(loc='upper right',prop={'size': 14})
                axs[0].tick_params(axis='both', which='both', length=10, direction='in'); axs[1].tick_params(axis='both', which='both', length=10, direction='in')
                axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both'); axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[0].get_yticklabels(), visible=True); plt.setp(axs[1].get_yticklabels(), visible=True)
            else:
                axs[0].plot(np.arange(nskips),overscanmedian,label='overscan median',color='teal'); axs[1].plot(np.arange(nskips),overscanMAD, label='overscan MAD',color='teal')
                axs[0].set_title('Overscan median and MAD with skip number'); axs[1].set_title('')
                axs[0].set(xlabel='skip number', ylabel='ADU'); axs[1].set(xlabel='skip number', ylabel='ADU')
                axs[0].legend(loc='upper right',prop={'size': 14}); axs[1].legend(loc='upper right',prop={'size': 14})
                axs[0].tick_params(axis='both', which='both', length=10, direction='in'); axs[1].tick_params(axis='both', which='both', length=10, direction='in')
                axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both'); axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
                plt.setp(axs[0].get_yticklabels(), visible=True); plt.setp(axs[1].get_yticklabels(), visible=True)
                #alternatively
                #fig, axs = plt.subplots(1, 1, figsize=(11,10), sharey=False, tight_layout=True)
                #axs.plot(np.arange(nskips),overscanmedian,label='overscan median')#; axs[1].plot(np.arange(nskips),overscanMAD, label='overscan mad')
                #axs.plot(np.arange(nskips),overscanMAD, label='overscan mad')
                #axs.set_yscale('log')#; axs[1].set_yscale('log')
                #axs.legend(loc='center right',prop={'size': 14})#; axs[1].legend(loc='upper right',prop={'size': 14})
                #axs.tick_params(axis='both', which='both', length=10, direction='in')#; axs[1].tick_params(axis='both', which='both', length=10, direction='in')
                #axs.grid(color='grey', linestyle=':', linewidth=1, which='both')#; axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
                #plt.setp(axs.get_yticklabels(), visible=True)#; plt.setp(axs[1].get_yticklabels(), visible=True)
                #axs.set_title('')#; axs[1].set_title('')
                #axs.set(xlabel='skip number', ylabel='ADU')#; axs[1].set(xlabel='skip number', ylabel='ADU')
            if reportQualityLogScale:
                for ax in axs.flat:
                    ax.set_yscale('log')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.99\linewidth'))
                if multipleimages: plot.add_caption('Overscan median and median absolute deviation (MAD) of images set')
                else: plot.add_caption('Overscan median and median absolute deviation (MAD) as functions of skip number')
            plt.clf()
            doc.append(NewPage())
            
            fig, axs = plt.subplots(4, 1, figsize=(11,10), sharey=False, tight_layout=True)
            if multipleimages:
                for i in range(upperindex-lowerindex+1):
                    if i == 0:
                        axs[0].plot(np.arange(ncolumns),[y for x in colmedian[i] for y in x], label='img column median',color='teal')
                        axs[1].plot(np.arange(ncolumns),[y for x in colmad[i] for y in x], label='img column MAD',color='teal')
                        axs[2].plot(np.arange(nrows),[y for x in rowmedian[i] for y in x],label='img row median',color='teal')
                        axs[3].plot(np.arange(nrows),[y for x in rowmad[i] for y in x],label='img row MAD',color='teal')
                    else:
                        axs[0].plot(np.arange(ncolumns),[y for x in colmedian[i] for y in x])
                        axs[1].plot(np.arange(ncolumns),[y for x in colmad[i] for y in x])
                        axs[2].plot(np.arange(nrows),[y for x in rowmedian[i] for y in x])
                        axs[3].plot(np.arange(nrows),[y for x in rowmad[i] for y in x])
            else:
                axs[0].plot(np.arange(ncolumns),colmedian,label='avg img column median',color='teal')
                axs[1].plot(np.arange(ncolumns),colmad,label='avg img column MAD',color='teal')
                axs[2].plot(np.arange(nrows),rowmedian,label='avg img row median',color='teal')
                axs[3].plot(np.arange(nrows),rowmad,label='avg img row MAD',color='teal')
            axs[0].annotate('median', xy=(650, 100), xycoords='axes points', size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w'))
            axs[1].annotate('MAD', xy=(650, 100), xycoords='axes points', size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w'))
            axs[2].annotate('median', xy=(650, 100), xycoords='axes points', size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w'))
            axs[3].annotate('MAD', xy=(650, 100), xycoords='axes points', size=14, ha='right', va='top',bbox=dict(boxstyle='round', fc='w'))
            axs[0].tick_params(axis='both', which='both', length=10, direction='in'); axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[2].tick_params(axis='both', which='both', length=10, direction='in'); axs[3].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both'); axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            axs[2].grid(color='grey', linestyle=':', linewidth=1, which='both'); axs[3].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True); plt.setp(axs[1].get_yticklabels(), visible=True)
            plt.setp(axs[2].get_yticklabels(), visible=True); plt.setp(axs[3].get_yticklabels(), visible=True)
            axs[0].set_title('Average image column and row median and MAD'); axs[1].set_title('')
            axs[2].set_title(''); axs[3].set_title('')
            axs[0].set(xlabel='column number', ylabel='ADU'); axs[1].set(xlabel='column number', ylabel='ADU')
            axs[2].set(xlabel='row number', ylabel='ADU'); axs[3].set(xlabel='row number', ylabel='ADU')
            if reportQualityLogScale:
                for ax in axs.flat:
                    ax.set_yscale('log')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.99\linewidth'))
                if multipleimages: plot.add_caption('Multiple images start skip median and median absolute deviation (MAD) as functions of column and row numbers')
                else: plot.add_caption('Average image median and median absolute deviation (MAD) as functions of column and row numbers')
            plt.clf()
            doc.append(NewPage())


#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure:
    if not multipleimages: reportname = 'reports/quality_'+sys.argv[2]
    if multipleimages: reportname = 'reports/quality_'+str(lowerindex)+'_'+str(upperindex)
else:
    if not multipleimages: reportname = 'quality_'+sys.argv[2]
    if multipleimages: reportname = 'quality_'+str(lowerindex)+'_'+str(upperindex)
doc.generate_pdf(reportname, clean_tex=False)
os.remove(reportname+'.tex')

end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')

