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
workingdirectory = config['working_directory']
default_directory_structure = config['raw_processed_header_reports_dir_structure']
iskipstart = config['skip_start']
iskipend = config['skip_end']
fixLeachReco = config['fix_leach_reconstruction']
reverse = config['reverse']
registersize = config['ccd_register_size']
analysisregion = config['analysis_region']
reportHeader = config['report'][-1]['header']
reportImage = config['report'][-1]['image']
reportPCD = config['report'][-1]['pcds']
reportChargeLoss = config['report'][-1]['chargeloss']
reportCalibrationDarkcurrent = config['report'][-1]['calibration_darkcurrent']
reportFFTskips = config['report'][-1]['fft_skips']
reportFFTrow = config['report'][-1]['fft_row']

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
from functions import make_colorbar_with_padding, gauss, factorial, convolutionGaussianPoisson, round_sig_2, reverse, analysisregion
import reconstruction
import chargeloss
import calibrationdc

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

image_data,skipper_image_start,skipper_image_end,skipper_averages,skipper_diff,skipper_diff_01,skipper_avg0,skipper_std = reconstruction.reconstructSkipperImage(image_file,arg2)

##############################################################################
#ESTIMATE NOISE AT SKIPS: 1, 10, 100 . . . 1000 ##############################
##############################################################################

startskipfitpar = functions.sigmaFinder(skipper_image_start, debug=False) #ampss, muss, stdss, stduncss
if reportPCD or reportCalibrationDarkcurrent:
    if nskips < 10: naverages = 0
    elif nskips < 100: naverages = 1; numberskips=[10]
    else:
        numberskips=[10]; index=1
        while index <= nskips/100:
            numberskips.append(index*100)
            naverages = index+1; index+=1
    ampmanyskip, mumanyskip, stdmanyskip, stduncmanyskip = [],[],[],[]
    for k in range(naverages): amp, mu, std, stdunc = functions.sigmaFinder(skipper_averages[:,:,k], debug=False); ampmanyskip.append(amp);mumanyskip.append(mu); stdmanyskip.append(std); stduncmanyskip.append(stdunc)

##############################################################################
#FIRST LAST SKIP CHARGE LOSS CHECK: KCL AND SKEW##############################
##############################################################################
    
if reportChargeLoss and nskips!=1:
    diff_image_core = functions.selectImageRegion(skipper_diff,'no_borders')
    PCDDstudyparameters = chargeloss.firstLastSkipPCDDCheck(diff_image_core, debug=False) #skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, muPCDD, stdPCDD
    kclsignificance = PCDDstudyparameters[2]/PCDDstudyparameters[3]
    if abs(kclsignificance) > 3: imageIsGood *= False; print('Kcl value flags probable charge loss')

##############################################################################
#ADU TO e- CALIBRATION AND DARK CURRENT ESTIMATES#############################
##############################################################################

if reportCalibrationDarkcurrent and nskips!=1:
    parametersDCfit, reducedchisquared, offset = calibrationdc.calibrationDC(skipper_avg0, stdmanyskip[-1], reverse, debug=False)
    calibrationconstant = parametersDCfit[0][5]; calibratedsigma = stdmanyskip[-1]/calibrationconstant
    skipper_avg_cal = -int(reverse)*(skipper_avg0 - offset)/calibrationconstant
    darkcurrentestimateAC = calibrationdc.anticlusteringDarkCurrent(functions.selectImageRegion(skipper_avg_cal,analysisregion), calibratedsigma, debug=False)

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

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report'))
doc.preamble.append(Command('author', 'DAMIC-M'))
doc.append(NoEscape(r'\maketitle'))
#############################################
#Print acqusition parameters value in report#
#############################################
if reportHeader:
    if default_directory_structure:
        fileheader = open(workingdirectory + 'header/' + sys.argv[2] + '.txt', 'r')
    else: fileheader = open(workingdirectory + sys.argv[2] + '.txt', 'r')
    lines = fileheader.readlines()
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

ampss, muss, stdss, stduncss = startskipfitpar #ss: start skip

if reportImage:

    centeredsstoplot = -int(reverse)*(skipper_image_start - muss)
    clustercandidates = reconstruction.findChargedPixelNoBorder(centeredsstoplot,stdss)
    isChargedCrown = True; coor = np.size(centeredsstoplot,0)//2, np.size(centeredsstoplot,1)//2
    for coor in clustercandidates:
        isChargedCrown = reconstruction.chargedCrown(coor,centeredsstoplot,stdss)
        if (isChargedCrown):
            print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
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
        import functions
        
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=True, tight_layout=True)
        
        skipper_image_start_region = functions.selectImageRegion(skipper_image_start,analysisregion)
        skipper_image_start_ravel = skipper_image_start_region.ravel()
        #instead of removing 0-entries from histogram use numpy mask to avoid discrepancies between gaussian and plotted PCD skipper_image0ravel
        #skipper_image = [s for s in skipper_image_start_ravel if s != 0]
        skipper_image_unsaturated = np.ma.masked_equal(skipper_image_start_ravel, 0.0, copy=False)
        skipper_imagehist, binedges = np.histogram(skipper_image_unsaturated, bins = 800, density=False)
        ampss = skipper_imagehist[np.argmax(skipper_imagehist)]
        axs[0].hist(skipper_image_start_ravel, 800, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='start skip pixel charge distribution')
        bincenters = np.arange(muss - 3*stdss, muss + 3*stdss + 6*stdss/100, 6*stdss/100) #last term in upper bound to get ~sym drawing
        axs[0].plot(bincenters, gauss(bincenters,ampss,muss,stdss), label='gaussian fit curve', linewidth=1, color='red')
        axs[0].legend(loc='upper left',prop={'size': 14})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        try: axs[0].set_title('First skip pixel charge distribution in '+analysisregion+': $\sigma_{0e^-}~=~$ ' + str(round(stdss,4)) + ' ADU; estimated noise: ' + str(round(stdss/calibrationconstant,4)) + ' $e^{-}$')
        except: axs[0].set_title('First skip pixel charge distribution in '+analysisregion+': $\sigma_{0e^-}~=~$ ' + str(round(stdss,4)) + ' ADU')

        
        if nskips!=1:
            averageimageoffset = functions.sigmaFinder(skipper_avg0, debug=False)[1]
            skipper_avg0_region = functions.selectImageRegion(skipper_avg0,analysisregion)
            avg_image_0ravel = skipper_avg0_region.ravel()
            avg_image_unsaturated = np.ma.masked_equal(avg_image_0ravel, 0.0, copy=False)
            avg_image_unsaturated = [s for s in avg_image_unsaturated if averageimageoffset - 5*calibrationconstant < s < averageimageoffset + calibrationconstant]
            avg_image_hist, binedges = np.histogram(avg_image_unsaturated, bins = 200, density=False)
            ampls = avg_image_hist[np.argmax(avg_image_hist)]
            bincenters = np.arange(averageimageoffset - 3*stdmanyskip[-1], averageimageoffset + 3*stdmanyskip[-1] + 6*stdmanyskip[-1]/200, 6*stdmanyskip[-1]/200)
            axs[1].plot(bincenters, gauss(bincenters,ampls,averageimageoffset,stdmanyskip[-1]), label='gaussian fit curve', linewidth=1, color='red')
            axs[1].hist(avg_image_0ravel, 200, range = (averageimageoffset - 5*calibrationconstant, averageimageoffset + calibrationconstant), density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')
            axs[1].legend(loc='upper left',prop={'size': 14})
            axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[1].get_yticklabels(), visible=True)
            axs[1].set_title('Average image pixel charge distribution in '+analysisregion+': $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip[-1],4)) + ' ADU; estimated noise: ' + str(round(stdmanyskip[-1]/calibrationconstant,4)) + ' $e^{-}$')
        
        plt.subplots_adjust(hspace=0.5)
        for ax in axs.flat:
            ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
        plt.clf()
        doc.append(NewPage())
        
        def r(ns):
            return stdss/np.sqrt(ns)
        fig, axs = plt.subplots(1, 1, figsize=(8,6), sharey=True, tight_layout=True)
        #numberSkips = [10,100,200,300,400,500,600,700,800,900,1000]
        ns = np.arange(1,1000,1)
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
        ax.axis([1, 1000, 0.1, 100])
        ax.loglog()
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.title('Resolution trend')
        
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Resolution trend computed on '+analysisregion+', as function of average image skip number.')
        plt.clf()
        doc.append(NewPage())
        
#############################################
#Charge loss indicators and skewness section#
#############################################
if reportChargeLoss and nskips!=1:
    with doc.create(Section('Charge-loss')):
        skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, muPCDD, stdPCDD = PCDDstudyparameters
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
        
        skipperdiffcoreravelled = diff_image_core.ravel()
        skipper_imagehist, binedges = np.histogram(skipper_image_unsaturated, bins = 800, density=False)
        axs[0].hist(skipperdiffcoreravelled, 400, density = False, histtype='step', linewidth=2, log = True, color='teal', label='pixel charge difference distribution')
        axs[0].legend(loc="upper right",prop={'size': 14})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[0].set_title('Estimated width : $\sigma_{dif}~=~$' + str(round(stdPCDD,4)) + 'ADU')
        
        skipperdiffcoreravelledinrange = [s for s in skipperdiffcoreravelled if s > muPCDD - 3*stdPCDD and s < muPCDD + 3*stdPCDD and s != 0]
        numbins = int(max(skipperdiffcoreravelledinrange) - min(skipperdiffcoreravelledinrange))
        skipperdiffcoreravelledinrangehist, binedges = np.histogram(skipperdiffcoreravelledinrange, numbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        pguess = [1E+2,muPCDD,stdPCDD]
        pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist, p0=pguess)
        PCDDhistfit = gauss(bincenters,*pfit)
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
            plot.add_caption('Full image pixel charge difference distribution (PCDD) in full and limited range (for fit). Entries at 0 (saturation digitizer range) might be masked for analysis purposes.')
        plt.clf()
        doc.append(NewPage())
        
        centeredskipperdiffcore = [s for s in skipperdiffcoreravelled-muPCDD if s != -muPCDD]
        plt.hist(centeredskipperdiffcore, 600, range = (-20*stdPCDD,10*stdPCDD), density = False, histtype='step', linewidth=2, log = True, color='teal',label='centered pixel charge difference distribution')
        plt.legend(loc='upper right',prop={'size': 20})
        plt.xlabel('pixel value [ADU]')
        plt.ylabel('counts per ADU')
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.title( '$k_{cl}~=~$' + str(round(kclPCDD,4)) + '$\pm$'+ str(round(kclPCDDuncertainty,4)) + ', $S(k_{cl})~=~$' + str(round(kclPCDD/kclPCDDuncertainty,4)) + ', skewness = ' + str(round(skewnessPCDD,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty,4)))
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Pedestal-subtracted full-image PCDD.')
        plt.clf()
        doc.append(NewPage())
        
#############################################
##Calibrated image and Dark Current section##
#############################################
if reportCalibrationDarkcurrent and nskips!=1:
    skipperavgcalibrated = skipper_avg_cal.ravel()
    try:#if calibration went wrong skipperavgcalibratedravel could be empty because limits are out of range
        if calibrationconstant == 10: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -10 and  s < 10]
        else: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -2 and  s < 4]
        nbins=50*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
    except:#if so we keep skipperavgcalibratedravel without range
        skipperavgcalibratedravel = skipperavgcalibrated
        nbins=50*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
    if nbins == 0: nbins=100
    skipperavgcalibratedravelhist, binedges = np.histogram(skipperavgcalibratedravel, nbins, density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    npeaksp = 3
    dcpar = parametersDCfit[0][0], npeaksp, parametersDCfit[0][2]/(50/0.5), parametersDCfit[0][3]/calibrationconstant
    #dcparunc has one more component (the gain) than dcpar (dcpar is an argument for the calibrated gaussian)
    try: dcparunc = parametersDCfit[1][0], parametersDCfit[1][1], parametersDCfit[1][2]/(50/0.5), parametersDCfit[1][3]/calibrationconstant, parametersDCfit[1][5]
    except: dcparunc = 0,0,0,0,0
    skipperavgcalibratedravelhistfit = convolutionGaussianPoisson(bincenters,*dcpar)
    #plt.plot(bincenters,skipperavgcalibratedravelhist,label='avg img calibrated pixel charge distribution', color='teal')
    plt.hist(skipperavgcalibratedravel, len(bincenters), density = False, histtype='step', linewidth=2, log = False, color = 'teal', label='avg image calibrated pixel charge distribution')
    plt.plot(bincenters, skipperavgcalibratedravelhistfit, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(reducedchisquared)), color='red')
    #plt.hist(skipperavgcalibrated.ravel(), 200, (-1,5), density = False, histtype='step', linewidth=2, log = True, color='teal')
    plt.legend(loc='upper right',prop={'size': 20})
    plt.xlabel('pixel value [e$^-$]')
    plt.ylabel('counts')
    plt.tick_params(axis='both', which='both', length=10, direction='in')
    plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
    #plt.setp(ax.get_yticklabels(), visible=True)
    plt.title('$I_{darkCF}~=~$' + str(round(dcpar[0],6)) + '$\pm$' + str(round_sig_2(dcparunc[0])) + ' $e^-$pix$^{-1}$, $I_{darkAC}~=~$' + str(round(darkcurrentestimateAC,6)) + ' $e^-$pix$^{-1}$')
    
    with doc.create(Section('Dark Current')):
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Calibrated pixel charge distribution. Dark current values computed with convolution fit (on full image) and anticlustering (on '+analysisregion+')')
        calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+'±'+str(round_sig_2(dcparunc[4]))+' ADU per electron.'
        doc.append(calibrationline)
        plt.clf()
        doc.append(NewPage())
        
#############################################
#######Fast Fourier Transform section #######
#############################################
if (reportFFTskips or reportFFTrow):
    import functions
    nallcolumns = hdr['NAXIS1']
    nrows = hdr['NAXIS2']
    nskips = hdr['NDCMS']
    samplet = hdr['MREAD']*0.001 #MREAD is in ms. Convert in s
    with doc.create(Section('Fourier Analysis')):

        if reportFFTskips and nskips!=1:
            samplet /= (nrows*nallcolumns)
            ncolumns = int(nallcolumns/nskips) # n of columns in the image
            functions.pixelFFT(image_data, nrows-1, ncolumns-1, nskips, samplet)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Full image Fast Fourier Transform (first to last skip).')
            plt.clf()
    
        if reportFFTrow:
            samplet *= nskips
            if nskips!=1: functions.rowFFT(skipper_avg0, nrows-1, ncolumns-1, samplet)
            else: functions.rowFFT(image_data, nrows-1, ncolumns-1, samplet)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Average image Fast Fourier Transform (all row pixels).')
            plt.clf()
            doc.append(NewPage())
    
#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure: reportname = 'reports/'+sys.argv[2]
else: reportname = sys.argv[2]
doc.generate_pdf(reportname, clean_tex=False)
os.remove(reportname+'.tex')
