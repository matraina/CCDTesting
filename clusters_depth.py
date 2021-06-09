#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina
Executable devoted to image cluster search and analysis for physics study and depth calibration.

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
calibrate = config['clusters_depth_analysis'][-1]['calibrate']
globalthreshold = config['clusters_depth_analysis'][-1]['global_threshold_in_sigma']
maximumthreshold = config['clusters_depth_analysis'][-1]['maximum_pixel_value_threshold_in_sigma']
usemask = config['clusters_depth_analysis'][-1]['use_mask']
if usemask: maskpath = config['clusters_depth_analysis'][-1]['mask_path']
else: maskpath = None
multipleimages = config['clusters_depth_analysis'][-1]['multiple_images'][-1]['use_multiple_images']
reportHeader = config['clusters_depth_analysis'][-1]['report'][-1]['header']
reportImage = config['clusters_depth_analysis'][-1]['report'][-1]['image']
reportCalibration = config['clusters_depth_analysis'][-1]['report'][-1]['calibration']
reportCluster = config['clusters_depth_analysis'][-1]['report'][-1]['clusters'][-1]['clusters_plots']
reportDepth = config['clusters_depth_analysis'][-1]['report'][-1]['depth_calibration_plots']

if test != 'clusters_depth_calibration':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for clusters and depth calibration analysis. Test selected in configuration file is different from 'clusters_depth_calibration': do you want to perform clustering and depth calibration analysis?\nPlease answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with clustering and depth calibration analysis')
        
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

from functions import sigmaFinder,selectImageRegion, make_colorbar_with_padding, gauss, round_sig_2, convolutionGaussianPoisson
from reconstruction import *
import calibrationdc
from clusters import *

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
#SINGLE IMAGE ANALYSIS: AVERAGE and CLUSTER SINGLE-SKIP IMG###################
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
    if nskips == 1:
        image_data = getSingleSkipImage(image_file)
        if reverse: image_data = reverseImage(image_data)
        ped_subtracted_image_data, row_pedestals, row_mads = subtractOvscPedestalRowByRow(image_data)
        ped_subtracted_image_data_exposed = selectImageRegion(ped_subtracted_image_data,'exposed_pixels')
        pedestal = np.mean(row_pedestals)
        mad = np.mean(row_mads)
        #cut = [200,500] #cut on value
        cut = [globalthreshold*mad,maximumthreshold*mad] #cut on value
        clusters = clusterImage(ped_subtracted_image_data_exposed,cut,mask=maskpath) #fix
        clusterssigmax,clusterssigmay = [],[]
        for i in range(len(clusters[0])):
            clusterssigmax.append(clusters[2][i][4])
            clusterssigmay.append(clusters[2][i][5])
        calibrationconstant = calibrationguess
        clustersenergy = [x * 0.00377/calibrationconstant for x in clusters[3]]
        #plt.hist(clusterssigmax,500,log=True)
        #plt.hist(clusterssigmay,500,log=False)
        ##plt.xlabel('energy (keV)')
        #plt.xlabel('$\sigma$ (pixels)')
        #plt.ylabel('entries')
        #plt.show()
    else:
        skipper_avg0 = getAverageSkipperImage(image_file)
        #plt.hist(skipper_avg0.ravel(),10000)
        #plt.show()
        mu_avg0, sigma_avg0 = sigmaFinder(skipper_avg0,debug=False)[1:3]
        if calibrate:
            parametersDCfit, reducedchisquared, offset = calibrationdc.calibrationDC(skipper_avg0, sigma_avg0, reverse, debug=False)
            calibrationconstant = parametersDCfit[0][5]; calibratedsigma = sigma_avg0/calibrationconstant
            skipper_avg = -int(reverse)*(skipper_avg0 - offset)
        else: skipper_avg = -int(reverse)*(skipper_avg0 - mu_avg0); calibrationconstant = calibrationguess
        skipper_avg_exposed = selectImageRegion(skipper_avg,'exposed_pixels')
        cut = [globalthreshold*sigma_avg0,maximumthreshold*sigma_avg0]
        clusters = clusterImage(skipper_avg_exposed,cut,mask=maskpath)
        clustersenergy = [x * 0.00377/calibrationconstant for x in clusters[3]]
        
##############################################################################
#MULTIPLE IMAGES ANALYSIS: CLUSTERING FOR MULTIPLE IMGS#######################
##############################################################################
if multipleimages:
    #If nskips > 1, average image is used for the study
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for depth calibration using multiple images. This code assumes multiple images have the same structure (size and number of skips). If this is not the case code may fail.\nDo you want to continue? Please answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with multiple images depth calibration analysis')
    lowerindex = config['depth_calibration_analysis'][-1]['multiple_images'][-1]['lower_index']
    upperindex = config['depth_calibration_analysis'][-1]['multiple_images'][-1]['upper_index']
    nameprefix = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    nimages = upperindex-lowerindex+1
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
    # WARNING: code assumes all images have same structure as first###############
    ##############################################################################
    # CLUSTER IMAGES STACK #######################################################
    ##############################################################################
    if nskips == 1:
        print('Stacking single skip images for depth calibration study')
        print('Following image structure expected: N. rows columns skips ',nrows,ncolumns,nskips)
        single_skip_image_stack = getManySingleImageStack(nameprefix,lowerindex,upperindex)
        image_data = single_skip_image_stack[:,:,0]
        for i in range(nimages):
            if reverse: single_skip_image_stack[:,:,i] = reverseImage(single_skip_image_stack[:,:,i])
        ped_subtracted_image_data_stack = np.zeros((nrows, ncolumns, nimages), dtype=np.float64)
        ncolumnsexp = min(ncolumns,registersize)
        ped_subtracted_image_data_exposed_stack = np.zeros((nrows, ncolumnsexp, nimages), dtype=np.float64)
        pedestals,mads,clusters = [],[],[]
        for i in range(nimages):
            ped_subtracted_image_data_stack[:,:,i],tmppedestals,tmpmads = subtractOvscPedestalRowByRow(single_skip_image_stack[:,:,i])
            ped_subtracted_image_data_exposed_stack[:,:,i] = selectImageRegion(ped_subtracted_image_data_stack[:,:,i],'exposed_pixels')
            pedestals.append(np.mean(tmppedestals)); mads.append(np.mean(tmpmads))
            cut = [globalthreshold*mads[-1],maximumthreshold*mads[-1]] #cut on value
            clusters.append(clusterImage(ped_subtracted_image_data_exposed_stack[:,:,i],cut,mask=maskpath))
        #print(clusters)
    else:
        print('WARNING: Stacking average images for depth calibration study')
        print('Following image structure expected: N. rows columns skips ',nrows,ncolumns,nskips)
        single_skip_image_stack = reconstructAvgImageStack(nameprefix,lowerindex,upperindex)
        skipper_avg0 = single_skip_image_stack[:,:,0]
        mu_avg0, sigma_avg0 = sigmaFinder(skipper_avg0,debug=False)[1:3]
        if calibrate:
            parametersDCfit, reducedchisquared, offset = calibrationdc.calibrationDC(single_skip_image_stack[:,:,0], sigma_avg0, reverse, debug=False)
            calibrationconstant = parametersDCfit[0][5]; calibratedsigma = sigma_avg0/calibrationconstant
        else: calibrationconstant = calibrationguess; offset = mu_avg0
        ncolumnsexp = min(ncolumns,registersize)
        skipper_avg_exposed_stack = np.zeros((nrows, ncolumnsexp, nimages), dtype=np.float64)
        pedestals,sigmas,clusters,clustersenergy = [],[],[],[]
        for i in range(nimages): #clustering performed in ADU regardless of calibration
            mutmp, sigmatmp = sigmaFinder(single_skip_image_stack[:,:,i],debug=False)[1:3]
            single_skip_image_stack[:,:,i] = -int(reverse)*(single_skip_image_stack[:,:,i] - mutmp) #reverse+img_ped_sub
            skipper_avg_exposed_stack[:,:,i] = selectImageRegion(single_skip_image_stack[:,:,i],'exposed_pixels')
            pedestals.append(mutmp); sigmas.append(sigmatmp)
            cut = [globalthreshold*sigmas[-1],maximumthreshold*sigmas[-1]] #cut on value
            clusters.append(clusterImage(skipper_avg_exposed_stack[:,:,i],cut,mask=maskpath))
        for i in range(nimages): clustersenergy.extend([x * 0.00377/calibrationconstant for x in clusters[i][3]])
        #plt.hist(clustersenergy,10000,log=True)
        #plt.xlabel('energy (keV)')
        #plt.ylabel('entries')
        #plt.show()











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

if not (reportHeader or reportImage or reportCalibration or reportCluster or reportDepth): print('No information to be reported. Report will not be produced. Exiting'); sys.exit()

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report on Clusters and Depth Calibration'))
doc.preamble.append(Command('author', 'DAMIC-M'))
doc.append(NoEscape(r'\maketitle'))

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

#############################################
#imagetoprint for image and calib reports#
if nskips == 1: imagetoprint = image_data; sigmatoprint = sigmaFinder(selectImageRegion(image_data,'overscan'),debug=False)[2]
else: imagetoprint = -int(reverse)*(skipper_avg0 - offset)/calibrationconstant; sigmatoprint = sigma_avg0/calibrationconstant
#############################################
###############Image section#################
#############################################
if reportImage:
    clustercandidates = findChargedPixelNoBorder(imagetoprint,sigmatoprint)
    isChargedCrown = True; coor = np.size(imagetoprint,0)//2, np.size(imagetoprint,1)//2
    for coor in clustercandidates:
        isChargedCrown = chargedCrown(coor,imagetoprint,sigmatoprint)
        if (isChargedCrown):
            #print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
            break
    if not isChargedCrown: coor = np.size(imagetoprint,0)//2, np.size(imagetoprint,1)//2
    with doc.create(Section('Images')):
        if np.size(imagetoprint,0) > 80: halfrangey = 40
        else: halfrangey = np.size(imagetoprint,0)//2
        if np.size(imagetoprint,1) > 80: halfrangex = 40
        else: halfrangex = np.size(imagetoprint,1)//2
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],2*halfrangey-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],2*halfrangex-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
        fig=plt.figure(figsize=(8,8))
        
        ax1=fig.add_subplot(111)
        plt.imshow(imagetoprint[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title('Calibrated average image')
        plt.ylabel("row")
        cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
        cb1 = plt.colorbar(cax=cax1)
        
        fig.tight_layout(pad=.001)
    
    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.99\linewidth'))
        plot.add_caption('Exposed pixels region for average image (first of stack if multiple images). Colorbar in electrons and ADU for many skip and single skip images, respectively.')
    plt.clf()
    doc.append(NewPage())
        
        
#############################################
#########Calibrated image section############
#############################################
if nskips == 1: reportCalibrationDarkcurrent = False
if reportCalibration:
    skipperavgcalibrated = skipper_avg_cal.ravel()
    try:#if calibration went wrong skipperavgcalibratedravel could be empty because limits are out of range
        if calibrationconstant == calibrationguess: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -10 and  s < 10]
        else: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -2 and  s < 4]
        nbins=50*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
    except:#if so we keep skipperavgcalibratedravel without range
        skipperavgcalibratedravel = skipperavgcalibrated
        nbins=50*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
    if nbins == 0: nbins=100
    skipperavgcalibratedravelhist, binedges = np.histogram(skipperavgcalibratedravel, nbins, density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    npeaksp = 3
    if calibrate: dcpar = parametersDCfit[0][0], npeaksp, parametersDCfit[0][2]/(50/0.5), parametersDCfit[0][3]/calibrationconstant
    #dcparunc has one more component (the gain) than dcpar (dcpar is an argument for the calibrated gaussian)
    try: dcparunc = parametersDCfit[1][0], parametersDCfit[1][1], parametersDCfit[1][2]/(50/0.5), parametersDCfit[1][3]/calibrationconstant, parametersDCfit[1][5]; skipperavgcalibratedravelhistfit = convolutionGaussianPoisson(bincenters,*dcpar); plt.plot(bincenters, skipperavgcalibratedravelhistfit, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(reducedchisquared)), color='red')
    except Exception as e: dcparunc = 0,0,0,0,0; print(e)
    plt.hist(skipperavgcalibratedravel, len(bincenters), density = False, histtype='step', linewidth=2, log = False, color = 'teal', label='avg image calibrated pixel charge distribution')
    plt.legend(loc='upper right',prop={'size': 14})
    plt.xlabel('pixel value [e$^-$]')
    plt.ylabel('counts')
    plt.tick_params(axis='both', which='both', length=10, direction='in')
    plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
    #plt.setp(ax.get_yticklabels(), visible=True)
    
    with doc.create(Section('Calibration')):
        fig.tight_layout(pad=.001)
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.90\linewidth'))
            plot.add_caption('Calibrated pixel charge distribution.')
        if calibrate: calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+'±'+str(round_sig_2(dcparunc[4]))+' ADU per electron. In case of multiple images the first image is shown (and, if selected, calibrated).'
        else: calibrationline = 'Guess calibration constant is: '+str(round(calibrationconstant,4))+' ADU per electron. In case of multiple images the first image is shown (and, if selected, calibrated).'
        doc.append(calibrationline)
        plt.clf()
        doc.append(NewPage())

#############################################
#########Clustering plots section#####
#############################################
if reportCluster:
    lowerEbound = config['depth_calibration_analysis'][-1]['report'][-1]['clusters'][-1]['lower_energy_bound_keV']
    upperEbound = config['depth_calibration_analysis'][-1]['report'][-1]['clusters'][-1]['upper_energy_bound_keV']
    fig,axs = plt.subplots(1,1)
    axs.hist(clustersenergy, 500, color='teal')#, label='Image cluster energy distribution')
    axs.set_yscale('log')
    #axs.yaxis.set_major_locator(MultipleLocator( 10**(ceil(log(max(np.abs(fftdata))-min(np.abs(fftdata)),10))-1) ))
    axs.tick_params(axis='both', which='both', length=10, direction='in')
    axs.grid(color='grey', linestyle=':', linewidth=1, which='both')
    plt.setp(axs.get_yticklabels(), visible=True)
    axs.set_xlim([lowerEbound,upperEbound])
    axs.set_xlabel('energy (keV)')
    axs.set_ylabel('entries per ' +str(round((max(clustersenergy)-min(clustersenergy))/500,4))+' keV')
    axs.set_title('Image cluster energy distribution')
    
    #plt.xlabel('$\sigma$ (pixels)')
    with doc.create(Section('Clusters')):
            fig.tight_layout(pad=.001)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.90\linewidth'))
                plot.add_caption('Cluster energy distribution.')
            plt.clf()
            doc.append(NewPage())

#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure:
    if not multipleimages: reportname = 'reports/clusters_'+sys.argv[2]
    if multipleimages: reportname = 'reports/clusters_'+str(lowerindex)+'_'+str(upperindex)
else:
    if not multipleimages: reportname = 'clusters_'+sys.argv[2]
    if multipleimages: reportname = 'clusters_'+str(lowerindex)+'_'+str(upperindex)
doc.generate_pdf(reportname, clean_tex=False)
os.remove(reportname+'.tex')

end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')
