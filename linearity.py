#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Executable devoted to assessing the linearity of the signal.
It can use one single image with high exposure, but also several images (check that (0-e) peak std dev increases with mean & accumulate statistics with cumulatePCDistributions method in reconstruction.py)

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
registersize = config['ccd_active_register_size']
prescan = config['prescan']
overscan = config['overscan']
analysisregion = 'full'
calibrationguess = config['calibration_constant_guess']
printheader = False
calibrate = config['linearity_analysis'][-1]['calibrate']
multipleimages = config['linearity_analysis'][-1]['multiple_images'][-1]['use_multiple_images']
measVSexp_e = config['linearity_analysis'][-1]['multiple_images'][-1]['measured_vs_expected_e']
transfercurve = config['linearity_analysis'][-1]['multiple_images'][-1]['transfer_curve']
if not multipleimages: measVSexp_e = False; transfercurve = False
elif (not measVSexp_e) and (not transfercurve): print('ERROR: You have selected multiple images analysis but none of the corresponding tests is set to true. Please update config file. Exiting'); sys.exit()
maxelectrons = config['linearity_analysis'][-1]['max_electrons']
reportHeader = config['linearity_analysis'][-1]['report'][-1]['header']
reportImage = config['linearity_analysis'][-1]['report'][-1]['image']
reportCalibrationDarkcurrent = config['linearity_analysis'][-1]['report'][-1]['calibration']
reportLinearityCurves = config['linearity_analysis'][-1]['report'][-1]['linearity_curves']

if test != 'linearity':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for linearity analysis. Test selected in configuration file is different from 'linearity': do you want to perform linearity analysis?\nPlease answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with linearity analysis')

import time
start = time.perf_counter()

if default_directory_structure:
    arg1 = 'raw/' + arg1
    arg2 = 'processed/' + arg2
    
if multipleimages:
    arg1 = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    arg2 = ''.join([i for i in arg2 if not i.isdigit()]).replace('.fits','')
    
##############################################################################
# Get Numpy and Scipy

import numpy as np
from scipy.optimize import curve_fit

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

from m_reconstruction import getAverageSkipperImage, reconstructAvgImageStack, cumulatePCDistributions, getADUMeansStds, findChargedPixelNoBorder, chargedCrown
from m_functions import sigmaFinder, convolutionGaussianPoisson, round_sig_2, linefunction, make_colorbar_with_padding
import m_calibrationdc

##############################################################################
# Specify path (can be out of the main tree)

import os
os.chdir(workingdirectory)

##############################################################################
# Import warnings for warning control

import warnings

##############################################################################
# WARN about image structure requirement
proceed = ''
while proceed != 'yes' and proceed !='no':
    proceed = input("You are running the code for linearity analysis. This code assumes resolved single electrons for input images (except when comparing means and std deviations of 0-e peaks). If this is not the case code may fail.\nDo you want to continue? Please answer 'yes' or 'no': ")
    if proceed == 'no': sys.exit()
    elif proceed == 'yes': print('Proceeding with image(s) linearity analysis')

##############################################################################
# Open the data image

if not multipleimages:
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
    ##############################################################################
    ##############################################################################
    ##############################################################################
    ##############################################################################
    # Start processing for linearity analysis
    # if leach: image is fixed in reconstruction module
    ##############################################################################
    #SINGLE IMAGE LINEARITY STUDY: MEASURED VS EXPECTED N ELECTRONS###############
    ##############################################################################
    skipper_avg0 = getAverageSkipperImage(image_file)
    offset, avg0_std = sigmaFinder(skipper_avg0, debug = False)[1:3]
    if calibrate:
        parametersDCfit, reducedchisquared, offset = m_calibrationdc.calibrationDC(skipper_avg0, avg0_std, reverse, debug=False)
        calibrationconstant = parametersDCfit[0][5]; calibratedsigma = parametersDCfit[0][3]/calibrationconstant
        skipper_avg_cal = reversign*(skipper_avg0 - offset)/calibrationconstant
    else:
        calibrationconstant = calibrationguess; calibratedsigma = avg0_std/calibrationconstant; print('WARNING: using calibration constant guess for linearity test')
        skipper_avg_cal = reversign*(skipper_avg0 - offset)/calibrationconstant
    skipper_avg_cal_ravelled = skipper_avg_cal.ravel()
    peakmus,peakstds,peakmuncs,peakstduncs = [],[],[],[]
    for npeakelectron in range(maxelectrons+1):
        npeakarray = [s for s in skipper_avg_cal_ravelled if s > npeakelectron - 3*calibratedsigma and s < npeakelectron + 3*calibratedsigma]
        if len(npeakarray) == 0: maxelectrons = npeakelectron - 1; break
        tmpmu, tmpstd, tmpmunc, tmpstdunc = sigmaFinder(npeakarray, debug = False)[1:5]
        #print(tmpmu, tmpstd)
        peakmus.append(tmpmu); peakstds.append(tmpstd); peakmuncs.append(tmpmunc); peakstduncs.append(tmpstdunc)

##############################################################################
#MULTIPLE IMAGES LINEARITY STUDY: STDDEVS VS MEANS & MEAS VS EXP NE###########
##############################################################################

if multipleimages:
    lowerindex = config['linearity_analysis'][-1]['multiple_images'][-1]['lower_index']
    upperindex = config['linearity_analysis'][-1]['multiple_images'][-1]['upper_index']
    nameprefix = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    hdr = fits.getheader(nameprefix+str(lowerindex)+'.fits',0)
    nskips = hdr['NDCMS']  # n of skips
    avgimagestack = reconstructAvgImageStack(nameprefix,lowerindex,upperindex)
    offset, avg0_std = sigmaFinder(avgimagestack[:,:,0], debug = False)[1:3]
    if transfercurve:
        print('I am going to compute the photon transfer curve for the selected images')
        means,stddevs,meansunc,stddevsunc = getADUMeansStds(avgimagestack,lowerindex,upperindex)
    if calibrate:
        parametersDCfit, reducedchisquared, offset = m_calibrationdc.calibrationDC(avgimagestack[:,:,0], avg0_std, reverse, debug=False)
        calibrationconstant = parametersDCfit[0][5]; calibratedsigma = parametersDCfit[0][3]/calibrationconstant
        avgimagestack_cal = reversign*(avgimagestack - offset)/calibrationconstant; skipper_avg_cal = avgimagestack_cal[:,:,0]
    else:
        calibrationconstant = calibrationguess; calibratedsigma = avg0_std/calibrationconstant; print('WARNING: using calibration constant guess for linearity test')
        avgimagestack_cal = reversign*(avgimagestack - offset)/calibrationconstant; skipper_avg_cal = avgimagestack_cal[:,:,0]
    if measVSexp_e:
        print('I am going to cumulate statistics from multiple images for linearity test'); skipper_avg_cal_ravelled = cumulatePCDistributions(avgimagestack_cal)
        peakmus,peakstds,peakmuncs,peakstduncs = [],[],[],[]
        for npeakelectron in range(maxelectrons+1):
            npeakarray = [s for s in skipper_avg_cal_ravelled if s > npeakelectron - 3*calibratedsigma and s < npeakelectron + 3*calibratedsigma]
            if len(npeakarray) == 0: maxelectrons = npeakelectron - 1; break
            tmpmu, tmpstd, tmpmunc, tmpstdunc = sigmaFinder(npeakarray, debug = False)[1:5]
            #print(tmpmu, tmpstd)
            peakmus.append(tmpmu); peakstds.append(tmpstd); peakmuncs.append(tmpmunc); peakstduncs.append(tmpstdunc)

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

if not (reportHeader or reportImage or reportCalibrationDarkcurrent or reportLinearityCurves): print('No information to be reported. Report will not be produced. Exiting'); sys.exit()

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report on Linearity'))
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
        

#############################################
###############Image section#################
#############################################
if reportImage:
    clustercandidates = findChargedPixelNoBorder(skipper_avg_cal,avg0_std/calibrationconstant)
    isChargedCrown = True; coor = np.size(skipper_avg_cal,0)//2, np.size(skipper_avg_cal,1)//2
    for coor in clustercandidates:
        isChargedCrown = chargedCrown(coor,skipper_avg_cal,avg0_std/calibrationconstant)
        if (isChargedCrown):
            #print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
            break
    if not isChargedCrown: coor = np.size(skipper_avg_cal,0)//2, np.size(skipper_avg_cal,1)//2
    with doc.create(Section('Images')):
        if np.size(skipper_avg_cal,0) > 80: halfrangey = 40
        else: halfrangey = np.size(skipper_avg_cal,0)//2
        if np.size(skipper_avg_cal,1) > 80: halfrangex = 40
        else: halfrangex = np.size(skipper_avg_cal,1)//2
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],2*halfrangey-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],2*halfrangex-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
        fig=plt.figure(figsize=(8,8))
        
        ax1=fig.add_subplot(111)
        plt.imshow(skipper_avg_cal[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title('Calibrated average image')
        plt.ylabel("row")
        cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
        cb1 = plt.colorbar(cax=cax1)
        
        fig.tight_layout(pad=.001)
    
    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.99\linewidth'))
        plot.add_caption('Exposed pixels region for average image (first of stack if multiple images).')
    plt.clf()
    doc.append(NewPage())
        
        
#############################################
#########Calibrated image section############
#############################################
if multipleimages and (not measVSexp_e): reportCalibrationDarkcurrent = False
if reportCalibrationDarkcurrent:
    if not multipleimages: skipperavgcalibrated = skipper_avg_cal.ravel()
    if measVSexp_e: skipperavgcalibrated = avgimagestack_cal[:,:,0]
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
    except: dcparunc = 0,0,0,0,0
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
#########Linearity curves section############
#############################################
if reportLinearityCurves:
    if maxelectrons>=0:
        if (not multipleimages) or measVSexp_e:
            nelectrons = np.arange(0,maxelectrons+1,1)
            warnings.filterwarnings("error")
            fit = True
            try: pfit, varmatrix = curve_fit(linefunction, nelectrons, peakmus, sigma=peakmuncs, absolute_sigma=True); punc = np.sqrt(np.diag(varmatrix))
            except:
                try: pfit, varmatrix = curve_fit(linefunction, nelectrons, peakmus); punc = np.sqrt(np.diag(varmatrix))
                except: fit = False; print('Linear fit failed. Reporting measurement without fit line')
            #print('Fit parameters:',pfit)
            #print('Fit parameters uncertainties:',punc)
            #print('Fit-corrected calibration constant:',pfit[1]*calibrationconstant)
            fig, axs = plt.subplots(1, 1, figsize=(8,6), sharey=True, tight_layout=True)
            from matplotlib.ticker import MaxNLocator
            axs.xaxis.set_major_locator(MaxNLocator(integer=True))
            if not multipleimages: axs.yaxis.set_major_locator(MaxNLocator(integer=True))
            #if multipleimages: axs.set_yscale('log')
            #print(nelectrons)
            #print(peakmus)
            #print(peakmuncs)
            if not multipleimages:
                linecomparison = plt.errorbar(nelectrons,peakmus,peakmuncs,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measurements')
                if fit: linecomparison = plt.plot(nelectrons,linefunction(nelectrons,pfit[0],pfit[1]),'k--',color='red',label='measurements weighed fit    line: '+str(round_sig_2(pfit[0]))+'+'+str(round(pfit[1],4))+'$\cdot N_e$')
                linecomparison = plt.plot(nelectrons,nelectrons,'k:',label='perfect linearity')
            if multipleimages:
                linecomparison = plt.errorbar(nelectrons,peakmus,peakmuncs,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measurements')
                if fit: linecomparison = plt.plot(nelectrons,linefunction(nelectrons,pfit[0],pfit[1]),'k--',color='red',label='measurements weighed fit    line: '+str(round_sig_2(pfit[0]))+'+'+str(round(pfit[1],4))+'$\cdot N_e$')
                linecomparison = plt.plot(nelectrons,nelectrons,'k:',label='perfect linearity')
            plt.legend(loc='upper left',prop={'size': 14})
            plt.ylabel('measured number of electrons [$e^-$]')
            plt.xlabel('expected number of electrons [$e^-$]')
            plt.tick_params(axis='both', which='both', length=10, direction='in')
            plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.title('Linearity study - measured vs expected $N_e$')
            with doc.create(Section('Linearity study')):
                fig.tight_layout(pad=.001)
                with doc.create(Figure(position='htb!')) as plot:
                    plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                    plot.add_caption('Measured electrons (gauss fit mean) vs expected electrons (peak number in sequence).')
                calibrationline2 = 'Corrected value of calibration constant '
                if fit: calibrationline2 += 'is: '+str(round(pfit[1]*calibrationconstant,4)); calibrationline2 += ' ADU per electron'
                if not fit: calibrationline2 += 'could not be estimated'
                doc.append(calibrationline2)
                plt.clf()
                doc.append(NewPage())
    if reportLinearityCurves and maxelectrons < 0: print('Linearity curves plots not produced: 0 points to plot. Check PCDs')
        
    if transfercurve:
        try:
        #sttdevsindices = np.argsort(means); means = np.sort(means); stddevs = stddevs[sttdevsindices]; sttdevsunc = stddevsunc[sttdevsindices]
            pfit, varmatrix = curve_fit(linefunction, means, stddevs, sigma=stddevsunc, absolute_sigma=True); punc = np.sqrt(np.diag(varmatrix))
            plt.plot(np.array(means),linefunction(np.array(means),pfit[0],pfit[1]),'k--',color='red',label='measurements weighed fit line: '+str(round_sig_2(pfit[0]))+'+'+str(round(pfit[1],4))+'$\cdot \mu_{0_e}$')
        except: pass
        stdvsmean = plt.errorbar(means,stddevs,stddevsunc,xerr=meansunc,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measurements')
        plt.legend(loc='upper left',prop={'size': 14})
        plt.ylabel('standard deviation of flat exposure peak [ADU]')
        plt.xlabel('mean (centroid) of flat exposure peak [ADU]')
        plt.tick_params(axis='both', which='both', length=10, direction='in')
        plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.title('Linearity study - photon transfer curve')
        with doc.create(Figure(position='htb!')) as plot:
            fig.tight_layout(pad=.001)
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Flat peaks std deviations vs means.')
        plt.clf()
        doc.append(NewPage())
    
#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure:
    if not multipleimages: reportname = 'reports/linearity_'+sys.argv[2]
    if multipleimages: reportname = 'reports/linearity_'+str(lowerindex)+'_'+str(upperindex)
else:
    if not multipleimages: reportname = 'linearity_'+sys.argv[2]
    if multipleimages: reportname = 'linearity_'+str(lowerindex)+'_'+str(upperindex)

doc.generate_pdf(reportname, clean_tex=True)

end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')

