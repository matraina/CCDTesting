#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
-------------------

*By: Michelangelo Traina

This module is devoted to assessing the linearity of the signal.
It can use one single image with high exposure, but also several images (accumulate statistics with cumulatePCDistributions method in reconstruction.py)

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
test = config['test']
workingdirectory = config['working_directory']
default_directory_structure = config['raw_processed_header_reports_dir_structure']
iskipstart = config['skip_start']
iskipend = config['skip_end']
fixLeachReco = config['fix_leach_reconstruction']
reverse = config['reverse']
registersize = config['ccd_register_size']
analysisregion = 'full'
calibrationguess = config['calibration_constant_guess']
printheader = False
calibrate = config['linearity_analysis'][-1]['calibrate']
multipleimages = config['linearity_analysis'][-1]['multiple_images'][-1]['use_multiple_images']
maxelectrons = config['linearity_analysis'][-1]['max_electrons']
reportHeader = config['linearity_analysis'][-1]['report'][-1]['header']
reportImage = config['linearity_analysis'][-1]['report'][-1]['image']
reportCalibrationDarkcurrent = config['linearity_analysis'][-1]['report'][-1]['calibration_darkcurrent']
reportLinearityCurves = config['linearity_analysis'][-1]['report'][-1]['linearity_curves']

if test != 'linearity':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for linearity analysis. Test selected in configuration file is different from 'linearity': do you want to perform linearity analysis? Please answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with linearity analysis')

import time
start = time.perf_counter()

if default_directory_structure:
    arg1 = 'raw/' + arg1
    arg2 = 'processed/' + arg2
    
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

from reconstruction import getAverageSkipperImage, reconstructAvgImageStack, cumulatePCDistributions
from functions import sigmaFinder, convolutionGaussianPoisson, round_sig_2, linefunction
import calibrationdc

##############################################################################
# Specify path (can be out of the main tree)

import os
os.chdir(workingdirectory)

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
    if __name__ == '__main__': print('N. rows columns skips ',nrows,ncolumns,nskips)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Start processing for linearity test
# if leach: image is fixed in reconstruction module
##############################################################################
#AVERAGE SKIPPER IMGS FROM ISKIPSTART TO ISKIPEND#############################
##############################################################################

if not multipleimages:
    skipper_avg0 = getAverageSkipperImage(image_file)
    offset, avg0_std = sigmaFinder(skipper_avg0, debug = False)[1:3]

##############################################################################
#CUMULATE STATISTICS FROM MANY SAME-PARAMETER AVERAGE IMAGES #################
##############################################################################

if multipleimages:
    lowerindex = config['linearity_analysis'][-1]['multiple_images'][-1]['lower_index']
    upperindex = config['linearity_analysis'][-1]['multiple_images'][-1]['upper_index']
    print('I am going to cumulate statistics from multiple images for linearity test')
    nameprefix = ''.join([i for i in arg1 if not i.isdigit()]).replace('.fits','')
    avgimagestack = reconstructAvgImageStack(nameprefix,lowerindex,upperindex)
    offset, avg0_std = sigmaFinder(avgimagestack[:,:,0], debug = False)[1:3]

##############################################################################
#ADU TO e- CALIBRATION AND DARK CURRENT ESTIMATES#############################
##############################################################################

if calibrate:
    if not multipleimages:
        parametersDCfit, reducedchisquared, offset = calibrationdc.calibrationDC(skipper_avg0, avg0_std, reverse, debug=False)
        calibrationconstant = parametersDCfit[0][5]; calibratedsigma = parametersDCfit[0][3]/calibrationconstant
        skipper_avg_cal = -int(reverse)*(skipper_avg0 - offset)/calibrationconstant
    if multipleimages:
        parametersDCfit, reducedchisquared, offset = calibrationdc.calibrationDC(avgimagestack[:,:,0], avg0_std, reverse, debug=False)
        calibrationconstant = parametersDCfit[0][5]; calibratedsigma = parametersDCfit[0][3]/calibrationconstant
        avgimagestack_cal = -int(reverse)*(avgimagestack - offset)/calibrationconstant
if not calibrate:
    calibrationconstant = calibrationguess; calibratedsigma = avg0_std/calibrationconstant; print('WARNING: using calibration constant guess for linearity test')
    if not multipleimages: skipper_avg_cal = -int(reverse)*(skipper_avg0 - offset)/calibrationconstant
    if multipleimages: avgimagestack_cal = -int(reverse)*(avgimagestack - offset)/calibrationconstant
    
if not multipleimages: skipper_avg_cal_ravelled = skipper_avg_cal.ravel()
if multipleimages: skipper_avg_cal_ravelled = cumulatePCDistributions(avgimagestack_cal)

##############################################################################
#CHECK LINEARITY UP TO MAX ELECTRONS #########################################
##############################################################################

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

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Report on Linearity'))
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
#########Calibrated image section############
#############################################
if reportCalibrationDarkcurrent:
    if not multipleimages: skipperavgcalibrated = skipper_avg_cal.ravel()
    if multipleimages: skipperavgcalibrated = avgimagestack_cal[:,:,0]
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
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Calibrated pixel charge distribution.')
        if calibrate: calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+'±'+str(round_sig_2(dcparunc[4]))+' ADU per electron. In case of multiple images the first image is shown (and, if selected, calibrated).'
        else: calibrationline = 'Guess calibration constant is: '+str(round(calibrationconstant,4))+' ADU per electron. In case of multiple images the first image is shown (and, if selected, calibrated).'
        doc.append(calibrationline)
        plt.clf()
        doc.append(NewPage())
        
#############################################
#########Linearity curves section############
#############################################
if reportLinearityCurves and maxelectrons>=0:
    nelectrons = np.arange(0,maxelectrons+1,1)
    import warnings
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
    if multipleimages: axs.set_yscale('log')
    #print(nelectrons)
    #print(peakmus)
    #print(peakmuncs)
    if not multipleimages:
        resolution = plt.errorbar(nelectrons,peakmus,peakmuncs,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measurements')
        if fit: resolution = plt.plot(nelectrons,linefunction(nelectrons,pfit[0],pfit[1]),'k--',color='red',label='measurements weighed fit    line: '+str(round_sig_2(pfit[0]))+'+'+str(round(pfit[1],4))+'$\cdot N_e$')
        resolution = plt.plot(nelectrons,nelectrons,'k:',label='perfect linearity')
    if multipleimages:
        resolution = plt.errorbar(nelectrons[1:],peakmus[1:],peakmuncs[1:],xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measurements')
        if fit: resolution = plt.plot(nelectrons[1:],linefunction(nelectrons[1:],pfit[0],pfit[1]),'k--',color='red',label='measurements weighed fit    line: '+str(round_sig_2(pfit[0]))+'+'+str(round(pfit[1],4))+'$\cdot N_e$')
        resolution = plt.plot(nelectrons,nelectrons,'k:',label='perfect linearity')
    plt.legend(loc='upper left',prop={'size': 14})
    plt.ylabel('measured number of electrons [$e^-$]')
    plt.xlabel('expected number of electrons [$e^-$]')
    plt.tick_params(axis='both', which='both', length=10, direction='in')
    plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
    plt.title('Linearity study')
    with doc.create(Section('Linearity study')):
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('.')
        calibrationline2 = 'Corrected value of calibration constant '
        if fit: calibrationline2 += 'is: '+str(round(pfit[1]*calibrationconstant,4)); calibrationline2 += ' ADU per electron'
        if not fit: calibrationline2 += 'could not be estimated'
        doc.append(calibrationline2)
        plt.clf()
        doc.append(NewPage())

if reportLinearityCurves and maxelectrons < 0: print('Linearity curves plots not produced: 0 points to plot. Check PCDs')
    
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
doc.generate_pdf(reportname, clean_tex=False)
os.remove(reportname+'.tex')

end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')

