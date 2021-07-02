#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Executable devoted to assessing the charge transfer efficiency of the CCD.
It uses a single image: single- or multiple-skip to assess parallel/serial or skip charge transfer efficiency (CTE), respectively. For simplicity EPER is not carried out on multiskip images

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
kclthreshold = config['kcl_threshold']
calibrationguess = config['calibration_constant_guess']
printheader = False
clocksCTE = config['transfer_analysis'][-1]['clocks_CTE_EPER']
skipsCTE = config['transfer_analysis'][-1]['skip_CTE_Kcl']
reportHeader = config['transfer_analysis'][-1]['report'][-1]['header']
reportImage = config['transfer_analysis'][-1]['report'][-1]['image']
reportCTE = config['transfer_analysis'][-1]['report'][-1]['CTE_plots']
if clocksCTE or skipsCTE: pass
else: print('ERROR: You are running the code for charge transfer analysis but none of the corresponding tests is set to true. Please update config file. Exiting'); sys.exit()

if clocksCTE and skipsCTE:
    whichCTE = ''
    while whichCTE != 'clocksCTE' and whichCTE != 'skipsCTE':
        whichCTE = input("ERROR: You are running the code for charge transfer analysis with both tests are set to true. Please select one. Which one do you want to carry out?\nPlease answer 'clocksCTE' or 'skipsCTE': ")
        if whichCTE == 'clocksCTE': skipsCTE = False
        elif whichCTE == 'skipsCTE': clocksCTE = False
    

if test != 'chargetransfer':
    proceed = ''
    while proceed != 'yes' and proceed !='no':
        proceed = input("You are running the code for charge transfer analysis. Test selected in configuration file is different from 'chargetransfer': do you want to perform charge transfer analysis?\nPlease answer 'yes' or 'no': ")
        if proceed == 'no': sys.exit()
        elif proceed == 'yes': print('Proceeding with charge transfer analysis')

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

from m_functions import sigmaFinder,selectImageRegion, make_colorbar_with_padding, gauss, round_sig_2
from m_reconstruction import getSingleSkipImage,findChargedPixelNoBorder,chargedCrown,reconstructSkipperImage
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
##############################################################################
##############################################################################
##############################################################################
##############################################################################
# Start processing for charge transfer analysis
# if leach: image is fixed in reconstruction module
##############################################################################
#RECONSTRUCT SINGLE SKIP IMAGE FOR EXTENDED PIXEL EDGE RESPONSE###############
##############################################################################

if clocksCTE and nskips==1:
    warnings.filterwarnings("error")
    image_data = getSingleSkipImage(image_file)
    try: image_eper = selectImageRegion(image_data,'EPER')
    except: print('ERROR: Image has no overscan. Cannot estimate parallel/serial with EPER'); sys.exit()
    warnings.filterwarnings("default")
    
    columnsarray = np.arange(prescan+registersize-10,prescan+registersize+10)#np.size(image_data,1))
    meanoverrows = []
    for i in range(20): meanoverrows.append(np.mean(image_eper[nrows-1,i]))
    meanfirstovsccolumn = meanoverrows[10]
    meanlastexposedcolumn = meanoverrows[9]
    eperCTE = 1 - meanfirstovsccolumn/(registersize*meanlastexposedcolumn)
    print('Average charge in first overscan column: '+str(meanfirstovsccolumn)+' ADU')
    print('Average charge in last exposed column: '+str(meanlastexposedcolumn)+' ADU')
    print('EPER-estimated CTE is: ',eperCTE)
elif clocksCTE and nskips!=1:  print('ERROR: You are running the code for single-skip charge transfer analysis but image has multiple skips. Please review image or config file. Exiting'); sys.exit()


##############################################################################
#COMPUTE CHARGE LOSS COEFFICIENT FOR MANY SKIP IMAGE##########################
##############################################################################

if skipsCTE and nskips!=1:
    skipper_diff,skipper_diff_01 = reconstructSkipperImage(image_file,arg2)[4:6]
    diff_image_exposed_01,diff_image_exposed = selectImageRegion(skipper_diff_01,'exposed_pixels'),selectImageRegion(skipper_diff,'exposed_pixels')
    PCDDstudyparameters01 = m_chargeloss.firstLastSkipPCDDCheck(diff_image_exposed_01, debug=False)
    PCDDstudyparameters = m_chargeloss.firstLastSkipPCDDCheck(diff_image_exposed, debug=False) #skewness, skewness_unc, kcl, kcl_unc, amp, mu, std
    kclsignificance01,kclsignificance = PCDDstudyparameters01[2]/PCDDstudyparameters01[3],PCDDstudyparameters[2]/PCDDstudyparameters[3]
    if abs(kclsignificance01) or abs(kclsignificance) > 3: print('Kcl value(s) flag probable charge loss')
elif skipsCTE and nskips==1: print('ERROR: You are running the code for multiskip charge transfer analysis but image only has one skip. Please review image or config file. Exiting'); sys.exit()

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

if not (reportHeader or reportImage or reportCTE): print('No information to be reported. Report will not be produced. Exiting'); sys.exit()

from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
from scipy.optimize import curve_fit
#setup document parameters
geometry_options = {'right': '2cm', 'left': '2cm'}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(Command('title', 'Image Analysis Report on Charge Transfer'))
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
###############Image section#################
#############################################
if nskips!=1: reportImage=False
if reportImage:
    stddev = sigmaFinder(image_data,False)[2]
    clustercandidates = findChargedPixelNoBorder(image_data,stddev)
    isChargedCrown = True; coor = np.size(image_data,0)//2, np.size(image_data,1)//2
    for coor in clustercandidates:
        isChargedCrown = chargedCrown(coor,image_data,stddev)
        if (isChargedCrown):
            #print(str(coor)+' 3x3 or larger cluster center surrounded by > 10*sigma crown. Plotting image of its surroundings')
            break
    if not isChargedCrown: coor = np.size(image_data,0)//2, np.size(image_data,1)//2
    with doc.create(Section('Images')):
        if np.size(image_data,0) > 80: halfrangey = 40
        else: halfrangey = np.size(image_data,0)//2
        if np.size(image_data,1) > 80: halfrangex = 40
        else: halfrangex = np.size(image_data,1)//2
        if coor[0] > halfrangey: deltay = halfrangey,halfrangey
        else: deltay = coor[0],2*halfrangey-coor[0]
        if coor[1] > halfrangex: deltax = halfrangex,halfrangex
        else: deltax = coor[1],2*halfrangex-coor[1]
        plotrange = [coor[0]-deltay[0],coor[0]+deltay[1],coor[1]-deltax[0],coor[1]+deltax[1]]
        fig=plt.figure(figsize=(8,8))
        
        ax1=fig.add_subplot(111)
        plt.imshow(image_data[plotrange[0]:plotrange[1],plotrange[2]:plotrange[3]],cmap=plt.cm.jet,extent=(plotrange[2],plotrange[3],plotrange[1],plotrange[0]))
        plt.title('Single skip image for CTE')
        plt.ylabel("row")
        cax1=make_colorbar_with_padding(ax1) # add a colorbar within its own axis the same size as the image plot
        cb1 = plt.colorbar(cax=cax1)

        fig.tight_layout(pad=.001)
    
    with doc.create(Figure(position='htb!')) as plot:
        plot.add_plot(width=NoEscape(r'0.99\linewidth'))
        plot.add_caption('Exposed pixels region for single skip image.')
    plt.clf()
    doc.append(NewPage())
    '''

#############################################
#########Calibrated image section############
#############################################
if multipleimages and (not measVSexp_e_multimg): reportCalibrationDarkcurrent = False
if reportCalibrationDarkcurrent:
    if not multipleimages: skipperavgcalibrated = skipper_avg_cal.ravel()
    if measVSexp_e_multimg: skipperavgcalibrated = avgimagestack_cal[:,:,0]
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
        '''
#############################################
#############CTE plots section###############
#############################################
if reportCTE:
    if clocksCTE:
        fig, axs = plt.subplots(2, 1, figsize=(7,6), sharey=False, tight_layout=True)
        
        for i in range(nrows-1): axs[0].plot(columnsarray, image_eper[i,:], linewidth=2, marker = 'o', color = 'red', linestyle = 'None')
        axs[0].plot(columnsarray, image_eper[nrows-1,:], linewidth=2, marker = 'o', color = 'red', linestyle = 'None', label = 'Row pixel charge')
        axs[0].plot(columnsarray, meanoverrows, linewidth=2, marker = 'o', color = 'teal', linestyle = 'None', label = 'Mean row pixel charge')
        axs[0].axvline(x=prescan+registersize,linestyle=':',color='red',label='First overscan column')
        axs[0].legend(loc='upper right',prop={'size': 14})
        axs[0].tick_params(axis='both', which='both', length=10, direction='in')
        axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[0].get_yticklabels(), visible=True)
        axs[0].set_title('Serial EPER')
        
        for i in range(nrows-1): axs[1].plot(columnsarray[9:-9], image_eper[i,9:-9], linewidth=2, marker = 'o', color = 'red', linestyle = 'None')
        axs[1].axvline(x=prescan+registersize,linestyle=':',color='red',label='First overscan column')
        axs[1].plot(columnsarray[9:-9], image_eper[nrows-1,9:-9], linewidth=2, marker = 'o', color = 'red', linestyle = 'None', label = 'Row pixel charge')
        axs[1].plot(columnsarray[9:-9], meanoverrows[9:-9], linewidth=2, marker = 'o', color = 'teal', linestyle = 'None', label = 'Mean row pixel charge')
        axs[1].legend(loc='upper center',prop={'size': 14})
        axs[1].tick_params(axis='both', which='both', length=10, direction='in')
        axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
        plt.setp(axs[1].get_yticklabels(), visible=True)
        axs[1].set_yscale('log')
        axs[1].set_title('Serial EPER')
        
        for axs in axs.flat:
            axs.set(xlabel='column number', ylabel='pixel value [ADU]')
        with doc.create(Section('Charge Transfer Efficiency')):
            fig.tight_layout(pad=.001)
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('EPER rows pixel charge values (mean and all rows) as function of column number.')
            plt.clf()
            epercteline = 'Serial CTE through EPER: CTE = '+ str(round(eperCTE,8))+'.'
            epercteline += ' Mean last exposed column row pixel value: '+str(meanlastexposedcolumn)+' ADU.'
            epercteline += ' Mean first overscan column row pixel value: '+str(meanfirstovsccolumn)+' ADU.'
            doc.append(epercteline)
            doc.append(NewPage())
            
    elif skipsCTE:
        with doc.create(Section('Charge-loss')):
            skewnessPCDD01, skewnessPCDDuncertainty01, kclPCDD01, kclPCDDuncertainty01, ampPCDD01, muPCDD01, stdPCDD01 = PCDDstudyparameters01
            skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, ampPCDD, muPCDD, stdPCDD = PCDDstudyparameters
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
            skipperdiffexposedravelled01 = diff_image_exposed_01.ravel()
            skipperdiffexposedravelled = diff_image_exposed.ravel()
            
            skipperdiffexposedravelledinrange01 = [s for s in skipperdiffexposedravelled01 if s > muPCDD01 - 3*stdPCDD01 and s < muPCDD01 + 3*stdPCDD01 and s != 0]
            numbins = int(max(skipperdiffexposedravelledinrange01) - min(skipperdiffexposedravelledinrange01))
            skipperdiffexposedravelledinrangehist01, binedges = np.histogram(skipperdiffexposedravelledinrange01, numbins, density=False)
            bincenters=(binedges[:-1] + binedges[1:])/2
            pguess = [ampPCDD01,muPCDD01,stdPCDD01]
            try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffexposedravelledinrangehist01, p0=pguess); PCDDhistfit01 = gauss(bincenters,*pfit)
            except: pfit = pguess; PCDDhistfit01 = gauss(bincenters,*pfit)
            axs[0].plot(bincenters, PCDDhistfit01, label='gaussian fit curve', linewidth=1, color='red')
            axs[0].hist(skipperdiffexposedravelledinrange01, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
            #axs[0].plot(bincenters,skipperdiffexposedravelledinrangehist01, label='pixel charge difference distribution', color='teal')
            axs[0].legend(loc='upper right',prop={'size': 14})
            axs[0].set_yscale('linear')
            axs[0].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            axs[0].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
            skipperdiffexposedravelledinrange = [s for s in skipperdiffexposedravelled if s > muPCDD - 3*stdPCDD and s < muPCDD + 3*stdPCDD and s != 0]
            numbins = int(max(skipperdiffexposedravelledinrange) - min(skipperdiffexposedravelledinrange))
            skipperdiffexposedravelledinrangehist, binedges = np.histogram(skipperdiffexposedravelledinrange, numbins, density=False)
            bincenters=(binedges[:-1] + binedges[1:])/2
            pguess = [ampPCDD,muPCDD,stdPCDD]
            try: pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffexposedravelledinrangehist, p0=pguess); PCDDhistfit = gauss(bincenters,*pfit)
            except: pfit = pguess; PCDDhistfit = gauss(bincenters,*pfit)
            axs[1].plot(bincenters, PCDDhistfit, label='gaussian fit curve', linewidth=1, color='red')
            axs[1].hist(skipperdiffexposedravelledinrange, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
                #axs[1].plot(bincenters,skipperdiffexposedravelledinrangehist, label='pixel charge difference distribution', color='teal')
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
            fitjusticationlineline = "NOTE: A good gaussian fit of pcdd's is essential for Kcl to be an effective charge loss classifier"
            doc.append(fitjusticationlineline)
            plt.clf()
            doc.append(NewPage())
            
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
            centeredskipperdiffexposed01 = [s for s in skipperdiffexposedravelled01-muPCDD01 if s != -muPCDD01]
            axs[0].hist(centeredskipperdiffexposed01, 600, range = (-20*stdPCDD01,10*stdPCDD01), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
            axs[0].legend(loc='upper right',prop={'size': 14})
            axs[0].tick_params(axis='both', which='both', length=10, direction='in')
            axs[0].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            axs[0].set_title('$k_{cl}~=~$' + str(round(kclPCDD01,4)) + '$\pm$'+ str(round(kclPCDDuncertainty01,4)) + ', $S(k_{cl})~=~$' + str(round(kclPCDD01/kclPCDDuncertainty01,4)) + ', skewness = ' + str(round(skewnessPCDD01,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty01,4)))
        
            centeredskipperdiffexposed = [s for s in skipperdiffexposedravelled-muPCDD if s != -muPCDD]
            axs[1].hist(centeredskipperdiffexposed, 600, range = (-20*stdPCDD,10*stdPCDD), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='centered pixel charge difference distribution')
            axs[1].legend(loc='upper right',prop={'size': 14})
            axs[1].tick_params(axis='both', which='both', length=10, direction='in')
            axs[1].grid(color='grey', linestyle=':', linewidth=1, which='both')
            plt.setp(axs[0].get_yticklabels(), visible=True)
            axs[1].set_title('$k_{cl}~=~$' + str(round(kclPCDD,4)) + '$\pm$'+ str(round(kclPCDDuncertainty,4)) + ', $S(k_{cl})~=~$' + str(round(kclPCDD/kclPCDDuncertainty,4)) + ', skewness = ' + str(round(skewnessPCDD,4)) + '$\pm$'+ str(round(skewnessPCDDuncertainty,4)))
            
            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Pedestal-subtracted full-image PCDDs: first and second skip (top) and second and end skip (bottom).')
                from scipy.stats import norm
                doc.append('First-second skip lower tail entries: '+str(len([s for s in centeredskipperdiffexposed01 if s < -kclthreshold*stdPCDD01]))+'. First-second skip upper tail entries: '+str(len([s for s in centeredskipperdiffexposed01 if s > kclthreshold*stdPCDD01]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffexposed01)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.\n Second-last skip lower tail entries: '+str(len([s for s in centeredskipperdiffexposed if s < -kclthreshold*stdPCDD]))+'. Second-last skip upper tail entries: '+str(len([s for s in centeredskipperdiffexposed if s > kclthreshold*stdPCDD]))+'. Both expected to be '+ str( int(round_sig_2( len(centeredskipperdiffexposed)*norm(loc = 0, scale = 1).cdf(-kclthreshold))) )+'.')
                
            plt.clf()
            doc.append(NewPage())
        
#############################################
#############Produce Report PDF##############
#############################################
import os
if default_directory_structure:
    reportname = 'reports/transfer_'+sys.argv[2]
else:
    reportname = 'transfer_'+sys.argv[2]
doc.generate_pdf(reportname, clean_tex=False)
os.remove(reportname+'.tex')

end = time.perf_counter()
print('Code execution took ' + str(round((end-start),4)) + ' seconds')

