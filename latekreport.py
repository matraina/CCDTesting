#function(s) to produce latex report with image quality information and plots

import sys
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
import functions
from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
import matplotlib.pyplot as plt  # noqa

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
    
def factorial(n):
    facto=1
    for f in range(2,n+1): facto*=f
    return facto
        
def convolutionGaussianPoisson(q, *p):
    dcratep, npeaksp, amplip, sigmap = p
    f = 0
    npeaksp = 3
    for peaks in range(npeaksp):
        f +=  ( (dcratep**peaks * np.exp(-dcratep) / factorial(peaks)) * (amplip / np.sqrt(2 * np.pi * sigmap**2)) * np.exp( - (q - peaks)**2 / (2 * sigmap**2)) )
    return f

import json
with open('config.json') as config_file:
    config = json.load(config_file)
registersize = config['ccd_register_size']
analysisregion = config['analysis_region']
reportHeader = config['report'][-1]['header']
reportPCD = config['report'][-1]['pcds']
reportChargeLoss = config['report'][-1]['chargeloss']
reportCalibrationDarkcurrent = config['report'][-1]['calibration_darkcurrent']
reportFFTskips = config['report'][-1]['fft_skips']
reportFFTrow = config['report'][-1]['fft_row']

from math import log10, floor
def round_sig_2(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def produceReport(image_file, image_data, skipper_image0, skipper_avg0, mufs, stdfs, stduncfs, mumanyskip, stdmanyskip, stduncmanyskip, skipperdiffcore, mudiff, stddiff, skew, skewuncertainty, kcl, kcluncertainty, offset, redchisquared, skipper_avg_cal, dcestimate2, *parametersDCfit):
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
        fileheader = open(sys.argv[3].replace('processed','header') + '.txt', 'r')
        lines = fileheader.readlines()
        with doc.create(Section('Image Acquisition Parameters')):
            with doc.create(Description()) as desc:
                for line in lines[0:70]:
                    if line.split()[0]!='COMMENT': desc.add_item(line,'')
                    #desc.add_item(line.split()[0].replace('=','')+'='+line.split()[-1],'')
                    if line.split()[0]=='MREAD': break
    doc.append(NewPage())
    
    #############################################
    #Pixel charge distribution and noise section#
    #############################################
    if reportPCD:
        with doc.create(Section('Pixel Charge Distributions and Noise')):
            import functions
            
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=True, tight_layout=True)
            calibrationconstant = parametersDCfit[0][5]
            
            skipper_image0 = functions.selectImageRegion(skipper_image0,analysisregion)
            skipper_image0ravel = skipper_image0.ravel()
            #skipper_image = [s for s in skipper_image0ravel if s != 0]
            #instead of removing 0-entries from histogram use numpy mask to avoid discrepancies between gaussian and plotted PCD skipper_image0ravel
            skipper_image_unsaturated = np.ma.masked_equal(skipper_image0ravel, 0.0, copy=False)
            skipper_imagehist, binedges = np.histogram(skipper_image_unsaturated, bins = 800, density=False)
            ampfs = skipper_imagehist[np.argmax(skipper_imagehist)]
            axs[0].hist(skipper_image0ravel, 800, density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='start skip pixel charge distribution')
            bincenters = np.arange(mufs - 3*stdfs, mufs + 3*stdfs + 6*stdfs/100, 6*stdfs/100) #last term in upper bound to get ~sym drawing
            axs[0].plot(bincenters, gauss(bincenters,ampfs,mufs,stdfs), label='gaussian fit curve', linewidth=1, color='red')
            axs[0].legend(loc='upper left',prop={'size': 14})
            axs[0].set_title('First skip pixel charge distribution: $\sigma_{0e^-}~=~$ ' + str(round(stdfs,4)) + ' ADU; estimated noise: ' + str(round(stdfs/calibrationconstant,4)) + ' $e^{-}$')
            
            correctoffset = functions.sigmaFinder(skipper_avg0, debug=False)[1]
            skipper_avg0 = functions.selectImageRegion(skipper_avg0,analysisregion)
            avg_image_0ravel = skipper_avg0.ravel()
            avg_image_unsaturated = np.ma.masked_equal(avg_image_0ravel, 0.0, copy=False)
            avg_image_unsaturated = [s for s in avg_image_unsaturated if correctoffset - 5*calibrationconstant < s < correctoffset + calibrationconstant]
            avg_image_hist, binedges = np.histogram(avg_image_unsaturated, bins = 200, density=False)
            ampls = avg_image_hist[np.argmax(avg_image_hist)]
            bincenters = np.arange(correctoffset - 3*stdmanyskip[-1], correctoffset + 3*stdmanyskip[-1] + 6*stdmanyskip[-1]/200, 6*stdmanyskip[-1]/200)
            axs[1].plot(bincenters, gauss(bincenters,ampls,correctoffset,stdmanyskip[-1]), label='gaussian fit curve', linewidth=1, color='red')
            axs[1].hist(avg_image_0ravel, 200, range = (correctoffset - 5*calibrationconstant, correctoffset + calibrationconstant), density = False, histtype='step', linewidth=2, log = True, color='teal', label = 'avg img pixel charge distribution')
            axs[1].legend(loc='upper left',prop={'size': 14})
            axs[1].set_title('Average image pixel charge distribution: $\sigma_{0e^-}~=~$ ' + str(round(stdmanyskip[-1],4)) + ' ADU; estimated noise: ' + str(round(stdmanyskip[-1]/calibrationconstant,4)) + ' $e^{-}$')
            
            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                #plot.add_caption('First skip and average image pixel charge distribution (PCD).')
            plt.clf()
            doc.append(NewPage())
            
            def r(ns):
                return stdfs/np.sqrt(ns)
            fig, axs = plt.subplots(1, 1, figsize=(8,6), sharey=True, tight_layout=True)
            numberSkips = [10,100,200,300,400,500,600,700,800,900,1000]
            ns = np.arange(1,1000,1)
            #resolution = plt.plot(1,stdfs,'ro',numberSkips[0:len(stdmanyskip)],stdmanyskip,'ro',ns,r(ns),'k-')
            resolution = plt.errorbar(numberSkips[0:len(stdmanyskip)],stdmanyskip,stduncmanyskip,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4, label='measured resolution in ADU')
            resolution += plt.errorbar(1,stdfs,stduncfs,xerr=None,fmt='.',ecolor='red',marker='o', mfc='red', mec='red', ms=4)
            resolution = plt.plot(ns,r(ns),'k--',label='expected $1/\sqrt(N_{skip})$ trend based on first skip sigma')
            plt.legend(loc='upper right',prop={'size': 14})
            plt.ylabel('resolution [ADU]')
            plt.xlabel('number of skips')
            plt.xscale('log')
            plt.yscale('log')
            ax.axis([1, 1000, 0.1, 100])
            ax.loglog()
            plt.title('Resolution trend')
            
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Resolution trend as function of average image skip number.')
            plt.clf()
            doc.append(NewPage())
        
    #############################################
    #Charge loss indicators and skewness section#
    #############################################
    if reportChargeLoss:
        with doc.create(Section('Charge-loss')):
            
            fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
            
            skipperdiffcoreravelled = skipperdiffcore.ravel()
            skipper_imagehist, binedges = np.histogram(skipper_image_unsaturated, bins = 800, density=False)
            axs[0].hist(skipperdiffcoreravelled, 400, density = False, histtype='step', linewidth=2, log = True, color='teal', label='pixel charge difference distribution')
            axs[0].legend(loc="upper right",prop={'size': 14})
            axs[0].set_title('Estimated width : $\sigma_{dif}~=~$' + str(round(stddiff,4)) + 'ADU')
            
            skipperdiffcoreravelledinrange = [s for s in skipperdiffcoreravelled if s > mudiff - 3*stddiff and s < mudiff + 3*stddiff and s != 0]
            numbins = int(max(skipperdiffcoreravelledinrange) - min(skipperdiffcoreravelledinrange))
            skipperdiffcoreravelledinrangehist, binedges = np.histogram(skipperdiffcoreravelledinrange, numbins, density=False)
            bincenters=(binedges[:-1] + binedges[1:])/2
            pguess = [1E+2,mudiff,stddiff]
            pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist, p0=pguess)
            pcddhistfit = gauss(bincenters,*pfit)
            axs[1].plot(bincenters, pcddhistfit, label='gaussian fit curve', linewidth=1, color='red')
            axs[1].hist(skipperdiffcoreravelledinrange, len(bincenters), density = False, histtype='step', linewidth=2, log = True, color = 'teal', label='pixel charge difference distribution')
            #axs[1].plot(bincenters,skipperdiffcoreravelledinrangehist, label='pixel charge difference distribution', color='teal')
            axs[1].legend(loc='upper right',prop={'size': 14})
            axs[1].set_yscale('linear')
            axs[1].set_title('$\mu_{PCDD}~=~$' + str(round(pfit[1],1)) + ' ADU, $\sigma_{PCDD}~=~$' + str(round(pfit[2],1)) + ' ADU')
            
            plt.subplots_adjust(hspace=0.5)
            for ax in axs.flat:
                ax.set(xlabel='pixel value [ADU]', ylabel='counts per ADU')
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Pixel charge difference distribution (PCDD) in full and limited range (for fit).')
            plt.clf()
            doc.append(NewPage())
            
            centeredskipperdiffcore = [s for s in skipperdiffcoreravelled-mudiff if s != -mudiff]
            plt.hist(centeredskipperdiffcore, 600, range = (-20*stddiff,10*stddiff), density = False, histtype='step', linewidth=2, log = True, color='teal',label='centered pixel charge difference distribution')
            plt.legend(loc='upper right',prop={'size': 20})
            plt.xlabel('pixel value [ADU]')
            plt.ylabel('counts per ADU')
            plt.title( '$k_{cl}~=~$' + str(round(kcl,4)) + '$\pm$'+ str(round(kcluncertainty,4)) + ', $S(k_{cl})~=~$' + str(round(kcl/kcluncertainty,4)) + ', skewness = ' + str(round(skew,4)) + '$\pm$'+ str(round(skewuncertainty,4)))
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Pedestal-subtracted PCDD.')
            plt.clf()
            doc.append(NewPage())
        
    #############################################
    ##Calibrated image and Dark Current section##
    #############################################
    if reportCalibrationDarkcurrent:
        skipperavgcalibrated = skipper_avg_cal.ravel()
        if calibrationconstant == 10: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -10 and  s < 10]
        else: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -2 and  s < 4]
        nbins=50*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
        if nbins == 0: nbins=100
        skipperavgcalibratedravelhist, binedges = np.histogram(skipperavgcalibratedravel, nbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        npeaksp = 3
        dcpar = parametersDCfit[0][0], npeaksp, parametersDCfit[0][2]/(50/0.5), parametersDCfit[0][3]/calibrationconstant
        #dcparunc has one more component (the gain) than dcpar (dcpar is an argument for the calibrated gaussian)
        dcparunc = parametersDCfit[1][0], parametersDCfit[1][1], parametersDCfit[1][2]/(50/0.5), parametersDCfit[1][3]/calibrationconstant, parametersDCfit[1][5]
        skipperavgcalibratedravelhistfit = convolutionGaussianPoisson(bincenters,*dcpar)
        #plt.plot(bincenters,skipperavgcalibratedravelhist,label='avg img calibrated pixel charge distribution', color='teal')
        plt.hist(skipperavgcalibratedravel, len(bincenters), density = False, histtype='step', linewidth=2, log = False, color = 'teal', label='avg image calibrated pixel charge distribution')
        plt.plot(bincenters, skipperavgcalibratedravelhistfit, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round_sig_2(redchisquared)), color='red')
        #plt.hist(skipperavgcalibrated.ravel(), 200, (-1,5), density = False, histtype='step', linewidth=2, log = True, color='teal')
        plt.legend(loc='upper right',prop={'size': 20})
        plt.xlabel('pixel value [e$^-$]')
        plt.ylabel('counts')
        plt.title('$I_{darkCF}~=~$' + str(round(dcpar[0],6)) + '$\pm$' + str(round_sig_2(dcparunc[0])) + ' $e^-$pix$^{-1}$, $I_{darkAC}~=~$' + str(round(dcestimate2,6)) + ' $e^-$pix$^{-1}$')
        
        with doc.create(Section('Dark Current')):
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Calibrated pixel charge distribution.')
            calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+'±'+str(round_sig_2(dcparunc[4]))+' ADU per electron.'
            doc.append(calibrationline)
            plt.clf()
            doc.append(NewPage())

    #############################################
    #######Fast Fourier Transform section #######
    #############################################
    if (reportFFTskips or reportFFTrow):
        import functions
        hdr = fits.getheader(image_file,0)
        nallcolumns = hdr['NAXIS1']
        nrows = hdr['NAXIS2']
        nskips = hdr['NDCMS']
    
    if reportFFTskips:
        samplet = hdr['MREAD']*0.001 #MREAD is in ms. Convert in s
        samplet /= (nrows*nallcolumns)
        ncolumns = int(nallcolumns/nskips) # n of columns in the image
        functions.pixelFFT(image_data, nrows-1, ncolumns-1, nskips, samplet)
        with doc.create(Section('Fourier Analysis')):
            with doc.create(Figure(position='htb!')) as plot:
                plot.add_plot(width=NoEscape(r'0.9\linewidth'))
                plot.add_caption('Full image Fast Fourier Transform (first to last skip).')
            plt.clf()
    
    if reportFFTrow:
        samplet *= nskips
        functions.rowFFT(skipper_avg0, nrows-1, ncolumns-1, samplet)
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Average image Fast Fourier Transform (all row pixels).')
        plt.clf()
        doc.append(NewPage())
     
    #############################################
    #############Produce Report PDF##############
    #############################################
    doc.generate_pdf(sys.argv[3].replace('processed/','reports/'), clean_tex=False)
    
    return 0

