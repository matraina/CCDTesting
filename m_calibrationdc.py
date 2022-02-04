# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Module devoted to the ADU-electron image calibration and dark current estimation

-------------------
'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import factorial
import lmfit
from m_functions import selectImageRegion

#functions performing gaussPoisson convolution fit. Using lmfit library. Adapted from A. Piers https://github.com/alexanderpiers/damicm-image-preproc/

import json
with open('config.json') as config_file:
    config = json.load(config_file)
    analysisregion = config['analysis_region']
    calibrationguess = config['calibration_constant_guess']
    
def computeGausPoissDist(avgimgravel, avgimgmu, avgimgstd, calibguess, darkcurrent, npoisson):

    avgimghist, binedges = np.histogram(avgimgravel, bins = int(0.5*(max(avgimgravel)-min(avgimgravel))), density=False)
    binwidth = np.diff(binedges)[0]
    bincenters = (binedges[:-1] + binedges[1:])/2
    
    ###part where estimate guesses for parameters below###

    # Set parameters to the fit
    params = lmfit.Parameters()
    if darkcurrent > 0:
        params.add('dcrate', value=darkcurrent, vary=False)
    else:
        params.add('dcrate', value=-1*darkcurrent, min = 0, vary=True)
    params.add('Nelectrons', value=npoisson, vary=False)
    params.add('Npixelspeak', value=len(avgimgravel), vary=True)
    params.add('sigma', value=avgimgstd, min = 0, vary=True)
    params.add('offset', value=avgimgmu, vary = True)
    if calibguess > 0:
        params.add('gain', value=calibguess, vary=True, min = 9.3)
    else:
        params.add('gain', value=-1*calibguess, min = 0)
    minimized = lmfit.minimize(lmfitGausPoisson, params, method='least_squares', args=(bincenters, avgimghist))
    
    # Operations on the returned values to parse into a useful format
    return minimized


def convolutionGaussianPoisson(q, *p):
    import numpy as np
    dcratep, nelectronsp, amplip, sigmap, offsetp, calibp = p
    f = 0
    for peaks in range(int(nelectronsp)):
        f +=  ( (dcratep**peaks * np.exp(-dcratep) / factorial(peaks)) * np.exp( - ( calibp*peaks - (q - offsetp))**2 / (2 * sigmap**2)) )
    return f * amplip / np.sqrt(2 * np.pi * sigmap**2)

def lmfitGausPoisson(param, x, data):
    '''
    LMFIT function for a gaussian convolved with a poisson distribution
    '''

    dcratep = param['dcrate']
    nelectronsp = param['Nelectrons']
    amplip = param['Npixelspeak']
    sigmap = param['sigma']
    offsetp = param['offset']
    calibp = param['gain']
    
    par = [dcratep, nelectronsp.value, amplip, sigmap, offsetp, calibp]

    model = convolutionGaussianPoisson(x, *par)

    # include uncertainties as weights
    sig=np.sqrt(data)
    # bins w uncertainty == 0 assigned uncertainty == 1, like bins w uncertainty == 1
    sig[sig==0]=1
    resids = model - data
    weighted = np.sqrt(resids ** 2 / sig ** 2)
    return weighted

    
def parseFitMinimum(fitmin):
    '''
        Takes to fit minimum and parses it into a dictionary of useful parameters
    '''
    params = fitmin.params
    output = {}
    output['dcrate'] = [ params['dcrate'].value,  params['dcrate'].stderr  ]
    output['sigma']  = [ params['sigma'].value, params['sigma'].stderr ]
    output['gain']    = [ params['gain'].value,   params['gain'].stderr   ]

    return output


def paramsToList(params):
    '''
        Converts lmfit.params to a list. Only for poiss + gauss function
    '''

    par = [ params['dcrate'].value, params['Nelectrons'].value, params['Npixelspeak'].value, params['sigma'].value, params['offset'].value, params['gain'].value]
    parunc = [ params['dcrate'].stderr, params['Nelectrons'].stderr, params['Npixelspeak'].stderr, params['sigma'].stderr, params['offset'].stderr, params['gain'].stderr]
    return par, parunc
    
def fitGuessesArray(centroid,calibguess,npeaks):
    pguesses = np.zeros((3,36), dtype=np.float64) #3 = fit parameters to tweak, 18 = set of guess
    i=0
    for npeaksguess in range(npeaks-2,npeaks+3,2):
        for calguess in np.arange(calibguess-0.5,calibguess+2.5,0.5):
            for zeroelectroncentroid in [0,centroid]:
                pguesses[:,i] = zeroelectroncentroid, calguess, npeaksguess
                i+=1
    return pguesses
    
def calibrationDC(avgimg,std,reverse,debug):

    avgimgravel=avgimg.ravel()
    nbins=int(0.5*(max(avgimgravel)-min(avgimgravel)))
    avgimghist, binedges = np.histogram([s for s in avgimgravel if s!=0], bins = nbins, density=False) #s!=0 only removes saturation coutns
    bincenters = (binedges[:-1] + binedges[1:])/2
    mu = bincenters[np.argmax(avgimghist)]
    if reverse: avgimg = mu - avgimg
    else: avgimg = avgimg - mu
    if analysisregion == 'arbitrary': avgimg = selectImageRegion(avgimg,analysisregion)
    avgimgravel=avgimg.ravel()
    avgimghist, binedges = np.histogram(avgimgravel, bins = nbins, density=False)
    bincenters = (binedges[:-1] + binedges[1:])/2
        
    pguesses = fitGuessesArray(mu,calibrationguess,npeaks=4)
    reducedchisquared = np.zeros(np.size(pguesses,1), dtype=np.float64)
    par = np.zeros((6,np.size(pguesses,1)), dtype=np.float64)
    parunc = np.zeros((6,np.size(pguesses,1)), dtype=np.float64)
    
    for tweakfitindex in range(np.size(pguesses,1)):
        fitminimized = computeGausPoissDist(avgimgravel, pguesses[0,tweakfitindex], std, pguesses[1,tweakfitindex], -1, pguesses[2,tweakfitindex])
        params = fitminimized.params
        #print(lmfit.fit_report(fitminimized))
        reducedchisquared[tweakfitindex] = fitminimized.redchi
        par[:,tweakfitindex] = np.array(paramsToList(params)[0])
        parprint = par[:,tweakfitindex]
        parunc[:,tweakfitindex] = np.array(paramsToList(params)[1])
        #debug
        '''
        if debug:
            #print(lmfit.fit_report(fitminimized))
            avgimghist, binedges = np.histogram(avgimgravel, bins = nbins, density=False)
            bincenters = (binedges[:-1] + binedges[1:])/2
            adu = np.linspace(bincenters[0], bincenters[-1], nbins)
            plt.plot(adu, convolutionGaussianPoisson(adu, *parprint), 'r')
            plt.yscale('log')
            plt.plot(adu, avgimghist, 'teal')
            plt.ylim(0.01, params['Npixelspeak'])
            plt.show()
         '''
    #find best fit
    optimalfitindex = np.argmin(abs(reducedchisquared))
    parmatrix = [par[:,optimalfitindex],parunc[:,optimalfitindex]]
            
    # save fit results
    bestfit = computeGausPoissDist(avgimgravel, pguesses[0,optimalfitindex], std, pguesses[1,optimalfitindex], -1, pguesses[2,optimalfitindex])
    print(parseFitMinimum(bestfit))
    darkcurrentestimate,offsetresidual,calibrationconstant = par[0,optimalfitindex],par[4,optimalfitindex],par[5,optimalfitindex]
    skipper_avg_cal = (avgimg - offsetresidual)/calibrationconstant
    if reverse: offset = mu - offsetresidual
    else: offset = mu + offsetresidual
    
    if debug:
        avgimghist, binedges = np.histogram(avgimgravel, bins = nbins, density=False)
        bincenters = (binedges[:-1] + binedges[1:])/2
        skipperavgfit = convolutionGaussianPoisson(bincenters,*par[:,optimalfitindex])
        plt.plot(bincenters,avgimghist, 'teal')
        plt.plot(bincenters,skipperavgfit, label='gauss-poisson convolution fit curve: '+'$\chi^2_{red}=$'+str(round(reducedchisquared[optimalfitindex],4)), color='red')
        plt.legend(loc = 'upper right')
        plt.yscale('log')
        plt.ylim(0.1)
        plt.show()
        print('Estimated offset is '+str(offset)+' ADU')
        #print(skipper_avg_cal.ravel())
    
    
    return parmatrix, reducedchisquared[optimalfitindex], offset





#functions perfoming anticlustering for dark current estimation. using calibration results obtain with above module

def crownFinder(pathindex, pixelrow, pixelcolumn):
    if pathindex == 0:
        pixelrow += 1
    elif pathindex == 1:
        pixelrow += 1
        pixelcolumn += 1
    elif pathindex == 2:
        pixelcolumn += 1
    elif pathindex == 3:
        pixelcolumn += 1
        pixelrow -= 1
    elif pathindex == 4:
        pixelrow -= 1
    elif pathindex == 5:
        pixelrow -= 1
        pixelcolumn -= 1
    elif pathindex == 6:
        pixelcolumn -= 1
    elif pathindex == 7:
        pixelcolumn -= 1
        pixelrow += 1
    else: print('pathindex is out of range (first crown travelled). Please assess usage')
    return pixelrow, pixelcolumn

#function emptyCrown() returns True if crown around pixel is empty, False otherwise. Its argument can be either border or bulk (of the image)
def emptyCrown(pixelrow, pixelcolumn, avgimginelectrons, imageregion, sigma, nrows, ncolumns):
    empty = True
    pathindex = 0
    while(empty and pathindex <= 7):
        tmppixelrow, tmppixelcolumn = crownFinder(pathindex, pixelrow, pixelcolumn)
        #print('crown finder moved me to: ');print(crownFinder(pathindex, pixelrow, pixelcolumn))
        if imageregion == 'bulk':
            if avgimginelectrons[tmppixelrow, tmppixelcolumn] < -2*sigma or avgimginelectrons[tmppixelrow, tmppixelcolumn] > 2*sigma:
                empty = False
            else: pathindex += 1
        elif imageregion == 'border':
            if (tmppixelrow>=0 and tmppixelrow<nrows and tmppixelcolumn>=0 and tmppixelcolumn<ncolumns):
                if (avgimginelectrons[tmppixelrow, tmppixelcolumn] < -2*sigma or avgimginelectrons[tmppixelrow, tmppixelcolumn] > 2*sigma):
                    empty = False
                else: pathindex += 1
            else: pathindex += 1
        else: print('Ill-defined image region. Please check code'); break
    return empty

def anticlusteringDarkCurrent(avgimginelectrons,std,debug):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    nrows, ncolumns = avgimginelectrons.shape
    #do border anti-clustering
    imageregion = 'border'
    zeroepixels,iso0epixels,oneepixels,iso1epixels,pixelscount,nclmn = 0,0,0,0,0,0
    for nrow in range (0,nrows):
        if avgimginelectrons[nrow,0] > -2*std and avgimginelectrons[nrow,0] < 2*std:
            zeroepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso0epixels += 1
        elif avgimginelectrons[nrow,0] > 1-2*std and avgimginelectrons[nrow,0] < 1+2*std:
            oneepixels += 1
            #print('pixel with one electron: '); print(nrow); print(nclmn)
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso1epixels += 1
        pixelscount += 1
    for nclmn in range (1,ncolumns):
        if avgimginelectrons[nrow,nclmn] > -2*std and avgimginelectrons[nrow,nclmn] < 2*std:
            zeroepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso0epixels += 1
        elif avgimginelectrons[nrow,nclmn] > 1-2*std and avgimginelectrons[nrow,nclmn] < 1+2*std:
            oneepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso1epixels += 1
        pixelscount += 1
    for nrow in range (nrows-2,-1,-1):
        if avgimginelectrons[nrow,nclmn] > -2*std and avgimginelectrons[nrow,nclmn] < 2*std:
            zeroepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso0epixels += 1
        elif avgimginelectrons[nrow,nclmn] > 1-2*std and avgimginelectrons[nrow,nclmn] < 1+2*std:
            oneepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso1epixels += 1
        pixelscount += 1
    for nclmn in range (ncolumns-2,0,-1):
        if avgimginelectrons[nrow,nclmn] > -2*std and avgimginelectrons[nrow,nclmn] < 2*std:
            zeroepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso0epixels += 1
        elif avgimginelectrons[nrow,nclmn] > 1-2*std and avgimginelectrons[nrow,nclmn] < 1+2*std:
            oneepixels += 1
            if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                iso1epixels += 1
        pixelscount += 1
    #quick, rough fix for 0- and 1-e crossing: all above just remove -2*std after 1 and insert factor 2 in print()'s
    if debug:
        #pixelscount checks if we measured the correct number of pixels
        print('Total number of border pixels checked: ', pixelscount)
        print('Total number of border ~empty pixels: ',zeroepixels)
        print('Total number of border isolated ~empty pixels: ',iso0epixels)
        print('Total number of border single-electron pixels: ',oneepixels)
        print('Total number of border isolated single-electron pixels: ',iso1epixels)
    
    #do bulk anti-clustering
    imageregion = 'bulk'
    for nrow in range (1,nrows-1):
        for nclmn in range (1,ncolumns-1):
            if avgimginelectrons[nrow,nclmn] > -2*std and avgimginelectrons[nrow,nclmn] < 2*std:
                zeroepixels += 1
                if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                    iso0epixels += 1
            elif avgimginelectrons[nrow,nclmn] > 1-2*std and avgimginelectrons[nrow,nclmn] < 1+2*std:
                oneepixels += 1
                if(emptyCrown(nrow, nclmn, avgimginelectrons, imageregion, std, nrows, ncolumns)):
                    iso1epixels += 1
            pixelscount += 1
        try: darkcurrentestimate = iso1epixels/iso0epixels
        except: darkcurrentestimate = float('nan')
    #darkcurrentestimate = darkcurrentestimate/(exposuretime+rdoutime)
    #darkcurrentestimate = darkcurrentestimate*2000 #mean and conversion into seconds
    #darkcurrentestimate = darkcurrentestimate*86400 #conversion into days
    
    if debug:
        #pixelscount checks if we measured the correct number of pixels
        print('Total number of checked pixels: ', pixelscount)
        print('Total number of ~empty pixels: ',zeroepixels)
        print('Total number of isolated ~empty pixels: ',iso0epixels)
        print('Total number of single-electron pixels: ',oneepixels)
        print('Total number of isolated single-electron pixels: ',iso1epixels)
        
    print('The anticlustering estimate of the dark current is: %.8f' % darkcurrentestimate,'e-/pix')#/day')
        
    return darkcurrentestimate
