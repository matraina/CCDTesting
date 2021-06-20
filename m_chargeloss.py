# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Charge loss check module. Will output kcl, skewness and respective uncertainties. Will also output plot for latek summary

-------------------
'''

import json
with open('config.json') as config_file:
    config = json.load(config_file)
kclthreshold = config['kcl_threshold']

from m_reconstruction import reverse

def gauss(x, *p):
    import numpy as np
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def firstLastSkipPCDDCheck(firstlastskipdifferenceimage, debug):
    import sys
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy import stats
    import matplotlib.pyplot as plt
    ravelleddifference = firstlastskipdifferenceimage.ravel()
    #use either negative or positive semiaxis as a range to estimate mean and stdev before actual fit, in order to exclude possible saturation peak at zero
    countpositive=countnegative=0
    for component in range(len(ravelleddifference)):
        if ravelleddifference[component] > 0: countpositive+=1
        elif ravelleddifference[component] < 0: countnegative+=1
    if countnegative > countpositive:
        rangeadhoc = (min(ravelleddifference),-5)
        nbins = int(-5 - min(ravelleddifference))
    elif countnegative < countpositive:
        rangeadhoc = (5, max(ravelleddifference))
        nbins = int(max(ravelleddifference) - 5)
    else: print('Few-skip image charge loss check failed at PCDD parameter estimation stage. Please review image'); sys.exit()
    #prepare PCDD histogram for mean (~pedestal) and sigma estimation probing bin value
    differencehistogram, binedges = np.histogram(ravelleddifference, int(nbins), range = rangeadhoc, density=False)
    mostlikelydifference = (binedges[np.argmax(differencehistogram)] + binedges[np.argmax(differencehistogram)+1])/2
    mostlikelydifferencecounts = differencehistogram[np.argmax(differencehistogram)]
    #estimate Half Maximum abscissa (first-last skip difference) to get FWHM and then std deviation
    bincounter = 1
    try:
        condition = np.ones(10)
        while(any(condition) and countnegative > countpositive):
            bincounter=bincounter+1
            for i in range (0,10): condition[i] = differencehistogram[np.argmax(differencehistogram)-bincounter-(i+1)] > 0.14*mostlikelydifferencecounts
        while(any(condition) and countnegative < countpositive):
            bincounter=bincounter+1
            for i in range (0,10): condition[i] = differencehistogram[np.argmax(differencehistogram)+bincounter+(i+1)] > 0.14*mostlikelydifferencecounts
    except: print('Search for half maximum abscissa for PCDD fit failed.')
    #estimate FWHM and then std deviation
    if countnegative > countpositive: twosigmadifference = ( binedges[np.argmax(differencehistogram)-bincounter] + binedges[np.argmax(differencehistogram)-bincounter - 1] ) / 2
    elif countnegative < countpositive: twosigmadifference = ( binedges[np.argmax(differencehistogram)+bincounter] + binedges[np.argmax(differencehistogram)+bincounter + 1] ) / 2
    #HM: estimate sigma using FWHM, twosigma: estimate sigma using 0.14Maximum (where 2*sigma should be)
    #HMdifferencecounts = differencehistogram[np.argmax(differencehistogram) - bincounter]
    twosigmadifferencecounts = differencehistogram[np.argmax(differencehistogram) - bincounter]
    
    #stdPCDDestimate = abs(mostlikelydifference - HMdifference)
    #stdPCDDestimate /= np.sqrt(2*np.log(2))
    stdPCDDestimate = abs(mostlikelydifference - twosigmadifference)/2
    
    #now find more accurate values for mean (~pedestal) and standard deviation by fitting PCDD in ad hoc range chosen based on estimates above
    #remove counts at zero from saturation
    ravelleddifferenceinrange = [s for s in ravelleddifference if s > mostlikelydifference - 3*stdPCDDestimate and s < mostlikelydifference + 3*stdPCDDestimate and s != 0]
    pguess = [mostlikelydifferencecounts,mostlikelydifference,stdPCDDestimate]
    ravelleddifferenceinrangehist, binedges = np.histogram(ravelleddifferenceinrange, bins = int(max(ravelleddifferenceinrange) - min(ravelleddifferenceinrange)), density=False)
    try:
        bincenters=(binedges[:-1] + binedges[1:])/2
        pfit, varmatrix = curve_fit(gauss, bincenters, ravelleddifferenceinrangehist, p0=pguess)
        pcddhistfit = gauss(bincenters,*pfit)
        ampPCDD, muPCDD, stdPCDD = pfit[0],pfit[1],pfit[2]
    except: ampPCDD, muPCDD, stdPCDD = mostlikelydifferencecounts, mostlikelydifference, stdPCDDestimate

    #skewness and uncertainty computation. Will use wider range to exclude isolated far outliers and retain all of the tails
    ravelleddifferenceinwiderrange = [s for s in ravelleddifference if s < mostlikelydifference + 8*stdPCDD]
    skewnessPCDD = stats.skew(ravelleddifferenceinwiderrange)
    #use expression for skewness var in case of sample extracted from normal distribution
    skewnessPCDDuncertainty = 6*( len(ravelleddifferenceinwiderrange) - 2 )
    skewnessPCDDuncertainty /= ( len(ravelleddifferenceinwiderrange) + 1 )
    skewnessPCDDuncertainty /= ( len(ravelleddifferenceinwiderrange) + 3 )
    skewnessPCDDuncertainty = np.sqrt(skewnessPCDDuncertainty)
    
    #charge loss coefficient and uncertainty computation. Quantity based on lower-upper tail symmetry in absence of charge loss
    #pedestal subtraction and sorting in ascending order
    centeredsortedravelleddifference = sorted(ravelleddifference - muPCDD)
    #remove saturation counts at muPCDD after pedestal subtraction
    centeredsortedravelleddifference = [s for s in centeredsortedravelleddifference if s!= - muPCDD]
    #count entries below and above symmetry threshold
    countsbelowsymmetrythreshold = countsabovesymmetrythreshold = 0
    component = 0
    while centeredsortedravelleddifference[component] < -kclthreshold*stdPCDD:
        countsbelowsymmetrythreshold += 1; component += 1
    component = len(centeredsortedravelleddifference) - 1
    while centeredsortedravelleddifference[component] > kclthreshold*stdPCDD:
        countsabovesymmetrythreshold += 1; component -= 1
    #compute charge loss coefficient
    if countsbelowsymmetrythreshold + countsabovesymmetrythreshold != 0:
        kcl = countsabovesymmetrythreshold - countsbelowsymmetrythreshold
        kcl /= countsbelowsymmetrythreshold + countsabovesymmetrythreshold
        kcluncertainty = ( np.sqrt( (countsabovesymmetrythreshold*countsbelowsymmetrythreshold**2) + (countsbelowsymmetrythreshold*countsabovesymmetrythreshold**2) ) )
        kcluncertainty = kcluncertainty*2/(countsabovesymmetrythreshold+countsbelowsymmetrythreshold)**2
    else:
        kcl = float('nan')
        kcluncertainty = float('nan')
        print('The charge loss coefficient cannot be computed as there is no counts on either tail beyond the set threshold.')
        

    if debug:
        if countnegative>countpositive:
            if reverse: print('negative baseline shift (more charge) in start skip')
            else: print('negative baseline shift (less charge) in start skip')
        elif countnegative<countpositive: 
            if reverse: print('positive baseline shift (less charge) in start skip')
            else: print('positive baseline shift (more charge) in start skip')
        print('Most likely difference (~mean and ~pedestal) is:', mostlikelydifference)
        print('Most likely difference counts are:', mostlikelydifferencecounts)
        print('2*sigma abscissa (first-last skip difference) is:', twosigmadifference)
        print('2*sigma ordinata (counts) is:', twosigmadifferencecounts)
        print('Value of first-last skip PCDD gaussian std estimate is:', round(stdPCDDestimate,4))
        print("Here's skewness of PCDD: ",skewnessPCDD,'+-',skewnessPCDDuncertainty)
        print('The entries below the symmetric threshold are:',countsbelowsymmetrythreshold)
        print('The entries above the symmetric threshold are:',countsabovesymmetrythreshold)
        print('The charge loss coefficient is:', kcl, '+-', kcluncertainty)
        plt.plot(bincenters,ravelleddifferenceinrangehist,label='pcdd')
        try: plt.plot(bincenters, pcddhistfit, label='fit curve')
        except: print('Gaussian fit not successful. Guess values printed in plot title')
        plt.title('$\mu_{pcdd}=$' + str(round(muPCDD,1)) + ' ADU, $\sigma_{pcdd}=$' + str(round(stdPCDD,1)) + ' ADU')
        plt.show()


    return skewnessPCDD, skewnessPCDDuncertainty, kcl, kcluncertainty, ampPCDD, muPCDD, stdPCDD
