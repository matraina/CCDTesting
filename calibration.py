#functions perfoming ADU to electron calibration through 0- and 1-electron peaks. If single electron resolution is not achieved, guess calibration constant is used
def gauss(x, *p):
    import numpy as np
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
    
def calibrationQualityAssess(averageimage): #improve (not a solid check)
    import numpy as np
    from scipy import signal
    import math
    #use find peaks in full range, if peaks size order altered by DC or others, result is false
    averageimagearrayinrange = [s for s in averageimage.ravel() if s > -4 and s < 6]
    #binwidth should be small enough to guarantee a couple bins between two peaks
    nbins = 20*int( max(averageimagearrayinrange) - min(averageimagearrayinrange) )
    if nbins==0: nbins=100
    averageimagearrayinrangehist, binedges = np.histogram(averageimagearrayinrange, nbins)
    #peaks should be distant at least half a unit
    #print(nbins, binedges[1]-binedges[0])
    peaksarray = signal.find_peaks( averageimagearrayinrangehist, distance = abs(0.7/(binedges[1]-binedges[0])), prominence = (10,math.inf) )
    try:
        if binedges[peaksarray[0][1]] > 0.8 and binedges[peaksarray[0][1]] < 1.2 and -0.2 < binedges[peaksarray[0][0]] < 0.2: goodcalibration = True
        else: goodcalibration = False
    except: goodcalibration = False; print("find_peaks() failed at calibration quality assess stage.")
    return goodcalibration

def calibrate(averageimage, calibrationguess, debug):
    import functions
    import numpy as np
    from scipy import signal
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    calibrationisgood = False
    #sigma finder returns a tuple with 2 elements: mu and sigma from fit. Retrieve mu as offset of average image PCD
    amplitude0electronpeak, offset, std0electronpeak = functions.sigmaFinder(averageimage,False)
    #discard entries below 0e- peak (<0 Ne axis==increasing ADU value) to ensure counting starts at 0e-
    avgimgarrayfrom0electron = [s for s in averageimage.ravel() if s > offset - 50*std0electronpeak and s < offset + 0.75*std0electronpeak]
    #find all peaks distant >= .5sigma for preliminary 1e- peak location
    nbins = int(max(avgimgarrayfrom0electron)-min(avgimgarrayfrom0electron))
    avgimgarrayfrom0electronhist, binedges = np.histogram(avgimgarrayfrom0electron, nbins)
    peaksarray = signal.find_peaks(avgimgarrayfrom0electronhist,distance = abs(1.5*std0electronpeak/(binedges[1]-binedges[0])))
    #if find peaks fails analysis does, to fix
    try:
        singlelectronpeak = binedges[peaksarray[0][-2]]
        singlelectronpeak += binedges[peaksarray[0][-2]+1]
        singlelectronpeak *= 0.5
        #fit 1e- peak for thorough calibration
        #remove pixels out of desired range: sigma is the same for all peaks
        avgimgarrayin1epeakrange = [s for s in avgimgarrayfrom0electron if s > singlelectronpeak - 2*std0electronpeak and s < singlelectronpeak + 2*std0electronpeak]
        pguess = [avgimgarrayfrom0electronhist[peaksarray[0][-2]],singlelectronpeak,std0electronpeak]
        nbins = int(max(avgimgarrayin1epeakrange)-min(avgimgarrayin1epeakrange))
        if nbins==0: nbins=100
        avgimgarrayin1epeakrangehist, binedges = np.histogram(avgimgarrayin1epeakrange, nbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        try:
            pfit, varmatrix = curve_fit(gauss, bincenters, avgimgarrayin1epeakrangehist, p0=pguess)
            avgimgarrayin1epeakrangehistfit = gauss(bincenters,*pfit)
            mu1electronpeak, std1electronpeak = pfit[1],pfit[2]
            calibrationconstant = offset - mu1electronpeak
            if debug:
                plt.plot(bincenters,avgimgarrayin1epeakrangehist,label='average pcd in 1e- range')
                plt.plot(bincenters, avgimgarrayin1epeakrangehistfit, label='fit curve')
                plt.title('Single electron peak fit: $\mu=$' + str(round(pfit[1],1)) + ' ADU, $\sigma=$' + str(round(pfit[2],1)) + ' ADU')
                plt.show()
        except: #when calibration fit fails use calibration guess as constant
            calibrationconstant = calibrationguess
            mu1electronpeak, std1electronpeak = offset - calibrationguess, std0electronpeak
            print("Single electron peak fit for calibration failed. Using guess calibration constant:",calibrationguess,"ADU per electron, and 0-electron peak   std:",round(std0electronpeak,2),"ADU")
    except: #when calibration fit fails use calibration guess as constant
        calibrationconstant = calibrationguess
        mu1electronpeak, std1electronpeak = offset - calibrationguess, std0electronpeak
        print("Single electron peak fit for calibration failed. Using guess calibration constant:",calibrationguess,"ADU per electron, and 0-electron peak std:",round(std0electronpeak,2),"ADU")
    if debug:
        print("Estimated 0e- peak location is:",round(offset,2),"ADU")
        print("Estimated 0e- peak std is:",round(std0electronpeak,2),"ADU")
        print("Requiring", round(abs(std0electronpeak/(binedges[1]-binedges[0])),4), "ADU distance between peaks in preliminary estimation of single electron peak location")
        print("Estimated 1e- peak location is:",round(singlelectronpeak,2),"ADU")
        print("Calibration constant is estimated (or guessed) to be:",round(calibrationconstant,4),"ADU per electron")
        plt.hist(averageimage.ravel(),2000,(17000,19000),log=True);plt.show()
    
    #now performing average image calibration
    averageimage = offset - averageimage
    averageimage = averageimage/calibrationconstant
    #plt.hist(averageimage.ravel(),200,(-4,6),log=True);plt.show()
    #last calibration check (find_peaks on full range avg img)
    calibrationisgood = calibrationQualityAssess(averageimage)
    if calibrationisgood: print("Calibration carried out successfully. Calibration constant value is:", round(calibrationconstant,4), "ADU/e-")
    else: print("Calibration with offset",round(offset,2),"ADU and constant",round(calibrationconstant,4),"ADU per electron seems inaccurate. Dark current estimation carried out nonetheless: result might be inaccurate. Please check image")
        
    #could in principle set calirbationisgood to false by default when guess is used, but it's very unlikely quality assessment fails in that case
        
    return averageimage, calibrationconstant, offset, calibrationisgood, mu1electronpeak, std1electronpeak
