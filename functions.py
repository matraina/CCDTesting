# VARIOUS FUNCTIONS

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def reject_outliers(data, m=2):                                                                            
#   return data[abs(data - median(data)) / mad(data, constant=1)< m]
    return data[abs(data - np.median(data)) < m * np.std(data)]   

def reject_outliers1(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def gauss(x, *p):
    import numpy as np
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def sigmaFinder(image, debug):
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    pcd = image.ravel()
    pcd = [s for s in pcd if s != 0]
    bins = int(max(pcd) - min(pcd))
    pcdhistogram, binedges = np.histogram(pcd, bins, density=False)
    while bins - np.argmax(pcdhistogram) < 30:
        bins += 10
        pcdhistogram, binedges = np.histogram(pcd, bins, density=False)
    mostlikelyentry = 0.5*(binedges[np.argmax(pcdhistogram)]+binedges[np.argmax(pcdhistogram)+1]) #pedestal estimation
    #find sigma using FWHM
    mostlikelyentrycounts = pcdhistogram[np.argmax(pcdhistogram)]
    bincounter = 1
    #loop towards right tail of gaussian to find bin of FWHM. Loop condition 10-fold checks that we're not <= HM for a fluctuation
    condition = np.ones(10)
    while(any(condition)):
        bincounter=bincounter+1
        for i in range (0,len(condition)): condition[i] = pcdhistogram[np.argmax(pcdhistogram)+bincounter+(i+1)] > 0.5*mostlikelyentrycounts
    #FWHM ADUs value
    fwhm = 0.5*( binedges[np.argmax(pcdhistogram)+bincounter] + binedges[np.argmax(pcdhistogram)+bincounter+1] )
    #find sigma using FWHM
    fwhmcounts = pcdhistogram[np.argmax(pcdhistogram) + bincounter]
    #sigma is: sigma  = 0.5*FWHM/sqrt(2ln2)
    sigma = abs(mostlikelyentry - fwhm)
    sigma = sigma/np.sqrt(2*np.log(2))
    #now find accurate mean and stdev by fitting in proper range
    fitrange = 2.
    pcdinrange = [s for s in pcd if s > mostlikelyentry - fitrange*sigma and s < mostlikelyentry + fitrange*sigma] #remove pixels out of desired range
    pguess = [mostlikelyentrycounts,mostlikelyentry,sigma]
    binsinrange = int(bins/int(max(pcd) - min(pcd)))*int(max(pcdinrange) - min(pcdinrange))
    pcdinrangehist, binedges = np.histogram(pcdinrange, binsinrange, density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    try:
        pfit, varmatrix = curve_fit(gauss, bincenters, pcdinrangehist, p0=pguess)
        pcdhistfit = gauss(bincenters,*pfit)
        amp,mu,std, = pfit[0],pfit[1],abs(pfit[2])
    except: amp, mu, std = pguess; pcdhistfit = gauss(bincenters,*pguess); print("Gaussian fit for noise evaluation failed. Fit guess values used")
    
    if debug:
        print("Most likely entry is:", mostlikelyentry)
        print("Most likely entry counts are:", mostlikelyentrycounts)
        print("FWHM is at:", fwhm)
        print("FWHM difference counts are:",fwhmcounts)
        print("Value of approximate gaussian std (noise) is:", round(sigma,4))
        print("Value of gaussian std (noise) is:", round(std,4))
        plt.plot(bincenters,pcdinrangehist,label='pcd')
        plt.plot(bincenters, pcdhistfit, label='fit curve')
        plt.title('$\mu=$' + str(round(mu,1)) + ' ADU, $\sigma=$' + str(round(std,3)) + ' ADU')
        plt.show()
        
    return amp, mu, std
    
def scanPlotsFile(scanparametername, firstskipnoise, avgimgnoise, kclsignificance, rscore, gain, dc1, dc2):
    import sys
    fsne = firstskipnoise/gain
    aine = avgimgnoise/gain
    fileplots = open(sys.argv[1]+"/reports/"+scanparametername+"_scan_output", "a+")
    fileplots.write(str(firstskipnoise)+' '+str(avgimgnoise)+' '+str(fsne)+' '+str(aine)+' '+str(kclsignificance)+' '+str(rscore)+' '+str(gain)+' '+str(dc1)+' '+str(dc2)+'\n')
    fileplots.close()
    return 0

def pixelFFT(skipimage, rows, columns, Nskips, samplet):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    
    fftdata = np.zeros(Nskips, dtype=np.float64)
    
    for row in range(rows):
        for clmn in range(columns):
            fftskips = fft(skipimage[row,clmn:clmn+Nskips]-skipimage[row,clmn:clmn+Nskips].mean())
            fftdata += np.abs(fftskips)
    
    xfreq = 1/samplet
    xfreq = np.arange(0,xfreq,xfreq/Nskips)
    fftdata /= (rows*columns)
    
    plt.plot(xfreq[1:int(Nskips/2)], np.abs(fftdata[1:int(Nskips/2)]), color="teal")
    plt.yscale("log")
    plt.ylabel("FFT magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title("Full Image Fast Fourier Transform")
    
    return 0

def rowFFT(avgimage, rows, columns, samplet):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    
    fftdata = np.zeros(columns, dtype=np.float64)
    
    for row in range(rows):
        fftrow = fft(avgimage[row,0:columns]-avgimage[row,0:columns].mean())
        fftdata += np.abs(fftrow)
    
    xfreq = 1/samplet
    xfreq = np.arange(0,xfreq,xfreq/columns)
    fftdata /= (rows)
    
    plt.plot(xfreq[1:int(columns/2)], np.abs(fftdata[1:int(columns/2)]), color="teal")
    plt.yscale("log")
    plt.ylabel("FFT magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title("Average Image Fast Fourier Transform")

    return 0
