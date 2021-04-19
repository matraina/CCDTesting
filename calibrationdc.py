import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import factorial
import lmfit


def computeGausPoissDist(avgimg, avgimgmu, avgimgstd, calibguess=-1, darkcurrent=-1, npoisson=10):

    avgimghist, binedges = np.histogram(avgimgravel, bins = int(max(avgimgravel)-min(avgimgravel)), density=False)
    bincenters = (binedges[:-1] + binedges[1:])/2
    
    ###part where estimate guesses for parameters below###

    # Set parameters to the fit
    params = lmfit.Parameters()
    params.add("sigma", value=avgimgstd)
    if darkCurrent > 0:
        params.add("dcrate", value=darkcurrent, vary=False)
    else:
        params.add("dcrate", value=-1*darkcurrent, min=0)
    params.add("offset", value=avgimgmu)
    if aduConversion > 0:
        params.add("gain", value=calibguess, vary=False)
    else:
        params.add("gain", value=-1*calibguess)
    params.add("Npixelspeak", value=avgimg.size)
    params.add("Nelectrons", value=npoisson, vary=False)
    minimized = lmfit.minimize(lmfitGausPoisson, params, args=(bincenters, avgimghist))

    # Operations on the returned values to parse into a useful format
    return minimized


def convolutionGaussianPoisson(q, *p):
    import numpy as np
    dcratep, npeaksp, amplip, sigmap, offsetp, calibp = p
    f = 0
    for peaks in range(npeaksp):
        f +=  ( (dcratep**peaks * np.exp(-dcratep) / factorial(peaks)) * (amplip / np.sqrt(2 * np.pi * sigmap**2)) * np.exp( - ((q - offsetp) - calibp*peaks)**2 / (2 * sigmap**2)) )
    return f

def lmfitGausPoisson(param, x, data):
    """
    LMFIT function for a gaussian convolved with a poisson distribution
    """

    dcratep = param["dcrate"]
    npeaksp = param["Nelectrons"]
    amplip = param["Npixelspeak"]
    sigmap = param["sigma"]
    offsetp = param["offset"]
    calibp = param["gain"]
    
    par = [dcratep, npeaksp, amplip, sigmap, offsetp, calibp]

    model = fGausPoisson(x, *par)
    return (data-model)


    
    
def parseFitMinimum(fitmin):
    """
        Takes to fit minimum and parses it into a dictionary of useful parameters
    """
    params = fitmin.params
    output = {}
    output["sigma"]  = [ params["sigma"].value, params["sigma"].stderr ]
    output["dcrate"] = [ params["dcrate"].value,  params["dcrate"].stderr  ]
    output["gain"]    = [ params["gain"].value,   params["gain"].stderr   ]

    return output


def paramsToList(params):
    """
        Converts lmfit.params to a list. Only for poiss + gauss function
    """

    par = [ params["sigma"].value, params["dcrate"].value, params["offset"].value, params["gain"].value, params["Npixelspeak"].value, params["Nelectrons"].value]
    return par
    
    
def calibrationDC(avgimg,mu,std):

    avgimgravel=avgimg.ravel()
    nbins=int(max(avgimgravel)-min(avgimgravel))
    avgimghist, binedges = np.histogram(avgimgravel, bins = nbins, density=False)
    bincenters = (binedges[:-1] + binedges[1:])/2
    
    # Perform poisson gaus fit to data
    fitminimized = computeGausPoissDist(avgimg, mu, std)
    params = fitminimized.params
    print(lmfit.fit_report(fitminimized))
    print(parseFitMinimum(fitminimized))

    # Plot fit results
    par = paramsToList(params)
    adu = np.linspace(bincenters[0], bincenters[-1], 2000)
    plt.plot(x, convolutionGaussianPoisson(adu, *par), "--r")
    plt.yscale("log")
    plt.ylim(0.01, params["Npixelspeak"])
    plt.show()
    
    return par




