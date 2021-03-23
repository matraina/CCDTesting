#functions perfoming anticlustering for dark current estimation. using calibration results obtain with calibration module

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
    else: print("pathindex is out of range (first crown travelled). Please assess usage")
    return pixelrow, pixelcolumn

#function emptyCrown() returns True if crown around pixel is empty, False otherwise. Its argument can be either border or bulk (of the image)
def emptyCrown(pixelrow, pixelcolumn, avgimginelectrons, imageregion, sigma, nrows, ncolumns):
    empty = True
    pathindex = 0
    while(empty and pathindex <= 7):
        tmppixelrow, tmppixelcolumn = crownFinder(pathindex, pixelrow, pixelcolumn)
        #print("crown finder moved me to: ");print(crownFinder(pathindex, pixelrow, pixelcolumn))
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
        else: print("Ill-defined image region. Please check code"); break
    return empty
    
    
def factorial(n):
    facto=1
    for f in range(2,n+1): facto*=f
    return facto
        
def convolutionGaussianPoisson(q, *p):
    import numpy as np
    dcratep, npeaksp, amplip, sigmap = p
    f = 0
    npeaksp = 3
    for peaks in range(npeaksp):
        f +=  ( (dcratep**peaks * np.exp(-dcratep) / factorial(peaks)) * (amplip / np.sqrt(2 * np.pi * sigmap**2)) * np.exp( - (q - peaks)**2 / (2 * sigmap**2)) )
    return f

def darkCurrentEstimations(avgimginelectrons,std,debug):
    import functions
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
            #print("pixel with one electron: "); print(nrow); print(nclmn)
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
        print("Total number of border pixels checked: ", pixelscount)
        print("Total number of border ~empty pixels: ",zeroepixels)
        print("Total number of border isolated ~empty pixels: ",iso0epixels)
        print("Total number of border single-electron pixels: ",oneepixels)
        print("Total number of border isolated single-electron pixels: ",iso1epixels)
    
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
    print("The anticlustering estimate of the dark current is: %.8f" % darkcurrentestimate,"e-/pix")#/day")
    
    if debug:
        #pixelscount checks if we measured the correct number of pixels
        print("Total number of checked pixels: ", pixelscount)
        print("Total number of ~empty pixels: ",zeroepixels)
        print("Total number of isolated ~empty pixels: ",iso0epixels)
        print("Total number of single-electron pixels: ",oneepixels)
        print("Total number of isolated single-electron pixels: ",iso1epixels)

    
    #############################################
    #PoissonGaussian fit dark current estimation#
    #############################################
    
    avgimginelectronsravel = avgimginelectrons.ravel()
    avgimginelectronsravel = [s for s in avgimginelectronsravel if s > -10 and s < 10]
    pguess = [0.01,2,1000,2]
    avgimginelectronshist, binedges = np.histogram(avgimginelectronsravel, bins = 50*int(max(avgimginelectronsravel) - min(avgimginelectronsravel)), density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    pfit, varmatrix = curve_fit(convolutionGaussianPoisson, bincenters, avgimginelectronshist, p0=pguess, bounds=([0,2,0,0], [10., 10, 10**10, 100]))
    avgimageinelectronshistfit = convolutionGaussianPoisson(bincenters,*pfit)
    darkcurrentestimate2,npeaks,npixels,sigmapeaks = pfit[0],pfit[1],pfit[2],pfit[3]
    #plt.plot(bincenters,avgimginelectronshist,label='aie')
    #plt.plot(bincenters, avgimageinelectronshistfit, label='fit curve')
    #plt.yscale('log')
    #plt.title('$I_{darkAC}=$' + str(round(darkcurrentestimate,6)) + ' $e^-$, $I_{darkCF}=$' + str(round(pfit[0],6)) + ' $e^-$')
    #plt.show()
    
    if debug:
        plt.plot(bincenters,avgimginelectronshist,label='aie')
        plt.plot(bincenters, avgimageinelectronshistfit, label='fit curve')
        #plt.yscale('log')
        plt.title('$I_{darkAC}=$' + str(round(darkcurrentestimate,6)) + ' $e^-$, $I_{darkCF}=$' + str(round(pfit[0],6)) + ' $e^-$')
        plt.show()
    
    return darkcurrentestimate, darkcurrentestimate2, pfit


    
