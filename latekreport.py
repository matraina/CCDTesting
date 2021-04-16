#function(s) to produce latex report with image quality information and plots
def gauss(x, *p):
    import numpy as np
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
    
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

def produceReport(image_file, image_data, skipper_image0, skipper_avg0, mufs, stdfs, mumanyskip, stdmanyskip, skipperdiffcore, mudiff, stddiff, skew, skewuncertainty, kcl, kcluncertainty, offset, calibrationconstant, dcestimate, dcestimate2, *dcfitpar):
    import sys
    from astropy.utils.data import get_pkg_data_filename
    from astropy.io import fits
    from scipy.optimize import curve_fit
    import numpy as np
    import matplotlib
    from pylatex import Document, Section, Figure, NoEscape, Math, Axis, NewPage, LineBreak, Description, Command
    matplotlib.use('Agg')  # Not to use X server. For TravisCI.
    import matplotlib.pyplot as plt  # noqa
    #setup document parameters
    geometry_options = {"right": "2cm", "left": "2cm"}
    doc = Document(geometry_options=geometry_options)
    doc.preamble.append(Command('title', 'Image Analysis Report'))
    doc.preamble.append(Command('author', 'DAMIC-M'))
    doc.append(NoEscape(r'\maketitle'))
    #############################################
    #Print acqusition parameters value in report#
    #############################################
    fileheader = open(sys.argv[3].replace("processed","header") + ".txt", "r")
    lines = fileheader.readlines()
    with doc.create(Section('Image Acquisition Parameters')):
        with doc.create(Description()) as desc:
            for line in lines[0:70]:
                if line.split()[0]!="COMMENT": desc.add_item(line,"")
                #desc.add_item(line.split()[0].replace("=","")+"="+line.split()[-1],"")
                if line.split()[0]=="MREAD": break
        doc.append(NewPage())
    
    #############################################
    #Pixel charge distribution and noise section#
    #############################################
    with doc.create(Section('Pixel Charge Distributions and Noise')):
        
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=True, tight_layout=True)
        
        skipper_image0ravel = skipper_image0.ravel()
        skipper_image = [s for s in skipper_image0ravel if s != 0]
        skipper_imagehist, binedges = np.histogram(skipper_image, bins = 800, density=False)
        ampfs = skipper_imagehist[np.argmax(skipper_imagehist)]
        axs[0].hist(skipper_image0ravel, 800, density = False, histtype='step', linewidth=2, log = True, color = "teal")
        bincenters = np.arange(mufs - 3*stdfs, mufs + 3*stdfs + 6*stdfs/100, 6*stdfs/100) #last term in upper bound to get ~sym drawing
        axs[0].plot(bincenters, gauss(bincenters,ampfs,mufs,stdfs), label='fit curve', linewidth=1, color="red")
        axs[0].set_title('First skip pixel charge distribution: $\sigma_{0e^-}~=~$ ' + str(round(stdfs,4)) + ' ADU')#; estimated noise: ' + str(round(stdfs/calibrationconstant,4)) + ' $e^{-}$')
        
        avg_image_0ravel = skipper_avg0.ravel()
        avg_image = [s for s in avg_image_0ravel if s != 0 and  offset - 5*calibrationconstant < s < offset + calibrationconstant]
        avg_image_hist, binedges = np.histogram(avg_image, bins = 200, density=False)
        ampls = avg_image_hist[np.argmax(avg_image_hist)]
        bincenters = np.arange(offset - 3*stdmanyskip[-1], offset + 3*stdmanyskip[-1] + 6*stdmanyskip[-1]/100, 6*stdmanyskip[-1]/100)
        axs[1].plot(bincenters, gauss(bincenters,ampls,offset,stdmanyskip[-1]), label='fit curve', linewidth=1, color="red")
        axs[1].hist(avg_image_0ravel, 200, range = (offset - 5*calibrationconstant, offset + calibrationconstant), density = False, histtype='step', linewidth=2, log = True, color="teal")
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
        resolution = plt.plot(1,stdfs,"ro",numberSkips[0:len(stdmanyskip)],stdmanyskip,"ro",ns,r(ns),"k-")
        plt.ylabel("resolution [ADU]")
        plt.xlabel("number of skips")
        plt.xscale("log")
        plt.yscale("log")
        ax.axis([1, 1000, 0.1, 100])
        ax.loglog()
        plt.title("Resolution trend")
        
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Resolution trend as function of average image skip number.')
        plt.clf()
        doc.append(NewPage())
        
    #############################################
    #Charge loss indicators and skewness section#
    #############################################
    with doc.create(Section('Charge-loss')):
        
        fig, axs = plt.subplots(2, 1, figsize=(11,10), sharey=False, tight_layout=True)
        
        skipperdiffcoreravelled = skipperdiffcore.ravel()
        skipper_imagehist, binedges = np.histogram(skipper_image, bins = 800, density=False)
        axs[0].hist(skipperdiffcoreravelled, 400, density = False, histtype='step', linewidth=2, log = True, color="teal")
        axs[0].set_title('Estimated width : $\sigma_{dif}~=~$' + str(round(stddiff,4)) + 'ADU')
        
        skipperdiffcoreravelledinrange = [s for s in skipperdiffcoreravelled if s > mudiff - 3*stddiff and s < mudiff + 3*stddiff and s != 0]
        numbins = int(max(skipperdiffcoreravelledinrange) - min(skipperdiffcoreravelledinrange))
        skipperdiffcoreravelledinrangehist, binedges = np.histogram(skipperdiffcoreravelledinrange, numbins, density=False)
        bincenters=(binedges[:-1] + binedges[1:])/2
        pguess = [1E+2,mudiff,stddiff]
        pfit, varmatrix = curve_fit(gauss, bincenters, skipperdiffcoreravelledinrangehist, p0=pguess)
        pcddhistfit = gauss(bincenters,*pfit)
        axs[1].plot(bincenters, pcddhistfit, label='fit curve', linewidth=1, color="red")
        axs[1].plot(bincenters,skipperdiffcoreravelledinrangehist, label='pcdd', color="teal")
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
        plt.hist(centeredskipperdiffcore, 600, range = (-20*stddiff,10*stddiff), density = False, histtype='step', linewidth=2, log = True, color="teal")
        plt.xlabel("pixel value [ADU]")
        plt.ylabel("counts per ADU")
        plt.title( "$k_{cl}~=~$" + str(round(kcl,4)) + "$\pm$"+ str(round(kcluncertainty,4)) + ", $S(k_{cl})~=~$" + str(round(kcl/kcluncertainty,4)) + ", skewness = " + str(round(skew,4)) + "$\pm$"+ str(round(skewuncertainty,4)))
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Pedestal-subtracted PCDD.')
        plt.clf()
        doc.append(NewPage())
        
    #############################################
    ##Calibrated image and Dark Current section##
    #############################################
    skipperavgcalibrated = (offset - skipper_avg0)/calibrationconstant
    if calibrationconstant == 10: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -10 and  s < 10]
    else: skipperavgcalibratedravel = [s for s in skipperavgcalibrated.ravel() if s > -2 and  s < 4]
    nbins=50*int(max(skipperavgcalibratedravel) - min(skipperavgcalibratedravel))
    if nbins == 0: nbins=100
    skipperavgcalibratedravelhist, binedges = np.histogram(skipperavgcalibratedravel, nbins, density=False)
    bincenters=(binedges[:-1] + binedges[1:])/2
    skipperavgcalibratedravelhistfit = convolutionGaussianPoisson(bincenters,*dcfitpar)
    plt.plot(bincenters,skipperavgcalibratedravelhist,label='aie', color="teal")
    plt.plot(bincenters, skipperavgcalibratedravelhistfit, label='fit curve', color="red")
    #plt.hist(skipperavgcalibrated.ravel(), 200, (-1,5), density = False, histtype='step', linewidth=2, log = True, color="teal")
    plt.xlabel("pixel value [e$^-$]")
    plt.ylabel("counts")
    plt.title('$I_{darkAC}~=~$' + str(round(dcestimate,6)) + ' $e^-$pix$^{-1}$, $I_{darkCF}~=~$' + str(round(dcestimate2,6)) + ' $e^-$pix$^{-1}$')
    
    with doc.create(Section('Dark Current')):
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Calibrated pixel charge distribution.')
        calibrationline = 'Calibration constant is: '+str(round(calibrationconstant,4))+' ADU per electron.'
        doc.append(calibrationline)
        plt.clf()
        doc.append(NewPage())

    #############################################
    #######Fast Fourier Transform section #######
    #############################################
    import functions
    hdr = fits.getheader(image_file,0)
    nallcolumns = hdr['NAXIS1']
    nrows = hdr['NAXIS2']
    nskips = hdr['NDCMS']
    samplet = hdr['MREAD']*0.001 #MREAD is in ms. Convert in s
    samplet /= (nrows*nallcolumns)
    ncolumns = int(nallcolumns/nskips) # n of columns in the image
    functions.pixelFFT(image_data, nrows-1, ncolumns-1, nskips, samplet)
    
    with doc.create(Section('Fourier Analysis')):
        with doc.create(Figure(position='htb!')) as plot:
            plot.add_plot(width=NoEscape(r'0.9\linewidth'))
            plot.add_caption('Full image Fast Fourier Transform (first to last skip).')
        plt.clf()
        
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
    doc.generate_pdf(sys.argv[3].replace("processed/","reports/"), clean_tex=False)
    
    return 0

