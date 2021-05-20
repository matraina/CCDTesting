# function(s) to reconstruct images of interest starting from raw .fits file
# *Adapted from plot_fits-image.py by: Lia R. Corrales, Adrian Price-Whelan, Kelle Cruz*
# *License: BSD*

import numpy as np
import sys
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import json
with open('config.json') as config_file:
    config = json.load(config_file)
default_directory_structure = config['raw_processed_header_reports_dir_structure']
workingdirectory = config['working_directory']
iskipstart = config['skip_start']
iskipend = config['skip_end']
fixLeachReco = config['fix_leach_reconstruction']
reverse = config['reverse']
registersize = config['ccd_register_size']
analysisregion = config['analysis_region']
printheader = config['print_header']

####################################################################
### function used to correct image from leach reconstruction bug ###
####################################################################
def fixLeachReconstruction(image_file):
    
    hdr = fits.getheader(image_file,0)
    nallcolumns = hdr['NAXIS1'] # n of pixels in the x axis, include the skips
    nrows = hdr['NAXIS2'] # n of pixels in the y axis, i.e. n of rows
    nskips = hdr['NDCMS']  # n of skips
    ncolumns = int(nallcolumns/nskips) # n of columns in the image
    ampl = hdr['AMPL']
    
    image_data0 = fits.getdata(image_file, ext=0)
    image_data = np.zeros((nrows, nallcolumns), dtype=np.float64)
    
    for y in range(0,nrows):
        if ampl == 'UL':
            ncoltot = int(nallcolumns/2)
        else:
            ncoltot = nallcolumns
        for x in range(2, ncoltot):
            image_data[y,x-2] = image_data0[y,x]
            if y < nrows-1:
                image_data[y,ncoltot-2] = image_data0[y+1,0]
                image_data[y,ncoltot-1] = image_data0[y+1,1]
            else:
                image_data[y,ncoltot-2] = image_data0[y,0]
                image_data[y,ncoltot-1] = image_data0[y,1]
         
    if ampl == 'UL':
        for x in range(nallcolumns-3,int(nallcolumns/2)-1,-1):
            image_data[y,x+2] = image_data0[y,x]
        if y < nrows-1:
            image_data[y,int(nallcolumns/2)] = image_data0[y+1,nallcolumns-2]
            image_data[y,int(nallcolumns/2)+1] = image_data0[y+1,nallcolumns-1]
        else:
            image_data[y,int(nallcolumns/2)] = image_data0[y,nallcolumns-2]
            image_data[y,int(nallcolumns/2)+1] = image_data0[y,nallcolumns-1]
            
    return image_data


###################################################################################
# primary reconstruction function: produces imgs for analysis and processed .fits #
###################################################################################
def reconstructSkipperImage(image_file,processedname):
    
    hdr = fits.getheader(image_file,0)
    nallcolumns = hdr['NAXIS1'] # n of pixels in the x axis, include the skips
    nrows = hdr['NAXIS2'] # n of pixels in the y axis, i.e. n of rows
    nskips = hdr['NDCMS']  # n of skips
    ncolumns = int(nallcolumns/nskips) # n of columns in the image
    ampl = hdr['AMPL']
    
    # Write header in a text file named just like output image, located (or not) in 'header' folder:
    workingdirectory = config['working_directory']
    headername = processedname.replace('fits','txt')
    if default_directory_structure:
        headername = headername.replace('processed','header')
        headername = workingdirectory + headername
    if printheader:
        fileHeader = open(headername, 'a')
        fileHeader.write(repr(fits.getheader(image_file, 0)))
        fileHeader.close()
    
    # Initialization of various arrays and parameters
    iskipstart = config['skip_start']
    iskipend = config['skip_end']
    if iskipstart < 0 or iskipstart > nskips: iskipstart = 0
    if iskipend < 0 or iskipend > nskips: iskipend = nskips - 1
    
    #declare numpy arrays for images
    #full image
    image_data = np.zeros((nrows, nallcolumns), dtype=np.float64)
    #image for skip 0
    skipper_image0 = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image for skip 1
    skipper_image1 = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image for skip 2
    skipper_image2 = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image for start skip
    skipper_image_start = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image for last skip
    skipper_image_end = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image for average of skip images
    skipper_avg0 = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image for standard deviation of skips
    skipper_std = np.zeros((nrows, ncolumns), dtype=np.float64)
    #image differences (first-second and second-last)
    skipper_diff_01 = np.zeros((nrows, ncolumns), dtype=np.float64)
    skipper_diff = np.zeros((nrows, ncolumns), dtype=np.float64)
    
    #fix leach image if necessary
    if fixLeachReco: image_data = fixLeachReconstruction(image_file)
    else: image_data = fits.getdata(image_file, ext=0)
    
    # Set naverages for resolution trend
    if nskips < 10: naverages = 0
    elif nskips < 100: naverages = 1
    else:
        index=1
        while index <= nskips/100:
            naverages = index+1; index+=1

    #create moving averages 3d image
    movingavg = np.zeros(naverages)
    skipper_averages = np.zeros((nrows, ncolumns, naverages), dtype=np.float64)

    #Fill the Skipper images
    #for loop produces y from 0 to nrows-1
    for y in range(0,nrows):
        xp = -1
        if nskips == 1:
            pedestaloneskiprow = np.median(image_data[y,0:nallcolumns:nskips]) #for pedestal subtraction in single-skip images
        for x in range(0, nallcolumns, nskips):
            xp = xp+1
            xeff = x
            xeffstart = xeff + iskipstart
            xeffend = xeff + iskipend #this is used as a range so will be OK when == nskips
            xeffp1 = xeffstart+1
            xeffp2 = xeffstart+2
            #redefine if UL case
            if ampl == 'UL' and xp>int(ncolumns/2)-1:
                xeff = x+nskips-1
                xeffend = xeff - iskipstart
                xeffstart = xeff - iskipend
                xeffp1 = xeffstart-1
                xeffp2 = xeffstart-2
            #averages and std of the skips of (y, xp) pixel
            index = 0
            if nskips > 1:
                if nskips >= 10:
                    movingavg[index] = (image_data[y,xeff:xeff+10].mean()); index+=1 #comment this line and while below to speed up
                    if nskips >= 100:
                        while index <= nskips/100: movingavg[index] = (image_data[y,xeff:xeff+(100*index)].mean()); index+=1
                avgpixval = image_data[y,xeffstart:xeffend].mean()
                stdpixval = image_data[y,xeffstart:xeffend].std()
                for k in range(naverages): skipper_averages[y, xp, k] = movingavg[k]  #comment along with if's above to speed up
                skipper_avg0[y,xp] = avgpixval
                skipper_std[y,xp] = stdpixval
                skipper_image0[y,xp] = image_data[y,xeff]
                skipper_image1[y,xp] = image_data[y,xeffp1]
                skipper_image2[y,xp] = image_data[y,xeffp2]
                skipper_image_start[y,xp] = image_data[y,xeffstart]
                skipper_image_end[y,xp] = image_data[y,xeffend]
                #check charge difference between first & second skips, and start+1 & end skip: charge loss feeds distribution at negative values, centroid value ~ pedestal: later subtracted
                skipper_diff_01[y,xp] = image_data[y,xeff] - image_data[y,xeff+1]
                skipper_diff[y,xp] = skipper_image1[y,xp] - skipper_image_end[y,xp]
            #pedestal subtraction for 1-skip images: subtract from every pixel relative row median
            elif nskips == 1:
                skipper_image_start[y,xp] = image_data[y,xp]
                image_data[y,xp] = image_data[y,xp] - pedestaloneskiprow
    
    processedfits = workingdirectory + processedname
    if nskips == 1: #processed image is pedestal-subtracted if nskip == 1
        hdr_copy = hdr.copy()
        hdu0 = fits.PrimaryHDU(data=image_data,header=hdr_copy)
        new_hdul = fits.HDUList([hdu0])
        new_hdul.writeto(processedfits, overwrite=True)
    # Output the skipper images, same header as original file
    else:
        hdr_copy = hdr.copy()
        hdu0 = fits.PrimaryHDU(data=skipper_image0,header=hdr_copy)
        hdu1 = fits.ImageHDU(data=skipper_image1)
        hdu2 = fits.ImageHDU(data=skipper_image2)
        hdu3 = fits.ImageHDU(data=skipper_image_start)
        hdu4 = fits.ImageHDU(data=skipper_image_end)
        hdu5 = fits.ImageHDU(data=skipper_avg0)
        hdu6 = fits.ImageHDU(data=skipper_std)
        hdu7 = fits.ImageHDU(data=skipper_diff_01)
        hdu8 = fits.ImageHDU(data=skipper_diff)
        new_hdul = fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4,hdu5,hdu6,hdu7,hdu8])
        new_hdul.writeto(processedfits, overwrite=True)
    
    
    return image_data, skipper_image_start, skipper_image_end, skipper_averages, skipper_diff, skipper_diff_01, skipper_avg0, skipper_std
    
###################################################################################
# lighter reconstruction function: produces average images only .fits #############
###################################################################################
def getAverageSkipperImage(image_file):
    
    hdr = fits.getheader(image_file,0)
    nallcolumns = hdr['NAXIS1'] # n of pixels in the x axis, include the skips
    nrows = hdr['NAXIS2'] # n of pixels in the y axis, i.e. n of rows
    nskips = hdr['NDCMS']  # n of skips
    ncolumns = int(nallcolumns/nskips) # n of columns in the image
    ampl = hdr['AMPL']
    
    if nskips <= 1: print('ERROR: Single skip image cannot be averaged over skips. Exiting'); sys.exit()
    
    # Initialization of various arrays and parameters
    iskipstart = config['skip_start']
    iskipend = config['skip_end']
    if iskipstart < 0 or iskipstart > nskips: iskipstart = 0
    if iskipend < 0 or iskipend > nskips: iskipend = nskips - 1
    
    #declare numpy arrays for images
    #full image
    image_data = np.zeros((nrows, nallcolumns), dtype=np.float64)
    #image for average of skip images
    skipper_avg0 = np.zeros((nrows, ncolumns), dtype=np.float64)
    
    #fix leach image if necessary
    if fixLeachReco: image_data = fixLeachReconstruction(image_file)
    else: image_data = fits.getdata(image_file, ext=0)
    
    #Fill the Skipper images
    #for loop produces y from 0 to nrows-1
    for y in range(0,nrows):
        xp = -1
        for x in range(0, nallcolumns, nskips):
            xp = xp+1
            xeff = x
            xeffstart = xeff + iskipstart
            xeffend = xeff + iskipend #this is used as a range so will be OK when == nskips
            #averages and std of the skips of (y, xp) pixel
            index = 0
            avgpixval = image_data[y,xeffstart:xeffend].mean()
            skipper_avg0[y,xp] = avgpixval
        
    return skipper_avg0

###################################################################################
################ fast cluster-finding for images plots in report ##################
###################################################################################
def findChargedPixelNoBorder(image,sigma):
    coordinates = []
    for row in range(1,np.size(image,0)-1):
        for column in range(1,np.size(image,1)-1):
            if image[row,column] > 20*sigma:
                coordinates.append([row,column])
    return coordinates

def chargedCrown(pixelcoor, image, sigma):
    from calibrationdc import crownFinder
    charged = True
    pathindex = 0
    while(charged and pathindex <= 7):
        tmppixelrow, tmppixelcolumn = crownFinder(pathindex, pixelcoor[0], pixelcoor[1])
        #print('crown finder moved me to: ');print(crownFinder(pathindex, pixelrow, pixelcolumn))
        if image[tmppixelrow, tmppixelcolumn] < 10*sigma:
            charged = False
        else: pathindex += 1
    return charged
            
###################################################################################
# append multiple img methods:  produce single array starting from same-size many #
###################################################################################

def reconstructAvgImageStack(imageprefix, lowerindex, upperindex):
    image = get_pkg_data_filename(imageprefix+str(lowerindex)+'.fits')
    hdr = fits.getheader(image,0)
    nrows = hdr['NAXIS2'] # n of pixels in the y axis, i.e. n of rows
    nallcolumns = hdr['NAXIS1'] # n of pixels in the x axis, include the skips
    nskips = hdr['NDCMS']  # n of skips
    nimages = abs(upperindex - lowerindex + 1)
    skipper_avg_stack = np.zeros((nrows, int(nallcolumns/nskips), nimages), dtype=np.float64)
    for i in range(lowerindex,upperindex+1):
        skipper_avg_stack[:,:,i-lowerindex] = getAverageSkipperImage(get_pkg_data_filename(imageprefix+str(i)+'.fits'))
        #print(skipper_avg_stack[:,:,i-lowerindex])
    return skipper_avg_stack

def cumulatePCDistributions(imagestack): #imagestack is the 3D stack of independent images
    ravelledimagestack = imagestack.ravel()
    return ravelledimagestack

    
    
    
