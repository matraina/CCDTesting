#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------

*By: Michelangelo Traina and Paolo Privitera to study skipper CCD data
*Adapted from plot_fits-image.py by: Lia R. Corrales, Adrian Price-Whelan, Kelle Cruz*
*License: BSD*

-------------------
"""
##############################################################################                             
# Input values from command line

import sys

#directory where raw images and derivati are found
arg1 = sys.argv[1]
#input FITS file
arg2 = sys.argv[2] + ".fits"
#output FITS file with skipper images
arg3 = sys.argv[3] + ".fits"
#skip to start average/std and difference of skips (after BS=BaselineShift (PedestalShift probably due to reset))
arg4 = sys.argv[4]
#skip to end average/std and difference of skips
arg5 = sys.argv[5]
#iskipend = nskips if iskipend set negative  (see lines below)
iskipstart, iskipend = int(arg4), int(arg5)

##############################################################################                             
# Get Numpy, and relevant Scipy

import numpy as np
import scipy.fftpack
from scipy import stats
from scipy.stats import norm
from scipy import signal

##############################################################################
# Set up matplotlib and use a nicer set of plot parameters

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

import functions
import chargeloss
import calibration
import darkcurrent
import latekreport

##############################################################################
# Specify path (can be out of the main tree)

import os
os.chdir(arg1)

##############################################################################                             
# Open the data image

image_file = get_pkg_data_filename(arg2)

##############################################################################
# Use `astropy.io.fits.info()` to display the structure of the file

fits.info(image_file)

##############################################################################
# Look at the header of the zero extension:

#print("Header Extension 0:")
#print(repr(fits.getheader(image_file, 0)))
#print()

##############################################################################
# Write header in a text file named just like output image, located in "header" folder:

fileHeader = open(sys.argv[3].replace("processed","header") + ".txt", "a")
fileHeader.write(repr(fits.getheader(image_file, 0)))
fileHeader.close()

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
print("N. rows columns skips ",nrows,ncolumns,nskips)

##############################################################################
# Generally the image information is located in the Primary HDU, also known
# as extension 0. Here, `astropy.io.fits.getdata()` reads the image
# data from this first extension using the keyword argument ``ext=0``:

image_data0 = fits.getdata(image_file, ext=0)

##############################################################################
# The data is now stored as a 2D numpy array. Print the dimensions using the
# shape attribute:

#print("Image ndim: ", image_data0.ndim)
#print("Image shape:", image_data0.shape)
#print("Image size: ", image_data0.size)
#print("Image dtype:", image_data0.dtype)

##############################################################################                             
# Create the Skipper images arrays
#row, column                                                                                    

# a copy of the image to correct for LEACH daq
image_data = np.zeros((nrows, nallcolumns), dtype=np.float64)
# image for skip 0
skipper_image0 = np.zeros((nrows, ncolumns), dtype=np.float64)
# image for skip 1
skipper_image1 = np.zeros((nrows, ncolumns), dtype=np.float64)
# image for skip 2
skipper_image2 = np.zeros((nrows, ncolumns), dtype=np.float64)
# image for last skip
skipper_image3 = np.zeros((nrows, ncolumns), dtype=np.float64)
# image for average of skip images
skipper_avg0 = np.zeros((nrows, ncolumns), dtype=np.float64)
# image for calibrated average of skip images (offset subtraction + gain rescaling)
skipper_avg_cal = np.zeros((nrows, ncolumns), dtype=np.float64)
# image for standard deviation of skips
skipper_std = np.zeros((nrows, ncolumns), dtype=np.float64)
# image difference skip 0 (or after BS) - avg
skipper_diff = np.zeros((nrows, ncolumns), dtype=np.float64)
# select high charge pixels
skipper_large_charge = np.zeros((nrows, ncolumns), dtype=np.bool)

##############################################################################
# Initialization of various arrays and parameters

# y = row   x = column
if iskipstart < 0  or iskipstart > nskips:
   iskipstart = 0
if iskipend < 0 or iskipend > nskips:
   iskipend = nskips - 1


##############################################################################    
# First fix the LEACH image

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

##############################################################################
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
         skipper_image0[y,xp] = image_data[y,xeffstart]
         skipper_image1[y,xp] = image_data[y,xeffp1]
         skipper_image2[y,xp] = image_data[y,xeffp2]
         skipper_image3[y,xp] = image_data[y,xeffend]
         #check charge difference between start and end skip: charge loss feeds distribution at negative values, centroid value ~ pedestal: later subtracted
         skipper_diff[y,xp] = skipper_image0[y,xp] - skipper_image3[y,xp]
      #pedestal subtraction for 1-skip images: subtract from every pixel relative row median
      elif nskips == 1:
         image_data[y,xp] = image_data[y,xp] - pedestaloneskiprow

##############################################################################
# Output the single skip image, same header of original file

if nskips == 1: #processed image is pedestal-subtracted if nskip == 1
   hdr_copy = hdr.copy()
   hdu0 = fits.PrimaryHDU(data=image_data,header=hdr_copy)
   new_hdul = fits.HDUList([hdu0])
   new_hdul.writeto(arg3, overwrite=True)
   sys.exit()

##############################################################################
#HEREON ONLY SKIPPER IMGS PROCESSING #########################################
##############################################################################

#imageIsGood = True

##############################################################################
#ESTIMATE NOISE AT SKIPS: 1, 10, 100 . . . 1000 ##############################
##############################################################################

ampfs, mufs, stdfs = functions.sigmaFinder(skipper_image0, False)
if mufs < 1E+3: imageIsGood *= False; print("Pedestal value is too small: LEACH might have failed.")
ampmanyskip, mumanyskip, stdmanyskip = [],[],[]
for k in range(naverages): amp, mu, std = functions.sigmaFinder(skipper_averages[:,:,k], False); ampmanyskip.append(amp); mumanyskip.append(mu); stdmanyskip.append(std)

##############################################################################
#FIRST LAST SKIP CHARGE LOSS CHECK: KCL AND SKEW##############################
##############################################################################

#can exclude image border over here if necessary
#charge loss check excluding border pixels
diff_image_core = np.zeros((nrows-2, ncolumns-2), dtype=np.float64) #row and column coordinates
for y in range(1,nrows-1):
    for x in range(1,ncolumns-1):
            diff_image_core[y-1][x-1] = skipper_diff[y][x]

skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, muPCDD, stdPCDD = chargeloss.firstLastSkipPCDDCheck(diff_image_core, False)
kclsignificance = kclPCDD/kclPCDDuncertainty
if abs(kclsignificance) > 3: imageIsGood *= False; print("Kcl value flags probable charge loss")

##############################################################################
#ADU TO e- CALIBRATION #######################################################
##############################################################################

skipper_avg_cal, calibrationconstant, offset, calibrationIsGood = calibration.calibrate(skipper_avg0, 10, False)[0:4]
if not calibrationIsGood: imageIsGood *= False; print("Calibration failed")

##############################################################################
#DARK CURRENT ESTIMATE #######################################################
##############################################################################

darkcurrentestimate, darkcurrentestimate2, dcfitparameters = darkcurrent.darkCurrentEstimations(skipper_avg_cal, stdmanyskip[-1]/calibrationconstant, False)

##############################################################################
#LATEK REPORTS ###############################################################
##############################################################################

latekreport.produceReport(image_file, image_data, skipper_image0, skipper_avg0, mufs, stdfs, mumanyskip, stdmanyskip, diff_image_core, muPCDD, stdPCDD, skewnessPCDD, skewnessPCDDuncertainty, kclPCDD, kclPCDDuncertainty, offset, calibrationconstant, darkcurrentestimate, darkcurrentestimate2, *dcfitparameters)

##############################################################################
#OUTPUT PROCESSED IMAGE ######################################################
##############################################################################

# Output the skipper images, same header as original file
hdr_copy = hdr.copy()
hdu0 = fits.PrimaryHDU(data=skipper_image0,header=hdr_copy)
hdu1 = fits.ImageHDU(data=skipper_image1)
hdu2 = fits.ImageHDU(data=skipper_image2)
hdu3 = fits.ImageHDU(data=skipper_image3)
hdu4 = fits.ImageHDU(data=skipper_avg0)
hdu5 = fits.ImageHDU(data=skipper_std)
hdu6 = fits.ImageHDU(data=skipper_diff)
new_hdul = fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4,hdu5,hdu6])
new_hdul.writeto(arg3, overwrite=True)
