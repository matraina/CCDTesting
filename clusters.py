#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina
Module devoted to image cluster search and analysis for physics study and depth calibration.

-------------------
'''


import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage import find_objects
from scipy.ndimage import sum as ndi_sum
from scipy.ndimage import maximum as ndi_max
from scipy.ndimage import center_of_mass as ndi_cms


class Cluster:
    
    def __init__(self,coordinates,pixel_electrons):
        self.coordinates = coordinates
        self.pixel_electrons = pixel_electrons
        
    def spatialMetrics(self):
        ipixel,avgy,avgx,avgy2,avgx2 = 0,0,0,0,0
        for coor in self.coordinates:
            avgy += coor[0]*self.pixel_electrons[ipixel]
            avgx += coor[1]*self.pixel_electrons[ipixel]
            avgy2 += (coor[0]**2)*self.pixel_electrons[ipixel]
            avgx2 += (coor[1]**2)*self.pixel_electrons[ipixel]
            #print(coor,self.pixel_electrons[ipixel])
            ipixel += 1
        avgy /= sum(self.pixel_electrons)
        avgx /= sum(self.pixel_electrons)
        avgy2 /= sum(self.pixel_electrons)
        avgx2 /= sum(self.pixel_electrons)
        stdx = avgx2 - avgx**2
        stdx = np.sqrt(stdx)
        stdy = avgy2 - avgy**2
        stdy = np.sqrt(stdy)
        return avgy, avgx, avgy2, avgx2, stdx, stdy
    
    def totalElectrons(self):
        return sum(self.pixel_electrons)



def clusterImage(image,cut,**kwargs):
    #cut = global_threshold,maximum_threshold
    #*args to make mask optional argument
    mask = kwargs.get('mask', None)
    
    from astropy.utils.data import get_pkg_data_filename
    from reconstruction import getSingleSkipImage
    import json
    with open('config.json') as config_file:
        config = json.load(config_file)
    usemask = config['depth_calibration_analysis'][-1]['use_mask']
    if usemask: maskpath = config['depth_calibration_analysis'][-1]['mask_path']
    else: maskpath = None

    s = generate_binary_structure(2,2)
    image_above_threshold = np.where(image > cut[0], 1, 0) #pixels above thr to 1, below to 0
    image_features, nclusters = label(image_above_threshold,s) #clusters (features) will be labelled with increasing integers up to nclusters
    #print(image_features,nclusters)
    
    cluster = find_objects(image_features) #get set of coordinates of smallest matrix containing each cluster

    #discard clusters where no pixel above maximum_threshold
    for icluster in range(nclusters):
        maximumincluster = ndi_max(image[cluster[icluster]],image_features[cluster[icluster]],icluster+1)
        if maximumincluster < cut[1]:
            image_features = np.where(image_features != icluster+1, image_features, 0)

    image_features, nclusters = label(image_features,s)
    #print(image_features,nclusters)

    print('I have found '+str(nclusters)+ ' clusters with threshold '+str(round(cut[0],2))+' ADU and maximum valued pixel with at least '+str(round(cut[1],2)) + ' ADU')
    
    cluster = find_objects(image_features)
    if mask is not None: mask = getSingleSkipImage(get_pkg_data_filename(maskpath))
    clusternpixels,clustertouchmask,clusterspatialmetrics,clusterenergy = [],[],[],[]
    for icluster in range(nclusters):
        clusterpixels = np.argwhere(image_features==icluster+1)
        npixels = len(clusterpixels)
        #print(clusterpixels,npixels)
        clusterinimage = image[cluster[icluster]]
        #print(clusterinimage)
        electronsinclusterpixels = image[clusterpixels[0:npixels,0],clusterpixels[0:npixels,1]]
        
        touchmask = False
        #pixeldistanceclosecluster = 1000
        for pixel in range(npixels):
            #check if cluster touches masked region
            if mask is not None:
                maskaroundpixel = mask[clusterpixels[pixel,0]-1:clusterpixels[pixel,0]+1,clusterpixels[pixel,1]-1:clusterpixels[pixel,1]+1]
                if len(maskaroundpixel[maskaroundpixel==1]) > 0 and not touchmask: touchmask = True
            #check minimum distance to other cluster
            #distanceclosecluster = 0
            #for idistpix in range(1,5):
            #    features_around = image_features[clusterpixels[pixel,0]-idistpix:clusterpixels[pixel,0]+idistpix,clusterpixels[pixel,1]-idistpix:clusterpixels[pixel,1]+idistpix].ravel()
            #    if len(features_around[(features_around != icluster+1) and (features_around>0)])>0 and distanceclosecluster==0:
            #        distanceclosecluster = idistpix
            #        break
            #if distanceclosecluster = 0: distanceclosecluster = 5
            #pixeldistanceclosecluster = min(pixeldistanceclosecluster,distanceclosecluster)
                    
        cluster_i = Cluster(clusterpixels,electronsinclusterpixels)
        clusternpixels.append(npixels)
        clustertouchmask.append(touchmask)
        clusterspatialmetrics.append(cluster_i.spatialMetrics())
        clusterenergy.append(cluster_i.totalElectrons())
        
    if maskpath is not None: print('I have found '+str(sum(clustertouchmask))+ ' clusters touching the masked pixels and columns')
        
    return clusternpixels, clustertouchmask, clusterspatialmetrics, clusterenergy
        




'''
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

image_file = get_pkg_data_filename('./raw/Image_Am241.fits')
from reconstruction import getSingleSkipImage

a = getSingleSkipImage(image_file)
cuts=[700,10000]
clusterenergies = clusterImage(a,cuts,mask=None)[2]
#import matplotlib.pyplot as plt
#plt.hist(clusterenergies,200,alpha=0.5)
#plt.show()
'''
