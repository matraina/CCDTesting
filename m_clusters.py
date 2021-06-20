#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
-------------------

*By: Michelangelo Traina (LPNHE, Sorbonne Universite) to study skipper CCD data
Module devoted to image cluster search and analysis for physics study and depth calibration.

-------------------
'''


import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage import find_objects
from scipy.ndimage import sum as ndi_sum
from scipy.ndimage import maximum as ndi_max
from scipy.ndimage import center_of_mass as ndi_cms
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def gaussian2d(coor,offset,A,mur,sigmar,muc,sigmac):
    import numpy as np
    #print(p)
    #print(*p)
    (r,c) = coor
    twodgauss = offset + A*(  np.exp(-(r-mur)**2/(2*sigmar**2))  +  np.exp(-(c-muc)**2/(2*sigmac**2))  )
    return twodgauss.ravel()

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
        return avgy, avgx, avgy2, avgx2, stdy, stdx
        
    def totalElectrons(self):
        return sum(self.pixel_electrons)

    def gaussianFit2D(self,offset,A,mur,sigmar,muc,sigmac):
        from scipy.optimize import curve_fit
        #print(p)
        #print(*p)
        #print(pguess)
        r,c=[],[]
        for coor in self.coordinates: r.append(coor[0]); c.append(coor[1]) #coords = coor[0],coor[1]
        #print(gaussian2d(coords, *pguess))
        #model = gaussian2d(*pguess)
        #pguess = (offset,A,mur,sigmar,muc,sigmac)
        p0=offset,A,mur,sigmar,muc,sigmac
        pfit, varmatrix = curve_fit(gaussian2d, (r,c), self.pixel_electrons.ravel(), p0)
        coords=r,c
        histfit = gaussian2d(coords,*pfit)
        
        #fit = plt.figure()
        #ax = fit.gca(projection='3d')
        #surf = ax.plot_trisurf(r,c, histfit, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        #surfdata = ax.plot_trisurf(r,c, self.pixel_electrons.ravel(), cmap=cm.inferno,linewidth=0, antialiased=False)
        
        #Customize the axes.
        #plt.xlabel('column')
        #ax.xaxis.set_major_locator(LinearLocator(4))
        #plt.ylabel('row')
        #ax.yaxis.set_major_locator(LinearLocator(4))
        #ax.set_zlim(0, 1900)
        #ax.set_zticks(np.arange(0,2375,475,dtype=int))
        #ax.set_title("Multivariate gaussian cluster fit", pad=20)
        #ax.pbaspect = [1., .33, 0.5]
        #ax.view_init(elev=35., azim=-70)
        #ax.yaxis.set_rotate_label(False)
        #ax.yaxis.label.set_rotation(0)
        #ax.zaxis.set_rotate_label(False)
        #ax.zaxis.label.set_rotation(0)
        #ax.dist = 10.5
        #fit.colorbar(surf, shrink=0.6, aspect=10)
        #plt.show()
        return pfit, varmatrix
        

def clusterImage(image,cut,**kwargs):
    #cut = global_threshold,maximum_threshold
    #*args to make mask optional argument
    mask = kwargs.get('mask', None)
    
    from astropy.utils.data import get_pkg_data_filename
    from m_reconstruction import getSingleSkipImage
    import json
    with open('config.json') as config_file:
        config = json.load(config_file)
    usemask = config['clusters_depth_analysis'][-1]['use_mask']
    if usemask: maskpath = config['clusters_depth_analysis'][-1]['mask_path']
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
    clusternpixels,clustertouchmask,clusterspatialmetrics,passcut,clusterenergy = [],[],[],[],[]
    sigmarfit,sigmacfit=[],[]
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
        
        avgy=clusterspatialmetrics[-1][0]
        stdy=clusterspatialmetrics[-1][5]
        avgx=clusterspatialmetrics[-1][1]
        stdx=clusterspatialmetrics[-1][4]
        amp=max(electronsinclusterpixels)
        offset=cut[0]
        #print(*p)
        
        parafittmp=[0,0,0,0,0,0]
        if sum(sum(image_features[cluster[icluster]]/(icluster+1)))/np.size(image_features[cluster[icluster]]) > 0.75:
            passcut.append(True)
        #    #print( sum(sum( image_features[cluster[icluster]]/(icluster+1))) )
        #    #print(image_features[cluster[icluster]])
        #    #print(np.size(image_features[cluster[icluster]]))
            if stdx > cut[2] and stdy > cut[2]:
                try:
                    parafittmp, covarmatrixtmp = cluster_i.gaussianFit2D(offset,amp,avgy,stdy,avgx,stdx)
                    sigmarfit.append(parafittmp[3]); sigmacfit.append(parafittmp[5])
                except: pass
        else: passcut.append(False)
            
        
    if maskpath is not None: print('I have found '+str(sum(clustertouchmask))+ ' clusters touching the masked pixels and columns')
        
    return clusternpixels, clustertouchmask, clusterspatialmetrics, passcut, sigmarfit, sigmacfit, clusterenergy, parafittmp



def sigmaMaxInRegions(clusterspatialmetrics,passcut,rows,columns,rows_subdivide,columns_subdivide):
    #define number of subdivision row and colwise as nearest larger integer to rc_subdivide/rc to accomodate all subimages
    addrs,addcs=0,0
    if rows%rows_subdivide != 0: addrs = 1
    if columns%columns_subdivide != 0: addcs = 1
    rowwisesubdivisions = int(rows/rows_subdivide+addrs)
    columnwisesubdivisions = int(columns/columns_subdivide+addcs)
    #consider only clusters that passed shape cut
    for k in range(len(passcut)):
        if not passcut[k]:
            clusterspatialmetrics[:].pop(k)
    #list of list to np array for easier manageability
    clusterspatialmetrics = np.array(clusterspatialmetrics)
    #loop across subimages and find parallel and serial sigmamaxs
    #s = [0,0]
    sigmamaxsparallel,sigmamaxsserial = [],[]
    for i in range(rowwisesubdivisions):
        lowerrow = i*rows_subdivide
        upperrow = lowerrow + rows_subdivide
        for j in range(columnwisesubdivisions):
            lowercol = j*columns_subdivide
            uppercol = lowercol + columns_subdivide
            tmpsigmamaxparallel = 0
            tmpsigmamaxserial = 0
            for icluster in range(len(clusterspatialmetrics[:])):
                if (lowerrow < clusterspatialmetrics[icluster][0] < upperrow) and (lowercol < clusterspatialmetrics[icluster][1] < uppercol):
                    tmpsigmamaxparallel = max(tmpsigmamaxparallel,clusterspatialmetrics[icluster][4])
                    tmpsigmamaxserial = max(tmpsigmamaxparallel,clusterspatialmetrics[icluster][5])
            sigmamaxsparallel.append(tmpsigmamaxparallel)
            sigmamaxsserial.append(tmpsigmamaxserial)
        #    s = [s[0],s[1]+1]
        #s = [s[0]+1,0]
    
    
    return sigmamaxsparallel,sigmamaxsserial
            
    
