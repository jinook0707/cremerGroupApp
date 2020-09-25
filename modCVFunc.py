# coding: UTF-8
"""
Functions from computer vision, using OpenCV, NumPy, SciPy, etc.

Dependency:
    Numpy (1.17),
    Scipy (1.5),
    OpenCV (4.1) 

last editted: 2020.09.24.
"""

import cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

DEBUG = False

#-------------------------------------------------------------------------------

def clustering(pt_list, threshold, criterion='distance'):
    """ Cluster given points

    Args:
        pt_list (list): List of points.
        threshold (int): Threshold to cluster.
        criterion (str): Criterion for fclusterdata function.

    Returns:
        nGroups (int): Number of groups.
        groups (list): List of groups.
    """
    if DEBUG: print("modCVFunc.clustering()")

    cRslt = []
    try: cRslt = list(fclusterdata(np.asarray(pt_list), 
                                   threshold, 
                                   criterion=criterion,
                                   metric='euclidean'))
    except: pass
    nGroups = 0
    groups = []
    if cRslt != []:
        groups = []
        nGroups = max(cRslt)
        for i in range(nGroups): groups.append([])
        for i in range(len(cRslt)):
             groups[cRslt[i]-1].append(pt_list[i])
    return nGroups, groups

#-------------------------------------------------------------------------------

def preProcImg(img, dilation=1, erosion=1, flagGrey=False):
    """ pre-processing image.

    Args:
        img (numpy.ndarray): Image (BGR image) array to process. 
        dilation (int): Number of iteration for cv2.dilate. 
        erosion (int): Number of iteration for cv2.erode. 
        flagGrey (bool): Turn to grey image.

    Returns:
        img (numpy.ndarray): image after pre-processing. 
    """
    if DEBUG: print("modCVFunc.preProcImg()")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    if dilation > 0:
        img = cv2.dilate(img, kernel, iterations=dilation)
    if erosion > 0:
        img = cv2.erode(img, kernel, iterations=erosion)
    if flagGrey:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#-------------------------------------------------------------------------------

def findColor(img, hsvMin, hsvMax, r=None):
    """ Find a color(range: hsvMin - hsvMax) in an area of an image.

    Args:
        img (numpy.ndarray): Input image (BGR).
        hsvMin (tuple): Minimum values to find the color.
        hsvMax (tuple): Maximum values to find the color.
        r (None/tuple): Rect (x1,y1,x2,y2). If None, use entire image.

    Returns:
        retImg (numpy.ndarray): Binary image indicating the found color.
    """
    if DEBUG: print("modCVFunc.findColor()")
    
    mask = np.zeros((img.shape[0], img.shape[1]) , dtype=np.uint8)
    if r == None:
        # Upper Left, Lower Left, Lower Right, Upper Right
        pts = [(0,0), (0,img.shape[0]), 
               (img.shape[1],img.shape[0]), (img.shape[1],0)]
    else:
        pts = [(r[0], r[1]), (r[0], r[3]), (r[2], r[3]), (r[2], r[1])] 
    cv2.fillConvexPoly(mask, np.asarray(pts), 255)
    tmpColImg = cv2.bitwise_and(img, img, mask=mask)
    tmpColImg = cv2.cvtColor(tmpColImg, cv2.COLOR_BGR2HSV)
    retImg = cv2.inRange(tmpColImg, hsvMin, hsvMax)
    __, retImg = cv2.threshold(retImg, 50, 255, cv2.THRESH_BINARY)
    return retImg

#-------------------------------------------------------------------------------

def getCentroid(img):
    """ Get the centroid of binary image. 

    Args:
        img (numpy.ndarray): Input binary image.

    Returns:
        (tuple): Coordinate of the centroid.
    """
    if DEBUG: print("modCVFunc.getCentroid()")

    m = cv2.moments(img)
    mx = int(m['m10']/m['m00'])
    my = int(m['m01']/m['m00'])
    return (mx, my)

#-------------------------------------------------------------------------------





