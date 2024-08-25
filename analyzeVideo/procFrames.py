# coding: UTF-8

"""
Computer vision processing on ant videos for FeatureDetector.

This program was coded and tested in Ubuntu 18.04.

Jinook Oh, Cremer group in Institute of Science and Technology Austria.
2021.May.
last edited: 2024-04-28

Dependency:
    wxPython (4.0)
    NumPy (1.17)
    OpenCV (4.1)

------------------------------------------------------------------------
Copyright (C) 2019-2020 Jinook Oh & Sylvia Cremer.
- Contact: jinook0707@protonmail.com

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
"""

import itertools
from time import time, sleep
from datetime import timedelta
from copy import copy
from glob import glob
from random import randint, choice
from os import path, mkdir
import logging
logging.basicConfig(
    format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
    level=logging.DEBUG
    )

import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans 
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist # Pairwise distances 
  # between observations in n-dimensional space. 
#from skimage.metrics import structural_similarity as ssim

from initVars import *
from modFFC import * 
from modCV import * 

FLAGS = dict(
                debug = False,
                )

_v = cv2.__version__.split("-")[0]
CV_Ver = [int(x) for x in _v.split(".")]

#===============================================================================

class ProcFrames:
    """ Class for processing a frame image using computer vision algorithms
    to code animal position/direction/behaviour
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ##### [begin] setting up attributes -----
        self.p = parent
        self.bg = None # background image of chosen video
        self.fSize = (960, 540) # default frame size
        self.cluster_cols = [
                                (200,200,200), 
                                (255,0,0), 
                                (0,255,0), 
                                (0,0,255), 
                                (255,100,100), 
                                (100,255,100), 
                                (100,100,255), 
                                (0,255,255),
                                (255,0,255), 
                                (255,255,0), 
                                (100,255,255),
                                (255,100,255), 
                                (255,255,100), 
                            ] # BGR color for each cluster in clustering
        #self.storage = {} # storage for previsouly calculated parameters 
        #  or temporary frame image sotrage, etc...
        ##### [end] setting up attributes -----

    #---------------------------------------------------------------------------
    
    def initOnLoading(self):
        """ initialize variables when input video was loaded
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.p.animalECase == "a2024":
            self.prevPGImg = None
            # store the last modification time
            self.videoMTime = getFileMTime(self.p.inputFP) 
            self.font= cv2.FONT_HERSHEY_SIMPLEX
            self.tScale = 0.5
            (self.tW, self.tH), self.tBL = cv2.getTextSize("0", self.font, 
                                                           self.tScale, 1) 
        
        else:
            self.prevPGImg = None

    #---------------------------------------------------------------------------
    
    def preProcess(self, q2m):
        """ pre-process video before running actual analysis 
        to obtain certain info such as ant's body length or background image.
        
        Args:
            q2m (None/queue.Queue): Queue to send data to main thread.
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.p
        nFrames = prnt.vRW.nFrames
        aecp = prnt.aecParam
        nF4pp = min(nFrames, 10) # number of frames for pre-processing 
        fIntv = max(1, int(nFrames/nF4pp))
        fH, fW = prnt.vRW.currFrame.shape[:2] # frame height & width
        nAnts = aecp["uAntsNum"] # number of ants in a container 
        nBroods = aecp["uBroodNum"] # number of brood items in a container 
        # number of rows & columns of continaers
        if "uCRows" in aecp.keys(): cRows = aecp["uCRows"]["value"]
        else: cRows = 1
        if "uCCols" in aecp.keys(): cCols = aecp["uCCols"]["value"]
        else: cCols = 1
        nC = cRows * cCols # number of containers
        
        ### HSV min & max of ant color
        colThr = {"min":[], "max":[]}
        for mm in ["min", "max"]:
            for hsv in ["H", "S", "V"]:
                colThr[mm].append(aecp[f'uColA-{mm}-{hsv}']['value'])
            colThr[mm] = tuple(colThr[mm])
        
        antLen = aecp["uAntLength"]["value"] # initial ant length
        # min. area for one ant blob to rule out errors in ant color detection
        minAArea = int(antLen * (antLen/3) * 0.75) 
        maxAArea = int(antLen * (antLen/3) * 1.5) 
        
        retData = {} # data to return
        if prnt.animalECase in ["[TBA"]:
        # certain cases which requires ant-body-length 
            retData["aLen"] = [] 

        if prnt.animalECase in ["[TBA]"]:
        # certain cases which requires background extraction
            retData["bgExtract"] = None 

        if len(retData) == 0:
            q2m.put(("finished", retData,), True, None)
            return 

        if "bgExtract" in retData.keys():
        # backgrouind extraction
            ext = "." + prnt.inputFP.split(".")[-1]
            bgFP = prnt.inputFP.replace(ext, "_bg.jpg")
            if path.isfile(bgFP):
                flagBgImg = True
                bg = cv2.imread(bgFP, cv2.IMREAD_GRAYSCALE)
            else:
                flagBgImg = False
                bg = np.zeros_like(prnt.vRW.currFrame, dtype=np.float32)

        fi = 0
        for idx in range(nF4pp):
            fi += fIntv 
            if idx % 10 == 0:
                msg = "Pre-processing ... "
                msg += f'{fi}/ {nFrames}; '
                msg += f'{int(idx/nF4pp*100)} %'
                q2m.put(("displayMsg", msg), True, None)
            ret = prnt.vRW.getFrame(fi)
            if not ret: continue
            frame = prnt.vRW.currFrame

            ##### [begin] determine the length of ant -----
            if "aLen" in retData.keys():
                ### get result image, detected with ant color
                antColRslt = findColor(frame, colThr["min"], colThr["max"])
                antColRslt= cv2.medianBlur(antColRslt, 5)
                
                # find blobs in the color-detection result image
                #   using connectedComponentsWithStats
                ccOutput = cv2.connectedComponentsWithStats(
                                                        antColRslt, 
                                                        connectivity=4
                                                        )
                nLabels = ccOutput[0] # number of labels
                labeledImg = ccOutput[1]
                # stats = [left, top, width, height, area]
                stats = list(ccOutput[2])
                lblsWArea = [] # labels with area
                for li in range(1, nLabels):
                    a = stats[li][4]
                    if a < minAArea or a > maxAArea: continue
                    lblsWArea.append([a, li])
                bLens = []
                for area, li in lblsWArea: # area and index
                    l, t, w, h, a = stats[li]
                    lcX = int(l + w/2) # center-point;x 
                    lcY = int(t + h/2) # center-point;y
                    ### calculate aligned line with the found blob
                    ptL = np.where(labeledImg==li) 
                    sPt = np.hstack((
                          ptL[1].reshape((ptL[1].shape[0],1)),
                          ptL[0].reshape((ptL[0].shape[0],1))
                          )) # stacked points
                    rr = cv2.minAreaRect(sPt)
                    lx, ly, lpts, rx, ry, rpts = calcLRPts(rr)
                    # calculate the length of the blob
                    bLen = np.sqrt((rx-lx)**2 + (ry-ly)**2)
                    # store the median length 
                    retData["aLen"].append(bLen) 
            ##### [end] determine the length of ant -----

            if "bgExtract" in retData.keys():
            # backgrouind image extraction
                if not flagBgImg: # bg image file doesn't exist
                    # accumulate the frame image
                    cv2.accumulateWeighted(frame, bg, 1/nFrames)

        if "aLen" in retData.keys():
            # get median value of all collected ant blob lengths
            retData["aLen"] = int(np.round(np.median(retData["aLen"])))

        if "bgExtract" in retData.keys():
            if not flagBgImg: # bg image file doesn't exist
                bg = cv2.convertScaleAbs(bg) # background image
                ### increase brightness of background image
                hsvF = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
                val = int(np.mean(hsvF[...,2])-np.mean(hsv[...,2]))
                vVal = hsv[...,2]
                hsv[...,2] = np.where((255-vVal)<val, 255, vVal+val)
                bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # removing details, then thresholding.
                __, bg = self.removeDetails(bg, None)
                # save as a file
                cv2.imwrite(bgFP, bg)
                retData["bgExtract"] = bg

        # return data 
        q2m.put(("finished", retData,), True, None)

    #---------------------------------------------------------------------------
    
    def proc_img(self, fImg, tD):
        """ Process frame image to code animal position/direction/behaviour
        
        Args:
            fImg (numpy.ndarray): Frame image array.
            tD (dict): temporary data to process such as 
              hD, bD, hPos, bPos, etc..
        
        Returns:
            fImg (numpy.ndarray): Image to return after processing.
            tD (dict): return data, including hD, bD, hPos, bPos, etc..
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        aec = self.p.animalECase
        fImg, tD = eval(f'self.proc_{aec}(tD, fImg)')
        return fImg, tD
  
    #---------------------------------------------------------------------------
    
    def drawStatusMsg(self, tD, fImg, fontP={}, pos=(0, 0),
                      drawAntSzRect=False, asrCol=(0,127,0)):
        """ draw status message and other common inforamtion on fImg
        
        Args:
            tD (dict): Dictionary to retrieve/store calculated data.
            fImg (numpy.ndarray): Frame image.
            fontP (dict): Font parameters. 
            pos (tuple): Position to write.
            drawAntSzRect (bool): Whether to draw ant-size rectangle.
            asrCol (tuple): Color for the ant size rect
        
        Returns:
            fImg (numpy.ndarray): Frame image array after drawing. 
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        prnt = self.p # parent 

        ### draw file-name & frame index
        if "font" in fontP.keys(): font = fontP["font"]
        else: font = cv2.FONT_HERSHEY_SIMPLEX
        if "scale" in fontP.keys(): scale = fontP["scale"]
        else: scale = 0.5 
        if "thck" in fontP.keys(): thck = fontP["thck"]
        else: thck = 1
        if "fCol" in fontP.keys(): fCol = fontP["fCol"]
        else: fCol = (0, 255, 0)
        fn = path.basename(prnt.inputFP)
        fi = prnt.vRW.fi
        nFrames = prnt.vRW.nFrames-1
        txt = "%s, %i/%i"%(fn, fi, nFrames)
        (tw, th), tBL = cv2.getTextSize(txt, font, scale, thck) 
        cv2.rectangle(fImg, pos, (pos[0]+tw+5,pos[1]+th+tBL), (0,0,0), -1)
        cv2.putText(fImg, txt, (pos[0]+5, pos[1]+th), font, 
                    fontScale=scale, color=fCol, thickness=thck)

        if drawAntSzRect:
            ### draw rectangle, representing ant size,
            ###   based on user-defined ant length
            antLen = prnt.aecParam["uAntLength"]["value"] # ant length
            aRectH = round(antLen/3)
            pos0 = (pos[0], pos[1]+th+tBL+10)
            txt = "ant rect size:"
            (tw, th), tBL = cv2.getTextSize(txt, font, scale, thck) 
            cv2.putText(fImg, txt, (pos0[0], pos0[1]+th), font, 
                        fontScale=scale, color=asrCol, thickness=thck)
            asrPos1 = (pos0[0]+tw+5, pos0[1])
            asrPos2 = (asrPos1[0]+antLen, asrPos1[1]+aRectH)
            cv2.rectangle(fImg, asrPos1, asrPos2, asrCol, -1)

        return fImg
     
    #---------------------------------------------------------------------------
    
    def proc_a2024(self, tD, fImg):
        """ Detecting motions of the ants in each ROI.
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            fImg (numpy.ndarray): Frame image.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            fImg (numpy.ndarray): Return image.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        prnt = self.p # parent
        aecp = prnt.aecParam # animal experiment case parameters
        fH, fW, __ = fImg.shape # frame shape
        passedMSec = timedelta(milliseconds=prnt.vRW.timestamp)
        addSec = timedelta(seconds=aecp["sec2timestamp"]["value"])
        fts = self.videoMTime + passedMSec + addSec
        dImgTyp = prnt.dispImgType
        # store the frame-timestamp
        tD["timestamp"] = fts 
        if dImgTyp != "Frame":
            # img other than frame image for showing results of 
            # certain detection algorithms
            dImg = np.zeros((fH,fW), dtype=np.uint8) 
        fi = prnt.vRW.fi # frame-index
        nFrames = prnt.vRW.nFrames # number of frames
        cRows = aecp["uCRows"]["value"] # rows of wells 
        cCols = aecp["uCCols"]["value"] # columns of wells 
        nC = cRows * cCols # n of containers 
        clrTags = ["ant", "brood"] # color targets A:Ants, B:Brood-items
        nAnts = aecp["uAntsNum"]["value"] # number of ants in a container
        nBroods = aecp["uBroodNum"]["value"] # number of broods in a container
        aLen = aecp["uAntLength"]["value"] # ant body length
        aArea = aLen * (aLen*0.333) # size of ant body
        # lower & upper thresholds for motion detection
        minMCont = aecp["motionCntThrMin"]["value"]/100 * aArea
        maxMotion = aecp["motionCntThrMax"]["value"]/100 * aArea

        # get pre-processed grey image
        __, pGImg = self.removeDetails(fImg, None)
        if dImgTyp == "Greyscale" : dImg = cv2.add(dImg, pGImg)

        if self.prevPGImg is None: # first run
            self.prevPGImg = pGImg.copy()
            mDiff = np.zeros(pGImg.shape, dtype=np.uint8)
        else:
            # difference image for motion detection
            mDiff = cv2.absdiff(pGImg, self.prevPGImg)
        self.prevPGImg = pGImg.copy()
        # apply cv2.threshold function in motion detection
        ret, mDiff = cv2.threshold(mDiff, aecp["motionThr"]["value"], 255, 
                                   cv2.THRESH_BINARY)
        if dImgTyp == "Motion": dImg = mDiff

        ### get result image, detected with ant color & brood
        imgClrD = {} # color-detection result images
        for clrT in clrTags:
            colThr = {"min":[], "max":[]}
            for mm in ["min", "max"]:
                for hsv in ["H", "S", "V"]:
                    key = f'uCol{clrT[0].upper()}-{mm}-{hsv}'
                    colThr[mm].append(aecp[key]['value'])
                colThr[mm] = tuple(colThr[mm])
            imgClrD[clrT] = findColor(fImg, colThr["min"], colThr["max"])
            imgClrD[clrT] = cv2.medianBlur(imgClrD[clrT], 1)

        mPts4D = [] # motion-points to draw on return image 
        mbr4D = {} # min. bounding rect to draw on return image 
        for clrT in clrTags: mbr4D[clrT] = []
        for ri in range(cRows):
            for ci in range(cCols):
            # go through rows & columnsn of wells 
                
                ### region-of-interest of this container
                roi = []
                for k in ["x", "y", "r"]:
                    if k == "x": refL = fW
                    else: refL = fH
                    roiV = aecp[f'uROI-{ri}-{ci}-{k}']["value"]/100 * refL 
                    roi.append(int(roiV))
                
                # key for storing motion-points
                mptk = f'motionPts{ri}{ci}' 
                ### key for storing color blob rects
                brk = {}
                for clrT in clrTags:
                    brk[clrT] = f'{clrT}BlobRectPts{ri}{ci}' 
                
                ##### [begin] find motion points -----
                _img = mDiff.copy()
                _img = maskImg(_img, [roi], 0)
                mPxls = int(np.sum(_img)/255)
                if mPxls > minMCont and mPxls < maxMotion:
                    ### get contours of motion
                    mode = cv2.RETR_EXTERNAL 
                    method = cv2.CHAIN_APPROX_SIMPLE
                    if getOpenCVVersion() >= 4.0:
                        cnts, hierarchy = cv2.findContours(_img, mode, method)
                    else:
                        _, cnts, hierarchy = cv2.findContours(_img, mode, 
                                                              method)
                    for cnti in range(len(cnts)):
                        mr = cv2.boundingRect(cnts[cnti])
                        # ignore too small contour
                        if mr[2]+mr[3] < minMCont: continue 
                        if dImgTyp == "Motion":
                            # draw the contour
                            cv2.drawContours(dImg, cnts, cnti, 255, 
                                             thickness=cv2.FILLED)
                        x = mr[0]+int(mr[2]/2)
                        y = mr[1]+int(mr[3]/2)
                        # store the contour's center point 
                        tD[mptk] += f'&{x}/{y}'
                        # store for drawing on return image later
                        mPts4D.append((x,y)) 
                    if tD[mptk] != "": tD[mptk] = tD[mptk].lstrip("&")
                ##### [end] find motion points -----

                #### [begin] find color blob -----
                for clrT in clrTags:
                    if clrT == "ant": nSubj = nAnts
                    elif clrT == "brood": nSubj = nBroods 
                    tDK = brk[clrT] # key for blob-rects in tD
                    tD[tDK] = "" # empty the data
                    ### get the color image of this container 
                    _img = imgClrD[clrT].copy()
                    _img = maskImg(_img, [roi], 0)
                    # find blobs in the color-detection result image
                    #   using connectedComponentsWithStats
                    ccOutput = cv2.connectedComponentsWithStats(_img, 
                                                                connectivity=8)
                    nLabels = ccOutput[0] # number of labels
                    labeledImg = ccOutput[1]
                    # stats = [left, top, width, height, area]
                    stats = list(ccOutput[2])
                    lblsWArea = [] # labels with area
                    for li in range(1, nLabels):
                        lblsWArea.append([stats[li][4], li])
                    # sort reverse by area
                    lblsWArea = sorted(lblsWArea, reverse=True)
                    for _, li in lblsWArea:
                        # enough data collected, break
                        if len(tD[tDK].split("&")) >= nSubj: break 

                        if dImgTyp == f'{clrT.capitalize()}-color':
                        # display image type is the color detection
                            dImg[labeledImg==li] = 255 # draw the blob
                        
                        l, t, w, h, a = stats[li]
                       
                        if clrT == "ant": minArea = aArea*0.333
                        elif clrT == "brood": minArea = aArea*0.5
                        maxArea = aArea * nSubj 
                        if a < minArea or a > maxArea:
                        # reached too small or too large ant-color blob
                            break
                        
                        ### calculate minimum bounding rect for this blob 
                        ptL = np.where(labeledImg==li) 
                        sPt = np.hstack((
                              ptL[1].reshape((ptL[1].shape[0],1)),
                              ptL[0].reshape((ptL[0].shape[0],1))
                              )) # stacked points
                        rotatedR = cv2.minAreaRect(sPt)

                        ### ignore this blob if it is too thin & long
                        ### (threshold is 6.0; approx. 2 ants 
                        ### in a straight line)
                        shThr = 6.0
                        lx, ly, lpts, rx, ry, rpts = calcLRPts(rotatedR)
                        dist1 = np.sqrt((lpts[0][0]-lpts[1][0])**2 + \
                                        (lpts[0][1]-lpts[1][1])**2)
                        dist2 = np.sqrt((lpts[0][0]-rpts[0][0])**2 + \
                                        (lpts[0][1]-rpts[0][1])**2)
                        if max(dist1,dist2)/min(dist1,dist2) > shThr:
                            continue

                        ### store to tD 
                        box = cv2.boxPoints(rotatedR)
                        box = np.int0(box)
                        for _i in range(4):
                            if tD[tDK] != "" and _i == 0: tD[tDK] += '&'
                            tD[tDK] += f'{box[_i][0]}/{box[_i][1]}'
                            if _i < 3: tD[tDK] += "/"

                        # store for drawing on return image later
                        mbr4D[clrT].append(box)
                ##### [end] find color blob ----- 

        if prnt.dispImgType != "Frame":
            # change frame image to the detection image 
            fImg = cv2.cvtColor(dImg, cv2.COLOR_GRAY2BGR) 

        #### [begin] display info on return image -----  
        # set colors for drawing
        c = dict(roi=(255,50,50), motion=(50,255,50),
                 ant=(0,255,255), brood=(255,255,0))

        ### draw status string
        if fImg.shape[0] < 1000: fontP = dict(scale=0.5, thck=1)
        else: fontP = dict(scale=1.0, thck=2)
        pos = (5, 5)
        fImg = self.drawStatusMsg(None, fImg, fontP, pos, True, c["ant"])

        for ri in range(cRows):
            for ci in range(cCols):
            # go through rows & columnsn of containers
                roi = {}
                for k in ["x", "y", "r"]:
                    val = aecp[f'uROI-{ri}-{ci}-{k}']["value"]
                    if k == "x": refL = fW
                    else: refL = fH
                    roi[k] = int(val/100 * refL)
                # draw circle around at container (ROI)
                cv2.circle(fImg, (roi["x"],roi["y"]), roi["r"], 
                           c["roi"], 2)
               
                ### draw continaer index
                txtPos = (roi["x"]-roi["r"]+int(self.tW*1.5), 
                          roi["y"]-roi["r"]+self.tH+self.tBL)
                cv2.putText(fImg, f'{ri} {ci}', txtPos, self.font,
                            fontScale=self.tScale, color=c["roi"], thickness=2)
        ### draw min. bounding rect of color-blobs
        for clrT in clrTags:
            for box in mbr4D[clrT]:
                cv2.drawContours(fImg, [box], 0, c[clrT], 2)
        ### draw motion-points
        r = max(1, int(aLen/12))
        for x, y in mPts4D:
            cv2.circle(fImg, (x,y), r, c["motion"], -1) 
        #### [end] display info on return image -----

        return fImg, tD 

    #---------------------------------------------------------------------------
    
    def storeFrameImg(self, fImg):
        """ store the frame image of the current frmae of the opened video 
        
        Args: 
            fImg (numpy.ndarray): Frame image.

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
       
        inputFP = self.p.vRW.fPath
        ext = "." + inputFP.split(".")[-1]
        folderPath = inputFP.replace(ext, "") + "_frames"
        if not path.isdir(folderPath): # folder doesn't exist
            mkdir(folderPath) # make one 
        fp = path.join(folderPath, "f%07i.jpg"%(self.p.vRW.fi)) # file path
        if not path.isfile(fp): # frame image file doesn't exist
            cv2.imwrite(fp, fImg) # save frame image 

    #---------------------------------------------------------------------------

    def removeDetails(self, img, bgImg):
        """ processing of removing details, then thresholding.

        Args:
            img (numpy.ndarray): image array to process.
            bgImg (numpy.ndarray/ None): background image to subtract.

        Returns:
            diff (numpy.ndarray): greyscale image after BG subtraction.
            edged (numpy.ndarray): greyscale image of edges in 'diff'.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if type(bgImg) == np.ndarray: 
            # get difference between the current frame and the background image 
            diffCol = cv2.absdiff(img, bgImg)
            diff = cv2.cvtColor(diffCol, cv2.COLOR_BGR2GRAY)
        else:
            diffCol = None 
            diff = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        aecp = self.p.aecParam
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        if "rdMExOIter" in aecp.keys() and aecp["rdMExOIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_OPEN, 
                        kernel, 
                        iterations=aecp["rdMExOIter"]["value"],
                        ) # to decrease noise & minor features
        if "rdMExCIter" in aecp.keys() and aecp["rdMExCIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_CLOSE, 
                        kernel, 
                        iterations=aecp["rdMExCIter"]["value"],
                        ) # closing small holes
        if "rdThres" in aecp.keys() and aecp["rdThres"]["value"] != -1:
            __, diff = cv2.threshold(
                            diff, 
                            aecp["rdThres"]["value"], 
                            255, 
                            cv2.THRESH_BINARY
                            ) # make the recognized part clear 
        return diffCol, diff
    
    #---------------------------------------------------------------------------
    
    def getEdge(self, greyImg):
        """ Find edges of greyImg using cv2.Canny and cannyTh parameters

        Args:
            greyImg (numpy.ndarray): greyscale image to extract edges.

        Returns:
            (numpy.ndarray): greyscale image with edges.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        return cv2.Canny(greyImg,
                         self.p.aecParam["cannyThMin"]["value"][0],
                         self.p.aecParam["cannyThMax"]["value"][1])
    
    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass

