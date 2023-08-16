# coding: UTF-8

"""
Computer vision processing on ant videos for FeatureDetector.

This program was coded and tested in Ubuntu 18.04.

Jinook Oh, Cremer group in Institute of Science and Technology Austria.
2021.May.
last edited: 2023-03-27

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

from modFFC import * 
from modCV import * 

FLAGS = dict(
                debug = False,
                )

_v = cv2.__version__.split("-")[0]
CV_Ver = [int(x) for x in _v.split(".")]

#===============================================================================

class ProcCV:
    """ Class for processing a frame image using computer vision algorithms
    to code animal position/direction/behaviour
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent):
        if FLAGS["debug"]: logging.info(str(locals()))
        
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
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.p.animalECase == "michaela20":
            aecp = self.p.aecParam # animal experiment case parameters
            nPDish = aecp["uNPDish"]["value"] # number of petri dishes
            nAnts = aecp["uNSubj"]["value"] # number of ants (in a petri-dish)
              # last frame-index where position of three ants were found  
            self.prevGPImg = [] # processed grey image of a petri-dish
              # in the previous frame
            self.pMovCntInfo = [] # contour info of movements in past
              # frames (2 dictionaries for 2 petri dishes)
            self.pBlobInfo = [] # ant blob info found in past frames
            self.aPos_lastFI = [] # last frame-index where ant's position 
              # is recorded.
            self.spots2ignore = [] # list of coordinates to ignore, due to
              # color on the plaster (confusing with ants' color markers)
            for pdi in range(nPDish):
                self.prevGPImg.append(None)
                self.pMovCntInfo.append({})
                self.pBlobInfo.append({})
                self.aPos_lastFI.append([])
                for ai in range(nAnts):
                    self.aPos_lastFI[-1].append(-1)
        
        elif self.p.animalECase in ["aggrMax21", "motion21"]:
            # pre-processed grey image of the previous frame
            self.prevPGImg = None  

        elif self.p.animalECase == "lindaWP21":
            # pre-processed grey image of the previous frame
            self.prevPGImg = None
            # previously detected center point of 4 petri dishes
            self.prevPDishCt = [(-1,-1)]*4 
            # ant position in the previous frame
            #self.prevAP = []
            # pupae position in the previous frame
            #self.prevPP = []
        
        elif self.p.animalECase == "sleepDet23":
            self.prevPGImg = None
            self.durNoMotion = None  
            self.font= cv2.FONT_HERSHEY_SIMPLEX
            self.tScale = 0.5
            (self.tW, self.tH), self.tBL = cv2.getTextSize("0", self.font, 
                                                           self.tScale, 1) 
            ### set ROI for each well 
            aecp = self.p.aecParam
            wRows = aecp["uWRows"]["value"]
            wCols = aecp["uWCols"]["value"]
            fH = self.p.vRW.currFrame.shape[0]
            self.wROI = {} # ROI (x,y,r) of each well
            rFrac = 1/wRows * 0.4
            rad = int(rFrac * fH)
            for ri in range(wRows):
                for ci in range(wCols):
                # go through rows & columnsn of wells 
                    key = f'w{ri}{ci}'
                    x = int((rFrac + ci*rFrac*2.65) * fH)
                    y = int((rFrac + ri*rFrac*2.945) * fH)
                    # set roi (x, y, rad) for each well
                    self.wROI[key] = [x, y, rad]

        else:
            self.prevPGImg = None

        self.mind = None

    #---------------------------------------------------------------------------
    
    def preProcess(self, q2m):
        """ pre-process video before running actual analysis 
        
        Args:
            q2m (None/queue.Queue): Queue to send data to main thread.
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        p = self.p

        if p.animalECase == "aggrMax21":
        # aggressive behavior for Max's experiment; backgrouind extraction
            ext = "." + p.inputFP.split(".")[-1]
            bgFP = p.inputFP.replace(ext, "_bg.jpg")

            if not path.isfile(bgFP):
                # set number of frames to run to extract BG
                nFrames = min(10001, p.vRW.nFrames) 
                bg = np.zeros_like(p.vRW.currFrame, dtype=np.float32)
                for fi in range(nFrames):
                    msg = "Background extraction... %.1f %%"%(fi/nFrames*100)
                    q2m.put(("displayMsg", msg), True, None)
                    ret = p.vRW.getFrame(-1)
                    frame = p.vRW.currFrame
                    cv2.accumulateWeighted(frame, bg, 1/nFrames)
                bg = cv2.convertScaleAbs(bg) # background image
                ### increase brightness of background image
                hsvF = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
                val = int(np.mean(hsvF[...,2])-np.mean(hsv[...,2]))
                vVal = hsv[...,2]
                hsv[...,2] = np.where((255-vVal)<val, 255, vVal+val)
                bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # get pre-processed gray image 
                __, bg = self.commonPreProc(bg, None)
                cv2.imwrite(bgFP, bg)
            else: 
                bg = cv2.imread(bgFP, cv2.IMREAD_GRAYSCALE)
            q2m.put(("finished", bg,), True, None)

        elif p.animalECase == "sleepDet23":
        # ant sleep-detection; go through first some frames (1000 or more?)
        # to determine ant's body length (head to gaster without 
        # legs or antennae by using more of cv2.medianBlur) in each well
            aecp = p.aecParam
            fH, fW = p.vRW.currFrame.shape[:2]
            wRows = aecp["uWRows"]["value"]
            wCols = aecp["uWCols"]["value"]
            roiOffsetX = int(aecp["uROIOffsetX"]["value"] * fH)
            roiOffsetY = int(aecp["uROIOffsetY"]["value"] * fH)
            nFrames = p.vRW.nFrames
            # n of frames for pre-processing 
            nF4pp = min(nFrames, max(100, int(nFrames * 0.001)))
            fIntv = max(1, int(p.vRW.nFrames/nF4pp))
            alData = {} # ant length data
            fi = 0
            for idx in range(nF4pp):
                fi += fIntv 
                if idx % 10 == 0:
                    msg = "estimating each ant's body length.. "
                    msg += f'{fi}/ {nFrames}; '
                    msg += f'{int(idx/nF4pp*100)} %'
                    q2m.put(("displayMsg", msg), True, None)
                ret = p.vRW.getFrame(fi)
                if not ret: continue
                frame = p.vRW.currFrame
                
                ### get result image, detected with ant color 
                colMin = tuple(aecp["uColA_min"]["value"])
                colMax = tuple(aecp["uColA_max"]["value"])
                rect = [0, 0, fW, fH]
                antColRslt = self.find_color(rect, frame, colMin, colMax)
                antColRslt= cv2.medianBlur(antColRslt, 5)

                for ri in range(wRows):
                    for ci in range(wCols):
                    # go through rows & columnsn of wells 
                        key = f'w{ri}{ci}' # well key
                        alK = f'uAntLen{ri}{ci}' # ant-length key
                        # init list for this key
                        if alK not in alData.keys(): alData[alK] = []

                        ##### [begin] find the ant blob -----
                        roi = copy(self.wROI[key])
                        ### apply offset to ROI
                        roi[0] += roiOffsetX
                        roi[1] += roiOffsetY
                        ### get the ant color image of this well 
                        acImg = antColRslt.copy()
                        acImg = maskImg(acImg, [roi], 0)
                        
                        # find blobs in the color-detection result image
                        #   using connectedComponentsWithStats
                        ccOutput = cv2.connectedComponentsWithStats(
                                                                acImg, 
                                                                connectivity=8
                                                                )
                        nLabels = ccOutput[0] # number of labels
                        labeledImg = ccOutput[1]
                        # stats = [left, top, width, height, area]
                        stats = list(ccOutput[2])
                        lblsWArea = [] # labels with area
                        for li in range(1, nLabels):
                            lblsWArea.append([stats[li][4], li])
                        lblsWArea = sorted(lblsWArea)
                        if lblsWArea != []:
                            # area and index of the largest blob
                            area, lli = lblsWArea[-1]
                            l, t, w, h, a = stats[lli]
                            lcX = int(l + w/2) # center-point;x 
                            lcY = int(t + h/2) # center-point;y
                            ### calculate aligned line with the found blob
                            ptL = np.where(labeledImg==lli) 
                            sPt = np.hstack((
                                  ptL[1].reshape((ptL[1].shape[0],1)),
                                  ptL[0].reshape((ptL[0].shape[0],1))
                                  )) # stacked points
                            rr = cv2.minAreaRect(sPt)
                            lx, ly, lpts, rx, ry, rpts = self.calcLRPts(rr)
                            # calculate length of the blob
                            bLen = np.sqrt((rx-lx)**2 + (ry-ly)**2)
                            # store the length 
                            alData[alK].append(bLen) 
                        ##### [end] find the ant blob ----- 
            
            ### get median value of all collected ant blob lengths
            for ri in range(wRows):
                for ci in range(wCols):
                    alK = f'uAntLen{ri}{ci}' # ant-length key
                    alData[alK] = int(np.round(np.median(alData[alK])))

            # send median ant-length data 
            q2m.put(("finished", alData,), True, None)

    #---------------------------------------------------------------------------
    
    def proc_img(self, frame_arr, animalECase, 
                 tD, flagMP=False, imgType='RGB-image'):
        """ Process frame image to code animal position/direction/behaviour
        
        Args:
            frame_arr (numpy.ndarray): Frame image array.
            animalECase (str): Animal experiment case.
            tD (dict): temporary data to process such as 
              hD, bD, hPos, bPos, etc..
            flagMP (bool): Whether some data positions were give manually. 
            imgType (str): Image type to return.
        
        Returns:
            tD (dict): return data, including hD, bD, hPos, bPos, etc..
            frame_arr (numpy.ndarray): Frame image to return after processing.
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        p = self.p # parent
        ecp = self.p.aecParam
        diff = None
        edged = None
                            
        ##### [begin] calculate data of the current frame ---
        d = p.oData[p.vRW.fi] # output data of the current frame
          # this might have already calculated data
        
        if animalECase == "michaela20":
            retG, frame_arr = self.proc_michaela20(frame_arr)
            tD = None # this one doesn't use tD and it uses 
              # multiple previous frames directly from 'oData' of 
              # the main frame.

        else:
            tD, retG, frame_arr = eval(
                                    "self.proc_%s(tD, frame_arr)"%(animalECase)
                                    )
        ##### [end] calculate data of the current frame ---
     
        if "debug" in imgType and type(retG) == np.ndarray:
            frame_arr = cv2.cvtColor(retG, cv2.COLOR_GRAY2BGR) 

        return tD, frame_arr 
  
    #---------------------------------------------------------------------------
    
    def drawStatusMsg(self, tD, frame_arr, drawAntSzRect=False, fontP={},
                      pos=(0, 0)):
        """ draw status message and other common inforamtion on frame_arr
        
        Args:
            tD (dict): Dictionary to retrieve/store calculated data.
            frame_arr (numpy.ndarray): Frame image array.
            drawAntSzRect (bool): Whether to draw ant-size rectangle.
            fontP (dict): Font parameters. 
            pos (tuple): Position to write.
        
        Returns:
            frame_arr (numpy.ndarray): Frame image array after drawing. 
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
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
        cv2.rectangle(frame_arr, pos, (pos[0]+tw+5,pos[1]+th+tBL), (0,0,0), -1)
        cv2.putText(frame_arr, txt, (pos[0]+5, pos[1]+th), font, 
                    fontScale=scale, color=fCol, thickness=thck)

        aecp = prnt.aecParam # animal experiment case parameters
        fH, fW = frame_arr.shape[:2]
       
        if drawAntSzRect:
            antLen = aecp["uAntLength"]["value"] # ant length
            ### draw rectangle, representing ant size,
            ###   based on user-defined ant length
            col = (0,127,0)
            mX = int(fW/2)
            aRectH = round(antLen/3)
            txt = "approx. ant size:"
            (tw, th), tBL = cv2.getTextSize(txt, font, scale, thck) 
            cv2.putText(frame_arr, txt, (pos[0]+mX-tw-5, pos[1]+th), font, 
                        fontScale=scale, color=col, thickness=thck)
            pt1 = (pos[0]+mX, pos[1]+th-aRectH)
            pt2 = (pos[0]+mX+antLen, pos[1]+pt1[1]+aRectH)
            cv2.rectangle(frame_arr, pt1, pt2, col, -1)

        return frame_arr
     
    #---------------------------------------------------------------------------
    
    def drawHeadDir(self, tD, frame_arr):
        """ draw head direction on frame_arr
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.
        
        Returns:
            frame_arr (numpy.ndarray): Frame image array after drawing. 
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if "hD" in tD and type(tD["hD"]) == int: # head direction available
            p = self.p # parent
            if p.ratFImgDispImg != None:
                r = 1.0/p.ratFImgDispImg
                lw = int(2 * r)
                cr = int(4 * r)
            pt1 = (tD["hPosX"], tD["hPosY"])
            pt2 = (tD["bPosX"], tD["bPosY"])
            cv2.line(frame_arr, pt1, pt2, (0,255,0), lw)
            cv2.circle(frame_arr, pt1, cr, (0,125,255), -1)
        return frame_arr

    #---------------------------------------------------------------------------

    def proc_rat15(self, tD, frame_arr):
        """ Calculate head direction of a rat 
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            diff (numpy.ndarray): grey image after background subtraction.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        diffCol, diff = self.commonPreProc(frame_arr, self.bg)
        edged = self.getEdged(diff)
        cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)

        if type(tD["p_hD"]) == int: 
            ### draw a far point with the known head direction 
            ###   from previous base point (bPos) 
            s = np.sin(np.deg2rad(tD["p_hD"]))
            c = np.cos(np.deg2rad(tD["p_hD"]))
            lL = max(cnt_br[2], cnt_br[3]) # length of line
            fpt = (int(tD["p_bPosX"]+lL*c),
                   int(tD["p_bPosY"]-lL*s)) # far front point away from head
            cv2.circle(frame_arr, fpt, 5, (50,50,50), -1)
            ### cluster subject points 
            dPts = np.where(diff==255)
            # coordinates of all the 255 pixels of diff
            dPts = np.hstack((dPts[1].reshape((dPts[1].shape[0],1)),
                              dPts[0].reshape((dPts[0].shape[0],1)))) 
            t_dPts = dPts.astype(np.float32)
            centroids, __ = kmeans(
                                obs=t_dPts,
                                k_or_guess=self.p.aecParam["uNKMC"]["value"]
                                ) # kmeans clustering
            ### calculate distances between the fpt and centroids of clusters,
            ### the closest cluster is supposed to be the head cluster
            d_cents = [] # (distance to fpt, x, y)
            for ci in range(len(centroids)):
                cx, cy = centroids[ci]
                dist = np.sqrt((cx-fpt[0])**2 + (cy-fpt[1])**2)
                d_cents.append([dist, cx, cy])
            d_cents = sorted(d_cents) # sort by distance
            centroids = np.array(d_cents, dtype=np.uint16)[:,1:]
            ### get points of the head cluster                            
            idx, __ = vq(dPts, np.asarray(centroids))
            t_pts = dPts[np.where(idx==0)[0]]
            #frame_arr[t_pts[:,1],t_pts[:,0]] = self.cluster_cols[0]
            ### determine hPos as the closest pixel toward fpt 
            dists = []
            for hci in range(len(t_pts)):
                ptx, pty = t_pts[hci]
                dists.append(np.sqrt((ptx-fpt[0])**2 + (pty-fpt[1])**2))
            idx = dists.index(min(dists))
            ### store hPos
            tD["hPosX"] = int(t_pts[idx][0])
            tD["hPosY"] = int(t_pts[idx][1])
            col = (200, 200, 200)
            cv2.circle(frame_arr, (tD["hPosX"],tD["hPosY"]), 3, col, -1)
            ### draw each cluster centroids and 
            ###   calculates distances between 
            ###   the head cluster and other clusters 
            dists = [] # distances between the head cluster and other clusters
            hcx, hcy = centroids[0] # head cluster x & y
            for ci in range(1, len(centroids)):
                cv2.circle(frame_arr, tuple(centroids[ci]), 3, col, -1)
                cx, cy = centroids[ci]
                dists.append(np.sqrt((hcx-int(cx))**2 + (hcy-int(cy))**2))
                t_pts = dPts[np.where(idx==ci)[0]]
                frame_arr[t_pts[:,1],t_pts[:,0]] = self.cluster_cols[ci]
            # index of the closets cluster to the head cluster
            bi = dists.index(min(dists)) + 1
            ### store bPos
            tD["bPosX"] = int(centroids[bi][0])
            tD["bPosY"] = int(centroids[bi][1])
        if tD["hPosX"] == 'None' or tD["bPosX"] == 'None':
            tD["hD"] = tD["p_hD"] 
        else:
            tD["hD"] = calc_line_angle((tD["bPosX"],tD["bPosY"]), 
                                 (tD["hPosX"],tD["hPosY"]))
            if self.p.vRW.fi > 1: # not the first frame
                if type(tD["p_hD"]) == int:
                    degDiffTol = self.p.aecParam["uDegTh"]["value"]
                    if calc_angle_diff(tD["p_hD"], tD["hD"]) > degDiffTol:
                    # differnce is too big. keep the previous hD
                        tD["hD"] = tD["p_hD"]
        
        self.drawHeadDir(tD, frame_arr) # draw head direction
        self.drawStatusMsg(tD, frame_arr) # draw status message on frame 
        
        return tD, diff, frame_arr

    #---------------------------------------------------------------------------
    
    def proc_tagless20(self, tD, frame_arr):
        """ detect ants' positions 
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        tMe = time()
        fSh = frame_arr.shape # frame shape
        fH = fSh[0] # frame height
        p = self.p # parent
        fi = p.vRW.fi # frame-index
        aecp = p.aecParam # animal experiment case parameters
        nAnts = aecp["uNSubj"]["value"] # number of ants
        gSz = aecp["uGasterSize"]["value"] # gaster size
        gr = int(np.sqrt(gSz/np.pi)) # radius of gaster
        antSz = gSz * 3 # approximate area of one ant 
        antLen = gr * 6
        col = [(255,0,0), (0,255,0), (0,0,255),
               (255,255,0), (255,0,255), (0,255,255),
               (127,0,0), (0,127,0), (0,0,127),
               (127,127,0),  (127,0,127), (0,127,127),
               (150,150,150), (255,255,255)]
        
        if not hasattr(self, "mask"):
            # make an empty mask image
            self.mask = np.zeros((fSh[0], fSh[1]), dtype=np.uint8)
        else:
            self.mask[:,:] = 0 # init
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

        # mask image
        frame_arr = maskImg(frame_arr, [aecp["uROI"]["value"]], (255,255,255)) 
        
        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None) 
        pGImg= cv2.bitwise_not(pGImg) # make ants white
        #eImg = self.getEdged(pGImg) # get image of edge-detection

        # mark gaster size
        cv2.circle(frame_arr, (int(fSh[1]/2), gr*2), gr, (127,127,127), -1)

        # find blobs using connectedComponentsWithStats
        ccOutput = cv2.connectedComponentsWithStats(pGImg, connectivity=4)
        nLabels = ccOutput[0] # number of labels
        labeledImg = ccOutput[1]
        stats = list(ccOutput[2]) # stat matrix(left, top, width, height, area)]
        fL = [] # labels which fit into our category (min. area)
        aVals = [] # area values
        for li in range(1, nLabels):
            if stats[li][4] >= gSz:
                fL.append(li)
                aVals.append(stats[li][4])
        #print(aVals, np.median(aVals)) 
        ai = 0
        bInfo = [] # ant blob info (blob-length, center-x, center-y, blob-angle)
        tmpImg = pGImg.copy()
        #tmpImg[tmpImg==255] = 100
        for i in range(len(fL)):
            li = fL[i] # label index
            # estimated number of individuals in this blob
            estN = int(max(1, np.floor(aVals[i]/antSz)))
            ptL = np.where(labeledImg==li) # points with the current label
            sPt = np.hstack((
                              ptL[1].reshape((ptL[1].shape[0],1)),
                              ptL[0].reshape((ptL[0].shape[0],1))
                              )) # stacked points
            if estN > 1:
                bImg = np.zeros(pGImg.shape, dtype=np.uint8)
                bImg[ptL] = pGImg[ptL]
                #skImg = drawSkeleton(bImg)
                if CV_Ver[0] >= 4: 
                    cnts, hierarchy = cv2.findContours(bImg, 
                                                cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
                else:
                    __, cnts, hierarchy = cv2.findContours(bImg, 
                                                cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
                
                ### find search starting points
                _e = 0.1
                while True:
                    epsilon = _e*cv2.arcLength(cnts[0],True)
                    approx = cv2.approxPolyDP(cnts[0],epsilon,True)
                    if approx.shape[0] >= estN*1.5: break
                    _e -= 0.005
                #print(fi, _e, approx.shape[0])
                approx = approx.reshape((approx.shape[0],2)).tolist()
                ''' 
                bImg[bImg==255] = 100
                for a in approx: cv2.circle(bImg, tuple(a), 2, 255, -1)
                return tD, bImg, frame_arr
                '''
                _app = list(approx)
                
                #cv2.imwrite("x.jpg", bImg)
                lines = []
                for _i in range(estN):
                    pSum = [] # sum of pixels and relevant info
                    for appI, pt1 in enumerate(approx):
                    # go through points from approxPolyDP on the blob 
                        pt1 = tuple(pt1)
                        for ang in range(-180, 179, 5):
                        # go through range of degrees
                            ### erase in a direction & store sum of pixels
                            _img = bImg.copy()
                            pt2 = calc_pt_w_angle_n_dist(ang, antLen, 
                                                bPosX=pt1[0], bPosY=pt1[1],
                                                flagScreen=True)
                            cv2.line(_img, pt1, pt2, 0, int(gr*1.0))
                            pSum.append((np.sum(_img), pt1, pt2, appI))
                    pSum = sorted(pSum)
                    # get the data when the largest area is erased
                    __, pt1, pt2, appI = pSum[0]
                    lines.append((pt1, pt2))
                    cv2.line(bImg, pt1, pt2, 0, gr*2)
                    #cv2.imwrite("x%02i.jpg"%(_i), bImg)
                    approx.pop(appI)
                    ai += 1
                
                ### mark ant line, its center and approxPolyDP points
                lCt = []
                for l in lines:
                    cv2.line(tmpImg, l[0], l[1], 100, gr*2) 
                    x1, y1 = l[0]
                    x2, y2 = l[1]
                    x = min(x1,x2) + int(abs(x1-x2)/2)
                    y = min(y1,y2) + int(abs(y1-y2)/2)
                    lCt.append((x,y))
                for ct in lCt:
                    cv2.circle(tmpImg, ct, 5, 150, -1)
                    # store the ant center point
                    bInfo.append((antLen, ct[0], ct[1]))
                for pt in _app:
                    cv2.circle(tmpImg, tuple(pt), 3, 200, -1)
            else: 
                ### calculate body line
                rr = cv2.minAreaRect(sPt)
                lx, ly, lpts, rx, ry, rpts = self.calcLRPts(rr)
                bLen = np.sqrt((rx-lx)**2 + (ry-ly)**2)
                x = int(lx + (rx-lx)/2)
                y = int(ly + (ry-ly)/2)
                bInfo.append((bLen, x, y))
                cv2.circle(tmpImg, (x,y), 5, 150, -1)
                ai += 1
        
        if ai > nAnts: # too many blobs are recognized as ant blobs 
            bInfo = sorted(bInfo, reverse=True) # sort by blob length
            bInfo = bInfo[:nAnts] # cut off smallest blobs 
        
        '''
        ### store each blob info 
        if fi == 0:
            for bi in range(len(bInfo)):
                tD["a%03iPosX"%(bi)] = bInfo[bi][1] 
                tD["a%03iPosY"%(bi)] = bInfo[bi][2]
        else:
            aIdx = list(range(nAnts))
            ### assign blob coordinate, if close enough to previous coordinate 
            thr = gr*1.5
            for thr in [gr*1.5, gr*3]:
                for _i, ai in enumerate(aIdx):
                    px = tD["p_a%03iPosX"%(ai)] 
                    py = tD["p_a%03iPosY"%(ai)] 
                    if px == 'None': continue
                    dists = []
                    for bi in range(len(bInfo)):
                        __, bx, by = bInfo[bi]
                        dist = np.sqrt((bx-px)**2 + (by-py)**2)
                        dists.append((dist, bi))
                    if dists == []: continue
                    dist, bi = sorted(dists)[0] # closest blob
                    if dist <= thr: # within threshold 
                        ### assign coordinate
                        tD["a%03iPosX"%(ai)] = bInfo[bi][1] 
                        tD["a%03iPosY"%(ai)] = bInfo[bi][2]
                        bInfo.pop(bi)
                        print(ai)
                        aIdx[_i] = None
                while None in aIdx: aIdx.remove(None)
            ### assign blob coordinate, if no previous coordinate is available
            for bi in range(len(bInfo)):
                for ai in aIdx:
                    if tD["p_a%03iPosX"%(ai)] == "None":
                        tD["a%03iPosX"%(ai)] = bInfo[bi][1] 
                        tD["a%03iPosY"%(ai)] = bInfo[bi][2]
                        aIdx.remove(ai)
                        break
            ### copy previous coordinate
            for ai in aIdx:
                tD["a%03iPosX"%(ai)] = copy(tD["p_a%03iPosX"%(ai)])
                tD["a%03iPosY"%(ai)] = copy(tD["p_a%03iPosY"%(ai)])
        
        ### display assigned ant indices 
        for ai in range(nAnts):
            cv2.putText(frame_arr, str(ai), 
                        (tD["a%03iPosX"%(ai)], tD["a%03iPosY"%(ai)]),
                        cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, 
                        color=(0,255,0), thickness=2)
        '''
        for bi in range(len(bInfo)):
            __, x, y = bInfo[bi]
            tD["a%03iPosX"%(bi)] = x 
            tD["a%03iPosY"%(bi)] = y
            cv2.circle(frame_arr, (x,y), 5, (0,255,0), -1)

        ### save images
        ext = self.p.vRW.fPath.split(".")[-1]
        folderPath = self.p.vRW.fPath.replace(".%s"%(ext), "")
        if not path.isdir(folderPath): # folder doesn't exist
            mkdir(folderPath) # make folder
        fp = path.join(folderPath, "f%06i.jpg"%(self.p.vRW.fi)) # file path
        cv2.imwrite(fp, frame_arr)
         
        self.drawStatusMsg(tD, frame_arr, True) # draw status message on frame 
        print(fi, time()-tMe) 
        return tD, tmpImg, frame_arr 
  
    #---------------------------------------------------------------------------
    
    def proc_tagless22(self, tD, frame_arr):
        """ detect ants' positions 
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.mind is None:
            self.mind = Mind(
                                self.p, # parent
                                object2mind = "ant", # object-to-mind is ant
                                )

        else:
            self.mind.find_ant(tD, frame_arr)
    
    #---------------------------------------------------------------------------
    
    def proc_michaela20(self, frame_arr):
        """ Detect ants' positions of three ants in a petri dish
        [experiment of Michaela Hoenigsberger, 2020]
        
        Use color tracking first to find color markings on ants, which
          will be trusted. Then, connected blobs will be sought to find ant(s).
        * Angle of ant blob is not used.
        
        Args:
            frame_arr (numpy.ndarray): Frame image array.
        
        Returns:
            diff (numpy.ndarray): grey image after background subtraction.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        ##### [begin] init constants & variables -----
        flagSaveFrameImgs = True 
        origGray = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
        fSh = frame_arr.shape # frame shape
        fH = fSh[0] # frame height
        p = self.p # parent
        pOutputData = p.oData # parent's output data (this function need access
          # to data of multiple previous frames
        pDataCols = p.dataCols # list of data columns
        fi = p.vRW.fi # frame-index
        nFrames = p.vRW.nFrames # number of frames
        aecp = p.aecParam # animal experiment case parameters
        aSFrame = aecp["uASFrame"]["value"] # analysis start frame 
        nPDish = aecp["uNPDish"]["value"] # number of petri dishes
        nAnts = aecp["uNSubj"]["value"] # number of ants (in a petri-dish)
        aMinArea = aecp["uGasterSize"]["value"] # minimum area of ant body
        aFullLen = int(np.sqrt(aMinArea)*3) # Full length of ant body.
          # It's multiplied by 3, because
          # aMinArea = approximate area of head or gaster of ant. 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        retGrayImg = np.zeros((fSh[0], fSh[1]), dtype=np.uint8)
        self.mask = np.zeros((fSh[0], fSh[1]), dtype=np.uint8)
        ##### [end] init constants & variables ----- 
        
        if flagSaveFrameImgs:
            self.storeFrameImg(frame_arr)
       
        # get pre-processed grey image
        __, procGImg = self.commonPreProc(frame_arr, None)

        if self.spots2ignore != []:
        # if spots to ignore (for color detection) exists
            rad = int(np.sqrt(aMinArea)/2) # radius of spot
            for spot in self.spots2ignore: # go through each spot
                # draw white color circle over the spot
                cv2.circle(frame_arr, spot, rad, (255,255,255), -1)
        
        for pdi in range(nPDish):
        # go through number of petri dishes 
            aplFI = self.aPos_lastFI[pdi]
            x = int(fSh[1] / 2)
            y = int(fSh[0] / 2)
            roiRad = min(x, y)
            offset = aecp["uROI%i_offset"%(pdi)]["value"]
            roiX = x + offset[0]
            roiY = y + offset[1]
            roiRad += offset[2]
            maskedImg = maskImg(procGImg.copy(), [(roiX,roiY,roiRad)], 255) 
            # display ROI area on frame image
            cv2.circle(frame_arr, (roiX, roiY), roiRad+2, (50,50,50), 4)

            if fi < aSFrame[pdi]: # analysis start frame didn't reach yet. 
                self.prevGPImg[pdi] = maskedImg.copy()
                continue # no need to analyze this frame
             
            ### get movements data in the current frame 
            if fi == 0: # first frame
                self.prevGPImg[pdi] = maskedImg.copy()
            else:
                m_diff = cv2.absdiff(maskedImg, self.prevGPImg[pdi])
                ret, m_diff = cv2.threshold(m_diff, 30, 255, cv2.THRESH_BINARY)
                edged = self.getEdged(m_diff)
                cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)
                # store movement contour info.
                self.pMovCntInfo[pdi][fi] = cnt_info 
                m_val = int(np.sum(m_diff)/255) # number of pixels, recognized 
                  # as movement (after cv2.threshold) in the petri dish
                mColIdx = pDataCols.index("movements%02i"%(pdi))
                pOutputData[fi][mColIdx] = str(m_val) # store movement value
                  # in a petri-dish
                self.prevGPImg[pdi] = maskedImg.copy() 

            # make binary image of darker color (= ants) as white (255)
            ret, antBlobImg = cv2.threshold(maskedImg,
                                            aecp["uAColTh"]["value"],
                                            255,
                                            cv2.THRESH_BINARY_INV)

            if aecp["uDilate"]["value"] > 0:
                # dilate ant blob image
                antBlobImg = cv2.dilate(antBlobImg, 
                                        kernel, 
                                        iterations=aecp["uDilate"]["value"]) 
            retGrayImg = antBlobImg.copy()

            ##### [begin] find ant blobs -----
            ccOutput = cv2.connectedComponentsWithStats(antBlobImg,
                                                        connectivity=4)
            nLabels = ccOutput[0] # number of labels (probably ant blobs)
            labeledImg = ccOutput[1]
            stats = list(ccOutput[2]) # stat [left, top, width, height, area]
            ccInfo = [] # connected component information (area, label index), 
              # which fit into the category of minimum area 
            for li in range(1, nLabels):
                if stats[li][4] >= aMinArea:
                    ccInfo.append((stats[li][4], li))
            ccInfo = sorted(ccInfo, reverse=True) # sort by area size 
            #print(ccInfo)
            antBlobCnt = 0
            bInfo = [] # ant blob info (blob-length, center-x, center-y)
              # angle is not used.
            for i in range(len(ccInfo)):
            # Go through found labels (ant blobs). 
                area = ccInfo[i][0] # area of connected component
                if area > aMinArea*3*4:
                # area is larger than approximate size of 4 ants
                    continue # ignore
                li = ccInfo[i][1] # label index
                # set estimated number of individuals in this blob
                if i == 0 and len(ccInfo) < nAnts:
                # this is the largest connected component (i == 0)
                #   & number of connected components is lower than 
                #   expected number of ants
                    estN = nAnts - (len(ccInfo)-1)
                else:
                    estN = 1
                dPts = np.where(labeledImg==li) # points with the current label
                dPts = np.hstack((dPts[1].reshape((dPts[1].shape[0],1)),
                                  dPts[0].reshape((dPts[0].shape[0],1)))) 
                if estN > 1:
                    # kmeans clustering with estimated number of ants
                    centroids, __ = kmeans(obs=dPts.astype(np.float32),
                                           k_or_guess=estN) 
                    idx, __ = vq(dPts, np.asarray(centroids))
                    for ci in range(len(centroids)):
                        cx, cy = centroids[ci]
                        dPts_k = dPts[np.where(idx==ci)]
                        # mark entire ant's body
                        #frame_arr[dPts_k[:,1],dPts_k[:,0]] = (100,100,255) 
                        ### calculate body line
                        rr = cv2.minAreaRect(dPts_k)
                        lx, ly, lpts, rx, ry, rpts = self.calcLRPts(rr)
                        # calculate body angle
                        #bAng = calc_line_angle((lx,ly), (rx,ry))
                        bLen = int(np.sqrt((rx-lx)**2 + (ry-ly)**2))
                        bInfo.append((bLen, int(cx), int(cy))) #, int(bAng)))
                        antBlobCnt += 1
                else:
                    ### calculate body line
                    rr = cv2.minAreaRect(dPts)
                    lx, ly, lpts, rx, ry, rpts = self.calcLRPts(rr)
                    # calculate body angle
                    #bAng = calc_line_angle((lx,ly), (rx,ry))
                    bLen = int(np.sqrt((rx-lx)**2 + (ry-ly)**2))
                    x = int(lx + (rx-lx)/2)
                    y = int(ly + (ry-ly)/2)
                    bInfo.append((bLen, x, y)) #, int(bAng)))
                    antBlobCnt += 1
             
            if antBlobCnt > nAnts: # too many blobs are recognized as ant blobs 
                bInfo = sorted(bInfo, reverse=True) # sort by blob length
                bInfo = bInfo[:nAnts] # cut off smallest blobs 

            self.pBlobInfo[pdi][fi] = bInfo # store the blob info.
            # At this point, number of blobs in bInfo should be same as 
            #   number of ants 
            #print(bInfo)
            ##### [end] find ant blobs -----
                    
            ##### [begin] find ants with color marks on gaster -----
            for ai in range(2): # there will be two color markers
            # index of two color markers = index of ant- 0 and 1.
                # last frame-index where position of this ant was found 
                lastIdx = aplFI[ai]
                # data-column-index of position-x of this ant 
                xColi = pDataCols.index("a%02i%02iPosX"%(pdi, ai))
                # data-column-index of position-y of this ant 
                yColi = pDataCols.index("a%02i%02iPosY"%(pdi, ai))
                colMin = tuple(self.p.aecParam["uCol%iMin"%(ai)]["value"])
                colMax = tuple(self.p.aecParam["uCol%iMax"%(ai)]["value"])
                rect = [roiX-roiRad, roiY-roiRad, roiX+roiRad, roiY+roiRad]
                fcRslt = self.find_color(rect, frame_arr, colMin, colMax)
                edged = self.getEdged(fcRslt)
                edged, cnts, hierarchy = cv2.findContours(
                                                  edged,
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE
                                                  )
                isColorFound = False
                if len(cnts) > 0:
                    cx = []
                    cy = []
                    for ci in range(len(cnts)):
                        br = cv2.boundingRect(cnts[ci])
                        x = br[0] + int(br[2]/2)
                        y = br[1] + int(br[3]/2)
                        dists = []
                        for bLen, bx, by in bInfo:
                            d = np.sqrt((bx-x)**2 + (by-y)**2)
                            dists.append(d)
                        if dists != []:
                            if min(dists) > aFullLen*2:
                            # this color marker is too far away
                            # from any of the ant blobs; incorrect detection(?) 
                              continue # ignore this color marker
                        cx.append(x)
                        cy.append(y)
                    if cx != [] and cy != []:
                        cmw = max(cx) - min(cx) # width of found color marker
                        cmh = max(cy) - min(cy) # height of found color marker
                        szTh = aFullLen/3
                        if cmw < szTh and cmh < szTh:
                        # size of found color marker should be small
                            isColorFound = True
                            if aplFI[ai] == -1 and fi > 0:
                            # This means that the first several frames
                            # were passed without successful color 
                            # marker finding. The 3rd ant position can't be
                            # determined without the two color marked ants.
                            # Thus, skip the first several frames for the
                            # 3rd ant as well.
                                aplFI[2] = fi
                            aplFI[ai] = fi # store the frame-index, where 
                              # color-marker was found
                            x = min(cx) + int(cmw/2)
                            y = min(cy) + int(cmh/2)
                            r = max(cmw, cmh)
                            if ai == 0: col = (100, 100, 255)
                            elif ai == 1: col = (100, 255, 255)
                            cv2.circle(frame_arr, (x, y), r, col, 1)
                            pOutputData[fi][xColi] = str(x) # store x-coordinate
                            pOutputData[fi][yColi] = str(y) # store y-coordinate
                            if lastIdx > -1 and lastIdx < fi-1:
                            # there are frame gaps between the current frame
                            #   and the last frame where color marker was found
                                px = int(pOutputData[fi-1][xColi])
                                py = int(pOutputData[fi-1][yColi])
                                d = np.sqrt((px-x)**2 + (py-y)**2)
                                if d > aFullLen*2: # distance between ant posi-
                                  # tion in the current and the previous frame
                                  # is too far away
                                    ### interpolate data bewtween lastIdx and fi
                                    nSteps = fi - lastIdx
                                    px = int(pOutputData[lastIdx][xColi])
                                    py = int(pOutputData[lastIdx][yColi])
                                    dX = (x-px) / nSteps
                                    dY = (y-py) / nSteps
                                    for step in range(1, nSteps+1):
                                        pfi = lastIdx + step
                                        nx = px + int(step*dX)
                                        ny = py + int(step*dY)
                                        pOutputData[pfi][xColi] = str(nx)
                                        pOutputData[pfi][yColi] = str(ny)
                            cv2.putText(frame_arr, 
                                        str(ai), 
                                        (x,y), 
                                        cv2.FONT_HERSHEY_DUPLEX, 
                                        fontScale=0.5, 
                                        color=(0,0,255),
                                        thickness=1) 
                if not isColorFound and fi > 0:
                # color marker was not found
                    ### store coordinate of the blob, which is close
                    ### to the position of ant in the previous frame
                    pxv = pOutputData[fi-1][xColi]
                    pyv = pOutputData[fi-1][yColi]
                    if not "None" in [pxv, pyv]: 
                        px = int(pxv)
                        py = int(pyv)
                        dists = []
                        for bi, (__, bx, by) in enumerate(bInfo):
                            d = np.sqrt((bx-px)**2 + (by-py)**2)
                            dists.append((d, bi))
                        dist, closeBI = sorted(dists)[0]
                        if dist < aFullLen*2:
                            __, x, y = bInfo[closeBI]
                        else: # found blob position is too far
                            ### copy position data from previous frame
                            x = int(pOutputData[fi-1][xColi])
                            y = int(pOutputData[fi-1][yColi])
                        pOutputData[fi][xColi] = str(x)
                        pOutputData[fi][yColi] = str(y)
                        cv2.circle(frame_arr, (x, y), 5, (175,175,175), 2)
            ##### [end] find ants with color marks on gaster ----- 
         
            ##### [begin] determine ant-2's position -----
            bCFIdx = min(aplFI[0], aplFI[1]) # last frame-index where 
              # both colors were found
            if bCFIdx > aplFI[2] or fi == nFrames-1:
            # there are frames where both colors were found,
            # while un-colored ant's position is not recorded.
            # OR
            # this is the last frame of the video.
                if bCFIdx > aplFI[2]:
                    startIdx = aplFI[2] + 1
                    endIdx = bCFIdx + 1
                elif fi == nFrames-1:
                    lastIdx = [] # last frame-index where data is available 
                    for ai in range(nAnts): lastIdx.append(aplFI[ai])
                    startIdx = min(lastIdx) + 1
                    endIdx = nFrames
                for pfi in range(startIdx, endIdx):
                # go through frames where ant-2 position can be determined
                    if bCFIdx > aplFI[2]:
                        ### remove blob info which are close to 
                        ###   the two colored ants' positions
                        _bInfo = self.pBlobInfo[pdi][pfi]
                        for cai in range(2): # through colored ants
                            xColi = pDataCols.index("a%02i%02iPosX"%(pdi, cai))
                            yColi = pDataCols.index("a%02i%02iPosY"%(pdi, cai))
                            cax = int(pOutputData[pfi][xColi])
                            cay = int(pOutputData[pfi][yColi])
                            distInfo = []
                            for bi in range(len(_bInfo)):
                                __, bx, by = _bInfo[bi]
                                dist = np.sqrt((bx-cax)**2 + (by-cay)**2)
                                distInfo.append((dist, bi))
                            if distInfo != []:
                                distInfo = sorted(distInfo)
                                dist, bi = distInfo[0]
                                _bInfo.pop(bi)
                        xColi = pDataCols.index("a%02i02PosX"%(pdi))
                        yColi = pDataCols.index("a%02i02PosY"%(pdi))
                        if len(_bInfo) > 0:
                            ### store the left blob's position as 
                            ###   the 3rd ant's position 
                            pOutputData[pfi][xColi] = str(_bInfo[0][1])
                            pOutputData[pfi][yColi] = str(_bInfo[0][2])
                        else:
                            ### copy the data from the previous frame
                            px = pOutputData[pfi-1][xColi]
                            py = pOutputData[pfi-1][yColi]
                            pOutputData[pfi][xColi] = copy(px)
                            pOutputData[pfi][yColi] = copy(py)
                    elif fi == nFrames-1:
                    # last frame
                        for ai in range(nAnts):
                            xColi = pDataCols.index("a%02i%02iPosX"%(pdi, ai))
                            yColi = pDataCols.index("a%02i%02iPosY"%(pdi, ai))
                            if pOutputData[pfi][xColi] == "None":
                            # if data is None
                                ### copy the last available data
                                px = pOutputData[lastIdx[ai]][xColi]
                                py = pOutputData[lastIdx[ai]][yColi]
                                #try:
                                px = int(px)
                                py = int(py)
                                #except:
                                #    pass
                                pOutputData[pfi][xColi] = copy(px)
                                pOutputData[pfi][yColi] = copy(py) 
                    del self.pBlobInfo[pdi][pfi]
                    if fi > 0:
                        ### determine individual ant's movements value
                        m_val = [0]*nAnts # movement values for each ant
                        for ci in range(len(self.pMovCntInfo[pdi][pfi])):
                            csz, cx, cy = self.pMovCntInfo[pdi][pfi][ci] 
                            distInfo = []
                            for ai in range(nAnts):
                                colTitle = "a%02i%02iPosX"%(pdi, ai)
                                xColi = pDataCols.index(colTitle)
                                colTitle = "a%02i%02iPosY"%(pdi, ai)
                                yColi = pDataCols.index(colTitle)
                                ax = int(pOutputData[pfi][xColi])
                                ay = int(pOutputData[pfi][yColi])
                                dist = np.sqrt((cx-ax)**2 + (cy-ay)**2)
                                distInfo.append((dist, ai))
                            distInfo = sorted(distInfo)
                            dist, ai = distInfo[0]
                            m_val[ai] += int(csz/2)
                        ### store the movements value of each ant 
                        for ai in range(nAnts):
                            mColi = pDataCols.index("a%02i%02imov"%(pdi, ai))
                            pOutputData[pfi][mColi] = str(m_val[ai])
                        del self.pMovCntInfo[pdi][pfi]
                aplFI[2] = bCFIdx # store the frame-index, where position
                  # of ant-2 (non-color marked one) is determined
            ##### [end] determine ant-2's position -----

            self.aPos_lastFI[pdi] = aplFI # update last frame index where
              # positions were found

        # draw line to check aFullLen is approximately same as ant length
        cv2.line(frame_arr, (10, 75), (10+aFullLen, 75), (50,50,50), 3)
        p.oData = pOutputData # update output data of the parent 
        # draw status message on frame 
        self.drawStatusMsg(None, frame_arr, True) 
        
        return retGrayImg, frame_arr 

    #---------------------------------------------------------------------------
    
    def proc_aggrMax21(self, tD, frame_arr):
        """ Finding aggression moments in a pair of ants
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        fSh = frame_arr.shape # frame shape
        fH = fSh[0] # frame height
        p = self.p # parent
        fi = p.vRW.fi # frame-index
        nFrames = p.vRW.nFrames # number of frames
        aecp = p.aecParam # animal experiment case parameters
        nPDish = aecp["uNPDish"]["value"] # number of petri dishes 
        antLen = aecp["uAntLength"]["value"] # ant length 
        gr = antLen / 6 # radius of gaster
        gSz = np.pi*gr**2 # gaster size
        antSz = gSz * 3 # approximate area of one ant 
        flagSaveFrameImgs = False 

        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None)
       
        ### get foreground (ant blobs in white)
        fg = cv2.absdiff(pGImg, self.bg)
        ret, antBlobImg = cv2.threshold(fg, aecp["uAColTh"]["value"], 255, 
                                        cv2.THRESH_BINARY)

        if fi > 0:
            '''
            ### get difference image for motion 
            m_diff = cv2.absdiff(pGImg, self.prevPGImg)
            ret, m_diff = cv2.threshold(m_diff, 30, 255, cv2.THRESH_BINARY)
            #m_val = int(np.sum(m_diff)/255) # number of pixels, recognized 
              # as movement (after cv2.threshold) in the petri dish
            '''
            ### get optical flow info 
            flow = cv2.calcOpticalFlowFarneback(self.prevPGImg,
                                        pGImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optFMag, optFAng = cv2.cartToPolar(flow[...,0], flow[...,1])
            # normalize magnitude
            optFMag = cv2.normalize(optFMag,None,0,255,cv2.NORM_MINMAX)
            # cut off too low magnitudes
            optFMag[optFMag<127] = 0
            #ret, optFMag = cv2.threshold(optFMag, 50, 255, cv2.THRESH_BINARY)
            optFAng = optFAng * 180 / np.pi / 2 # radian to range of 0-180
            ### divide angels into low number of sections 
            nAngSec = 8 # number of sections in angle
            _deg = 180 / nAngSec
            optFAng = np.uint8(optFAng/_deg) * _deg
            ### draw the found optical flow
            optFImg = np.zeros_like(frame_arr)
            optFImg[...,0] = optFAng 
            optFImg[...,1] = 255
            optFImg[...,2] = optFMag 
            optFImg = cv2.cvtColor(optFImg,cv2.COLOR_HSV2BGR)
        self.prevPGImg = pGImg.copy()

        retGrayImg = pGImg
        for pdi in range(nPDish):
        # go through number of petri dishes
            ### prepare ROI parameters
            roiX = int(fSh[1] / 2)
            roiY = int(fSh[0] / 2)
            roiRad = min(roiX, roiY)
            offset = aecp["uROI%i_offset"%(pdi)]["value"]
            roiX += offset[0]
            roiY += offset[1]
            roiRad += offset[2]
            # display ROI area on frame image
            cv2.circle(frame_arr, (roiX, roiY), roiRad+1, (255,255,0), 1)
            if fi == 0: continue 
            
            ### mask image
            abImg = antBlobImg.copy()
            abImg = maskImg(abImg, [(roiX,roiY,roiRad)], 0)

            # find blobs using connectedComponentsWithStats
            ccOutput = cv2.connectedComponentsWithStats(abImg, 
                                                        connectivity=8)
            nLabels = ccOutput[0] # number of labels
            labeledImg = ccOutput[1]
            stats = list(ccOutput[2]) # stat (left, top, width, height, area)
            abLst = []
            for li in range(1, nLabels):
                l, t, w, h, area = stats[li]
                #print(fi, pdi, li, area, gSz, antSz*6)
                # ignore if the component is too small or large
                if area < gSz or area > antSz*6: continue
                abLst.append([area, li, w, h])
            abLst = sorted(abLst, reverse=True) # large blob first
            nB = min(len(abLst), 2) # number of blobs (one or two ant blobs)
            if nB > 0: 
            # one or more blobs are found.
            # (one blob, when two ants are touching each other)
                brW = 0 # width of rect, bounding one or two blobs
                brH = 0 
                mar = [] # list of corner points of min. area rect
                for bi in range(nB):
                # a single blob or two ant blobs
                    ### get min. area rect
                    pts = np.where(labeledImg==abLst[bi][1])
                    pts = np.hstack((pts[1].reshape((pts[1].shape[0],1)),
                                     pts[0].reshape((pts[0].shape[0],1)))) 
                    rr = cv2.minAreaRect(pts)
                    box = np.int64(cv2.boxPoints(rr))
                    mar.append(box)
                    brW += abLst[bi][2]
                    brH += abLst[bi][3]
                cpt = [0, 0] 
                isAntClose = False
                if nB == 1: # only one blob found
                    isAntClose = True
                    for i1 in range(4):
                        cpt[0] += mar[0][i1][0]
                        cpt[1] += mar[0][i1][1]
                else: # two blobs are found
                    ### determine if it's close enough by the closest distance
                    dists = []
                    for i1 in range(4):
                        x1, y1 = mar[0][i1]
                        cpt[0] += x1
                        cpt[1] += y1
                        for i2 in range(4):
                            x2, y2 = mar[1][i2]
                            if i1 == 0:
                                cpt[0] += x2
                                cpt[1] += y2
                            dists.append(np.sqrt((x1-x2)**2 + (y1-y2)**2))
                    if min(dists) <= antLen: isAntClose = True
                
                if isAntClose:
                # if two ants are very close
                    mask = np.zeros_like(abImg)
                    for bi in range(nB):
                        cv2.polylines(frame_arr, [mar[bi]], True, (0,0,255), 1)
                        # mask optical flow image where ants are
                        cv2.drawContours(mask,[mar[bi]],0,255,-1)
                    aa = np.sum(mask)/255 # area around ants
                    _img = optFImg.copy()
                    _img = cv2.bitwise_and(_img, _img, mask=mask)

                    _grey = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                    nz = np.where(_grey>0, 1, 0)
                    # motion area where motion detected via optical flow
                    ma = np.sum(nz) 
                    mRat = ma / aa # ratio of motion, compared to ant-area
                    if mRat > 0.6:
                        # center point of all points
                        cpt = (int(cpt[0]/(nB*4)), int(cpt[1]/(nB*4)))
                        r = int(max(brW, brH)/2)
                        cv2.circle(frame_arr, cpt, r, (0,0,255), 2)
                        tD["p%03iAggression"%(pdi)] = 1 
                    frame_arr = cv2.add(frame_arr, _img)

            retGrayImg = cv2.add(retGrayImg, abImg)

        ### draw line for displaying user-defined ant length
        pt1 = (int(fSh[1]/2 - antLen/2), round(gr*4))
        pt2 = (int(fSh[1]/2 + antLen/2), round(gr*6))
        cv2.rectangle(frame_arr, pt1, pt2, (0,255,0), -1) 
        cv2.putText(frame_arr, "user-defined ant length", 
                    (int(fSh[1]/2+antLen), round(gr*6)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.3, color=(0,255,0), thickness=1)
        ### draw frame index
        cv2.putText(frame_arr, "%i/%i"%(fi, nFrames), 
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.3, color=(0,255,0), thickness=1)

        if flagSaveFrameImgs:
            self.storeFrameImg(frame_arr)
        
        return tD, retGrayImg, frame_arr

    #---------------------------------------------------------------------------
    
    def proc_egoCent21(self, tD, frame_arr):
        """ Finding ant blobs and motion in egocentric video 
        cropped by Christoph Sommer
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        fSh = frame_arr.shape # frame shape
        fH = fSh[0] # frame height
        p = self.p # parent
        fi = p.vRW.fi # frame-index
        nFrames = p.vRW.nFrames # number of frames
        aecp = p.aecParam # animal experiment case parameters
        antLen = aecp["uAntLength"]["value"] # ant length 
        gr = antLen / 6 # radius of gaster
        gSz = np.pi*gr**2 # gaster size
        antSz = gSz * 3 # approximate area of one ant  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None)
        # get ant blobs in white
        ret, antBlobImg = cv2.threshold(pGImg, aecp["uAColTh"]["value"], 255,
                                        cv2.THRESH_BINARY_INV)

        ### prepare ROI parameters
        roiX = int(fSh[1] / 2)
        roiY = int(fSh[0] / 2)
        roiRad = min(roiX, roiY)
        offset = aecp["uROI0_offset"]["value"]
        roiX += offset[0]
        roiY += offset[1]
        roiRad += offset[2] # ROI radius for image for detecting ant blobs
        roiRadM = int(roiRad/2) # ROI radius for motion image
         
        # mask image
        antBlobImg = maskImg(antBlobImg, [(roiX,roiY,roiRad)], 0)
         
        # dialte for nullifying light grey color crosshair
        antBlobImg = cv2.dilate(antBlobImg, kernel, iterations=1) 

        m_val = 0
        if fi > 0:
            # get difference image for motion
            m_diff = cv2.absdiff(pGImg, self.prevPGImg)
            ret, m_diff = cv2.threshold(m_diff, 30, 255, cv2.THRESH_BINARY)
            # mask motion image
            m_diff = maskImg(m_diff, [(roiX,roiY,roiRadM)], 0)
            m_val = int(np.sum(m_diff)/255) # motion value
            if m_val > 1000: m_val = 0 # hard-coded max. motion value
            retGrey = m_diff
        else:
            retGrey = np.zeros(antBlobImg.shape, dtype=np.uint8) 
        tD["motion"] = m_val # store motion value
        self.prevPGImg = pGImg.copy()

        # find blobs using connectedComponentsWithStats
        ccOutput = cv2.connectedComponentsWithStats(antBlobImg, 
                                                    connectivity=8)
        nLabels = ccOutput[0] # number of labels
        labeledImg = ccOutput[1]
        stats = list(ccOutput[2]) # stat (left, top, width, height, area)
        abLst = []
        fCtX = int(fSh[1]/2) # center-x of frame image
        fCtY = int(fSh[0]/2) # center-y of frame image 
        for li in range(1, nLabels):
            l, t, w, h, area = stats[li] 
            #print(fi, pdi, li, area, antSz*6)
            # ignore if the component is too small or large
            if area < antSz/2 or area > antSz*3: continue
            ctX = l + int(w/2)
            ctY = t + int(h/2)
            # distance from frame center
            dist = np.sqrt((ctX-fCtX)**2 + (ctY-fCtY)**2) 
            abLst.append([dist, li, area])
        abLst = sorted(abLst, reverse=True) # close blob (to center) first
        nB = len(abLst) # number of blobs

        if (nB == 1) and (abLst[0][2] > antSz*1.5):
        # there's a single blob and it has more than a single ant
        #   (= two ants are touching each other)
            # store whether the focal ant is close with another 
            tD["closeWithAnother"] = "True" 
        elif nB > 1:
        # there're more than a single blob
            ### get min. area rect of the focal ant and the closest ant blob
            minR = []
            for bi in range(2):
                pts = np.where(labeledImg==abLst[bi][1])
                pts = np.hstack((pts[1].reshape((pts[1].shape[0],1)),
                                 pts[0].reshape((pts[0].shape[0],1)))) 
                rr = cv2.minAreaRect(pts)
                box = np.int64(cv2.boxPoints(rr))
                minR.append(box)

            ### get distances between all min. area rect points 
            dists = []
            for i1 in range(4):
                x1, y1 = minR[0][i1]
                for i2 in range(4):
                    x2, y2 = minR[1][i2]
                    dists.append(np.sqrt((x1-x2)**2 + (y1-y2)**2))
            if min(dists) <= antLen/3: # if they are close enough
                tD["closeWithAnother"] = "True"

        #### [begin] display info on frame_arr -----
        # display ROI area on frame image
        cv2.circle(frame_arr, (roiX, roiY), roiRad+1, (255,255,0), 1)
        cv2.circle(frame_arr, (roiX, roiY), roiRadM+1, (255,255,0), 1)
        ### draw rectangle, representing ant size
        ###   , based on user-defined ant length
        pt1 = (int(fSh[1]/2 - antLen/2), round(gr))
        pt2 = (int(fSh[1]/2 + antLen/2), pt1[1]+round(gr*2))
        cv2.rectangle(frame_arr, pt1, pt2, (0,127,0), -1)
        ### draw frame index
        cv2.putText(frame_arr, "%i/%i"%(fi, nFrames),
                    (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3, color=(0,255,0), thickness=1)
        ### draw bounding rect of detected blobs
        for dist, li, area in abLst: 
            l, t, w, h, area = stats[li]
            cv2.rectangle(frame_arr, (l, t), (l+w, t+h), (0,255,0), 1)
        ### draw whether the focal ant is close with another ant 
        if tD["closeWithAnother"] == "True":
            cv2.rectangle(frame_arr, (0, 0), (fSh[1], fSh[0]), (0,0,255), 5)
        #### [end] display info on frame_arr -----

        return tD, retGrey, frame_arr

    #---------------------------------------------------------------------------
    
    def proc_lindaWP21(self, tD, frame_arr):
        """ Finding whether ant is close to pupae.
        Additionally, records amount of motion 
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        fH, fW, __ = frame_arr.shape # frame shape
        prnt = self.p # parent
        fi = prnt.vRW.fi # frame-index
        nFrames = prnt.vRW.nFrames # number of frames
        aecp = prnt.aecParam # animal experiment case parameters
        nPDish = aecp["uNPDish"]["value"] # number of petri dishes 
        # number of ants; "uNSubj" includes number of pupae 
        nAnts = int(aecp["uNSubj"]["value"] / 2)
        pDRad = int(aecp["uPDishRad"]["value"]/100*fH) # radius of petri dish
        antLen = aecp["uAntLength"]["value"] # ant length 
        gr = round(antLen / 6) # radius of gaster
        gSz = np.pi*gr**2 # gaster size
        antSz = gSz * 3 # approximate area of one ant  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 

        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None)
        # prepare grey image to return
        retGrey = np.zeros(pGImg.shape, dtype=np.uint8)

        if nPDish == 1: # there's only one container in the video
            pDishCt = [(int(fW/2), int(fH/2))] # center of petri dish

        elif nPDish == 4: # there are four containers in the video
            ### find petri-dish edge color 
            colMin = tuple(aecp["uColD_min"]["value"])
            colMax = tuple(aecp["uColD_max"]["value"])
            rect = [0, 0, fW, fH]
            fcRsltD = self.find_color(rect, frame_arr, colMin, colMax)
            fcRsltD = cv2.morphologyEx(fcRsltD, cv2.MORPH_CLOSE, kernel, 
                                       iterations=3) # closing small holes
            fcRsltD = cv2.medianBlur(fcRsltD, 11)
            if "petri-dish" in prnt.dispImgType.lower(): retGrey = fcRsltD 
            
            ### detect circles
            minDist = int(fH/6) 
            param1 = 30
            param2 = 30 # smaller -> more false circles
            minRadius = int(fH/6)
            maxRadius = int(fH/4)
            circles = cv2.HoughCircles(fcRsltD, cv2.HOUGH_GRADIENT, 1, 
                                       minDist, param1=param1, param2=param2, 
                                       minRadius=minRadius, maxRadius=maxRadius)
            ##### [begin] calculate center points of 4 petri-dishes -----
            pDishCt = copy(self.prevPDishCt)
            if circles is None:
                abortCalcPDCt = True # abort calculation of petri-dish centers
            else:
                circles = np.uint16(np.around(circles))[0,:]
                if circles.shape[0] >= 3:
                # three or four petri-dishes are found
                    ### determine center of four petri-dishes
                    minX = np.min(circles[:,0])
                    maxX = np.max(circles[:,0])
                    minY = np.min(circles[:,1])
                    maxY = np.max(circles[:,1])
                    ctFX = int(np.mean((minX, maxX)))
                    ctFY = int(np.mean((minY, maxY)))
                    if "white" in prnt.dispImgType.lower():
                        cv2.circle(retGrey, (ctFX, ctFY), 20, 175, 5)
                    abortCalcPDCt = False 
                else:
                # two or less petri-dish is found
                    ### find white paper & lighting
                    colMin = tuple(aecp["uColWh_min"]["value"])
                    colMax = tuple(aecp["uColWh_max"]["value"])
                    rect = [0, 0, fW, fH]
                    fcRsltWh = self.find_color(rect, frame_arr, colMin, colMax)
                    fcRsltWh = cv2.medianBlur(fcRsltWh, 1) 
                    if "white" in prnt.dispImgType.lower(): retGrey = fcRsltWh

                    ### get center of the white color area (entire setup)
                    edged = self.getEdged(fcRsltWh)
                    cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)
                    ctFX = cnt_br[0] + int(cnt_br[2]/2)
                    ctFY = cnt_br[1] + int(cnt_br[3]/2)
                    if "white" in prnt.dispImgType.lower():
                        cv2.rectangle(retGrey, tuple(cnt_br[:2]), 
                              (cnt_br[0]+cnt_br[2],cnt_br[1]+cnt_br[3]), 175, 5)

                    ### determine whether the found white area is large enough
                    ###   (to avoid incorrect petri-dish assignment when
                    ###    hand is occluding the camera view partially)
                    if cnt_br[2] < pDRad*6 or cnt_br[3] < pDRad*4: 
                        abortCalcPDCt = True 
                    else:
                        abortCalcPDCt = False 
            if abortCalcPDCt: 
                pDishCt = [(-1,-1)]*4 
            else:
                for x, y, r in circles:
                    ### determine petri-dish index
                    if x < ctFX:
                        if y < ctFY: pdi = 0
                        else: pdi = 2 
                    else:
                        if y < ctFY: pdi = 1
                        else: pdi = 3
                    # store the petri-dish center point
                    pDishCt[pdi] = (int(x), int(y))
                    # updating petri dish center in class variable
                    self.prevPDishCt[pdi] = (int(x), int(y))
                    if "petri-dish" in prnt.dispImgType.lower():
                        cv2.circle(retGrey, pDishCt[pdi], pDRad, 175, 2)
            ##### [end] calculate center points of 4 petri-dishes -----

        ### get difference image for motion
        if fi > 0:
            m_diff = cv2.absdiff(pGImg, self.prevPGImg)
            mThr = aecp["motionThr"]["value"]
            ret, m_diff = cv2.threshold(m_diff, mThr, 255, cv2.THRESH_BINARY)
        self.prevPGImg = pGImg.copy()

        aRect = [] # ant rects in each petri-dish
        pRect = [] # pupae rects in each petri-dish
        # whether ant and pupae is close in each petri-dish
        apClose = [False]*nPDish 
        
        for pdi in range(nPDish):
        # go through number of petri dishes

            ### list for rects of ants and pupae in a petri-dish
            aRect.append([])
            pRect.append([])
            
            # the petri-dish in this position has not recognized
            if pDishCt[pdi] == (-1, -1): continue
            roiX, roiY = pDishCt[pdi] 

            m_val = 0
            if fi > 0:
                # mask motion image
                _diff = maskImg(m_diff, [(roiX,roiY,pDRad)], 0)
                m_val = int(np.sum(_diff)/255) # motion value
                if m_val > antSz*2*nAnts:
                # too large motion value > ignore 
                    m_val = 0 
                if "motion" in prnt.dispImgType.lower(): retGrey = m_diff
            tD["motion_%02i"%(pdi)] = m_val # store motion value

            # mask color image
            colImg = maskImg(frame_arr, [(roiX,roiY,pDRad)], (127,127,127))

            ### get ant blob by color
            colMin = tuple(aecp["uColA_min"]["value"])
            colMax = tuple(aecp["uColA_max"]["value"])
            rect = [0, 0, fW, fH]
            fcRsltA = self.find_color(rect, colImg, colMin, colMax)
            fcRsltA = cv2.dilate(fcRsltA, kernel, 1) 

            ### determine ant position
            # find blobs using connectedComponentsWithStats
            ccOutput = cv2.connectedComponentsWithStats(fcRsltA, connectivity=8)
            nLabels = ccOutput[0] # number of labels
            labeledImg = ccOutput[1]
            # stat matrix(left, top, width, height, area)]
            stats = list(ccOutput[2]) 
            fL = [] # labels which fit into the category (size)
            for li in range(1, nLabels):
                # set upper limit of blob size with some margin.
                # (* nAnts = blob could be bundled ants)
                # (* 2 = margin; blob size could be much larger depending on
                #    posture and its shadow)
                upperL = antSz * nAnts * 2 
                if gSz*0.75 <= stats[li][4] < upperL:
                    fL.append([stats[li][4], li])
            fL = sorted(fL, reverse=True) # large blob first
            if len(fL) > 0:
                if len(fL) > nAnts: fL = fL[:nAnts]
                for fli in range(len(fL)):
                    li = fL[fli][1]
                    aRect[pdi].append(stats[li][:4])
                    if "ant" in prnt.dispImgType.lower():
                        ### draw image for debugging
                        r = aRect[pdi][-1]
                        cv2.rectangle(retGrey, tuple(r[:2]), 
                                      (r[0]+r[2],r[1]+r[3]), 150, 1)

            ### get pupae blob by color
            colMin = tuple(aecp["uColP_min"]["value"])
            colMax = tuple(aecp["uColP_max"]["value"])
            rect = [0, 0, fW, fH]
            fcRsltP = self.find_color(rect, colImg, colMin, colMax)
            fcRsltP = cv2.dilate(fcRsltP, kernel, 1) 

            ### determine pupae position
            # find blobs using connectedComponentsWithStats
            ccOutput = cv2.connectedComponentsWithStats(fcRsltP, connectivity=8)
            nLabels = ccOutput[0] # number of labels
            labeledImg = ccOutput[1]
            # stat matrix(left, top, width, height, area)]
            stats = list(ccOutput[2]) 
            fL = [] # labels which fit into the category (size/area)
            for li in range(1, nLabels):
                if gSz*0.25 <= stats[li][4] < antSz*nAnts*2:
                # size fits
                    fL.append([stats[li][4], li])
            fL = sorted(fL, reverse=True) # large blob first
            if len(fL) > 0:
                if len(fL) > nAnts: fL = fL[:nAnts]
                for fli in range(len(fL)):
                    li = fL[fli][1]
                    pRect[pdi].append(stats[li][:4])
                    if "pupae" in prnt.dispImgType.lower():
                        ### draw image for debugging
                        r = pRect[pdi][-1]
                        cv2.rectangle(retGrey, tuple(r[:2]), 
                                      (r[0]+r[2],r[1]+r[3]), 150, 1)

            ### determine whether ant is close to pupae
            def isAPClose(aRect, pRect):
                if aRect != [] and pRect != []:
                    for ai in range(len(aRect)):
                        for pi in range(len(pRect)):
                            l, t, w, h = aRect[ai]
                            aPts = [(l,t), (l+w,t), (l,t+h), (l+w,t+h), 
                                    (l+int(w/2),t+int(h/2))]
                            l, t, w, h = pRect[pi]
                            pPts = [(l,t), (l+w,t), (l,t+h), (l+w,t+h),
                                    (l+int(w/2),t+int(h/2))]
                            for aPt in aPts:
                                for pPt in pPts:
                                    ax, ay = aPt
                                    px, py = pPt
                                    dist = np.sqrt((ax-px)**2 + (ay-py)**2)
                                    if dist < antLen/2:
                                        return True
                return False                            
            ret = isAPClose(aRect[pdi], pRect[pdi])
            if ret:
                tD["moi0_%02i"%(pdi)] = "True"
                apClose[pdi] = True

            ### add certain greyscale image for debugging purpose
            if "ant" in prnt.dispImgType.lower():
                retGrey = cv2.add(retGrey, fcRsltA)
            if "pupae" in prnt.dispImgType.lower(): 
                retGrey = cv2.add(retGrey, fcRsltP)

        #### [begin] display info on frame_arr -----
        ### draw ROI(petri-dish area) & its index 
        for pdi, (x, y) in enumerate(pDishCt):
            if x == -1: continue
            cv2.circle(frame_arr, (x,y), pDRad, (255,255,0), 2)
            txtPos = (x-pDRad, y-int(pDRad*0.75))
            cv2.putText(frame_arr, "%i"%(pdi), txtPos, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(255,255,0), thickness=2)
        # draw status & user defined ant size 
        frame_arr = self.drawStatusMsg(None, frame_arr, True) 
        ### draw rects around ant 
        for _aRect in aRect:
            for l, t, w, h in _aRect:
                cv2.rectangle(frame_arr, (l,t), (l+w,t+h), (0,255,0), 1)
        ### draw rects around pupae 
        for _pRect in pRect:
            for l, t, w, h in _pRect:
                cv2.rectangle(frame_arr, (l,t), (l+w,t+h), (0,255,255), 1)
        ### draw if ant and pupae is close
        for pdi, isClose in enumerate(apClose):
            if isClose:
                cv2.circle(frame_arr, pDishCt[pdi], pDRad+10, (0,255,0), 10)
        #### [end] display info on frame_arr -----

        return tD, retGrey, frame_arr

    #---------------------------------------------------------------------------
    
    def proc_motion21(self, tD, frame_arr):
        """ records amount of motion on each frame in each ROI
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        fH, fW, __ = frame_arr.shape # frame shape
        prnt = self.p # parent
        fi = prnt.vRW.fi # frame-index
        nFrames = prnt.vRW.nFrames # number of frames
        aecp = prnt.aecParam # animal experiment case parameters
        nPDish = aecp["uNPDish"]["value"] # number of petri dishes 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  

        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None)

        # prepare grey image to return
        retGrey = np.zeros(pGImg.shape, dtype=np.uint8)  

        ### get difference image for motion
        if fi > 0:
            m_diff = cv2.absdiff(pGImg, self.prevPGImg)
            mThr = aecp["motionThr"]["value"]
            ret, m_diff = cv2.threshold(m_diff, mThr, 255, cv2.THRESH_BINARY)
        
        self.prevPGImg = pGImg.copy()

        for pdi in range(nPDish):
        # go through number of petri dishes (here, square-shape chambers)
            m_val = 0
            if fi > 0:
                roi = aecp["uROI%i"%(pdi)]["value"]
                # mask motion image
                _diff = maskImg(m_diff, [roi], 0)
                m_val = int(np.sum(_diff)/255) # motion value
                thr = roi[2]*roi[3]*0.1
                if m_val > thr: m_val = 0 # too large motion value, ignore 
                retGrey = cv2.add(retGrey, m_diff)
            tD["motion_%02i"%(pdi)] = m_val # store motion value

        #### [begin] display info on frame_arr -----
        ### draw ROI & its index
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0
        thck = 2 
        for pdi in range(nPDish):
            l, t, w, h = aecp["uROI%i"%(pdi)]["value"]
            cv2.rectangle(frame_arr, (l,t), (l+w,t+h), (255,255,0), 2)
            (tw, th), tBL = cv2.getTextSize("0", font, scale, thck) 
            txtPos = (l-tw*2, t+th+tBL)
            cv2.putText(frame_arr, "%i"%(pdi), txtPos, font,
                        fontScale=scale, color=(255,255,0), thickness=thck)
        ### draw status 
        fontP = dict(scale=2.0, thck=2)
        frame_arr = self.drawStatusMsg(None, frame_arr, False, fontP)
        #### [end] display info on frame_arr -----
        
        return tD, retGrey, frame_arr

    #---------------------------------------------------------------------------
    
    def proc_aos21(self, tD, frame_arr):
        """ Make motion data like data from AntOS.
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        fH, fW, __ = frame_arr.shape # frame shape
        prnt = self.p # parent
        fi = prnt.vRW.fi # frame-index
        nFrames = prnt.vRW.nFrames # number of frames
        aecp = prnt.aecParam # animal experiment case parameters
        nPDish = aecp["uNPDish"]["value"] # number of petri dishes 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  

        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None)

        # prepare grey image to return
        retGrey = np.zeros(pGImg.shape, dtype=np.uint8)  

        ### get difference image for motion
        if fi > 0:
            m_diff = cv2.absdiff(pGImg, self.prevPGImg)
            mThr = aecp["motionThr"]["value"]
            ret, m_diff = cv2.threshold(m_diff, mThr, 255, cv2.THRESH_BINARY)
        
        self.prevPGImg = pGImg.copy()

        tD["Timestamp"] = get_time_stamp(True)
        motionPts = "" # motion point (center of contour) data as string 
        for pdi in range(nPDish):
        # go through number of petri dishes (here, square-shape chambers)
            m_val = 0
            if fi > 0:
                roi = aecp["uROI%i"%(pdi)]["value"]
                # mask motion image
                _diff = maskImg(m_diff, [roi], 0)
                edged = self.getEdged(_diff)
                cnt_info, cnt_pts, cnt_br, cnt_cpt = self.getCntData(edged)
                for wh, x, y in cnt_info:
                    motionPts += "(%i/ %i)"%(x, y)
        tD["Value"] = motionPts # store motion point string

        #### [begin] display info on frame_arr -----
        ### draw ROI & its index
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.0
        thck = 2 
        for pdi in range(nPDish):
            l, t, w, h = aecp["uROI%i"%(pdi)]["value"]
            cv2.rectangle(frame_arr, (l,t), (l+w,t+h), (255,255,0), 2)
            (tw, th), tBL = cv2.getTextSize("0", font, scale, thck) 
            txtPos = (l-tw*2, t+th+tBL)
            cv2.putText(frame_arr, "%i"%(pdi), txtPos, font,
                        fontScale=scale, color=(255,255,0), thickness=thck)
        ### draw status 
        fontP = dict(scale=2.0, thck=2)
        frame_arr = self.drawStatusMsg(None, frame_arr, False, fontP)
        #### [end] display info on frame_arr -----
        
        return tD, retGrey, frame_arr

    #---------------------------------------------------------------------------
    
    def proc_sleepDet23(self, tD, frame_arr):
        """ For detecting sleep of the ant in each ROI.
        (Each ROI has a single ant.)
        
        Args:
            tD (dict): dictionary to retrieve/store calculated data
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            tD (dict): received 'tD' dictionary, but with calculated data.
            frame_arr (numpy.ndarray): Frame image array.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        fH, fW, __ = frame_arr.shape # frame shape
        prnt = self.p # parent
        fi = prnt.vRW.fi # frame-index
        nFrames = prnt.vRW.nFrames # number of frames
        aecp = prnt.aecParam # animal experiment case parameters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
        wRows = aecp["uWRows"]["value"] # rows of wells 
        wCols = aecp["uWCols"]["value"] # columns of wells 
        roiOffsetX = int(aecp["uROIOffsetX"]["value"] * fH)
        roiOffsetY = int(aecp["uROIOffsetY"]["value"] * fH)
        nWells = wRows * wCols # n of wells
        wColor = (150, 0, 0) # color for drawing info of well
        frac4motion = 0.15 # fraction of ant body area to determine 
                           #   ant's meaningful motion
        a2refRatThr = [0.8, 1.5] # threshold to determine sleep; 
                                 # found ant body length to its reference length

        # get pre-processed grey image
        __, pGImg = self.commonPreProc(frame_arr, None)

        # prepare grey image to return
        retGrey = np.zeros(pGImg.shape, dtype=np.uint8)

        if prnt.dispImgType == "Greyscale(debug)" : retGrey = pGImg

        wasPrevPGImgWasNone = False 
        if self.prevPGImg is None: 
            wasPrevPGImgWasNone = True
            self.prevPGImg = pGImg.copy() 

        if self.durNoMotion is None or self.wROI is None:
            self.durNoMotion = {} # duration of no-motion in each well 
            self.wROI = {} # ROI (x,y,r) of each well
            rFrac = 1/wRows * 0.4
            rad = int(rFrac * fH)
            for ri in range(wRows):
                for ci in range(wCols):
                # go through rows & columnsn of wells 
                    key = f'w{ri}{ci}'
                    # init. duration of no-motion in each well
                    self.durNoMotion[key] = 0
                    x = int((rFrac + ci*rFrac*2.65) * fH)
                    y = int((rFrac + ri*rFrac*2.945) * fH)
                    # set roi (x, y, rad) for each well
                    self.wROI[key] = [x, y, rad]

        ### get difference image for motion
        if wasPrevPGImgWasNone:
            mDiff = cv2.absdiff(pGImg, np.zeros(pGImg.shape, dtype=np.uint8))
        else:
            mDiff = cv2.absdiff(pGImg, self.prevPGImg)
        self.prevPGImg = pGImg.copy()
        mThr = aecp["motionThr"]["value"]
        ret, mDiff = cv2.threshold(mDiff, mThr, 255, cv2.THRESH_BINARY)
        #mDiff = cv2.medianBlur(mDiff, 3)

        ### get result image, detected with ant color 
        colMin = tuple(aecp["uColA_min"]["value"])
        colMax = tuple(aecp["uColA_max"]["value"])
        rect = [0, 0, fW, fH]
        antColRslt = self.find_color(rect, frame_arr, colMin, colMax)
        antColRslt= cv2.medianBlur(antColRslt, 1)

        flags = {}
        antMBR = {} # minimum-bounding-rect of the found ant blob in each well
        aLenRef = {} # ant body length reference for each well
        a2refRat = {} # found ant body to reference length ratio
        for ri in range(wRows):
            for ci in range(wCols):
            # go through rows & columnsn of wells 
                pdi = (ri*wCols) + ci # index for storing resultant moi & motion
                key = f'w{ri}{ci}'

                alK = f'uAntLen{ri}{ci}' # ant-length key
                aLenRef[alK] = aecp[alK]["value"] # ant's body length  reference
                antWid = int(aLenRef[alK]/3) # approximate ant's width

                # flag for this well 
                flags[key] = dict(longNoMotion=False, lenInRange=False)

                ##### [begin] find the ant blob -----
                roi = copy(self.wROI[key])
                ### apply offset to ROI
                roi[0] += roiOffsetX
                roi[1] += roiOffsetY
                ### get the ant color image of this well 
                acImg = antColRslt.copy()
                acImg = maskImg(acImg, [roi], 0)
                frame_arr[acImg==255] = (0,0,200) # mark pixels of ant color 
                
                # find blobs in the color-detection result image
                #   using connectedComponentsWithStats
                ccOutput = cv2.connectedComponentsWithStats(acImg, 
                                                            connectivity=8)
                nLabels = ccOutput[0] # number of labels
                labeledImg = ccOutput[1]
                # stats = [left, top, width, height, area]
                stats = list(ccOutput[2])
                lblsWArea = [] # labels with area
                for li in range(1, nLabels):
                    lblsWArea.append([stats[li][4], li])
                lblsWArea = sorted(lblsWArea)
                if lblsWArea != []:
                    # area and index of the largest blob
                    area, lli = lblsWArea[-1]
                    l, t, w, h, a = stats[lli]
                    lcX = int(l + w/2) # center-point;x of the largest blob
                    lcY = int(t + h/2) # center-point;y of the largest blob
                    #maxLen = np.sqrt(w**2 + h**2) 
                    #if maxLen > aLenRef[alK]/3:
                    ## check minimum length of the found blob 
                    if prnt.dispImgType == "(debug) Ant-color":
                        acImg[:,:] = 0 # erase ant-color image
                        acImg[labeledImg==lli] = 255 # draw the largest blob
                        retGrey = cv2.add(retGrey, acImg)
        
                    ''' 
                    ### change label indices of close enough blobs
                    ###   to the index of the largest blob
                    for li in range(1, nLabels):
                        l, t, w, h, a = stats[li]
                        _cX = int(l + w/2)
                        _cY = int(t + h/2)
                        _dist = np.sqrt((lcX-_cX)**2 + (lcY-_cY)**2)
                        if _dist <= aLenRef[alK]:
                            labeledImg[labeledImg==li] = lli
                    '''
                    
                    frame_arr[labeledImg==lli] = (0,0,255) # mark the ant blob 
                    
                    ### calculate aligned line with the found blob
                    ptL = np.where(labeledImg==lli) 
                    sPt = np.hstack((
                          ptL[1].reshape((ptL[1].shape[0],1)),
                          ptL[0].reshape((ptL[0].shape[0],1))
                          )) # stacked points
                    rotatedR = cv2.minAreaRect(sPt)
                    lx, ly, lpts, rx, ry, rpts = self.calcLRPts(rotatedR)
                    # calculate length of the blob
                    bLen = np.sqrt((rx-lx)**2 + (ry-ly)**2)
                    antMBR[key] = [(lx, ly), lpts, (rx, ry), rpts, bLen]
                ##### [end] find the ant blob ----- 

                ### get motion image of this well 
                m_diff = mDiff.copy()
                m_diff = maskImg(m_diff, [roi], 0)
                if prnt.dispImgType == "(debug) Motion":
                    retGrey = cv2.add(retGrey, m_diff)

                m_pixel = int(np.sum(m_diff)/255)
                _thr = aLenRef[alK]*antWid*frac4motion # fraction of ant area
                if m_pixel > _thr:
                # if motion pixels are more than threshold;_thr value
                    tD[f'motion_{pdi:02d}'] = m_pixel # store motion result
                    self.durNoMotion[key] = 0 
                    continue # to the next well

                self.durNoMotion[key] += 1 # increase no-motion duration
                if self.durNoMotion[key] <= aecp["uNoMotionThr"]["value"]:
                # the duration has not passed the threshold yet
                    continue # to the next well

                flags[key]["longNoMotion"] = True

                if key in antMBR.keys():
                    # store the ratio 
                    a2refRat[key] = np.round(antMBR[key][-1]/aLenRef[alK], 3)
                    if a2refRatThr[0] <= a2refRat[key] < a2refRatThr[1]:
                        flags[key]["lenInRange"] = True # store flag
                        # both long-no-motion and lenInRange are True
                        # store sleep in result dict.
                        tD[f'moi0_{pdi:02d}'] = 1 

        #### [begin] display info on frame_arr -----  
        ### draw well circle and its flag circles
        for ri in range(wRows):
            for ci in range(wCols):
            # go through rows & columnsn of wells 
                wk = f'w{ri}{ci}'

                x, y, r = self.wROI[wk]
                x += roiOffsetX
                y += roiOffsetY
                # draw circle around well 
                cv2.circle(frame_arr, (x,y), r, wColor, 1)
               
                colors = dict(longNoMotion=(255,0,0), lenInRange=(0,0,200))
                for fKey in ["longNoMotion", "lenInRange"]:
                    r += 5 
                    if flags[wk][fKey]: _color = colors[fKey] 
                    else: _color = (127, 127, 127)
                    # draw circle for the flag 
                    cv2.circle(frame_arr, (x,y), r, _color, 2)

                if flags[wk]["longNoMotion"] and flags[wk]["lenInRange"]:
                    # mark that the ant is in sleep
                    cv2.putText(frame_arr, "sleep", (x-int(self.tW*2.5),y), 
                                self.font, fontScale=self.tScale, 
                                thickness=2, color=(30,30,30))
                    
                ### draw well index
                txtPos = (x-r+int(self.tW*1.5), y+self.tH+self.tBL)
                cv2.putText(frame_arr, f'{ri}{ci}', txtPos, self.font,
                            fontScale=self.tScale, color=wColor, thickness=2)

                if wk in antMBR.keys():
                    pt1, lpts, pt2, rpts, bLen = antMBR[wk]
                    ang = calc_line_angle(pt1, pt2)
                    
                    # draw ant body line 
                    cv2.line(frame_arr, pt1, pt2, (0,255,0), 2) 

                    ### draw ant body length reference
                    alK = f'uAntLen{ri}{ci}' # ant-length key
                    pt1, pt2 = lpts
                    if pt1[0] > pt2[0]: pt = pt1
                    else: pt = pt2
                    ax, ay = calc_pt_w_angle_n_dist(ang, aLenRef[alK], 
                            bPosX=pt[0], bPosY=pt[1], flagScreen=True)
                    cv2.line(frame_arr, pt, (ax,ay), (200,200,0), 2) 

                    ### draw ratio of ant-body-len to reference ant-len
                    rx = pt2[0] + int(aLenRef[alK]/3)
                    ry = pt2[1]
                    txt = f'{bLen/aLenRef[alK]:.3f}'
                    cv2.putText(frame_arr, txt, (rx,ry), self.font, 
                                fontScale=self.tScale, color=(0,127,0), 
                                thickness=1)
        ### draw status string 
        fontP = dict(scale=0.5, thck=1)
        pos = (0, int(fH/2-self.tH/2))
        frame_arr = self.drawStatusMsg(None, frame_arr, False, fontP, pos) 
        #### [end] display info on frame_arr -----

        fi = prnt.vRW.fi
        nFrames = prnt.vRW.nFrames-1
        #if fi == nFrames: # this is the last frame
        
        return tD, retGrey, frame_arr

    #---------------------------------------------------------------------------
    
    def storeFrameImg(self, frame_arr):
        """ store the frame image of the current frmae of the opened video 
        
        Args: 
            frame_arr (numpy.ndarray): Frame image array.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
       
        inputFP = self.p.vRW.fPath
        ext = "." + inputFP.split(".")[-1]
        folderPath = inputFP.replace(ext, "") + "_frames"
        if not path.isdir(folderPath): # folder doesn't exist
            mkdir(folderPath) # make one 
        fp = path.join(folderPath, "f%07i.jpg"%(self.p.vRW.fi)) # file path
        if not path.isfile(fp): # frame image file doesn't exist
            cv2.imwrite(fp, frame_arr) # save frame image

    #---------------------------------------------------------------------------
    
    def calcLRPts(self, rr): 
        """
        Calculating line along a blob

        Args:
            rr (OpenCV RotatedRect box): Result of cv2.minAreaRect on 
              points of a blob.

        Returns:
            lx (int): 'x' of middle point of two points on left side of rect 
            ly (int): 'y' of middle point of two points on left side of rect 
            lpts (tuple): two points of rect on left side
            rx (int): 'x' of middle point of two points on right side of rect 
            ry (int): 'y' of middle point of two points on right side of rect 
            rpts (tuple): two point of rect on right side
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        box = cv2.boxPoints(rr)
        box = np.int64(box)
        ### group 2 closer points
        s1=[]; s2=[]
        _d = []
        _d.append(np.sqrt(
            (box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2
            ))
        _d.append(np.sqrt(
            (box[0][0]-box[2][0])**2 + (box[0][1]-box[2][1])**2
            ))
        _d.append(np.sqrt(
            (box[0][0]-box[3][0])**2 + (box[0][1]-box[3][1])**2
            ))
        # index of closest point to the first box point
        iC20 = _d.index(min(_d)) + 1 
        s2Idx = list(range(1,4)); s2Idx.remove(iC20)
        s1.append(tuple(box[0])); s1.append(tuple(box[iC20]))
        for s2i in s2Idx: s2.append(tuple(box[s2i]))
        ### get center point of each group and calc.
        ### left and right side points
        s1x = int((s1[0][0]+s1[1][0])/2)
        s1y = int((s1[0][1]+s1[1][1])/2)
        s2x = int((s2[0][0]+s2[1][0])/2)
        s2y = int((s2[0][1]+s2[1][1])/2)
        if s1x < s2x: lx=s1x; ly=s1y; lpts=s1; rx=s2x; ry=s2y; rpts=s2
        else: lx=s2x; ly=s2y; lpts=s2; rx=s1x; ry=s1y; rpts=s1
        return lx, ly, lpts, rx, ry, rpts

    #---------------------------------------------------------------------------
    
    def getYForFittingLine(self, pts, br, lx, rx, width):
        ''' get left and right side y-coordinates to draw fitting line
        '''
        if FLAGS["debug"]: logging.info(str(locals()))
        
        (vx, vy, x, y) = cv2.fitLine(pts, 
                                     distType=cv2.DIST_L2, 
                                     param=0, 
                                     reps=0.01, 
                                     aeps=0.01) # DIST_L2: least square 
                                                #   distance p(r) = r^2/2
        yunit = float(vy)/vx
        ly = int(-x*yunit + y) # left-most y-corrdinate
        ry = int((width-x)*yunit + y) # right-most y-coordinate
        if lx != 0: ly = int(lx*yunit + ly)
        if rx != width: ry = int((rx-lx)*yunit + ly)
        ### remove parts of the fitting line, 
        ###   which go out of the overall bounding rect
        if ly < br[1]:
            rat = float(br[1]-ly)/(ry-ly)
            lx = lx + int((rx-lx)*rat)
            ly = br[1]
        elif ly > br[1]+br[3]:
            rat = float((br[1]+br[3])-ly)/(ry-ly)
            lx = lx + int((rx-lx)*rat)
            ly = br[1]+br[3]
        if ry < br[1]:
            rat = float(br[1]-ly)/(ry-ly)
            rx = lx + int((rx-lx)*rat)
            ry = br[1]
        elif ry > br[1]+br[3]:
            rat = float((br[1]+br[3])-ly)/(ry-ly)
            rx = lx + int((rx-lx)*rat)
            ry = br[1]+br[3] 
        return int(ly), int(ry), yunit
    
    #---------------------------------------------------------------------------
    
    def find_color(self, rect, inImage, HSV_min, HSV_max):
        """ Find a color(range: 'HSV_min' ~ 'HSV_max') in an area('rect') of 
        an image('inImage') 'rect' here is (x1,y1,x2,y2)
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        # points (upper Left, lower Left, lower Right, upper Right)
        pts_ = [(rect[0], rect[1]), (rect[0], rect[3]),
                (rect[2], rect[3]), (rect[2], rect[1])] 
        mask = np.zeros((inImage.shape[0], inImage.shape[1]) , dtype=np.uint8)
        tmp_grey_img = np.zeros((inImage.shape[0], inImage.shape[1]), 
                                dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.asarray(pts_), 255)
        tmp_col_img = cv2.bitwise_and(inImage, inImage, mask=mask)
        HSV_img = cv2.cvtColor(tmp_col_img, cv2.COLOR_BGR2HSV)
        tmp_grey_img = cv2.inRange(HSV_img, HSV_min, HSV_max)
        ret, tmp_grey_img = cv2.threshold(tmp_grey_img, 50, 255, 
                                          cv2.THRESH_BINARY)
        if HSV_min[0] == 0 and HSV_max[0] < 40: # finding red color
            HSV_min = (180-HSV_max[0], HSV_min[1], HSV_min[2])
            HSV_max = (180, HSV_max[1], HSV_max[2])
            tmp_grey_img1 = cv2.inRange(HSV_img, HSV_min, HSV_max)
            ret, tmp_grey_img1 = cv2.threshold(tmp_grey_img1, 50, 255, 
                                               cv2.THRESH_BINARY)
            tmp_grey_img = cv2.add(tmp_grey_img, tmp_grey_img1)
        return tmp_grey_img

    #---------------------------------------------------------------------------

    def commonPreProc(self, img, bgImg):
        """ image pre-processing.

        Args:
            img (numpy.ndarray): image array to process.
            bgImg (numpy.ndarray/ None): background image to subtract.

        Returns:
            diff (numpy.ndarray): greyscale image after BG subtraction.
            edged (numpy.ndarray): greyscale image of edges in 'diff'.
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if type(bgImg) == np.ndarray: 
            # get difference between the current frame and the background image 
            diffCol = cv2.absdiff(img, bgImg)
            diff = cv2.cvtColor(diffCol, cv2.COLOR_BGR2GRAY)
        else:
            diffCol = None 
            diff = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ecp = self.p.aecParam
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        if "preMExOIter" in ecp.keys() and ecp["preMExOIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_OPEN, 
                        kernel, 
                        iterations=ecp["preMExOIter"]["value"],
                        ) # to decrease noise & minor features
        if "preMExCIter" in ecp.keys() and ecp["preMExCIter"]["value"] != -1:
            diff = cv2.morphologyEx(
                        diff, 
                        cv2.MORPH_CLOSE, 
                        kernel, 
                        iterations=ecp["preMExCIter"]["value"],
                        ) # closing small holes
        if "preThres" in ecp.keys() and ecp["preThres"]["value"] != -1:
            __, diff = cv2.threshold(
                            diff, 
                            ecp["preThres"]["value"], 
                            255, 
                            cv2.THRESH_BINARY
                            ) # make the recognized part clear 
        return diffCol, diff
    
    #---------------------------------------------------------------------------
    
    def getEdged(self, greyImg):
        """ Find edges of greyImg

        Args:
            greyImg (numpy.ndarray): greyscale image to extract edges.

        Returns:
            (numpy.ndarray): greyscale image with edges.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        return cv2.Canny(greyImg,
                         self.p.aecParam["cannyTh"]["value"][0],
                         self.p.aecParam["cannyTh"]["value"][1])
    
    #---------------------------------------------------------------------------
    
    def getCntData(self, img):
        """ Get some useful data from contours in a given image.

        Args:
            img (numpy.ndarray): greyscale image to get contour data.

        Returns:
            cnt_info (list): contour info list. 
                each item is a tuple (size, center-X, center-Y) of a contour.
                'size' is width + height.
            cnt_pts (list): list of every pixels in all contours.
            cnt_br (CvRect): bounding rect of all contour points
            cnt_cpt (tuple): center point of all contours.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ### find contours
        if CV_Ver[0] >= 4: 
            cnts, hierarchy = cv2.findContours(img, 
                                               cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
        else:
            img, cnts, hierarchy = cv2.findContours(img, 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        
        cnt_info = [] # contour info (size, center-X, center-Y)
        cnt_pts = [] # put points of all contours into this list
        for ci in range(len(cnts)):
            mr = cv2.boundingRect(cnts[ci])
            if mr[2]+mr[3] < self.p.aecParam["contourThr"]["value"]: continue
            #cv2.circle(img, (mr[0]+mr[2]/2,mr[1]+mr[3]/2), mr[2]/2, 125, 1)
            cnt_info.append((mr[2]+mr[3], 
                             mr[0]+int(mr[2]/2), 
                             mr[1]+int(mr[3]/2)))
            cnt_pts += list(cnts[ci].reshape((cnts[ci].shape[0], 2)))
        if len(cnt_pts) > 0:
            # rect bounding all contour points
            cnt_br = cv2.boundingRect(np.asarray(cnt_pts))
            # calculate center point of all contours
            cnt_cpt = (cnt_br[0]+int(cnt_br[2]/2), cnt_br[1]+int(cnt_br[3]/2))
        else:
            cnt_br = (-1, -1, -1, -1)
            cnt_cpt = (-1, -1)

        return cnt_info, cnt_pts, cnt_br, cnt_cpt
    
    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass

