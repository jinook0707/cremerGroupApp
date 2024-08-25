# coding: UTF-8
"""
This module is for DataVisualizer app to process data from AnVid

Dependency:
    Numpy (1.17)
    SciPy (1.4)
    OpenCV (3.4)
    tsmoothie (1.0)

last edited: 2024-05-16
"""

import sys, csv, ctypes, string
from os import path, remove
from glob import glob
from copy import copy
from time import time
from datetime import datetime, timedelta
from random import randint

import wx, cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, welch, periodogram
from scipy.cluster.vq import vq, kmeans, kmeans2
from scipy.spatial.distance import cdist 
from scipy import stats
# package for anomaly detection
from tsmoothie.smoother import LowessSmoother

from initVars import *
from modFFC import *
from modCV import *

DEBUG = False

#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(int(ctypes.c_ulong(-1).value//2))

#===============================================================================

class ProcAnVidRslt:
    """ Class for processing data from AnVid and generate graph.
    
    Args:
        parent (wx.Frame): Parent frame
    
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    
    def __init__(self, mainFrame, parent):
        if DEBUG: MyLogger.info(str(locals()))

        ##### [begin] setting up attributes on init. -----
        self.mainFrame = mainFrame # main object with wx.Frame
        self.parent = parent 
        # to store data and other variables after running initOnDataLoading
        self.v = {}
        # list of replicate IDs either in group1 or 2.
        self.rIdLst = dict(group1 = [3, 4, 11, 13, 15, 
                                     23, 24, 29, 30, 31, 
                                     32, 39, 40],
                           group2 = [6, 7, 17, 18, 19, 
                                     20, 27, 28, 33, 35, 
                                     36, 37, 38])
        # list of replicate IDs which has only one ROI in the video
        self.singleRLst = [6, 7, 11, 33]
        ### which ROI-index is the replicate to consider in analysis 
        self.singleRLstROI = {}
        self.singleRLstROI[6] = "roi01"  
        self.singleRLstROI[7] = "roi00"  
        self.singleRLstROI[11] = "roi01"  
        self.singleRLstROI[33] = "roi00"  
        ##### [end] setting up attributes on init. -----

    #---------------------------------------------------------------------------
  
    def initOnDataLoading(self, q2m=None):
        """ init. process when input file was loaded

        Args:
            q2m (None/queue.Queue): Queue to send data to main thread.

        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        retMsg = ""
        main = self.mainFrame

        ### To return; this will get into self.v
        fImg = {} # first frame image of each video file
        data = {} # extracted data from each file
        timestamp = {} # timestamp of data from each file
        tunnel = {} # tunnel position (initially;automatically detected)
        colTitles = [] # column title list in CSV data file 
        tHW = -1
        tHH = -1

        ### gather file path info 
        flagFileErr = False
        fpLst = []
        fPath, fn = path.split(main.inputFP)
        if "before" in fn:
            fpLst = [path.join(fPath, fn),
                     path.join(fPath, fn.replace("before", "after"))]
        elif "after" in fn:
            fpLst = [path.join(fPath, fn.replace("after", "before")),
                     path.join(fPath, fn)]
        if len(fpLst) != 2: flagFileErr = True
        for fp in fpLst:
            ### Check whether there's a video with the same file-name.
            vfp = None
            for ext in ["mp4", "mkv", "mov", "avi"]:
                _vfp = fp.replace(".csv", f'.{ext}')
                if path.isfile(_vfp):
                    vfp = _vfp
                    break
            if vfp is None: flagFileErr = True
        if flagFileErr:
            retMsg = "ERROR: There must be two files with _before and _after"
            retMsg += " tag in the filename and also two video files with the"
            retMsg += " corresponding filenames."
            # return data
            ret = dict(fImg=fImg, data=data, timestamp=timestamp, 
                       tunnel=tunnel, tHW=tHW, tHH=tHH, colTitles=colTitles, 
                       fnKIdx=0)
            output = ("finished", ret, dict(retMsg=retMsg))
            q2m.put(output, True, None)

        for fpi, fp in enumerate(fpLst):
        # go through file list
            fnK = path.basename(fp).replace(".csv", "")
            
            msg = f'Reading CSV data from {fnK}.. {fpi+1}/{len(fpLst)}'
            q2m.put(("displayMsg", msg), True, None)
            
            data[fnK] = {}
            tunnel[fnK] = {} 
            timestamp[fnK] = []

            mPtX = {} # list of x of all motion poiints
            mPtY = {} # list of y of all motion poiints

            fh = open(fp, "r")
            lines = fh.readlines()
            fh.close()
            
            for line in lines:
            # read lines
                if line.strip().startswith("frame-index"):
                    if colTitles == []:
                        ### store column title
                        colTitles = [cT.strip() for cT in line.split(",")]
                    for cT in colTitles:
                        if cT in ["frame-index", "timestamp"]: continue
                        # data-key; 
                        #   'motionPts', 'antBlobRectPts', 'broodBlobRectPts'
                        dataK = cT.rstrip(string.digits)
                        # ROI index; 00, 01 and so on
                        #   with row & column index from AnVid
                        roiK = "roi" + cT.replace(dataK, "")
                        if not roiK in data[fnK].keys():
                            data[fnK][roiK] = {} # dict with each ROI
                            mPtX[roiK] = []
                            mPtY[roiK] = []
                        # list for each data-type
                        data[fnK][roiK][dataK] = []
                    continue
                
                items = [it.strip() for it in line.split(",")]
                if len(items) < len(colTitles): continue
                
                for ii, item in enumerate(items):
                    if ii >= len(colTitles): continue
                    if colTitles[ii] == "frame-index": continue
                    
                    if colTitles[ii] == "timestamp":
                        timestamp[fnK].append(item)
                        continue

                    dataK = colTitles[ii].rstrip(string.digits) 
                    roiK = "roi" + colTitles[ii].replace(dataK, "")
                    if item == "":
                        data[fnK][roiK][dataK].append(None)
                    else:
                        ''' Split by coordinates.
                        motionPts are stored as x/y&x/y& ...
                        ant- or broodBlobRectPts are stored as 
                            x1/y1/x2/y2/x3/y3/x4/y4&x1/y1/x2/y2/ ...
                        '''
                        _data = []
                        coords = item.split("&")
                        for coord in coords:
                            if coord.strip() == "": continue
                            coord = [int(x) for x in coord.split("/")]
                            if dataK == "motionPts":
                                mPtX[roiK].append(coord[0])
                                mPtY[roiK].append(coord[1])
                            _data.append(coord)
                        data[fnK][roiK][dataK].append(_data) 

            vRW = VideoRW(self) # video reading/writing
            ### get a video file path of the first csv data file
            for ext in ["mp4", "mkv", "mov", "avi"]:
                vfp = fp.replace(".csv", f'.{ext}')
                if path.isfile(vfp): break
            vRW.initReader(vfp) # load a video file 
            ret = vRW.getFrame() # read one frame
            fImg[fnK] = vRW.currFrame.copy() # store the frame image

            fH, fW = fImg[fnK].shape[:2]
            tHW = int(fW/6) # half of tunnel area width
            tHH = int(fH/15) # half of tunnel area height

            ### detect circles (arena) 
            gImg = makeImgDull(fImg[fnK], 1, 1, True)
            minDist = int(fH*0.5) 
            minRadius = int(fH*0.25)
            maxRadius = int(fH*0.5)
            param1 = 25
            param2 = 100 # (smaller -> more false circles)
            cir = cv2.HoughCircles(gImg, cv2.HOUGH_GRADIENT, 1, 
                                   minDist, param1=param1, param2=param2, 
                                   minRadius=minRadius, maxRadius=maxRadius)
            if cir is not None: # arena found
                cir = pd.DataFrame(cir[0,:].astype('i'))
                # sort to have the large circles first
                cir = cir.sort_values(2, ascending=False)
                cir = cir.values.tolist()
                if len(cir) > 2: cir = cir[:2]
           
                for roiI, roiK in enumerate(data[fnK].keys()):
                    
                    ### determine the found circle position and radius
                    ###   which matches with the current ROI 
                    mx = np.mean(mPtX[roiK])
                    my = np.mean(mPtY[roiK])
                    dists = []
                    for ci, (cx, cy, cr) in enumerate(cir):
                        dists.append(np.sqrt((cx-mx)**2 + (cy-my)**2))
                    cx, cy, cr = cir[dists.index(np.min(dists))]
                    cv2.circle(fImg[fnK], (cx, cy), cr, (255,255,100), 1)
                    
                    mpx = np.asarray(mPtX[roiK])
                    mpy = np.asarray(mPtY[roiK])
                    dists = np.sqrt((mpx-cx)**2 + (mpy-cy)**2)
                    # get indices of points out of the arena 
                    idx = (dists > cr*1.05).nonzero()[0]
                    if len(idx) == 0:
                        retMsg = "No motion-points out of the arena found."
                        retMsg += " Tunnel position cannot be determined."
                        x = int(np.mean(mpx))
                        y = int(np.mean(mpy))
                        pt1 = (x-tHW, y-tHH)
                        pt2 = (x+tHW, y+tHH)
                        tunnel[fnK][roiK] = (pt1, pt2)
                    else:
                        ##### [begin] get center point of tunnels for stimulus
                        '''
                        ### mark points outside of the arena
                        for i in idx:
                            cv2.circle(fImg[fnK], (mpx[i], mpy[i]), 2, 
                                       (50,50,100), -1)
                        '''

                        _x = mpx[idx]
                        _y = mpy[idx]
                        pts = np.hstack((_x.reshape((_x.shape[0],1)),
                                         _y.reshape((_y.shape[0],1)))) 
                        pts = pts.astype(np.float32)
                        _my = np.mean(_y)
                        # inital 3 centroids
                        initCt = np.array([[np.min(_x), _my], 
                                           [np.median(_x), _my], 
                                           [np.max(_x), _my]])
                        # k-means clustering, assuming 3 tunnels
                        ctr, __ = kmeans(obs=pts, k_or_guess=initCt)
                        xs = []
                        ys = []
                        for i, (x, y) in enumerate(ctr):
                            ### check the cluster centroid whether it's 
                            ###   too far away from the other centroids
                            flagDrop = True 
                            for j, (x1, y1) in enumerate(ctr):
                                if i == j: continue
                                dist = np.sqrt((x-x1)**2 + (y-y1)**2)
                                if dist < cr: flagDrop = False
                            # it's too far away from the others, ignore
                            if flagDrop: continue
                            xs.append(x)
                            ys.append(y)
                            '''
                            # mark the centroid
                            cv2.circle(fImg[fnK], (int(x),int(y)), 5, 
                                       (0,0,255), -1)
                            '''
                        x = int(np.mean(xs))
                        y = int(np.mean(ys)) 
                        pt1 = (x-tHW, y-tHH)
                        pt2 = (x+tHW, y+tHH)
                        # mark the tunnel area 
                        #cv2.rectangle(fImg[fnK], pt1, pt2, clr["tuC"], 2)
                        ##### [end] get center point of tunnels for stimulus
                        # store the tunnel center-position 
                        tunnel[fnK][roiK] = (pt1, pt2)

        # [multiple measures; Not being used;2024-04]
        # graph line color for multiple measures
        self.gC = [(255,150,150), (100,255,100), (127,127,255),
                   (100,255,255), (255,255,100), (255,100,255)]

        # return data
        ret = dict(fImg=fImg, data=data, timestamp=timestamp, 
                   tunnel=tunnel, tHW=tHW, tHH=tHH, colTitles=colTitles, 
                   fnKIdx=0)

        output = ("finished", ret, dict(retMsg=retMsg))
        q2m.put(output, True, None)

    #---------------------------------------------------------------------------

    def drawGraph(self, q2m, gSIdx=0):
        """ Draw graph with data from AnVid 
        
        Args:
            q2m (queue.Queue): Queue to send data to main thread.
            gSIdx (int): Start index for this graph.
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        retMsg = ""

        main = self.mainFrame # wx.Frame in visualizer.py
        prnt = self.parent # graph processing module (procGraph.py)
        fnK = list(self.v["data"].keys())[self.v["fnKIdx"]] # filename key
        data = self.v["data"][fnK] # list of lines in CSV result file
        timestamp = self.v["timestamp"][fnK] # data timestamp 
        #tunnel = self.v["tunnel"][fnK] # center of tunnels
        fImg = self.v["fImg"][fnK].copy() # frame images 
        fH, fW = fImg.shape[:2]
        tunnel = {}
        tHW = self.v["tHW"] # half of tunnel area width
        tHH = self.v["tHH"] # half of tunnel area height
        if "before" in fnK.lower(): bOrA = "Before"
        else: bOrA = "After"
        for i in range(2): # for two ROIs
            w = wx.FindWindowByName(f'tunnel{bOrA}{i}Pt_txt', main.panel["ml"])
            x, y = (int(x) for x in w.GetValue().split(","))
            tunnel[f'roi{i:02d}'] = ((x-tHW, y-tHH), (x+tHW, y+tHH))
        w = wx.FindWindowByName("process_cho", main.panel["ml"])
        proc2run = w.GetString(w.GetSelection()) # process to run 
        # list of processes to draw bar-graph
        procLst2drawBarGraph = ["intensity", "intensityT", "presenceT", 
                                "intensityP", "intensityPABR", "dist2EO", 
                                "dist2EOCh", "distP2T", "distP2A", "distA2T"] 
        ### proc2run lists for other data instead of motion only
        # antBlobRectPts
        p2rLst4ABR = ["dist2EO", "dist2EOCh", "distA2T", "spAHeatmapABR", 
                      "presenceT"] 
        # broodBlobRectPts
        p2rLst4BBR = ["distP2T"] 
        # both ant and brood rects
        p2rLst4AnB = ["intensityPABR", "distP2A", "spAHeatmapPABR"] 
        # Motion and broodBlobRectPts
        p2rLst4MnB = ["intensityP", "spAHeatmapP"] 
        cvFont = cv2.FONT_HERSHEY_PLAIN # font for writing info on graph image
        rImg = None # graph image to return
        dPtIntvSec = None # bundle-interval
        additionalImg = None # additional image (such as floating legend)
        fsPeriod = {} # to store sample frequency and its period
                      #   (for showing on mouse-move event
                      #    in power spectrum density)
        mg = dict(l=0, r=0, t=0, b=0) # graph margin (left, right, top, bottom)
        timeLbl = "" # start and end time for heatmap
        rows = -1
        cols = -1 
        gSDT = None # starting datetime of this graph
        gEIdx = -1 # data index where this graph ends
        gEDT = None # end datetime of this graph
        bData = {} # container for lists of bundled data (key = ROI)
        bD_dt = {} # datetime for the bundled data (key = ROI)
        gInfo = {} # addition graph info. different depending on graph type
        roiKeyLst = list(data.keys())
        for roiK in roiKeyLst: 
            bData[roiK] = []
            bD_dt[roiK] = []
        hmArr = None # array for heatmap 
        gMsg = ""
       
        if proc2run == "initImg":
            ##### [begin] initial image with the first frame image ----- 
            ### get font for the graph
            fThck = 2
            thP = max(int(fH*0.03), 12)
            fScale, txtW, txtH, txtBl = getFontScale(
                                    cvFont, thresholdPixels=thP, thick=fThck
                                    )
            rImg = fImg
            ### write filename
            tx = 5
            ty = 5 + txtH + txtBl
            cv2.putText(rImg, fnK, (tx, ty), cvFont, fontScale=fScale,
                        color=(0,0,0), thickness=fThck)
           
            ### draw ant's length reference
            w = wx.FindWindowByName("antLen_txt", main.panel["ml"])
            aLen = str2num(widgetValue(w), 'int')
            pt1 = (int(fW/2-aLen/2), int(fH/2-aLen/6))
            pt2 = (pt1[0]+aLen, pt1[1]+int(aLen/3))
            cv2.rectangle(fImg, pt1, pt2, (50,255,50), -1)

            ### draw tunnel area
            for i in range(2):
                pt1, pt2 = tunnel[f'roi{i:02d}']
                cv2.rectangle(fImg, pt1, pt2, (0,127,255), 2)

            '''
            ### prepare the return image with frame images of before & after 
            rImg = np.zeros((fH*2, fW, 3), np.uint8)
            y = 0
            for tag in ["before", "after"]:
                ### put the frame image into the return image
                if tag == "before":
                    _fnK = fnK
                    rImg[y:y+fH, 0:fW] = fImg
                elif tag == "after":
                    _fnK = fnK.replace("before", "after")
                    rImg[y:y+fH, 0:fW] = self.v["fImg"][_fnK]
                ### write filename
                tx = 5
                ty = y + int(5 + txtH * 2)
                cv2.putText(rImg, _fnK, (tx, ty), cvFont, fontScale=fScale,
                            color=(0,0,0), thickness=fThck)
                y += fH
            #  this initImg include both before and after; increase fnKIdx
            self.v["fnKIdx"] += 1
            '''
            ##### [end] initial image with the first frame image -----

            rawData = [None]

        else:
            ##### [begin] data-processing (bundling) -----  
            ### get interval to bundle one data point
            w = wx.FindWindowByName("dataPtIntv_cho", main.panel["ml"])
            dPtIntvSec = int(w.GetString(w.GetSelection()))
            dPtIntv = timedelta(seconds=dPtIntvSec)
            ### get interval to generate heatmap
            w = wx.FindWindowByName("heatMapIntv_cho", main.panel["ml"])
            hmIntvMin = int(w.GetString(w.GetSelection()))
            hmIntv = timedelta(seconds=hmIntvMin*60)
            if hmIntvMin == -1: gSIdx = 0 
            if proc2run.startswith("spAHeatmap"):
                # init array for heatmap
                hmArr = np.zeros(fImg.shape[:2], dtype=np.uint32) 
                ### get heatmap point radius
                ###   if this is -1, it will be a single pixel 
                txt = wx.FindWindowByName("hmPt_txt", main.panel["ml"])
                hmRad = int(txt.GetValue())

            for i, roiK in enumerate(roiKeyLst):
                ### bundle data with interval
                if proc2run in p2rLst4ABR: _d = data[roiK]["antBlobRectPts"]
                elif proc2run in p2rLst4BBR: _d = data[roiK]["broodBlobRectPts"]
                elif proc2run in p2rLst4AnB:
                    _d = dict(ant=data[roiK]["antBlobRectPts"], 
                              brood=data[roiK]["broodBlobRectPts"])
                elif proc2run in p2rLst4MnB:
                    _d = dict(motion=data[roiK]["motionPts"], 
                              brood=data[roiK]["broodBlobRectPts"])
                else: _d = data[roiK]["motionPts"]
                args = (proc2run, fnK, _d, timestamp, tunnel[roiK], \
                        bData[roiK], bD_dt[roiK], gSIdx, dPtIntvSec, dPtIntv, \
                        hmIntvMin, hmIntv, hmArr, q2m)
                ret = self.bundleData(args)
                bData[roiK], bD_dt[roiK], hmArr = ret[:3]
                if i == 0:
                    gSDT, gEIdx, gEDT = ret[3:]
            ##### [end] data-processing (bundling) -----  

            ##### [begin] drawing graph ----- 
            
            ### Get user-defined graph default (a single graph) size.
            ''' This is just a default value. The actual size of the graph
            will be dynamically determined in the graph drawing function.
            '''
            txt = wx.FindWindowByName("graphSz_txt", main.panel["ml"])
            try:
                graphW, graphH = txt.GetValue().split(",")
                graphW = int(graphW)
                graphH = int(graphH)
            except:
                graphW = 1500
                graphH = 500

            if proc2run.startswith("spAHeatmap"):
                ### drawing heatmap
                args = (fnK, hmArr, main, prnt, bData, bD_dt, \
                        cvFont, hmRad, gSDT, gEDT, q2m, fImg)
                rImg, rawData = self.drawHeatmap(args)
            
            else:
                # graph background color
                gBg = (30, 30, 30) 
                rawData = {}
                nDCols = {}
                y = 0
                for roiK in roiKeyLst:
                    if proc2run in procLst2drawBarGraph: 
                        ### drawing bar graph
                        args = (proc2run, fnK, roiK, bData[roiK], dPtIntvSec, \
                                graphW, graphH, gBg, cvFont, \
                                bD_dt[roiK], main, q2m, gSDT, gEDT)
                        ret = self.drawBarGraph(args) # draw graph for this ROI
                        retGImg, nDCols[roiK], rawData[roiK] = ret

                    elif "PSD" in proc2run: 
                        # drawing PSD;
                        # Welch's method for estimating Power Spectral Density 
                        #   - smoothing over non-systematic noise
                        #   - being robust to some non-stationarities
                        args = (proc2run, main, bData[roiK], bD_dt[roiK], \
                                dPtIntvSec, graphW, graphH, gBg, cvFont, \
                                fsPeriod)
                        retGImg, fsPeriod, mg, retMsg = self.drawPSD(args)
                        rawData = [None]

                    if rImg is None:
                        ### init the return image
                        gH, gW = retGImg.shape[:2] # size of a graph for one ROI
                        iH = gH * len(roiKeyLst) # height of the return image
                        rImg = np.zeros((iH, gW, 3), dtype=np.uint8)

                    rImg[y:y+gH] = retGImg # store the returned graph image
                    y += gH 
                    # store the height of each ROI graph
                    gInfo["roiGraphHght"] = iH
            ##### [end] drawing graph -----
            
        q2m.put(("displayMsg", "storing image and data..",), True, None)
       
        ### store graph info,
        gInfo["fnK"] = fnK # filename 
        gInfo["proc2run"] = proc2run # process 
        gInfo["bData"] = bData # bundled data
        gInfo["bD_dt"] = bD_dt # datetime for bundled data
        gInfo["gSIdx"] = gSIdx # graph start index 
        gInfo["gEIdx"] = gEIdx # graph end index 
        gInfo["dPtIntvSec"] = dPtIntvSec # bundle inverval in seconds 
        gInfo["timeLbl"] = timeLbl
        gInfo["mg"] = mg 

        if proc2run in procLst2drawBarGraph: 
            gInfo["nDataInRow"] = nDCols # number of data columns 
                                         #   in a row of daily data
        elif "PSD" in proc2run:
            gInfo["fsPeriod"] = fsPeriod

        ad = dict(additionalImg=additionalImg, retMsg=retMsg)
        ### send the results
        output = ("finished", rImg, rawData, gInfo, ad)
        q2m.put(output, True, None) 

    #---------------------------------------------------------------------------

    def initTBin(self, proc2run):
        """ init. temporary lists, depending on proc2run 
        
        Args:
            proc2run (str): Process to run
         
        Returns:
            None
        """ 
        #if DEBUG: MyLogger.info(str(locals()))
   
        tBin = {}
        
        if proc2run.startswith("intensity") or proc2run.startswith("presence"):
            tBin = dict(val=[])
        
        elif proc2run.startswith("spA"):
            tBin = dict(pts=[], ts=[])
        
        elif proc2run.startswith("dist"):
            tBin = dict(dists=[])

        return tBin

    #---------------------------------------------------------------------------

    def fillEmptyData(self, proc2run, bD, bD_dt, dt):
        """ fill empty data depending on proc2run 
        
        Args:
            proc2run (str): Process to run
            bD (dict): Bundled data
            bD_dt (list): Datetimes of bundled data
            dt (datetime): Datetime of the data
         
        Returns:
            None
        """ 
        #if DEBUG: MyLogger.info(str(locals()))
        
        if proc2run.startswith("spAHeatmap"):
        # heatmap; has list of motion-points in each bundle
            fillItem = []
        elif proc2run.startswith("dist2"):
        # distance measure 
            fillItem = -1
        else:
            fillItem = 0
        bD.append(fillItem)
        
        bD_dt.append(dt)
        
        return bD, bD_dt

    #---------------------------------------------------------------------------

    def bundleData(self, args):
        """ Bundle data with time interval
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))
        
        main = self.mainFrame

        proc2run, fnK, data, timestamp, tunnel, bD, bD_dt, gSIdx, dPtIntvSec, \
          dPtIntv, hmIntvMin, hmIntv, hmArr, q2m = args

        if proc2run in ["distP2T", "distA2T"]:
            (x1, y1), (x2, y2) = tunnel
            tunnelCt = (int(x1 + (x2-x1)/2), int(y1 + (y2-y1)/2))

        if proc2run in ["intensityP", "intensityPABR", 
                        "spAHeatmapP", "spAHeatmapPABR"]:
            w = wx.FindWindowByName(f'antLen_txt', main.panel["ml"])
            aLen = str2num(widgetValue(w), 'int')

        ### get hours to ignore & process
        h2p = {}
        for k in ["h2ignore", "h2proc"]:
            w = wx.FindWindowByName(f'{k}_txt', main.panel["ml"])
            if w is None: h2p[k] = 0
            else: h2p[k] = str2num(widgetValue(w), 'float')
         
        firstDT = get_datetime(timestamp[0])
        if "before" in fnK and h2p["h2proc"] > 0:
        # this data file was recorded before an experimental treatment, &
        # there's a h2proc limit
        # (For 'before' video, h2proc goes backward from the end (treatment))
            dt = get_datetime(timestamp[-1]) # datetime of the last timestamp
            startDT = dt - timedelta(hours=h2p["h2proc"])
            for tsi in range(len(timestamp)-2, 0, -1):
            # go backward in timestamp
                _dt = get_datetime(timestamp[tsi])
                if _dt <= startDT: # reached the start-datetime
                    if gSIdx == 0: # this is the first graph
                        gSIdx = tsi # change the graph-start-index
                    firstDT = copy(_dt)
                    break
         
        is1stData = True
        gSDT = None # starting datetime of this graph
        gEIdx = -1 # data index where this graph ends
        gEDT = None # end datetime of this graph
        # init temporary lists to bundle data for each data point
        tBin = self.initTBin(proc2run)

        if type(data) == list: dLen = len(data)
        elif type(data) == dict: dLen = len(data[list(data.keys())[0]])
        for di in range(gSIdx, dLen):
            if di%1000 == 0:
                msg = f'processing data.. [{fnK}]  {di+1}/ {dLen}'
                q2m.put(("displayMsg", msg,), True, None)
            dt = get_datetime(timestamp[di]) # datetime of the timestamp

            elapsedHour = ((dt-firstDT).total_seconds())/60/60 
            
            if "after" in fnK and h2p["h2ignore"] > 0:
                # ignore some hours at the beginning
                if elapsedHour < h2p["h2ignore"]: continue

            if h2p["h2proc"] > 0:
            # there's a set limited hour to process
                ### finish data processing if enough hours passed 
                if "before" in fnK: hr = h2p["h2proc"]
                elif "after" in fnK: hr = h2p["h2ignore"] + h2p["h2proc"]
                if elapsedHour >= hr:
                    gEIdx = -1 # notify it reached the end
                    gEDT = dt
                    break

            if is1stData: 
            # the 1st data to process
                gSDT = copy(dt) # store the datetime of start of the graph
                sDT = copy(gSDT) 
                sDI = copy(di) # store starting index of this data bundle 
                is1stData = False

            if dt-sDT >= dPtIntv: # elapsed time is over the interval
            ##### [begin] process data of the past interval -----
                nIntv = int((dt-sDT).total_seconds()/dPtIntvSec)
                if nIntv > 1:
                    ### fill zero data when time passed over one interval
                    for __ in range(nIntv-1):
                        bD, bD_dt = self.fillEmptyData(
                                            proc2run, bD, bD_dt, sDT
                                            )
                        sDT = sDT + dPtIntv

                ##### [begin] store data of this bundled data ---
                if proc2run.startswith("intensity") or \
                  proc2run.startswith("presence"):
                    bD.append(sum(tBin["val"])) 
                
                elif proc2run.startswith("spAHeatmap"):
                    if len(tBin["pts"]) > 0:
                        ### add to heatmap array
                        for pt in tBin["pts"]:
                            hmArr[pt[1],pt[0]] += 1
                    # append all motion points
                    bD.append(tBin["pts"])

                elif proc2run.startswith("dist"):
                    if tBin["dists"] == []:
                        bD.append(-1)
                    else:
                        bD.append(int(np.median(tBin["dists"])))
                ##### [end] store data of this bundled data ---
                    
                ##### [begin] store timestamp of this bundled data ---
                if proc2run == "spAHeatmap":
                    if len(tBin["ts"]) > 0:
                        _ts = tBin["ts"][0]
                        bD_dt.append(get_datetime(_ts))
                
                else:
                    bD_dt.append(sDT)
                ##### [end] store timestamp of this bundled data ---
                 
                # init temporary data
                tBin = self.initTBin(proc2run)
                # update starting datetime
                sDT += dPtIntv
                # update starting index
                sDI = copy(di)
            ##### [end] process data of the past interval -----
         
            if (type(data) == list and data[di] is not None) or \
              type(data) == dict:
            
                if proc2run == "intensity":
                    # store number of motions for this data bundle 
                    tBin["val"].append(len(data[di]))

                elif proc2run in ["intensityT", "presenceT"]:
                    (x1, y1), (x2, y2) = tunnel
                    cnt = 0
                    if proc2run == "intensityT":
                        for x, y in data[di]:
                            if x1 < x < x2 and y1 < y < y2: cnt += 1
                    elif proc2run == "presenceT":
                        cts = self.getCtOfBR(data[di]) # get center-points of BR
                        for x, y in cts:
                            if x1 < x < x2 and y1 < y < y2: cnt += 1
                    tBin["val"].append(cnt)
                
                elif proc2run in ["spAHeatmap", "spAHeatmapABR"]:
                    ### store motion points
                    _ts = timestamp[di].strip()
                    if proc2run == "spAHeatmapABR":
                    # heatmap with ant-blob-rects
                        # store center-points of ABR
                        tBin["pts"] += self.getCtOfBR(data[di]) 
                    else:
                    # heatmap with motion
                        tBin["pts"] += data[di] 
                    if len(data[di]) > 0:
                        tBin["ts"].append(_ts)

                elif proc2run in ["intensityP", "intensityPABR",
                                  "spAHeatmapP", "spAHeatmapPABR"]:
                # heatmap with motion/ABR only around pupae
                    if proc2run.endswith("ABR"):
                        if data["ant"][di] == None: mOrA = None
                        else: mOrA = self.getCtOfBR(data["ant"][di])
                    else:
                        mOrA = data["motion"][di]
                    brD = data["brood"][di]
                    if None not in [mOrA, brD]:
                        # center-points of brood-blob-rects 
                        bCts = self.getCtOfBR(brD)
                        cnt = 0
                        for mx, my in mOrA:
                            for bx, by in bCts:
                                dist = np.sqrt((mx-bx)**2 + (my-by)**2)
                                if dist <= aLen:
                                # if this motion occurred close enough
                                #   to one of the pupae
                                    if proc2run.startswith("intensity"):
                                        cnt += 1
                                    elif proc2run.startswith("spAHeatmap"):
                                        tBin["pts"].append([mx, my])
                        if proc2run.startswith("intensity"):
                            tBin["val"].append(cnt)
                        elif proc2run.startswith("spAHeatmap"):
                            tBin["ts"].append(timestamp[di].strip())

                elif proc2run.startswith("dist2EO"):
                    cts = self.getCtOfBR(data[di]) # get center-points of BR
                    dist = 0 # sum of min. distances to each other (ants)
                    for ctsI, pt in enumerate(cts):
                        _cts = copy(cts)
                        _cts.pop(ctsI) # remove the current data (pt)
                        # calculates distances to other blob-center-points
                        _dists = cdist([pt], _cts)  
                        dist += np.min(_dists) # add the min. distance
                    tBin["dists"].append(dist)
                
                elif proc2run in ["distP2T", "distA2T"]:
                    cts = self.getCtOfBR(data[di]) # get center-points of BR
                    dists = []
                    for pt in cts:
                        _dist = np.sqrt((pt[0]-tunnelCt[0])**2 + \
                                        (pt[1]-tunnelCt[1])**2)
                        dists.append(_dist)
                    # append mean-distance of pupae (or ants) 
                    #   to tunnel-area-center
                    tBin["dists"].append(np.mean(dists))
                
                elif proc2run == "distP2A":
                    if None not in [data["ant"][di], data["brood"][di]]:
                        ### get center-points of BR
                        aCts = self.getCtOfBR(data["ant"][di])
                        bCts = self.getCtOfBR(data["brood"][di])
                        dists = []
                        for bCt in bCts:
                            dist2A = []
                            for aCt in aCts:
                                dist2A.append(np.sqrt((aCt[0]-bCt[0])**2 + \
                                              (aCt[1]-bCt[1])**2))
                            dists.append(min(dist2A))
                        # append distance-mean of pupae to closest-ant 
                        tBin["dists"].append(np.mean(dists))

            if di == dLen-1: # data reached the end of data
                gEIdx = -1 # notify it reached the end
                gEDT = dt
            else:
                if proc2run.startswith("spAHeatmap"):
                    if hmIntvMin != -1 and dt-gSDT > hmIntv:
                    # elapsed time is over the heatmap interval
                        gEIdx = sDI-1 # store the end index
                        gEDT = sDT # store the end datetime of this graph 
                        break 

        if proc2run == "dist2EOCh":
        # post-processing for changes of values
            pBD = [-1]
            for i in range(1, len(bD)):
                prev = bD[i-1]
                curr = bD[i]
                if prev == -1 or curr == -1: pBD.append(-1)
                else: pBD.append(curr-prev)
            bD = pBD

        return (bD, bD_dt, hmArr, gSDT, gEIdx, gEDT)

    #---------------------------------------------------------------------------

    def drawBarGraph(self, args):
        """ draw bar graph such as 'intensity'
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        proc2run, fnK, roiK, bD, dPtIntvSec, graphW, graphH, \
          gBg, cvFont, bD_dt, main, q2m, gSDT, gEDT = args 

        dLst = {} # temporary data to draw graph  
        dLst[proc2run] = bD 
        
        ### prepare graph image array & smoothed data 
        sdLst = {}
        sPolyOrder = 3
        if dPtIntvSec <= 60:
            sWinLen = 13
        elif dPtIntvSec == 300:
            sWinLen = 5
        elif dPtIntvSec >= 600:
            sWinLen = 3
            sPolyOrder = 2
        
        w = wx.FindWindowByName("barGMax_txt", main.panel["ml"])
        try: maxV = int(w.GetValue())
        except: maxV = -1
        if maxV == -1: maxV = max(bD)
        bdLen = len(bD)
        dayCols = int((24 * 60 * 60) / dPtIntvSec) # number of columns for a day
        if bdLen < dayCols: cols = bdLen 
        else: cols = dayCols
        if cols < graphW: p2col = int(graphW / cols) # pixel-to-column 
        else: p2col = 1
        # get smoothed data
        sdLst[proc2run] = savgol_filter(dLst[proc2run], 
                                        window_length=sWinLen, 
                                        polyorder=sPolyOrder)
        gRows = copy(graphH)
        if maxV == 0: yMul = 1 
        else: yMul = gRows / maxV
        img = np.zeros((gRows, bdLen*p2col, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), 
                      gBg, -1) # bg-color 
        
        ### get font for the graph
        fThck = 2
        _thP = max(int(gRows*0.07), 15)
        fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                 thresholdPixels=_thP,
                                                 thick=fThck)
        fCol = (255, 255, 255)
       
        peaks = {}
        if proc2run == "intensity":
            d4rDE = {} # for storing data for output numpy array raw data
            ''' [[!! currently (2024-03-15) not using !!]]
            d4rDE["peakIntvSec"] = {} # interval between peaks
            d4rDE["peakIntvSecDt"] = {} # date time when peak interval occurs
            d4rDE["mBoutSec"] = {} # lengths (seconds) of a motion bout
            d4rDE["mBoutSecDt"] = {} # when the motion bout starts
            d4rDE["inactivitySec"] = {} # inactivity duration (seconds)
                                        #   between two continuous motion bouts.
            d4rDE["inactivitySecDt"] = {} # when the inactivity duration starts
            '''
        ### whether to draw outlier or not
        w = wx.FindWindowByName("outlier_chk", main.panel["ml"])
        flagOutlier = w.GetValue() 
        ### whether to draw smooth data-line or not
        w = wx.FindWindowByName("smoothLine_chk", main.panel["ml"])
        flagSmoothLine = w.GetValue() 
        ### whether to draw peak points 
        w = wx.FindWindowByName("peak_chk", main.panel["ml"])
        flagDrawPeaks = w.GetValue()

        ### write some graph info 
        tx = txtW * 2 
        ty = txtH + txtBl + 5 
        _txt = f'[{fnK};{roiK}] {proc2run} (intv:{dPtIntvSec} s)'
        cv2.putText(img, _txt, (tx, ty), cvFont,
                    fontScale=fScale, color=fCol, thickness=fThck)
        
        olIdx = [] # outlier indices
        for mi, mk in enumerate(dLst.keys()):
        # [multiple measures; Not being used;2024-04]
        # go through data (for when multiple measures are in the data) 
            cD = dLst[mk] # current data
            max_cD = int(np.max(cD))
            zStartedIdx = -1 # index where zero value started
            nzStartedIdx = -1 # index where non-zero value started
            
            if proc2run == "intensity":
                for k in d4rDE.keys(): # for each data keys for raw-data-output
                    d4rDE[k][mk] = [] # make a list for this measure

            if proc2run == "intensity" and flagOutlier:
                ##### [begin] outlier detection ----- 
                _tmp = copy(cD) # temporary data for outlier detection
                
                #print(_tmp)
                #np.save("tmp.npy", np.asarray(_tmp))

                #_tmp = np.asarray(_tmp)
                #_tmp = np.sqrt(_tmp) # square-root for transforming 
                                     #   skewed data to normal distribution
                #_tmp = np.log10(_tmp)
                #print(list(_tmp))
                ### Generalized Extreme Studentized Deviate
                """
                detector = OutlierDetection_GESD(_tmp,
                             alpha=0.05, maxOutliers=0.1, verbose=False)
                # detect outlier index
                isNormal, olIdx = detector.detect()
                olIdx = sorted(olIdx)
                if not isNormal:
                    msg = "!!! Note !!! The array (after removing"
                    msg += " outliers)"
                    msg += " is NOT from normal distribution."
                    print(msg)
                """
                ### Median Absolute Deviation
                """
                # How to detect and handle outliers
                # , Iglewicz & Hoaglin (1993)
                model = MAD(threshold=3.5) # 
                model.fit(_tmp.reshape(-1, 1))
                pred = model.labels_
                olIdx = list(np.where(pred==1)[0])
                _tmp = list(_tmp)
                ### nomality test after removing outliers
                for _oli in olIdx: _tmp[_oli] = None
                while None in _tmp: _tmp.remove(None)
                stat, p = stats.normaltest(_tmp)
                #stat, p = stats.kstest(_tmp, "norm")
                #stat, p = stats.shapiro(_tmp)
                print("Stat.:",stat, "p-value:",p)
                msg = "\nThe array after removing outliers is "
                alpha = 0.05 #1e-3
                if p < alpha: msg += "NOT "
                msg += "from normal distribution; "
                msg += "p={}, alpha={}\n".format(p, alpha)
                print(msg)
                """
                ### Outlier detection using 
                # LOWESS (Locally Weighted Scatterplot Smoothing);
                # non-parametric regression, fitting a unique linear 
                # regression for every data point by including nearby 
                # data points to estimate the slope and intercept.
                # * smooth_fraction: Between 0 and 1. The smoothing span. 
                #     A larger value of smooth_fraction will 
                #     result in a smoother curve. 
                smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
                smoother.smooth(_tmp)
                low, up = smoother.get_intervals('prediction_interval', 
                                                 confidence=0.05)
                pts = smoother.data[0]
                upPts = up[0]
                lowPts = low[0]
                olIdx = []
                for pi in range(len(pts)):
                    if pts[pi] > upPts[pi] or pts[pi] < lowPts[pi]:
                        olIdx.append(pi)
                ##### [end] outlier detection ----- 
            
            ##### [begin] draw data lines -----
            p_dt = copy(bD_dt[0])
            for x, val in enumerate(cD):
                
                if x%100 == 0:
                    msg = "drawing data-point %i/ %i"%(x, bdLen)
                    q2m.put(("displayMsg", msg,), True, None)

                _x1 = x * p2col

                c_dt = bD_dt[x]
                if c_dt - p_dt >= timedelta(hours=1):
                    ### draw hour-line
                    cv2.line(img, (_x1, 0), (_x1, gRows), (127,127,127), 1)
                    p_dt = copy(c_dt)

                ''' [[!! currently (2024-03-15) not using !!]]
                if proc2run == "intensity":
                    ##### [begin] store motion bout data -----
                    if val == 0:
                    # zero value
                        # store zero starting index
                        if zStartedIdx == -1: zStartedIdx = copy(x)

                        if nzStartedIdx != -1:
                        # there were some preceding non zero values
                            ### store activity bout duration 
                            _zST = bD_dt[nzStartedIdx]
                            _zET = copy(c_dt) 
                            _et = (_zET-_zST).total_seconds()
                            d4rDE["mBoutSec"][mk].append(_et)
                            d4rDE["mBoutSecDt"][mk].append(_zST)
                            nzStartedIdx = -1
                    else:
                    # non-zero value
                        # store non-zero starting index
                        if nzStartedIdx == -1: nzStartedIdx = copy(x)

                        if zStartedIdx != -1:
                        # there were some preceding zero values
                            ### store inactivity duration 
                            _zST = bD_dt[zStartedIdx]
                            _zET = copy(c_dt)
                            _et = (_zET-_zST).total_seconds()
                            d4rDE["inactivitySec"][mk].append(_et)
                            d4rDE["inactivitySecDt"][mk].append(_zST)
                            zStartedIdx = -1
                    ##### [end] store motion bout data -----
                '''

                y = gRows - int(val * yMul)
                sy = gRows - int(sdLst[mk][x] * yMul) # y of smoothed data
                _x2 = _x1 + p2col
                
                ### draw data line
                _col = (150, 150, 150)
                if len(olIdx) > 0: # there's outlier
                    if x == olIdx[0]: # outlier index matches
                        # different color for outlier data point
                        _col = (0,0,255)
                        olIdx.pop(0)
                if p2col == 1:
                    cv2.line(img, (_x1, y), (_x1, gRows), _col, 1)  
                else:
                    cv2.rectangle(img, (_x1, y), (_x2, gRows), 
                                  _col, -1)

                if flagSmoothLine and x > 0:
                    _col = self.gC[mi]
                    # draw smoothed line
                    cv2.line(img, (px, psy), (_x1, sy), _col, 1) 

                px = copy(_x1)
                py = copy(y)
                psy = copy(sy)
            ##### [end] draw data lines ----- 
            
            syArr = np.asarray(sdLst[mk])
            # determine value for prominence
            mm = np.std(syArr[syArr>0])
            #cv2.line(img, (0, gRows-mm), (bdLen-1, gRows-mm), (0,0,0), 1)
            ### draw peak points
            peaks[mk], _ = find_peaks(sdLst[mk], prominence=mm) 
            prevPeakTime = None
            pmLen = max(1, int(min(gRows,bdLen)*0.03)) # peak marker len
            pmThck = max(1, int(pmLen*0.2)) # peak marker line thickness
            for peak in peaks[mk]:
                y = gRows - int(sdLst[mk][peak] * yMul)
                _peak = peak * p2col
                pts = np.array([[_peak, y], 
                                [_peak-int(pmLen/2), y-pmLen],
                                [_peak+int(pmLen/2), y-pmLen]], np.int32)
                pts = pts.reshape((-1,1,2))
                if flagDrawPeaks:
                    #cv2.polylines(img, [pts], True, self.gC[mi], pmThck)
                    cv2.fillPoly(img, [pts], self.gC[mi])
                ###
                ''' [[!! currently (2024-03-15) not using !!]]
                currDT = bD_dt[peak]
                if proc2run == "intensity" and prevPeakTime is not None:
                    d4rDE["peakIntvSecDt"][mk].append(bD_dt[peak])
                    _et = (bD_dt[peak]-prevPeakTime).total_seconds()
                    d4rDE["peakIntvSec"][mk].append(_et)
                prevPeakTime = currDT  
                '''
           
            ### write some info
            _txt = "" 
            if len(dLst) > 1: _txt = f' [{mk[:3]}]'
            _txt += f' max: {max_cD}'
            _txt += f', sum: {np.sum(cD)/1000:.1f}k'
            _txt += f', peaks: {len(peaks[mk])}'
            ty += txtH + txtBl + 5
            cv2.putText(img, _txt, (tx, ty), cvFont, fontScale=fScale,
                        color=fCol, thickness=fThck)
         

        ##### [begin] prepare raw data -----

        ### prepare intensity 
        maxVal = []
        if type(bD) == list:
            ks = [proc2run]
            maxVal.append(np.max(bD))
            dLen = len(bD)
        elif type(bD) == dict: # might have multiple data lists
        # In this case, dictionary-key will be the data column
            for k in dLst.keys(): maxVal.append(np.max(dLst[mk]))
            dLen = len(dLst[k])
        ### set data type depending on the max value of each list
        bdType = []
        for mv in maxVal:
            if "dist" in proc2run or proc2run == "saMVec":
                bdType.append(getNumpyDataType(mv, flagIntSign=True)) 
            else:
                bdType.append(getNumpyDataType(mv)) 
        _dtype = [("datetime", "U26")]
        for mi, mk in enumerate(dLst.keys()): _dtype.append((mk, bdType[mi]))
        rD = np.zeros(dLen, dtype=_dtype) # initiate array
        #rD1 = np.zeros(dLen, dtype=_dtype) # initiate array
        rD["datetime"] = np.array([str(x) for x in bD_dt])
        #rD1["datetime"] = rD["datetime"] 
        for mi, mk in enumerate(dLst.keys()):
            rD[mk] = np.array(dLst[mk], dtype=bdType[mi])
            #rD1[mk] = np.array(sdLst[mk], dtype=bdType[mi])
        #rawData = dict(i=rD, iSmooth=rD1)
        rawData = dict(i=rD)
        
        if proc2run == "intensity":
        ### prepare other raw data for "intensity" process.
            _dtype = [("datetime", "U26"), ("max", np.int32), 
                      ("sum", np.int32), ("peak", np.int32)]

            ''' 
            [!!! multiple measures; Not being used;2024-04]
            rawData["dailySum"] = np.zeros(len(inten24hD["sum"]), 
                                           dtype=_dtype) # initiate array
            for i, k in enumerate(inten24hD.keys()):
                rawData["dailySum"][k] = np.array(
                                            inten24hD[k], dtype=_dtype[i][1]
                                            )
            
            for dk in d4rDE.keys():
            # go through keys of data for raw output
                if dk.endswith("Dt"): continue
                for mk in dLst.keys():
                # go through data (for when multiple measures are in the data) 
                    _data = d4rDE[dk][mk]
                    _dtData = d4rDE[dk+"Dt"][mk]
                    
                    _rDK = "%s%s"%(mk, dk.capitalize())
                    _dLen = len(_data)
                    if _dLen == 0: _dtype = np.uint8
                    else: _dtype = getNumpyDataType(np.max(_data))
                    _dataType = [("datetime", "U26"), (_rDK, _dtype)]
                    rawData[_rDK] = np.zeros(_dLen, dtype=_dataType) 
                    rawData[_rDK][_rDK] = np.asarray(_data)
                    rawData[_rDK]["datetime"] = np.array(
                                                    [str(x) for x in _dtData]
                                                    )
            '''
        ##### [end] prepare raw data -----
        
        return img, cols, rawData 

    #---------------------------------------------------------------------------

    def drawPSD(self, args):
        """ draw Power Spectrum Density graph 
         
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        proc2run, main, bData, bD_dt, dPtIntvSec, graphW, graphH, gBg, \
          cvFont, fsPeriod = args

        retMsg = ""

        key = proc2run.rstrip("PSD")
        cho = wx.FindWindowByName("psdLen_cho", main.panel["ml"])
        # length of data for PSD
        psdLen = cho.GetString(cho.GetSelection())
        cho = wx.FindWindowByName("psdNPerSeg_cho", main.panel["ml"])
        # n per segment for PSD as ratio to the leng of data
        psdNPerSeg = int(cho.GetString(cho.GetSelection())) 

        ### get beginning & end data index for each bundle (such as 1 day)
        if psdLen == "entire input data":
            psdDI = [[0, len(bData)-1]]
        else:
            psdDI = [[0, -1]]
            if psdLen.endswith(" d"):
                thr = timedelta(days=int(psdLen.replace(" d","")))
            elif psdLen.endswith(" h"):
                thr = timedelta(hours=int(psdLen.replace(" h","")))
            for i, dt in enumerate(bD_dt):
                if i == 0: _bDT = copy(dt) # store beginning datetime
                if (dt-_bDT) > thr: # time for a PSD passed
                    psdDI[-1][1] = i-1 # store end index for PSD
                    psdDI.append([i, -1]) # store beginning index
                    _bDT = copy(dt) # new beginning datetime
            if psdDI[-1][1] == -1: psdDI.pop(-1)
        if len(psdDI) == 0:
            retMsg = "!!! ERROR:: No data indices are found for PSD"

        ### prepare result image
        mg = dict(l=100, r=0, t=50, b=50) # left,right,top,bottom margin
        cols = graphW + mg["l"] + mg["r"] 
        rows = (graphH + mg["b"] + mg["t"]) * max(1, len(psdDI))
        img = np.zeros((rows, cols, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), 
                      gBg, -1) # bg-color 

        ### get font for the graph
        fThck = 1
        _thP = 14 
        fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                 thresholdPixels=_thP,
                                                 thick=fThck)

        for psdi in range(len(psdDI)):
        # go through each PSD
            ### get data for PSD
            idx0, idx1 = psdDI[psdi]
            psdData = np.asarray(bData[idx0:idx1+1])
            if len(psdData) < psdNPerSeg: continue
            psdData -= round(np.mean(psdData))
            _txt = "begin:%s, end:%s"%(bD_dt[idx0], bD_dt[idx1])
            _txt += ", dataLen:%i"%(len(psdData))
            _txt += ", %.3f"%(len(psdData)/psdNPerSeg)
            print(_txt)
            ''' 
            ### only for testing purpose
            import wave
            wave = wave.open("__/5kHz_sinewave.wav", "rb") # this file is 
                                    # a recorded tone sound (5 kHz sinewave)
            wFr = wave.readframes(wave.getnframes())
            _dtype = "int%i"%(wave.getsampwidth()*8)
            psdData = np.frombuffer(wFr, dtype=_dtype)
            '''                

            inputFS = len(psdData)
            # calculate PSD
            fs, pxx = welch(psdData, fs=inputFS, nperseg=psdNPerSeg)
            #fs, pxx = periodogram(psdData, fs=inputFS)
            if (np.sum(pxx) == 0): continue
            ### to log-scale
            '''
            _nzi = np.where(pxx!=0)
            pxx[_nzi] = np.log10(pxx[_nzi])
            '''
            #print(np.min(pxx), np.max(pxx), np.mean(pxx), np.std(pxx))
            #print(inputFS/fs*dPtIntvSec)

            ### normalize data to PSD height
            pxxMin = np.min(pxx)
            pxxMax = np.max(pxx)
            zeroY = (mg["t"]+graphH+mg["b"]) * (psdi+1) - mg["b"]
            if pxxMin < 0:
                pxx4draw = pxx * (graphH/ (pxxMax-pxxMin))
                zeroY = round(zeroY + np.min(pxx4draw))
            else:
                pxx4draw = pxx * (graphH/ pxxMax)
        
            ##### [begin] drawing -----
            color = dict(axis=(127,127,127), data=(0,255,255), 
                         txt=(255,255,255))
            # top Y-coord of this PSD
            topY = (mg["t"]+graphH+mg["b"]) * psdi
            pt1 = (mg["l"], topY+mg["t"]) 
            pt2 = (mg["l"], topY+mg["t"]+graphH)
            pt3 = (cols, topY+mg["t"]+graphH)
            ### draw x & y -axis
            cv2.line(img, pt1, pt2, color["axis"], 1)
            cv2.line(img, pt2, pt3, color["axis"], 1)
            ### draw some parameters
            _txt = "%s, %s, %s"%(proc2run, psdLen, psdNPerSeg)
            _txt += ", seg.-Len: %i"%(psdNPerSeg)
            _txt += ", data-interval: %i s"%(dPtIntvSec)
            _pt = (int(cols/2-txtW*len(_txt)/2), pt1[1]-txtH)
            cv2.putText(img, _txt, _pt, cvFont, fontScale=fScale, 
                        color=color["txt"], thickness=fThck)
            ### draw x-label
            _txt = "frequency"
            _pt = (int(mg["l"]+graphW/2-txtW*len(_txt)/2), pt2[1]+txtH*2)
            cv2.putText(img, _txt, _pt, cvFont, fontScale=fScale, 
                        color=color["txt"], thickness=fThck)
            ### draw y-label
            _pt = (txtW, int(rows/2+txtH/2))
            cv2.putText(img, "PSD", _pt, cvFont, fontScale=fScale, 
                        color=color["txt"], thickness=fThck)
            ### draw min and max value of y-axis
            '''
            _txt = "10**%i"%(pxxMin)
            _pt = (mg["l"]-txtW*(len(_txt)+1), pt2[1]+txtH)
            cv2.putText(img, _txt, _pt, cvFont, fontScale=fScale, 
                        color=color["txt"], thickness=fThck)
            _txt = "10**%i"%(pxxMax)
            _pt = (mg["l"]-txtW*(len(_txt)+1), pt1[1]+txtH)
            cv2.putText(img, _txt, _pt, cvFont, fontScale=fScale, 
                        color=color["txt"], thickness=fThck)
            '''

            ### draw data line
            prevFPI = 0 
            prevFPTxt = "" 
            for i, val in enumerate(pxx4draw):
                x = mg["l"] + round(fs[i]/inputFS*2*graphW)
                y = zeroY - round(val)
                ### store period of this frequency
                ###   (for showing the value in mouse-move event)
                _txt = "" #"%.3f; "%(fs[i])
                if i == 0:
                    fsPeriod[0] = "" 
                else:
                    _totDur = len(psdData) * dPtIntvSec
                    period = _totDur / fs[i] # inputFS = len(psdData)
                    if period <= 60: _txt += "%i s"%(period)
                    elif 60 < period <= 60*60: _txt += "%.1f m"%(period/60)
                    else: _txt += "%.1f h"%(period/60/60)
                    _idx = round(fs[i] / (inputFS/2) * graphW)
                    fsPeriod[_idx] = _txt
                    _half = prevFPI+1 + int((_idx-prevFPI)/2)
                    for _i in range(prevFPI+1, _half):
                        fsPeriod[_i] = prevFPTxt 
                    for _i in range(_half, _idx):
                        fsPeriod[_i] = _txt 
                    prevFPI = _idx
                    prevFPTxt = _txt
                ### draw line
                if i > 0:
                    cv2.line(img, (prevX, prevY), (x, y), color["data"], 1)
                ### store coordinate
                prevX = x
                prevY = y 
            
            ### draw peak points
            #mm = int(abs(np.mean(pxx4draw) - np.min(pxx4draw))*0.5)
            mm = int(np.std(pxx4draw))
            peaks, _ = find_peaks(pxx4draw, prominence=mm)
            pmLen = max(2, int(min(graphW, graphH)*0.01)) # peak marker len
            for peak in peaks:
                ### draw peak marker
                x = mg["l"] + round(fs[peak]/inputFS*2*graphW)
                y = zeroY - round(pxx4draw[peak])
                pts = np.array([[x, y], 
                                [x-int(pmLen/2), y-pmLen],
                                [x+int(pmLen/2), y-pmLen]], np.int32)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(img, [pts], True, color["data"], 1)
                ### draw its value
                _txt = fsPeriod[round(fs[peak]/(inputFS/2)*graphW)]
                cv2.putText(img, _txt, (x+int(pmLen/2), y), cvFont, 
                            fontScale=fScale, color=color["txt"], 
                            thickness=fThck)
            ##### [end] drawing -----

        return img, fsPeriod, mg, retMsg 

    #---------------------------------------------------------------------------

    def drawHeatmap(self, args):
        """ draw heatmap graph 
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        fnK, hmArr, main, prnt, bD, bD_dt, cvFont, hmRad, \
          gSDT, gEDT, q2m, fImg = args

        if np.sum(hmArr) == 0:
            img = np.zeros(fImg.shape, np.uint8)
            rawData = dict(heatMapArr=hmArr)
            return img, rawData 

        rows, cols = hmArr.shape[:2]
        r = [0, 0, cols, rows]

        fThck = 2 
        _thP = int(rows*0.025) 
        fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                 thresholdPixels=_thP,
                                                 thick=fThck)

        ### determine heatmap level ranges and its colors
        hmLvlRngs = {}
        hmCols = {}
        w = wx.FindWindowByName("heatmapMax_txt", main.panel["ml"])
        try: hmMaxVal = int(w.GetValue())
        except: hmMaxVal = -1
        if hmMaxVal == -1: hmMaxVal = np.max(hmArr) 
        numHMR = min(5, hmMaxVal) # number of heatmap ranges
        intv = hmMaxVal / numHMR
        step = numHMR * 0.4
        _def = 100
        _stepVal = 255 - _def
        for i in range(numHMR): # go through ranges
            _min = int(np.ceil(intv*(i))) + 1
            _max = intv*(i+1) + 1
            key = "%i - %i"%(_min, _max-1)
            # set heatmap level range
            hmLvlRngs[key] = (_min, _max)
            ### set color for this heatmap range
            if i < step: # 1st/2nd: red
                c = int(_def+ (i+1) * (_stepVal/step))
                hmCols[key] = (0, 0, c)
            elif i < step*2: # 3rd/4th: yellow
                c = int(_def+ (i+1-step) * (_stepVal/step))
                hmCols[key] = (0, c, c)
            else: # last: white
                hmCols[key] = (255, 255, 255)

        _lbl1 = str(gSDT).split(".")[0] # remove microseconds from datetime
        _lbl2 = str(gEDT).split(".")[0] 
        _lbl1 = _lbl1.replace(":", "").replace("-", "")
        _lbl2 = _lbl2.replace(":", "").replace("-", "")
        _lbl1 = _lbl1.replace(" ","T")
        _lbl2 = _lbl2.replace(" ","T")
        titleLbl = f'[{fnK}] {_lbl1}_{_lbl2}'
        titleLbl += f' set-max.: {hmMaxVal}'
        msg = "generating heatmap image.. [%s]"%(titleLbl)
        q2m.put(("displayMsg", msg,), True, None)
        ### make base image to draw heatmap
        fSh = fImg.shape
        baseImg = self.makeBaseImg(fImg) 
        img = baseImg.copy()
        ### convert data type if applicable
        if np.max(hmArr) < np.iinfo(np.uint8).max:
            hmArr = hmArr.astype(np.uint8)
        elif np.max(hmArr) < np.iinfo(np.uint16).max:
            hmArr = hmArr.astype(np.uint16)
        ### draw heatmap 
        img = prnt.drawHeatmapImg(hmArr, (0,0,0), img, hmLvlRngs, 
                                  hmCols, titleLbl, hmRad)

        chk = wx.FindWindowByName("saveHMVideo_chk", main.panel["ml"])
        if chk.GetValue():
            q2m.put(("displayMsg", "making heatmap video..",), True, None)
            args = (main.inputFP, fnK, copy(fImg), bD, bD_dt, cvFont, fThck, \
                    fScale, txtW, txtH, txtBl)
            # make heatmap video file
            self.makeHeatmapVideo(args)

        rawData = dict(heatMapArr=hmArr)

        return img, rawData 

    #---------------------------------------------------------------------------

    def makeHeatmapVideo(self, args):
        """ Make heatmap video 
        
        Args:
            args (tuple): Arguments
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        flag = "gradual" # gradual/ flash/ ani

        inputFP, fnK, fImg, bData, bD_dt, \
          cvFont, fThck, fScale, txtW, txtH, txtBl = args

        fn = f'heatmap_{fnK}_{get_time_stamp()}.avi'
        _fp0, _fp1 = path.split(inputFP)
        if _fp1 == "": _fp0 = path.split(_fp0)[0]
        fPath = path.join(_fp0, fn)
        if path.isfile(fPath): remove(fPath)
        print(f"\n{fPath}\n")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #rad = int(round(img.shape[0] * 0.002))
        #col = (255,255,0)
        w = wx.FindWindowByName('heatMapVideoFPS_cho', 
                                self.mainFrame.panel["ml"])
        videoFPS = int(w.GetString(w.GetSelection())) 
        '''
        ### cut a part of data to make the video
        _i0 = 25000; _i1 = 25300
        bData = bData[_i0:_i1] 
        bD_dt = bD_dt[_i0:_i1] 
        '''
        
        fSh = fImg.shape
        snapshot = np.zeros(fSh, dtype=np.uint8)
        grey = np.zeros(tuple(fSh[:2]), dtype=np.int16) 
        baseImg = self.makeBaseImg(fImg) 
        frame = baseImg.copy()
        if flag == "gradual":
            inc = 85 # increase of pixel value per each motion
            dec = 5 # 1/5/17 decrease of all pixel values per each frame
            mPtRad = 2 # when '0', it means single pixel value change
        elif flag == "flash":
            mPtRad = 0
        # init video writer
        video_rec = cv2.VideoWriter(fPath, fourcc=fourcc, fps=videoFPS, 
                            frameSize=(frame.shape[1], frame.shape[0]), 
                            isColor=True)
        
        if flag in ["gradual", "ani"] and mPtRad > 0:
            tmpGrey = np.zeros(tuple(fSh[:2]), dtype=np.uint8)

        dLen = []
        for roiK in bData.keys(): dLen.append(len(bData[roiK]))
        dLen = max(dLen)
        for fi in range(dLen):
            if flag != "ani" and fi%10 == 0:
                msg = f'writing video {fi+1}/ {dLen}'
                print("\r", msg, end="          ", flush=True)
            
            if flag == "flash": frame = baseImg.copy()
            
            for roiK in bData.keys():
                _bD = bData[roiK][fi]
            
                ### draw motion points 
                if flag == "gradual":
                # in accumulation with slow decrease
                    grey -= dec
                    if mPtRad > 0: tmpGrey[:,:] = 0
                    for x, y in _bD:
                        #x -= ssROI[0]
                        if mPtRad == 0:
                            grey[y,x] += inc
                        else:
                            cv2.circle(tmpGrey, (x, y), mPtRad, inc, -1)
                    if mPtRad > 0: grey += tmpGrey
                    grey[grey>255] = 255
                    grey[grey<0] = 0
                
                elif flag == "flash":
                    for x, y in _bD:
                        if mPtRad == 0:
                            frame[y,x] = (255, 255, 255)
                        else:
                            cv2.circle(frame, (x, y), mPtRad, (255,255,255), -1)
                    if mPtRad > 0: grey += tmpGrey
                    grey[grey>255] = 255
                    grey[grey<0] = 0

                if flag != "flash":
                    frame = baseImg.copy()
                    frame += cv2.cvtColor(grey.astype(np.uint8), 
                                          cv2.COLOR_GRAY2BGR)

            if flag != "ani":
                dtStr = ""
                for roiK in bData.keys():
                    if fi < len(bD_dt[roiK]):
                        dtStr = str(bD_dt[roiK][fi])
                        break
                if dtStr != "":
                    tx = int(frame.shape[1]/2) - int(txtW*19/2)
                    ty = txtH + txtBl + 5
                    # write datetime
                    cv2.putText(frame, str(bD_dt[roiK][fi]), (tx, ty), cvFont, 
                        fontScale=fScale, color=(255,255,255), thickness=fThck)
                
            # write this frame into video
            video_rec.write(frame)

        video_rec.release()

    #---------------------------------------------------------------------------

    def makeBaseImg(self, img):
        """ make base image for other data drawing such as heatmap  
        
        Args:
            img (numpy.ndarray): Input image.
         
        Returns:
            img (numpy.ndarray): Output image.
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 
                               iterations=3) # decrease minor features 
        img = cv2.Canny(img, 40, 80)
        img = img.astype(np.int16)
        img -= 230
        img[img<0] = 0
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return img

    #---------------------------------------------------------------------------

    def getCtOfBR(self, pts):
        """ get center-points of a bounding rect points (ant-bounding-rect 
        detected by color)
        
        Args:
            args (list): List of rect-points 
         
        Returns:
            cts (list): List of center-points 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        cts = [] # list of center-point of blobs 
        for x1, y1, x2, y2, x3, y3, x4, y4 in pts:
            xMean = int(np.mean([x1, x2, x3, x4]))
            yMean = int(np.mean([y1, y2, y3, y4]))
            cts.append((xMean, yMean))
        return cts

    #---------------------------------------------------------------------------

    def getSavFilePath(self, gIdx, sExt, rawDataFP=""):
        """ get file path to save graph or raw data
        
        Args:
            gIdx (int): Graph index 
            sExt (str): File extenstion to save
            rawDataFP (str): This function was called to verify 
              whether the raw data with the given 'rawDataFP' is 
              valid to save.
         
        Returns:
            fp4sav (None/str): Save file path.
                               If it's not valid, returns None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        oPath, fn = path.split(self.mainFrame.inputFP) 
        fnK = self.parent.graphImg[gIdx]["fnK"] 

        if rawDataFP == "":
            ### determine save file path
            newFN = copy(fnK)
            proc2run = self.parent.graphImg[gIdx]["proc2run"]
            newFN += f'_{proc2run}'
            if proc2run == "spAHeatmap": newFN += f'_{gIdx}' 
            newFN += f'{sExt}'
            # get the 1st replicate-index from data filename
            replIdx = int(fnK.split("_")[0].split("-")[0].lstrip("ID"))
            for gK in self.rIdLst.keys():
                if replIdx in self.rIdLst[gK]:
                # if the replicate-index is in this group-key
                    _path = path.join(oPath, gK)
                    if path.isdir(_path): # if the folder exists
                        oPath = _path 
                    break
            fp4sav = path.join(oPath, newFN)

        else:
            ### verify whether this raw data file path is valid to save
            replIdx = int(fnK.split("_")[0].split("-")[0].lstrip("ID")) 
            if replIdx in self.singleRLst:
            # if this replicate belongs to the singleRLst;
            #   list of replicate IDs which has only one ROI in the video
                _ext = "." + rawDataFP.split(".")[-1]
                roi = rawDataFP.replace(_ext,"").split("_")[-1]
                if self.singleRLstROI[replIdx] == roi: fp4sav = rawDataFP
                else: fp4sav = None
            else:
                fp4sav = rawDataFP
        return fp4sav

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ != "__main__":
    pass


