# coding: UTF-8
"""
This module is for DataVisualizer app to process data from AntOS of Jinook Oh.

Dependency:
    Numpy (1.17)
    SciPy (1.4)
    OpenCV (3.4)
    tsmoothie (1.0)

    #PYOD (0.9)

last edited: 2023-07-18
"""

import sys, csv, ctypes
from os import path, remove
from glob import glob
from copy import copy
from time import time
from datetime import datetime, timedelta
from random import randint

import wx, cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter, welch, periodogram
from scipy.cluster.vq import vq, kmeans 
from scipy import stats
### packages for anomaly detection
#from pyod.models.mad import MAD # Median Absolute Deviation
from tsmoothie.smoother import LowessSmoother

from modFFC import *
from modCV import *

MyLogger = setMyLogger("visualizer")

DEBUG = False

#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(int(ctypes.c_ulong(-1).value//2))

#===============================================================================

class ProcAntOS:
    """ Class for processing data and generate graph of AntOS.
    
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

        self.camIdx = None 
        self.camIdx4i = None 
        self.imgFiles = None 
        self.data = None 
        self.ci = None 
        self.keys = None 
        self.temperature = None 
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

        main = self.mainFrame
        
        # camera index for certain region of interest
        camIdx4i = dict(
                            roi0=99 # ROI-0 was recorded by cam-index 99
                        ) 
        
        # gather paths of all files under the input folder, 
        # including sub-folders.
        fileLst = getFilePaths_recur(main.inputFP)
        
        rsltFP = []
        logFP = []
        imgFiles = {} 
        camIdx = []
        for fp in fileLst:
            fn = path.basename(fp)
            if fn.startswith("rslt"): rsltFP.append(fp)
            elif fn.startswith("log_"):
                logFP.append(fp) # store log file path
            elif fn.split(".")[-1] == "jpg":
                try:
                    ci = fp.replace(".jpg","").split("_cam-")[-1]
                    ci = int(ci)
                    if not ci in camIdx:
                        camIdx.append(ci)
                        imgFiles[ci] = []
                    imgFiles[ci].append(fp) # store image file path
                except:
                    continue
       
        if len(rsltFP) == 0: return False
        data = []
        _d = []
        for i, rfp in enumerate(rsltFP):
            msg = "Loading CSV files.. %i/ %i"%(i+1, len(rsltFP))
            q2m.put(("displayMsg", msg), True, None)
            _d = list(csv.reader(open(rsltFP[i], "r"))) # load CSV data
            if i == 0: dataHeader = copy(_d[0]) # store csv header
            _d.pop(0)
            data += _d 

        '''
        ### !!! TEMPORARY for flatChamber data (202107) !!! -----
        for di in range(len(data)):
            if len(data[di]) == 3:
                data[di].append("(-1/ -1)")
        ### -----------------------------------------------------
        '''

        ### store indices of each column-title
        ci = {}
        for _ci in range(len(dataHeader)):
            key = dataHeader[_ci].strip()
            ci[key] = _ci 

        ### store keys
        keys = []
        for d in data: 
            keys.append(d[ci["Key"]].strip())
            # break loop when it's a new data-set (keys are rotated once)
            if len(keys) > 1 and keys[0] == keys[-1]: break

        ### Remove motionPts 
        ###   where too many motions points are recorded in a single 
        ### data line (false motion due to lighting condition changes)
        msg =  "Removing motionPts by external causes ..."
        q2m.put(("displayMsg", msg), True, None)
        lst = [] # temporary list of number of motion-points
        idx = [] # index of data
        for di in range(len(data)):
            d = data[di]
            k = d[ci["Key"]].strip()
            if k == "motionPts":
                v = d[ci["Value"]]
                lst.append(len(v.split(")(")))
                idx.append(di)
        lst = np.asarray(lst) 
        w = wx.FindWindowByName("maxMotionThr_txt", main.panel["ml"])
        try: thr = int(w.GetValue())
        except: thr = 100
        idxI = list(np.where(lst >= thr)[0]) # index of idx where too many
                                             # motion points are recorded
        for i, _idx in enumerate(idxI):
            if i % 100 == 0:
                p = (i+1) / len(idxI) * 100 
                msg =  "Validating motionPts (%.1f %%)..."%(p)
                q2m.put(("displayMsg", msg), True, None)
            idx2pop = idx[_idx]-i
            print("[DROP] ", data[idx2pop][1]) # print timestamp of dropped data
            data.pop(idx2pop)
        msg = "Motion points Mean: %i"%(int(np.mean(lst)))
        msg += ", Median: %i"%(int(np.median(lst)))
        msg += ", Max: %i"%(int(np.max(lst)))
        msg += ", Number of dropped frames: %i"%(len(idxI))
        print(msg)

        # store the datetime of the first data 
        self.firstDT = get_datetime(data[0][ci["Timestamp"]])

        ### store temperature data with datetime, max and min value.
        msg =  "Storing temperature data ..."
        q2m.put(("displayMsg", msg), True, None)
        temperature = dict(dt=[], t=[], ma=-1, mi=-1)
        for i, lfp in enumerate(logFP):
            fh = open(lfp, "r")
            lines = fh.readlines()
            fh.close()
            for line in lines:
                if "Temperature:" in line:
                    try: t = float(line.split("Temperature:")[1])
                    except: continue
                    ts = line.split(",")[0]
                    dt = get_datetime(ts)
                    temperature["dt"].append(dt)
                    temperature["t"].append(t)
        if temperature["t"] == []:
            temperature["mi"] = -1
            temperature["ma"] = -1
        else:
            temperature["mi"] = np.min(temperature["t"])
            temperature["ma"] = np.max(temperature["t"]) 

        # location labels
        self.loc = ["brood", "food", "garbage"]
        # location color
        self.locC = dict(brood=(0,255,255),
                         food=(255,255,0),
                         garbage=(255,50,255))
        # graph line color for multiple measures
        self.gC = [(255,150,150), (100,255,100), (127,127,255),
                   (100,255,255), (255,255,100), (255,100,255)]
        
        ret = dict(camIdx=camIdx, camIdx4i=camIdx4i, imgFiles=imgFiles,
                   data=data, ci=ci, keys=keys, temperature=temperature)
        q2m.put(("finished", ret,), True, None)

    #---------------------------------------------------------------------------

    def drawGraph(self, q2m, gSIdx=0):
        """ Draw graph with data from AntOS
        
        Args:
            q2m (queue.Queue): Queue to send data to main thread.
            gSIdx (int): Start index for this graph.
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        main = self.mainFrame # wx.Frame in visualizer.py
        prnt = self.parent # graph processing module (procGraph.py)
        data = self.data # list of lines in CSV result file
        ci = self.ci # column index of CSV data
        cho = wx.FindWindowByName("camIdx_cho", main.panel["ml"])
        try: camIdx = int(cho.GetString(cho.GetSelection())) # camera index 
        except: camIdx = self.camIdx4i["roi0"]
        imgFileLst = self.imgFiles[camIdx] # image file list
        cho = wx.FindWindowByName("process_cho", main.panel["ml"])
        proc2run = cho.GetString(cho.GetSelection()) # process to run 
        tempMeasureIntv = 600 # temperature measure interval in seconds
        cvFont = cv2.FONT_HERSHEY_PLAIN # font for writing info on graph image
        loc = self.loc  # ptoi/location keywords
        locC = self.locC # location colors
        img = None # graph image to return
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
        days = [] # days in this graph
        ### get region of interest (x, y, w, h) rects
        roi = dict(roi0=[-1, -1, -1, -1])
        roiCt = {}
        for k in roi.keys():
            if camIdx == self.camIdx4i[k]:
                txt = wx.FindWindowByName("%sROI_txt"%(k), main.panel["ml"])
                r = txt.GetValue() 
                try: r = [int(_x) for _x in r.split(",")]
                except: pass
                if len(r) == 4: roi[k] = r 
                # store center of the ROI
                roiCt[k] = (int(r[0]+r[2]/2), int(r[1]+r[3]/2))
        ### get spAGrid info 
        spAGrid = dict(rect=[-1,-1,-1,-1])
        w = wx.FindWindowByName("spAGridRows_cho", main.panel["ml"])
        rows = int(w.GetString(w.GetSelection())) 
        w = wx.FindWindowByName("spAGridCols_cho", main.panel["ml"])
        cols = int(w.GetString(w.GetSelection()))
        spAGrid["rows"] = rows
        spAGrid["cols"] = cols 
        w = wx.FindWindowByName("spAGridMExIter_cho", main.panel["ml"])
        spAGrid["MExIter"] = int(w.GetString(w.GetSelection())) 
        w = wx.FindWindowByName("spAGridROI_txt", main.panel["ml"])
        r = w.GetValue()
        try: r = [int(_x) for _x in r.split(",")]
        except: pass
        if len(r) == 4: spAGrid["rect"] = r
        if spAGrid["rect"] != [-1,-1,-1,-1]:
            x1, y1 = r[:2]
            x2 = x1 + r[2] 
            y2 = y1 + r[3] 
            spAGrid["cW"] = int((max(x1, x2)-min(x1, x2)) / cols)
            spAGrid["cH"] = int((max(y1, y2)-min(y1, y2)) / rows)
        ### get points of interests
        ptoi = {}
        for k in loc:
            ptoi[k] = [-1, -1]
            txt = wx.FindWindowByName("%sPt_txt"%(k), main.panel["ml"])
            pt = txt.GetValue() 
            try: pt = [int(_x) for _x in pt.split(",")]
            except: pass
            if len(pt) == 2: ptoi[k] = pt
        ### get threshold to determine location boundary around ptoi
        w = wx.FindWindowByName("locThr_txt", main.panel["ml"])
        txtVal = w.GetValue().split(",")
        locThr = {}
        for i, k in enumerate(loc):
            try: locThr[k] = float(txtVal[i])
            except: locThr[k] = 0.5
        ### get distance of location-boundary for each ptoi
        locDist = {}
        for k in loc:
            locDist[k] = -1
            if ptoi[k] == [-1, -1]: continue
            x1, y1 = ptoi[k]
            dists = []
            for _k in loc:
                if k == _k or ptoi[_k] == [-1, -1]: continue
                x2, y2 = ptoi[_k]
                dists.append(np.sqrt((x1-x2)**2 + (y1-y2)**2))
            if len(dists) > 0:
                locDist[k] = np.min(dists) * locThr[k]
        if "_" in proc2run:
        # "_" in process name indicates multiple measurements 
            bDataLst = {}
            for k in proc2run.split("_"): bDataLst[k] = []
        elif proc2run == "intensityL":
        # multiple measures based on location
            bDataLst = {}
            for k in loc: bDataLst[k] = []
        elif proc2run == "saMVec":
        # single-ant's movement vector (direction, distance & position)
            bDataLst = dict(direction=[], dist=[], posX=[], posY=[])
        elif proc2run.startswith("spAGrid"):
        # activity measures of each cell in a grid 
            bDataLst = {}
            for r in range(spAGrid["rows"]):
                for c in range(spAGrid["cols"]):
                    k = "[%i][%i]"%(r, c)
                    bDataLst[k] = []
        else:
            bDataLst = [] # list of bundled data
        bD_dt = [] # datetime for the bundled data
        hmArr = None # array for heatmap (or other spatial-analysis)
        ### make the list of process which measures multiple items 
        # local-intensity
        self.multipleMeasures = ["intensityL", "saMVec"]
        # process which has the underbar is measuring multiple items 
        if "_" in proc2run: self.multipleMeasures.append(proc2run)
        # spatial-analysis-grid measures in each grid-cell 
        if "spAGrid" in proc2run: self.multipleMeasures.append(proc2run)
        
        if proc2run == "initImg":
        ##### [begin] data-processing for initImg ----- 
            img = cv2.imread(imgFileLst[0]) # load 1st image
          
            ### set colors
            clr = {}
            clr["roi0"] = (200, 75, 75) # roi0
            clr["bl"] = (0,0,200) # to mark ant's body length

            ### draw ROI rects
            for k in roi.keys():
                if roi[k] != [-1, -1, -1, -1]:
                    x1, y1 = roi[k][:2]
                    x2 = x1 + roi[k][2] 
                    y2 = y1 + roi[k][3] 
                    cv2.rectangle(img, (x1, y1), (x2, y2), clr["roi0"], 3)

            ### draw the grid for spatial-analysis
            thck = 2
            if spAGrid["rect"] != [-1,-1,-1,-1]:
                x1, y1 = spAGrid["rect"][:2]
                x2 = x1 + spAGrid["rect"][2] 
                y2 = y1 + spAGrid["rect"][3] 
                cv2.rectangle(img, (x1, y1), (x2, y2), clr["roi0"], thck)
                for r in range(1, spAGrid["rows"]):
                    ry = y1 + r*spAGrid["cH"]
                    cv2.line(img, (x1, ry), (x2, ry), clr["roi0"], thck)
                for c in range(1, spAGrid["cols"]):
                    cx = x1 + c*spAGrid["cW"]
                    cv2.line(img, (cx, y1), (cx, y2), clr["roi0"], thck)

            ### draw the center of ROI0
            r = int(min(img.shape[0], img.shape[1]) * 0.015)
            cv2.circle(img, tuple(roiCt["roi0"]), r, clr["roi0"], -1)
            
            ### draw points of interest & its boundary
            for k in ptoi.keys():
                if ptoi[k] != [-1, -1]:
                    r = int(min(img.shape[0], img.shape[1]) * 0.05)
                    # draw point of interest
                    cv2.circle(img, tuple(ptoi[k]), r, locC[k], 3)
                    if locDist[k] != -1:
                        ### draw location boundary
                        r = int(locDist[k]*0.75)
                        cv2.circle(img, tuple(ptoi[k]), r, locC[k], 2)
                        r = int(locDist[k])
                        col = tuple([min(x+50, 255) for x in locC[k]])
                        cv2.circle(img, tuple(ptoi[k]), r, col, 1)

            ### draw body length of an ant in the center of image
            iH, iW = img.shape[:2]
            w = wx.FindWindowByName("antLen_txt", main.panel["ml"])
            self.antLen = int(w.GetValue())
            pt1 = (int(iW/2)-1, int(iH/2)-int(self.antLen/2))
            pt2 = (int(iW/2)+1, int(iH/2)+int(self.antLen/2))
            cv2.rectangle(img, pt1, pt2, clr["bl"], -1)

            rawData = [None]
        ##### [end] data-processing for initImg ----- 

        elif proc2run == "sanityChk":
        ##### [begin] data-processing for sanityChk ----- 
            rdi = 0 # for keeping read index of the CSV data
            for imgFile in imgFileLst:
                img = cv2.imread(imgFile)
                fn = path.basename(imgFile)
                q2m.put(("displayMsg", "processing - %s"%(fn),), True, None)
                ts = fn.split("_cam-")[0] # time-stamp of the image
                imgDT = get_datetime(ts)
                for di in range(rdi, len(data)):
                    ### check whether this data belongs to the target cam-index
                    try: _camIdx = int(data[di][ci["Cam-idx"]])
                    except: continue
                    if camIdx != _camIdx: continue
                    # time-stamp of this data 
                    ts = data[di][ci["Timestamp"]].strip() 
                    # datetime of the time-stamp
                    dt = get_datetime(ts) 
                    if dt > imgDT:
                        for _di in range(di, len(data)):
                            _ts = data[_di][ci["Timestamp"]].strip() # timestamp
                            if _ts != ts: break
                            _key = data[_di][ci["Key"]].strip() 
                            _value = data[_di][ci["Value"]].strip()
                            if _key == "colCent":
                                pt = [int(_x) for _x in _value.split("/")]
                                cv2.circle(img, tuple(pt), 10, (200,0,0), -1)
                            elif _key == "motionPts":
                                pts = _value.strip("()").split(")(")
                                for pt in pts:
                                    pt = [int(_x) for _x in pt.split("/")]
                                    cv2.circle(img, tuple(pt), 5, (0,0,200), -1)
                        break
                prnt.storeGraphImg(img, main.pi["mp"]["sz"])
                rdi = di
            rawData = [None]
        ##### [end] data-processing for sanityChk ----- 
        else:
        ##### [begin] data-processing (bundling) for actual measurements ----- 
            ### get user-defined graph default size 
            txt = wx.FindWindowByName("graphSz_txt", main.panel["ml"])
            try:
                graphW, graphH = txt.GetValue().split(",")
                graphW = int(graphW)
                graphH = int(graphH)
            except:
                graphW = 1500
                graphH = 500
            ### zero padding to make daily graph
            if proc2run in ["intensity", "intensityL", "spAHeatmap",
                            "motionSpread", "proxMCluster", 
                            "dist2b", "dist2c", "dist2ca",
                            "saMVec"]:
                flagZeroPadding = True
            else:
                flagZeroPadding = False
            ### get interval to bundle one data point 
            cho = wx.FindWindowByName("dataPtIntv_cho", main.panel["ml"])
            dPtIntvSec = int(cho.GetString(cho.GetSelection()))
            dPtIntv = timedelta(seconds=dPtIntvSec)
            ### get interval to generate heatmap
            cho = wx.FindWindowByName("heatMapIntv_cho", main.panel["ml"])
            hmIntvMin = int(cho.GetString(cho.GetSelection()))
            hmIntv = timedelta(seconds=hmIntvMin*60)
            if hmIntvMin == -1: gSIdx = 0
            # graph background color
            gBg = (30, 30, 30) 
            cutIdx = None
            ### init temporary lists to bundle data for each data point
            if proc2run in ["intensity", "intensityPSD"]:
                tBin = dict(intensity=[])
            elif proc2run == "intensityL":
                tBin = dict(mx=[], my=[]) # motion point x & y coordinates
            elif proc2run == "saMVec":
                tBin = dict(mx=[], my=[], prevPos=[-1,-1]) 
            elif proc2run == "spAHeatmap":
                tBin = dict(m_pts=[], ts=[]) 
                ### make array for heatmap
                fImg = cv2.imread(imgFileLst[0])
                # init array for heatmap
                hmArr = np.zeros(fImg.shape[:2], dtype=np.uint32) 
                ### get heatmap point radius
                ###   if this is -1, it will be a single pixel 
                txt = wx.FindWindowByName("hmPt_txt", main.panel["ml"])
                hmRad = int(txt.GetValue())
            elif proc2run.startswith("spAGrid"):
                tBin = dict(m_pts=[])
                fImg = cv2.imread(imgFileLst[0])
                if proc2run == "spAGridDenseActivity":
                    # hmArr array will be used 
                    #   for measuring something in each cell of the grid
                    hmArr = np.zeros(fImg.shape[:2], dtype=np.uint8) 
                else:
                    hmArr = np.zeros(fImg.shape[:2], dtype=np.uint32)
            elif proc2run.startswith("dist2"):
                tBin = dict(mx=[], my=[]) 
            elif proc2run == "motionSpread":
                tBin = dict(
                    spreadValue = [], # motion spread values
                    pMPts = [], # motion-points in previous data-line 
                    pMACents = [] # centroids of moved ants in the previous line
                    )
                fImg = cv2.imread(imgFileLst[0])
                # hmArr array will be used for debugging purpose
                hmArr = np.zeros(fImg.shape[:3], dtype=np.uint8) 
            elif proc2run == "proxMCluster":
                tBin = dict(interactingMCluster = [])
                fImg = cv2.imread(imgFileLst[0])
                hmArr = np.zeros(fImg.shape[:2], dtype=np.int16)
            ### bundle data with interval
            args = (camIdx, roi, proc2run, data, ci, bDataLst, bD_dt, gSIdx, \
                    flagZeroPadding, dPtIntvSec, dPtIntv, hmIntvMin, hmIntv, \
                    tBin, ptoi, roiCt, locDist, hmArr, spAGrid, q2m)
            ret = self.bundleData(args) 
            bDataLst, bD_dt, gSDT, gEIdx, gEDT, zPadIdx, days, hmArr = ret
        ##### [end] data-processing (bundling) for actual measurements ----- 

        ##### [begin] drawing graph ----- 
        if proc2run in ["intensity", "intensityL", "motionSpread", 
                        "proxMCluster", "dist2b", "dist2c", "dist2ca"]:
            args = (proc2run, bDataLst, dPtIntvSec, graphW, graphH, gBg, \
                    cvFont, bD_dt, tempMeasureIntv, main, flagZeroPadding, \
                    q2m, days, gSDT, gEDT, zPadIdx)
            img, nDCols, rawData = self.drawBarGraph(args)

        elif "PSD" in proc2run: 
        # Applying Welch's method for estimating power spectral density 
        #   - smoothing over non-systematic noise
        #   - being robust to some non-stationarities
            args = (proc2run, main, bDataLst, dPtIntvSec, graphW, graphH, \
                    gBg, cvFont, bD_dt, fsPeriod)
            img, fsPeriod, mg = self.drawPSD(args)
            rawData = [None]
        
        elif proc2run == "spAHeatmap":
            args = (hmArr, main, prnt, bDataLst, bD_dt, cvFont, hmRad, camIdx, \
                    gSDT, gEDT, q2m, fImg)
            img, rawData = self.drawHeatmap(args)

        elif proc2run.startswith("spAGrid"):
            args = (proc2run, bDataLst, dPtIntvSec, graphH, gBg, \
                    cvFont, bD_dt, main, spAGrid, q2m)
            img, cpdImg, rawData = self.drawGridMeasure(args)

        elif proc2run == "saMVec":
            img = cv2.imread(imgFileLst[0])
            img = self.makeBaseImg(img) 
            args = (proc2run, bDataLst, bD_dt, img, q2m)
            img, rawData = self.drawMVec(args)
        ##### [end] drawing graph -----
            
        q2m.put(("displayMsg", "storing image and data..",), True, None)
       
        ### store additional info, different depending which type of graph
        gInfo = dict(bDataLst=bDataLst, bD_dt=bD_dt, gSIdx=gSIdx, gEIdx=gEIdx,
                     timeLbl=timeLbl, days=days, mg=mg, dPtIntvSec=dPtIntvSec)
        if proc2run in ["intensity", "intensityL", "motionSpread", 
                        "proxMCluster", "dist2b", "dist2c", "dist2ca"]:
            gInfo["nDataInRow"] = nDCols # number of data columns 
                                         #   in a row of daily data
            gInfo["initImgHeight"] = img.shape[0]

        elif proc2run.startswith("spAGrid"):
            gInfo["spAGrid"] = spAGrid
            additionalImg = cpdImg

        elif "PSD" in proc2run:
            gInfo["initImgWidth"] = img.shape[1]
            gInfo["fsPeriod"] = fsPeriod
       
        ### send the results
        output = ("finished", img, rawData, gInfo, additionalImg)
        q2m.put(output, True, None) 

    #---------------------------------------------------------------------------

    def fillEmptyData(self, proc2run, bDataLst, bD_dt, dt):
        if proc2run == "spAHeatmap":
        # heatmap; has list of motion-points in each bundle
            fillItem = []
        elif proc2run == "saMVec":
        # saMVec (single-ant movement vector)
            fillItem = dict(direction=-1, dist=0, posX=-1, posY=-1)
        elif proc2run.startswith("dist2"):
        # distance measure 
            fillItem = -1
        else:
            fillItem = 0
        if proc2run in self.multipleMeasures:
        # multiple measurements
            for k in bDataLst.keys():
                if type(fillItem) == dict: bDataLst[k].append(fillItem[k])
                else: bDataLst[k].append(fillItem)
        else:
            bDataLst.append(fillItem)
        bD_dt.append(dt)
        return bDataLst, bD_dt

    #---------------------------------------------------------------------------

    def bundleData(self, args):
        """ Bundle data with time interval
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        camIdx, roi, proc2run, data, ci, bDataLst, bD_dt, gSIdx, \
          flagZeroPadding, dPtIntvSec, dPtIntv, hmIntvMin, hmIntv, tBin, \
          ptoi, roiCt, locDist, hmArr, spAGrid, q2m = args

        main = self.mainFrame
        ### get hours to ignore & process
        h2p = {}
        for k in ["h2ignore", "h2proc"]:
            w = wx.FindWindowByName(f'{k}_txt', main.panel["ml"])
            h2p[k] = int(widgetValue(w))
       
        days = [] # days in this graph
        gSDT = None # starting datetime of this graph
        gEIdx = -1 # data index where this graph ends
        gEDT = None # end datetime of this graph
        zPadIdx = [-1, -1] # zero padding index at the beginning and end
        if proc2run == "spAGridOFlow":
            pImg = None # for storing data image from the previous data bundle

        dLen = len(data)
        for di in range(gSIdx, dLen):
            if di%1000 == 0:
                msg = "processing data %i/ %i"%(di+1, dLen)
                q2m.put(("displayMsg", msg,), True, None)
            ### check whether this data belongs to the target cam-index
            try: _camIdx = int(data[di][ci["Cam-idx"]])
            except: continue
            if camIdx != _camIdx: continue # ignore data with 
                                           #   different cam index
            ts = data[di][ci["Timestamp"]].strip() # timestamp of the data
            dt = get_datetime(ts) # datetime of the timestamp

            elapsedHour = ((dt-self.firstDT).total_seconds())/60/60 
            if h2p["h2ignore"] > 0:
                # ignore some hours at the beginning
                if elapsedHour < h2p["h2ignore"]: continue
            if h2p["h2proc"] > 0:
                ### finish data processing if enough hours passed 
                if elapsedHour >= h2p["h2ignore"] + h2p["h2proc"]:
                    gEIdx = -1 # notify it reached the end
                    gEDT = dt
                    break

            if gSDT is None: 
            # the 1st data to process
                gSDT = dt # store the datetime of start of the graph
                sDI = copy(di) # store starting index of this data bundle 
                days.append(datetime(year=dt.year, month=dt.month, 
                                     day=dt.day))
                # datetime of the beginning of the day
                _dayB = datetime(year=dt.year, month=dt.month, day=dt.day,
                                 hour=0, minute=0, second=0)
                if flagZeroPadding:
                    ### fill the missing data in the starting date 
                    while _dayB+dPtIntv < dt:
                        bDataLst, bD_dt = self.fillEmptyData(
                                            proc2run, bDataLst, bD_dt, _dayB
                                            )
                        _dayB = _dayB + dPtIntv
                    # store where zero padding ended at the beginning 
                    zPadIdx[0] = len(bDataLst) 
                    sDT = _dayB # store starting datetime of this data bundle
                else:
                    sDT = copy(gSDT)

            if dt-sDT >= dPtIntv: # elapsed time is over the interval
            ##### [begin] process data of the past interval -----
                nIntv = int((dt-sDT).total_seconds()/dPtIntvSec)
                if nIntv > 1:
                    ### fill zero data when time passed over one interval
                    for __ in range(nIntv-1):
                        bDataLst, bD_dt = self.fillEmptyData(
                                            proc2run, bDataLst, bD_dt, sDT
                                            )
                        sDT = sDT + dPtIntv
                
                ##### [begin] store data & TS of this bundled data ---
                if proc2run in ["intensity", "intensityPSD"]:
                    _inten = sum(tBin["intensity"]) 
                    bDataLst.append(_inten)

                elif proc2run == "motionSpread":
                    bDataLst.append(sum(tBin["spreadValue"]))

                elif proc2run == "proxMCluster":
                    bDataLst.append(sum(tBin["interactingMCluster"]))

                elif proc2run == "intensityL":
                    mx = np.asarray(tBin["mx"])
                    my = np.asarray(tBin["my"])
                    for k in ptoi.keys():
                        if -1 in ptoi[k]:
                            bDataLst[k].append(0) 
                        else:
                            dist_m = np.sqrt((mx-ptoi[k][0])**2 + 
                                             (my-ptoi[k][1])**2)
                            ptW = (locDist[k] - dist_m) / locDist[k]
                            ptW[ptW<0] = 0
                            bDataLst[k].append(np.sum(ptW))

                elif proc2run.startswith("dist2"):
                    if tBin["mx"] == []:
                        bDataLst.append(-1)
                    else:
                        if proc2run == "dist2b":
                            _pt = ptoi["brood"]
                        elif proc2run in ["dist2c", "dist2ca"]:
                            _pt = roiCt["roi0"]
                        '''
                        cx = np.mean(tBin["mx"])
                        cy = np.mean(tBin["my"])
                        dist = np.sqrt((_pt[0]-cx)**2 + (_pt[1]-cy)**2)
                        bDataLst.append(int(round(dist)))
                        '''
                        if proc2run in ["dist2b", "dist2c"]:
                            dists = []
                            for mpti, x in enumerate(tBin["mx"]):
                                y = tBin["my"][mpti]
                                dists.append(
                                        np.sqrt((_pt[0]-x)**2 + (_pt[1]-y)**2)
                                        )
                            bDataLst.append(int(max(dists)))
                        elif proc2run == "dist2ca":
                            ### store data with all the motion points,
                            ### regardless of data-bundle-interval.
                            for mpti, x in enumerate(tBin["mx"]):
                                y = tBin["my"][mpti]
                                dist = np.sqrt((_pt[0]-x)**2 + (_pt[1]-y)**2)
                                # store data
                                bDataLst.append(int(dist))
                                # [*] store timestamp for this data
                                bD_dt.append(sDT)
                            bD_dt.pop(-1) # [*] remove the last timestamp, 
                                # because one will be added for all types of 
                                # 'proc2run'

                elif proc2run == "saMVec": 
                    flagNonMov = False
                    if tBin["mx"] == []:
                        flagNonMov = True 
                    else:
                        xs = np.asarray(tBin["mx"])
                        ys = np.asarray(tBin["my"])
                        if tBin["prevPos"] == [-1, -1]:
                            tBin["prevPos"] = [int(np.mean(xs)), 
                                               int(np.mean(ys))]
                            flagNonMov = True 
                        else:
                            px, py = tBin["prevPos"]
                            dists = np.sqrt((xs-px)**2 + (ys-py)**2) 
                            # furthest distance from the previous position
                            fDist = np.max(dists)
                            if fDist < self.antLen*1.2:
                            # didn't move long enough
                                flagNonMov = True 
                            else:
                                fIdx = np.where(dists==fDist)[0][0]
                                # the furthest point from the previous position
                                x = xs[fIdx]; y = ys[fIdx]
                                # get the degree of direction
                                deg = calc_line_angle((px,py), (x, y))
                                deg = convt_180_to_360(deg)
                                ''' 
                                ### [debugging]
                                _img = np.zeros((1440, 550), np.uint8)
                                cv2.line(_img, (px,py), (x,y), 255, 5)
                                cv2.circle(_img, (x,y), 5, 255, -1)
                                cvFont = cv2.FONT_HERSHEY_PLAIN
                                cv2.putText(_img, f'[{deg}]', (5, 20), cvFont, 
                                    fontScale=1.0, color=255, thickness=1)
                                fn = "%s.jpg"%(str(sDT).replace(" ","T"))
                                if deg == 90:
                                    fn = fn.replace(".jpg", "___90___.jpg")
                                cv2.imwrite(fn, _img)
                                ''' 
                                # store direction
                                bDataLst["direction"].append(deg)
                                # store distance
                                bDataLst["dist"].append(fDist/self.antLen)
                                ### store the current position
                                bDataLst["posX"].append(x)
                                bDataLst["posY"].append(y)
                                # update the prev. position in temp. bin
                                tBin["prevPos"] = (x, y)
                    if flagNonMov:
                        bDataLst["direction"].append(-1)
                        bDataLst["dist"].append(0)
                        bDataLst["posX"].append(-1)
                        bDataLst["posY"].append(-1)

                if proc2run == "spAHeatmap":
                    if len(tBin["m_pts"]) > 0:
                        ### add to heatmap array
                        for pt in tBin["m_pts"]:
                            hmArr[pt[1],pt[0]] += 1
                    # append all motion points
                    bDataLst.append(tBin["m_pts"])
                        
                elif proc2run.startswith("spAGrid"):
                    ##### [begin] spatial-analysis with grid on nest -----
                    ##### [DEBUG;begin] store processed image -----
                    flagImgSav4Debug = False 
                    def imgSav4Debug(img, spAGrid, fn):
                        ### draw grid
                        if len(img.shape) == 2: col = 127
                        else: col = (255,255,255)
                        thck = 1
                        x1, y1 = spAGrid["rect"][:2]
                        x2 = x1 + spAGrid["rect"][2] 
                        y2 = y1 + spAGrid["rect"][3]  
                        cv2.rectangle(img, (x1, y1), (x2, y2), col, thck)
                        for r in range(1, spAGrid["rows"]):
                            ry = y1 + r*spAGrid["cH"]
                            cv2.line(img, (x1, ry), (x2, ry), col, thck)
                        for c in range(1, spAGrid["cols"]):
                            cx = x1 + c*spAGrid["cW"]
                            cv2.line(img, (cx, y1), (cx, y2), col, thck)
                        # store the processed image of this bundle
                        cv2.imwrite(fn, img)
                    ##### [DEBUG;end] store processed image -----
                    
                    if len(tBin["m_pts"]) > 0: 
                        hmArr[:,:] = 0 # init the array
                        cellArea = spAGrid["cW"] * spAGrid["cH"]

                        if proc2run.startswith("spAGridHeat") or \
                          proc2run == "spAGridOFlow":
                        # spatial-analysis; heatmap in each cell of grid
                            ### add to heatmap array
                            for x, y in tBin["m_pts"]:
                                #hmArr[y,x] += 1
                                ### increase the motion point 
                                ###   & its surrounding pixels
                                _x1 = max(0, x-1)
                                _x2 = min(x+1, hmArr.shape[1]-1)
                                _y1 = max(0, y-1)
                                _y2 = min(y+1, hmArr.shape[0]-1)
                                hmArr[_y1:_y2,_x1:_x2] += 1

                        else:
                            img = hmArr # instead of heatmap array,
                              # will be used to leave only group activity chunks
                            for x, y in tBin["m_pts"]:
                                # draw motion point and surrounding pixels
                                _pt1 = (x-1, y-1)
                                _pt2 = (x+1, y+1)
                                cv2.rectangle(img, _pt1, _pt2, 255, -1)
                            if flagImgSav4Debug:
                                fn = "%s_d1.jpg"%(str(sDT).replace(" ","T"))
                                imgSav4Debug(img, spAGrid, fn) 

                        x1, y1 = spAGrid["rect"][:2]
                        x2 = x1 + spAGrid["rect"][2] 
                        y2 = y1 + spAGrid["rect"][3]

                        if proc2run == "spAGridDenseActivity":
                        # spatial-analysis; dense activity in each cell of grid
                            if spAGrid["MExIter"] > 0:
                                kernel = cv2.getStructuringElement(
                                                cv2.MORPH_RECT,(3,3)
                                                )
                                # close the motion point clouds;
                                #   forming chunk from close motions points
                                img = cv2.morphologyEx(
                                        img,
                                        cv2.MORPH_CLOSE,
                                        kernel,
                                        iterations=1
                                        )
                                if flagImgSav4Debug:
                                    fn = "%s_d2.jpg"%(str(sDT).replace(" ","T"))
                                    imgSav4Debug(img, spAGrid, fn)
                                # for removing minor motion point clouds
                                img = cv2.morphologyEx(
                                        img,
                                        cv2.MORPH_OPEN,
                                        kernel,
                                        iterations=spAGrid["MExIter"]
                                        )
                                if flagImgSav4Debug:
                                    fn = "%s_d3.jpg"%(str(sDT).replace(" ","T"))
                                    imgSav4Debug(img, spAGrid, fn)
                            
                            for r in range(spAGrid["rows"]):
                            # go through each row
                                for c in range(spAGrid["cols"]):
                                # and column to calculate in each cell
                                    _x1 = x1 + c*spAGrid["cW"]
                                    _x2 = _x1 + spAGrid["cW"]
                                    _y1 = y1 + r*spAGrid["cH"]
                                    _y2 = _y1 + spAGrid["cH"]
                                    # pixel count of white color in this cell
                                    wc = np.sum(img[_y1:_y2,_x1:_x2]/255)
                                    # fraction of white area in the cell
                                    frac = wc / cellArea 
                                    ### store the fraction 
                                    k = "[%i][%i]"%(r, c)
                                    bDataLst[k].append(frac)

                        elif proc2run.startswith("spAGridHeat"):
                        # spatial-analysis; heatmap in each cell of grid
                            for r in range(spAGrid["rows"]):
                            # go through each row
                                for c in range(spAGrid["cols"]):
                                # and column to calculate in each cell
                                    _x1 = x1 + c*spAGrid["cW"]
                                    _x2 = _x1 + spAGrid["cW"]
                                    _y1 = y1 + r*spAGrid["cH"]
                                    _y2 = _y1 + spAGrid["cH"]
                                    ### store the value in this cell 
                                    k = "[%i][%i]"%(r, c)
                                    if proc2run == "spAGridHeatMean":
                                        mv = np.mean(hmArr[_y1:_y2,_x1:_x2])
                                    elif proc2run == "spAGridHeatMax":
                                        mv = np.max(hmArr[_y1:_y2,_x1:_x2])
                                    if np.isnan(mv): mv = 0
                                    bDataLst[k].append(mv)
                            if flagImgSav4Debug:
                                img = hmArr.astype(np.float16)
                                img = (img/np.max(img)*255).astype(np.uint8)
                                fn = "%s.jpg"%(str(sDT).replace(" ","T"))
                                imgSav4Debug(img, spAGrid, fn)

                        elif proc2run == "spAGridOFlow":
                        # spatial-analysis; optical-flow in each cell of grid
                          
                            # modify hmArr; 
                            #   the max. value of the current bundle as 255
                            hmArr = hmArr / np.max(hmArr) * 255
                            img = hmArr.astype(np.uint8)
                            if flagImgSav4Debug:
                                fn = "%s_d1.jpg"%(str(sDT).replace(" ","T"))
                                imgSav4Debug(img, spAGrid, fn)

                            if pImg is None: # first data bundle
                                optFAng = np.zeros(hmArr.shape, np.uint8)
                                optFMag = optFAng 
                            else:
                                ### get optical flow info 
                                flow = cv2.calcOpticalFlowFarneback(
                                           pImg, img, None,
                                           0.5, 3, 15, 3, 5, 1.2, 0
                                           )
                                optFMag, optFAng = cv2.cartToPolar(flow[...,0], 
                                                                   flow[...,1])
                                # normalize magnitude to 0-255
                                optFMag = cv2.normalize(optFMag, None, 0, 255,
                                                        cv2.NORM_MINMAX)
                                optFMag[np.isnan(optFMag)] = 0
                                # cut-off low magnitude
                                #optFMag[optFMag<1] = 0
                                ret, optFMag = cv2.threshold(optFMag, 50, 255, 
                                                             cv2.THRESH_BINARY)
                                
                                # radian to range of 0-360
                                optFAng = optFAng / np.pi / 2 * 360

                                ### divide angels into low number of sections 
                                #nAngSec = 8 # number of sections in angle
                                #_deg = 360 / nAngSec
                                #optFAng = np.uint8(optFAng/_deg) * _deg
                                
                                # 0 is reserved for no optical-flow
                                optFAng[optFAng==0] = 1 

                                if flagImgSav4Debug:
                                    optFAng = optFAng/2 # max.value 360 to 180 
                                    ### draw the found optical flow
                                    _shape = (img.shape[0], img.shape[1], 3)
                                    optFImg = np.zeros(_shape, np.uint8)
                                    optFImg[...,0] = optFAng 
                                    optFImg[...,1] = 255
                                    optFImg[...,2] = optFMag  
                                    optFImg = cv2.cvtColor(optFImg,
                                                           cv2.COLOR_HSV2BGR)
                                    fn = "%s_d2.jpg"%(str(sDT).replace(" ","T"))
                                    imgSav4Debug(optFImg, spAGrid, fn)
                                    sleep(0.1)
                            pImg = img.copy()
                           
                            for r in range(spAGrid["rows"]):
                            # go through each row
                                for c in range(spAGrid["cols"]):
                                # and column to calculate in each cell
                                    _x1 = x1 + c*spAGrid["cW"]
                                    _x2 = _x1 + spAGrid["cW"]
                                    _y1 = y1 + r*spAGrid["cH"]
                                    _y2 = _y1 + spAGrid["cH"]
                                    ### store the median angle value 
                                    ###   in this cell 
                                    k = "[%i][%i]"%(r, c)
                                    _arrA = optFAng[_y1:_y2,_x1:_x2]
                                    _arrM = optFMag[_y1:_y2,_x1:_x2]
                                    _c255 = np.sum(_arrM) / 255
                                    if _c255 >= cellArea*0.1:
                                    # optical-flow in this cell is large enough 
                                        mv = np.median(_arrA[_arrM==255])
                                        # 0 is reserved for no optical-flow
                                        if mv == 0: mv += 1 
                                    else:
                                        mv = 0
                                    bDataLst[k].append(mv)
                   
                    else: # no motion is recorded in this bundle
                        for k in bDataLst.keys():
                            if proc2run == "spAGridOFlow":
                                bDataLst[k].append(0)
                            else:
                                bDataLst[k].append(0.0)
                    ##### [end] spatial-analysis with grid on nest ----- 

                ### store timestamp for this data 
                if proc2run == "spAHeatmap":
                    if len(tBin["ts"]) > 0:
                        _ts = tBin["ts"][0]
                        bD_dt.append(get_datetime(_ts))
                else:
                    bD_dt.append(sDT)
                    if days[-1].day != sDT.day:
                        days.append(datetime(year=sDT.year, 
                                             month=sDT.month, 
                                             day=sDT.day))
                ##### [end] store data & TS of this bundled data ---
                ### init temporary data
                if proc2run in ["intensity", "intensityPSD"]: 
                    tBin = dict(intensity=[])
                elif proc2run in ["intensityL", "dist2b", "dist2c", "dist2ca"]:
                    tBin = dict(mx=[], my=[]) 
                elif proc2run == "saMVec":
                    tBin["mx"] = []
                    tBin["my"] = []
                elif proc2run.startswith("spA"):
                    tBin = dict(m_pts=[], ts=[])
                elif proc2run == "motionSpread":
                    tBin = dict(spreadValue=[], pMPts=[], pMACents=[])
                elif proc2run == "proxMCluster":
                    tBin = dict(interactingMCluster=[])
                # update starting datetime
                sDT = sDT + dPtIntv
                # update starting index
                sDI = copy(di)
            ##### [end] process data of the past interval -----
            key = data[di][ci["Key"]].strip()
            value = data[di][ci["Value"]].strip()
            if key == "motionPts":
                m_pts_str = value.strip("()").split(")(")
                ### get motion points which are in one of ROIs
                m_pts = []
                for _pt in m_pts_str:
                    pt = [int(_x) for _x in _pt.split("/")]
                    for rk in roi.keys():
                        if not camIdx == self.camIdx4i[rk]: continue
                        _r = roi[rk]
                        if _r[0] <= pt[0] <= _r[0]+_r[2] and \
                          _r[1] <= pt[1] <= _r[1]+_r[3]:
                            m_pts.append(pt)
                            break
                if proc2run in ["intensity", "intensityPSD"]:
                    # store number of motions for this data bundle 
                    tBin["intensity"].append(len(m_pts))
                if proc2run.startswith("spA"):
                    ### store motion points
                    _ts = data[di][ci["Timestamp"]].strip()
                    tBin["m_pts"] += m_pts
                    if proc2run == "spAHeatmap" and len(m_pts) > 0:
                        tBin["ts"].append(_ts)
                if proc2run in ["intensityL", "dist2b", "dist2c", "dist2ca",
                                "saMVec"]:
                    if len(m_pts) > 0:
                        m_pts_arr = np.asarray(m_pts)
                        tBin["mx"] += list(m_pts_arr[:,0]) 
                        tBin["my"] += list(m_pts_arr[:,1])
                if proc2run == "motionSpread":
                    ##### [begin] process data-line for motionSpread -----
                    flagImgSav4Debug = False 
                    # (constant) maximum expected number of motion points 
                    #   of one clusterred group (of one ant) 
                    #   in a data-line (0.4-0.6s).
                    maxEInCluster = 4 
                    if flagImgSav4Debug:
                        _ts = data[di][ci["Timestamp"]].strip()
                        fn = "%s.jpg"%(_ts) # image filename to save
                    hmArr[:,:,:] = 0 # init image
                    # motion-points in previous data-line
                    pMPts = tBin["pMPts"]
                    # centroids of moved ants in the previous data-line
                    pMACents = tBin["pMACents"]
                    # list of moving ants' centroids in the current data-line
                    cMACents = [] 
                    if len(pMPts) > 0:
                    # there was some motion points 
                    #   in the previous data-line (0.4-0.6s)
                        # cluster motion points based on ant's body length
                        nG, g = clustering(m_pts, int(self.antLen*0.9), 
                                           criterion='distance') 
                        for gi in range(len(g)):
                        # go through each clusterred group
                            # get approximate number of ants in this 
                            #   motion-point group
                            nAntsInG = np.ceil(len(g[gi])/maxEInCluster)
                            nAntsInG = int(nAntsInG)
                            if nAntsInG > 1:
                            # it seems that more than one ant in this group
                                dPts = np.asarray(g[gi], dtype=np.float32)
                                # re-group this group with the estimated 
                                #   number of ants using k-means
                                centroids, __ = kmeans(obs=dPts,
                                                       k_or_guess=nAntsInG) 
                                idx, __ = vq(dPts, np.asarray(centroids))
                                for _ci in range(len(centroids)): 
                                    tPt = dPts[np.where(idx==_ci)]
                                    tPt = tPt.astype(np.uint16)
                                    # store the re-grouping results
                                    g.append(tPt.tolist()) 
                                    #color = 55 + _ci*int(200/len(centroid))
                                    #hmArr[tPt[:,1],tPt[:,0]] = color 
                                g[gi] = None # remove original group
                        # remove original group after re-grouping
                        while None in g: g.remove(None) 
                        
                        ### store the center of moving ant's motion-points
                        for gi in range(len(g)):
                            if len(g[gi]) == 1:
                                mx, my = g[gi][0]
                            elif len(g[gi]) > 1:
                                _a = np.asarray(g[gi])
                                mx = np.mean(_a[:,0])
                                my = np.mean(_a[:,1])
                            cMACents.append((int(mx), int(my)))

                        if flagImgSav4Debug:
                        # debugging 
                            rad = int(self.antLen/2) 
                            ### draw reference length of an ant
                            iH, iW = hmArr.shape[:2]
                            pt1 = (int(iW/2)-1, int(iH/2)-int(self.antLen/2))
                            pt2 = (int(iW/2)+1, int(iH/2)+int(self.antLen/2))
                            cv2.rectangle(hmArr, pt1, pt2, (0,0,200), -1)
                            ### draw clustering results
                            if nG > 0:
                                color = (200,200,200)
                                for pt in cMACents:
                                    cv2.circle(hmArr, pt, rad, color, 1)
                            ### draw motion points
                            color = (0,255,255)
                            for pt in m_pts:
                                #hmArr[pt[1],pt[0]] = 255
                                pt1 = (pt[0]-1, pt[1]-1)
                                pt2 = (pt[0]+1, pt[1]+1)
                                cv2.rectangle(hmArr, pt1, pt2, color, -1)

                    ### calcuate spread-value
                    spreadV = 0
                    for pcx, pcy in pMACents:
                    # go through centroids of moved ants in prev. data-line  
                        color = (randint(100,255), 
                                 randint(100,255), 
                                 randint(100,255))
                        for cci in range(len(cMACents)):
                        # centroids of moving ants in the current data-line 
                            ccx, ccy = cMACents[cci]
                            _cd = np.sqrt((pcx-ccx)**2 + (pcy-ccy)**2)
                            if self.antLen*0.5 < _cd <= self.antLen*1.5:
                            # if two centroids are in neighboring distance 
                                spreadV += 1 # increase spread value
                                cMACents[cci] = None # no further counting 
                                                     #   of this centroid
                                if flagImgSav4Debug:
                                # debugging
                                    pt1 = (pcx, pcy)
                                    pt2 = (ccx, ccy)
                                    cv2.line(hmArr, pt1, pt2, color, 1)
                                    cv2.circle(hmArr, pt1, 5, color, -1)
                                    cv2.circle(hmArr, pt2, rad, color, 2)
                        while None in cMACents: cMACents.remove(None)

                    if flagImgSav4Debug:
                    # debugging
                        print("[img saving] ", fn)
                        cv2.imwrite(fn, hmArr) # save image
                   
                    # append spread-value
                    tBin["spreadValue"].append(spreadV)
                    # store moved ants' centroids 
                    tBin["pMACents"] = cMACents
                    # store motion points
                    tBin["pMPts"] = m_pts
                    ##### [end] process data-line for motionSpread -----

                if proc2run == "proxMCluster":
                    if di == gSIdx:
                    # the 1st data to process
                        ### init 
                        # save images for debugging purpose
                        flagImgSav4Debug = False 
                        # (constant) maximum expected number of motion points 
                        #   of one clusterred group (of one ant) 
                        #   in a data-line (approx. one second). 
                        maxEInCluster = 5 
                        # temporary for drawing motion clusters
                        tmpGrey = np.zeros(tuple(hmArr.shape[:2]), 
                                           dtype=np.uint8) 
                        th4c = self.antLen # distance for clustering
                        th4i = 3 # threshold value to recognize as the cluster 
                        inc = 2 # increase of pixel value per each motion
                        dec = 1 # decrease of all pixel values per each frame
                    
                    tBin["interactingMCluster"].append(0) # init value
                    # cluster motion points based on ant's body length
                    nG, g = clustering(m_pts, th4c, criterion='distance') 
                    tmpGrey[:,:] = 0
                    for gi in range(len(g)):
                    # go through each clusterred group
                        # get approximate number of ants in this 
                        #   motion-point group
                        nAntsInG = np.ceil(len(g[gi])/maxEInCluster)
                        if nAntsInG > 1: 
                        # more than one ant in this motion cluster 
                            _a = np.asarray(g[gi])
                            mx = int(np.mean(_a[:,0]))
                            my = int(np.mean(_a[:,1]))
                            cv2.circle(tmpGrey, (mx, my), th4c, inc, -1)
                    hmArr += tmpGrey # increase
                    hmArr -= dec # decrease 
                    hmArr[hmArr<0] = 0
                    hmArr[hmArr>th4i] = th4i 
                    _val = np.sum(hmArr==th4i) # count pixels, recognized
                                               # as prox. motion cluster 
                    # store the interacting motion cluster value
                    tBin["interactingMCluster"][-1] += _val
                    
                    if flagImgSav4Debug:
                    # debugging
                        _ts = data[di][ci["Timestamp"]].strip()
                        fn = "%s.jpg"%(_ts) # image filename to save
                        debugImg = hmArr.copy().astype(np.uint8)
                        debugImg *= int(255/th4i)
                        _idx = np.where(debugImg==255)
                        debugImg = cv2.cvtColor(debugImg, cv2.COLOR_GRAY2BGR)
                        debugImg[_idx] = (0,0,255)
                        ### draw motion points
                        color = (0,255,255)
                        for pt in m_pts:
                            pt1 = (pt[0]-1, pt[1]-1)
                            pt2 = (pt[0]+1, pt[1]+1)
                            cv2.rectangle(debugImg, pt1, pt2, color, -1)
                        print("[img saving] ", fn)
                        cv2.imwrite(fn, debugImg) # save image

            '''
            elif key == "colCent":
                if "distCent" in proc2run:
                    ### store ant color centroid
                    pt = [int(_x) for _x in value.split("/")]
                    tBin["colCentX"].append(pt[0])
                    tBin["colCentY"].append(pt[1])
            '''
            
            if di == len(data)-1: # data reached the end of data
                gEIdx = -1 # notify it reached the end
                gEDT = dt
            else:    
                if proc2run == "spAHeatmap":
                    if hmIntvMin != -1 and dt-_dayB > hmIntv:
                    # elapsed time is over the heatmap interval
                        gEIdx = sDI-1 # store the end index
                        gEDT = sDT # store the end datetime of this graph 
                        break
      
        if flagZeroPadding:
            ### fill data until end of the end date 
            zPadIdx[1] = len(bDataLst) # store index where zero padding 
                                       # ended at the end 
            _dateEnd = datetime(year=dt.year, month=dt.month, day=dt.day)
            _dateEnd = _dateEnd + timedelta(days=1) - timedelta(seconds=1)
            while dt < _dateEnd:                   
                dt += timedelta(seconds=dPtIntvSec)
                bDataLst, bD_dt = self.fillEmptyData(
                                        proc2run, bDataLst, bD_dt, dt
                                        ) 
        return (bDataLst, bD_dt, gSDT, gEIdx, gEDT, zPadIdx, days, hmArr)

    #---------------------------------------------------------------------------

    def drawBarGraph(self, args):
        """ draw bar graph such as 'intensity'
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        proc2run, bDataLst, dPtIntvSec, graphW, graphH, gBg, cvFont, \
          bD_dt, tempMeasureIntv, main, flagZeroPadding, q2m, days, \
          gSDT, gEDT, zPadIdx = args 

        # 'origDLst' is for storing original data in 'intensityL'; 
        #   data will be normalized in 'intensityL'.
        origDLst = {} 
        dLst = {} # temporary data to draw graph  
        if proc2run in self.multipleMeasures: # multiple measurements
            for k in bDataLst.keys():
                if sum(bDataLst[k]) > 0:
                    if proc2run == "intensityL":
                        origDLst[k] = bDataLst[k]
                        ### normalize measure to 0.0-1.0
                        d = np.asarray(bDataLst[k])
                        dLst[k] = list(d/np.max(d))
                    else:
                        dLst[k] = bDataLst[k]
        else:
            dLst[proc2run] = bDataLst
        
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
        cols = int((24 * 60 * 60) / dPtIntvSec) # number of columns for a day
        if cols < graphW: p2col = int(graphW / cols) # pixel-to-column 
        else: p2col = 1
        if proc2run in self.multipleMeasures: # multiple measurements
            k = list(dLst.keys())[0]
            _cols = len(dLst[k])
            maxVal = 0 # max. value of entire graph
            for k in dLst.keys():
                # get smoothed data
                sdLst[k] = savgol_filter(dLst[k], 
                                         window_length=sWinLen, 
                                         polyorder=sPolyOrder)
                ### get max. value
                if proc2run == "intensityL":
                    _maxVal = 1.0
                else:
                    _maxVal = max(sdLst[k])
                if maxVal < _maxVal: maxVal = copy(_maxVal)
        else:
            _cols = len(bDataLst) 
            # get smoothed data
            sdLst[proc2run] = savgol_filter(dLst[proc2run], 
                                            window_length=sWinLen, 
                                            polyorder=sPolyOrder)
            maxVal = max(bDataLst)
        _rows = copy(graphH) 
        yMul = _rows / maxVal 
        _img = np.zeros((_rows, _cols*p2col, 3), dtype=np.uint8)
        cv2.rectangle(_img, (0,0), (_img.shape[1], _img.shape[0]), 
                      gBg, -1) # bg-color 
        
        ### get font for the graph
        fThck = 2
        _thP = max(int(_rows*0.07), 15)
        fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                 thresholdPixels=_thP,
                                                 thick=fThck)
       
        ### mark day time
        """
        x1 = -1; x2 = -1
        dayCol = (100, 50, 50)
        for x in range(_cols):
            h = bD_dt[x].hour
            if x1 == -1 and h == 6: x1 = copy(x)
            if x1 != -1 and x2 == -1 and h == 18: x2 = copy(x)
            if x1 != -1 and x2 != -1:
                _x1 = x1 * p2col
                _x2 = x2 * p2col
                cv2.rectangle(_img, (_x1,0), (_x2,_rows), dayCol, -1)
                x1 = -1; x2 = -1
        if x1 != -1 and x2 == -1:
            _x = x * p2col
            _x1 = x1 * p2col
            cv2.rectangle(_img, (_x1,0), (_x,_rows), dayCol, -1)
        """
        
        ### mark temperature
        tempLbl = ""
        """
        t = self.temperature["t"]
        tLen = len(t)
        tdt = self.temperature["dt"]
        tlv = self.temperature["mi"]
        tRng = self.temperature["ma"] - tlv
        tempLbl = "Temp.: min. %.1f C"%(tlv)
        tempLbl += ", max. %.1f C"%(self.temperature["ma"])
        if not proc2run in self.multipleMeasures: # not multiple measurements
            ### draw temperature line
            ti = 0 # index for temperature data
            tPrevPt = (-1, -1)
            for x in range(_cols):
                if len(tdt) == 0: break
                dt = bD_dt[x]
                if dt >= tdt[ti]:
                    while (dt-tdt[ti]).total_seconds() > tempMeasureIntv:
                        ti += 1
                        if ti == tLen: break
                    if ti == tLen: break
                    y = _rows - int((t[ti]-tlv)/tRng*_rows)
                    _x = x * p2col
                    if ti > 0:
                        cv2.line(_img, tPrevPt, (_x, y), (0,0,175), 2)
                    tPrevPt = (_x, y)
        """
        
        peaks = {}
        if proc2run == "intensity":
            d4rDE = {} # for storing data for output numpy array raw data
            d4rDE["peakIntvSec"] = {} # interval between peaks
            d4rDE["peakIntvSecDt"] = {} # date time when peak interval occurs
            d4rDE["mBoutSec"] = {} # lengths (seconds) of a motion bout
            d4rDE["mBoutSecDt"] = {} # when the motion bout starts
            d4rDE["inactivitySec"] = {} # inactivity duration (seconds)
                                        #   between two continuous motion bouts.
            d4rDE["inactivitySecDt"] = {} # when the inactivity duration starts
        ### whether to draw outlier or not
        chk = wx.FindWindowByName("outlier_chk", main.panel["ml"])
        flagOutlier = chk.GetValue() 
        ### whether to draw smooth data-line or not
        if proc2run in self.multipleMeasures:
            flagSmoothLine = True
        else:
            chk = wx.FindWindowByName("smoothLine_chk", main.panel["ml"])
            flagSmoothLine = chk.GetValue() 
        ### whether to draw peak points 
        chk = wx.FindWindowByName("peak_chk", main.panel["ml"])
        flagDrawPeaks = chk.GetValue() 
        olIdx = [] # outlier indices
        for mi, mk in enumerate(dLst.keys()):
        # go through data (for when multiple measures are in the data) 
            _d = dLst[mk]
            zStartedIdx = -1 # index where zero value started
            nzStartedIdx = -1 # index where non-zero value started
            
            if proc2run == "intensity":
                for k in d4rDE.keys(): # for each data keys for raw-data-output
                    d4rDE[k][mk] = [] # make a list for this measure

            if proc2run == "intensity" and flagOutlier:
                ##### [begin] outlier detection ----- 
                _tmp = copy(_d) # temporary data for outlier detection
                if flagZeroPadding: # data was zero-padded on both ends.
                    # cut off the padded data
                    _tmp = _tmp[zPadIdx[0]:zPadIdx[1]]
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
                for _i in range(len(olIdx)): olIdx[_i] += zPadIdx[0] 
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
                        olIdx.append(pi+zPadIdx[0])
                ##### [end] outlier detection ----- 
            
            ##### [begin] draw data lines ----- 
            for x, val in enumerate(_d):
                if x%100 == 0:
                    msg = "drawing data-point %i/ %i"%(x, _cols)
                    q2m.put(("displayMsg", msg,), True, None)
               
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
                            _zET = bD_dt[x] 
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
                            _zET = bD_dt[x]
                            _et = (_zET-_zST).total_seconds()
                            d4rDE["inactivitySec"][mk].append(_et)
                            d4rDE["inactivitySecDt"][mk].append(_zST)
                            zStartedIdx = -1
                    ##### [end] store motion bout data -----

                y = _rows - int(val * yMul)
                sy = _rows - int(sdLst[mk][x] * yMul) # y of smoothed data
                _x1 = x * p2col
                _x2 = _x1 + p2col
                
                if not proc2run in self.multipleMeasures: # single measurement
                    ### draw data line
                    _col = (150, 150, 150)
                    if len(olIdx) > 0: # there's outlier
                        if x == olIdx[0]: # outlier index matches
                            # different color for outlier data point
                            _col = (0,0,255)
                            olIdx.pop(0)
                    if p2col == 1:
                        cv2.line(_img, (_x1, y), (_x1, _rows), _col, 1)  
                    else:
                        cv2.rectangle(_img, (_x1, y), (_x2, _rows), 
                                      _col, -1)

                if flagSmoothLine and x > 0:
                    if proc2run == "intensityL": _col = self.locC[mk]
                    else: _col = self.gC[mi]
                    if proc2run in self.multipleMeasures:
                        # draw data line
                        #cv2.line(_img, (px, py), (_x1, y), _col, 1)
                        # draw smoothed line
                        cv2.line(_img, (px, psy), (_x1, sy), _col, 1)
                    else:
                        # draw smoothed line
                        cv2.line(_img, (px, psy), (_x1, sy), _col, 1)

                px = copy(_x1)
                py = copy(y)
                psy = copy(sy)
            ##### [end] draw data lines -----

            syArr = np.asarray(sdLst[mk])
            # determine value for prominence
            mm = np.std(syArr[syArr>0])
            #cv2.line(_img, (0, _rows-mm), (_cols-1, _rows-mm), (0,0,0), 1)
            ### draw peak points
            peaks[mk], _ = find_peaks(sdLst[mk], prominence=mm) 
            prevPeakTime = None
            pmLen = max(1, int(min(_rows,_cols)*0.03)) # peak marker len
            pmThck = max(1, int(pmLen*0.2)) # peak marker line thickness
            for peak in peaks[mk]:
                y = _rows - int(sdLst[mk][peak] * yMul)
                _peak = peak * p2col
                pts = np.array([[_peak, y], 
                                [_peak-int(pmLen/2), y-pmLen],
                                [_peak+int(pmLen/2), y-pmLen]], np.int32)
                pts = pts.reshape((-1,1,2))
                if flagDrawPeaks:
                    #cv2.polylines(_img, [pts], True, self.gC[mi], pmThck)
                    cv2.fillPoly(_img, [pts], self.gC[mi])
                ###
                currDT = bD_dt[peak]
                if proc2run == "intensity" and prevPeakTime is not None:
                    d4rDE["peakIntvSecDt"][mk].append(bD_dt[peak])
                    _et = (bD_dt[peak]-prevPeakTime).total_seconds()
                    d4rDE["peakIntvSec"][mk].append(_et)
                prevPeakTime = currDT  
        
        ### reshape the result graph image (cut by each day)
        ###   & write some info on data of the day
        cols = int((24 * 60 * 60) / dPtIntvSec)
        rows = _rows * len(days)
        img = np.zeros((rows, cols*p2col, 3), dtype=np.uint8)
        if proc2run in self.multipleMeasures: # multiple measurements
            pi = {}
            for mk in dLst.keys(): pi[mk] = 0 # peak index for measure key
        else:
            pi = 0 # peak index
        # dict for saving data as file; 
        #   currently saving only for "intensity" process
        inten24hD = {"datetime":[], "max":[], "sum":[], "peak":[], 
                     "temperature":[]}
        tempIdx = 0 # index for temperature data
        for _di in range(len(days)):
            x1 = _di * cols
            x2 = min(_img.shape[1], (_di+1) * cols) 
            y1 = _di * _rows
            y2 = (_di+1) * _rows
            _x1 = x1 * p2col
            _x2 = x2 * p2col
            img[y1:y2,0:_x2-_x1] = _img[:,_x1:_x2] # crop for the day row
            ### write some info of the day
            for mi, mk in enumerate(dLst.keys()):
            # go through data (for when multiple measures are in the data) 
                if proc2run == "intensityL":
                    fCol = self.locC[mk]
                    _d = np.asarray(origDLst[mk][x1:x2])
                else:
                    fCol = tuple([max(0, c-50) for c in self.gC[mi]])
                    _d = np.asarray(dLst[mk][x1:x2])
                _mv = int(np.max(_d))
                _txt = "[%s] max: %i"%(mk[:3], _mv)
                if mk in ["intensity"]:
                    _sum = np.sum(_d)
                    _txt += ", sum: %.1fk"%(_sum/1000)
                elif mk == "reach":
                    _d = _d[_d>0] # drop non-zero values
                    _med = np.median(_d)
                    _txt += ", med: %.1f"%(_med)
                if flagDrawPeaks:
                    peakCnt = 0
                    tmp = []
                    if proc2run in self.multipleMeasures: pRng0 = pi[mk]
                    else: pRng0 = pi
                    for _pi in range(pRng0, len(peaks[mk])):
                        tmp.append(peaks[mk][_pi])
                        if x1 <= peaks[mk][_pi] < x2: peakCnt += 1
                        else: break
                    _txt += ", peaks: %i"%(peakCnt)
                    if flagDrawPeaks:
                        if proc2run in self.multipleMeasures: pi[mk] = copy(_pi)
                        else: pi = copy(_pi)
                tx = txtW * 2 
                ty = int(y1 + txtH * 2 * (2+mi))
                cv2.putText(img, _txt, (tx, ty), cvFont, fontScale=fScale,
                            color=fCol, thickness=fThck)
                if proc2run in ["intensity"]:
                    flagDiscard = False
                    if _di == 0 and \
                      (gSDT-bD_dt[x1]) >= timedelta(hours=1):
                        flagDiscard = True
                    if _di == len(days)-1 and \
                      (bD_dt[x2-1]-gEDT) >= timedelta(hours=1):
                        flagDiscard = True
                    if not flagDiscard:
                        t = self.temperature["t"]
                        tdt = self.temperature["dt"]
                        tmpTemp = []
                        for ti in range(tempIdx, len(t)):
                            _tdt = datetime(year=tdt[ti].year,
                                            month=tdt[ti].month,
                                            day=tdt[ti].day)
                            if _tdt > days[_di]: break
                            tmpTemp.append(t[ti])
                        if tmpTemp == []: # if temperature data is empty
                            if len(inten24hD["temperature"]) > 0:
                                # copy previous day's avg. temperature
                                tmpTemp = [inten24hD["temperature"][-1]]
                        tempIdx = ti
                        ### store data
                        inten24hD["datetime"].append(str(days[_di]))
                        inten24hD["max"].append(_mv)
                        inten24hD["sum"].append(_sum)
                        if flagDrawPeaks: inten24hD["peak"].append(peakCnt)
                        else: inten24hD["peak"].append(-1)
                        inten24hD["temperature"].append(np.mean(tmpTemp)) 

        ### mark center line
        mx = int(cols*p2col/2)
        cv2.line(img, (mx, 0), (mx, rows), (200,200,200), 1)
       
        '''
        ### mark hour line
        _gap = cols*p2col/24
        for _i in range(24):
            x = int(_i * _gap)
            cv2.line(img, (x, 0), (x, rows), (127,127,127), 1)
        '''

        ### write some graph info 
        _txt = "%s/ intv:%i s"%(proc2run, dPtIntvSec)
        if tempLbl != "": _txt += "/ %s"%(tempLbl)
        fCol = (255, 255, 255)
        cv2.putText(img, _txt, (5, txtH+10), cvFont,
                    fontScale=fScale, color=fCol, thickness=fThck)

        ##### [begin] prepare raw data -----

        ### prepare intensity and its smoothed data
        maxVal = []
        if type(bDataLst) == list:
            ks = [proc2run]
            maxVal.append(np.max(bDataLst))
            dLen = len(bDataLst)
        elif type(bDataLst) == dict: # might have multiple data lists
        # In this case, dictionary-key will be the data column
            for k in dLst.keys(): maxVal.append(np.max(dLst[mk]))
            dLen = len(dLst[k])
        ### set data type depending on the max value of each list
        bdType = []
        for mv in maxVal:
            if proc2run in ["saMVec", "dist2b", "dist2c", "dist2ca"]:
                bdType.append(getNumpyDataType(mv, flagIntSign=True)) 
            else:
                bdType.append(getNumpyDataType(mv)) 
        _dtype = [("datetime", "U26")]
        for mi, mk in enumerate(dLst.keys()): _dtype.append((mk, bdType[mi]))
        rD = np.zeros(dLen, dtype=_dtype) # initiate array
        rD1 = np.zeros(dLen, dtype=_dtype) # initiate array
        rD["datetime"] = np.array([str(x) for x in bD_dt])
        rD1["datetime"] = rD["datetime"] 
        for mi, mk in enumerate(dLst.keys()):
            rD[mk] = np.array(dLst[mk], dtype=bdType[mi])
            rD1[mk] = np.array(sdLst[mk], dtype=bdType[mi])
        rawData = dict(i=rD, iSmooth=rD1)
        
        if proc2run == "intensity":
        ### prepare other raw data for "intensity" process.
            _dtype = [("datetime", "U26"), ("max", np.int32), 
                      ("sum", np.int32), ("peak", np.int32),
                      ("temperature", np.float16)]
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
                    
                    if flagZeroPadding and dk == "inactivitySec":
                    # if it's zero-padded and inactivity data
                        _data = _data[1:] # remove the first data
                        _dtData = _dtData[1:]

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

        proc2run, main, bDataLst, dPtIntvSec, graphW, graphH, gBg, cvFont, \
          bD_dt, fsPeriod = args

        key = proc2run.rstrip("PSD")
        cho = wx.FindWindowByName("psdLen_cho", main.panel["ml"])
        # length of data for PSD
        psdLen = cho.GetString(cho.GetSelection())
        cho = wx.FindWindowByName("psdNPerSeg_cho", main.panel["ml"])
        # n per segment for PSD as ratio to the leng of data
        psdNPerSeg = int(cho.GetString(cho.GetSelection())) 

        ### get beginning & end data index for each bundle (such as 1 day)
        if psdLen == "entire input data":
            psdDI = [[0, len(bDataLst)-1]]
        else:
            psdDI = [[0, -1]]
            if psdLen.endswith(" d"):
                thr = timedelta(days=int(psdLen.replace(" d","")))
            elif psdLen.endswith(" h"):
                thr = timedelta(hours=int(psdLen.replace(" h","")))
            _bDT = None 
            for i, dt in enumerate(bD_dt):
                if i == 0:
                    _bDT = copy(dt) # store beginning datetime
                if (dt-_bDT) > thr: # time for a PSD passed
                    psdDI[-1][1] = i-1 # store end index for PSD
                    psdDI.append([i, -1]) # store beginning index
                    _bDT = copy(dt) # new beginning datetime
            if psdDI[-1][1] == -1: psdDI.pop(-1)
        if len(psdDI) == 0:
            print("!!! ERROR:: No data indices are found for PSD")

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
            psdData = np.asarray(bDataLst[idx0:idx1+1])
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

        return img, fsPeriod, mg

    #---------------------------------------------------------------------------

    def drawHeatmap(self, args):
        """ draw heatmap graph 
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        hmArr, main, prnt, bDataLst, bD_dt, cvFont, hmRad, camIdx, \
          gSDT, gEDT, q2m, fImg = args

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
        hmMaxVal = np.max(hmArr) 
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
        timeLbl = "%s_%s"%(_lbl1, _lbl2)
        msg = "generating heatmap image.. [%s]"%(timeLbl)
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
                                  hmCols, timeLbl, hmRad) 

        chk = wx.FindWindowByName("saveHMVideo_chk", main.panel["ml"])
        if chk.GetValue():
            q2m.put(("displayMsg", "making heatmap video..",), True, None)
            # make heatmap video file
            self.makeHeatmapVideo((main.inputFP, 
                            copy(self.imgFiles[camIdx]), bDataLst, bD_dt,
                            cvFont, fThck, fScale, txtW, txtH, txtBl))

        rawData = dict(heatMapArr=hmArr)

        return img, rawData 

    #---------------------------------------------------------------------------

    def drawGridMeasure(self, args):
        """ draw measure of each cell in the grid
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        proc2run, bDataLst, dPtIntvSec, graphH, gBg, cvFont, \
          bD_dt, main, spAGrid, q2m = args
        flagDrawPeaks = False 

        origDLst = {}
        dLst = bDataLst

        ### get max. value in each cell
        maxValLst = []
        for k in dLst.keys(): maxValLst.append(np.max(dLst[k]))

        if not proc2run == "spAGridDenseActivity":
            ### get max. value in all grid cells
            if proc2run.startswith("spAGridHeat"): 
                mvInGrid = float(np.max(maxValLst)) 
            elif proc2run == "spAGridOFlow":
                mvInGrid = 360
            ### normalize each cell data to 0.0-1.0
            for k in dLst.keys():
                origDLst[k] = bDataLst[k] # store the original data
                d = np.asarray(bDataLst[k])
                dLst[k] = list(d/mvInGrid)
       
        '''
        sdLst = {} # smoothed data
        sPolyOrder = 3
        if dPtIntvSec <= 60:
            sWinLen = 13 
        elif dPtIntvSec == 300:
            sWinLen = 5 
        elif dPtIntvSec >= 600:
            sWinLen = 3 
            sPolyOrder = 2 
        ### get smoothed data
        for k in dLst.keys():
            sdLst[k] = savgol_filter(dLst[k], 
                                     window_length=sWinLen, 
                                     polyorder=sPolyOrder)
        '''

        maxVal = 1.0 # max. value of the graph
        k = list(dLst.keys())[0]
        cols = len(dLst[k])
        p2col = 1 # number of pixels to draw each data column
        rows = max(graphH, 1000)
        nCells = len(dLst) # number of cells
        cellGH = int(np.round(rows / nCells)) # height of each cell graph
        yMul = rows / maxVal / nCells
        bottomMargin = spAGrid["rows"]+1
        # set up image array to return
        img = np.zeros((rows+bottomMargin, cols*p2col, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), 
                      gBg, -1) # bg-color 
        
        ### get font for the graph
        fThck = 2
        _thP = max(int(rows*0.01), 15)
        fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                 thresholdPixels=_thP,
                                                 thick=fThck)
       
        peaks = {}
        peakIntvMin = {} # interval (minutes) between peaks
        # color for marking (graph-info, peaks, etc)
        iCol = (255, 255, 255)
        ### whether to draw outlier or not
        chk = wx.FindWindowByName("outlier_chk", main.panel["ml"])
        flagOutlier = chk.GetValue() 
        olIdx = [] # outlier indices
        for i, k in enumerate(dLst.keys()):
            _d = dLst[k]
            rowIdx, colIdx = k.split("][")
            rowIdx = int(rowIdx[1:])
            colIdx = int(colIdx[:-1])
            yBase = (rowIdx*spAGrid["cols"]+colIdx+1) * cellGH

            if flagOutlier:
                ##### [begin] outlier detection ----- 
                _tmp = copy(_d) # temporary data for outlier detection
                #print(_tmp)
                #np.save("tmp.npy", np.asarray(_tmp))
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
            for x, val in enumerate(_d):
                if x%100 == 0:
                    msg = "drawing data-point %i/ %i"%(x, cols)
                    q2m.put(("displayMsg", msg,), True, None)
                y = yBase - int(val * yMul)
                #sy = yBase - int(sdLst[k][x] * yMul) # y of smoothed data
                _x1 = x * p2col
                _x2 = _x1 + p2col
                
                ### draw data line
                _col = (200, 200, 200)
                if len(olIdx) > 0: # there's outlier
                    if x == olIdx[0]: # outlier index matches
                        # different color for outlier data point
                        _col = (0,0,255)
                        olIdx.pop(0)
                if proc2run == "spAGridOFlow" and val > 0:
                # the value indicates an optical-flow
                    _col = (50, 255, 50)
                if p2col == 1:
                    cv2.line(img, (_x1, y), (_x1, yBase), _col, 1)  
                else:
                    cv2.rectangle(img, (_x1, y), (_x2, yBase), _col, -1)
                
                '''
                if x > 0:
                    # draw smoothed line
                    cv2.line(img, (px, psy), (_x1, sy), iCol, 1)
                '''

                px = copy(_x1)
                py = copy(y)
                #psy = copy(sy)
            ##### [end] draw data lines -----

            '''
            if flagDrawPeaks:
                syArr = np.asarray(sdLst[k])
                # determine value for prominence
                mm = np.std(syArr[syArr>0])
                ### draw peak points
                peaks[k], _ = find_peaks(sdLst[k], prominence=mm)
                peakIntvMin[k] = {} 
                prevPeakTime = None
                pmLen = max(1, int(min(rows,cols)*0.03)) # peak marker len
                pmThck = max(1, int(pmLen*0.2)) # peak marker line thickness
                for peak in peaks[k]:
                    y = yBase - int(sdLst[k][peak] * yMul)
                    _peak = peak * p2col
                    pts = np.array([[_peak, y], 
                                    [_peak-int(pmLen/2), y-pmLen],
                                    [_peak+int(pmLen/2), y-pmLen]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    #cv2.polylines(img, [pts], True, iCol, pmThck)
                    cv2.fillPoly(img, [pts], iCol)
                    ###
                    currDT = bD_dt[peak]
                    if prevPeakTime is not None:
                        if peakIntvMin[k] == {} or \
                         currDT.day != prevPeakTime.day:
                            ### new list for a day
                            pimK = "%i-%i"%(currDT.month, currDT.day)
                            peakIntvMin[k][pimK] = []
                        elapsedTime = (bD_dt[peak]-prevPeakTime).total_seconds()
                        peakIntvMin[k][pimK].append(elapsedTime/60)
                    prevPeakTime = currDT 
                print("Intervals of %s peaks: %s"%(k, str(peakIntvMin[k])))
            '''
            
        ### write some graph info 
        _txt = "%s/ intv:%i s"%(proc2run, dPtIntvSec)
        if proc2run.startswith("spAGridHeat"):
            if type(mvInGrid) == int: _txt += "/ maxValue:%i"%(mvInGrid)
            elif type(mvInGrid) == float: _txt += "/ maxValue:%.3f"%(mvInGrid)
        fCol = (255, 255, 255)
        cv2.putText(img, _txt, (5, txtH+10), cvFont,
                    fontScale=fScale, color=fCol, thickness=fThck)
 
        ### draw cell-position-dot; 
        ###   for showing which grid-cell each row represents
        cpdImg = np.zeros((rows, cellGH, 3), np.uint8) # init image
        dotDiameter = int(cellGH / (spAGrid["rows"]*2))
        dotRad = int(dotDiameter / 2)
        color = (255, 255, 255)
        for cIdx in range(nCells):
            yBase = cIdx * cellGH
            cr = int(cIdx / spAGrid["cols"])
            cc = cIdx % spAGrid["cols"]
            if cIdx > 0:
                # draw base line
                cv2.line(cpdImg, (0, yBase), (cellGH, yBase), color, 1)
            for r in range(spAGrid["rows"]):
                for c in range(spAGrid["cols"]):
                    x = c*dotDiameter*2 + dotDiameter
                    y = yBase + r*dotDiameter*2 + dotDiameter
                    if r == cr and c == cc: # dot for the current row
                        cv2.circle(cpdImg, (x,y), dotRad+1, color, -1)
                    else: # other dots
                        cv2.circle(cpdImg, (x,y), dotRad, color, 1)

        ### prepare raw data
        bdType = []
        for mv in maxValLst:
        # set data type depending on the max value of each cell 
            if "." in str(mv): # float values
                if mv < np.finfo(np.float16).max: bdType.append(np.float16)
                else: bdType.append(np.float32)
            else: # integer values
                if mv < np.iinfo(np.uint8).max: bdType.append(np.uint8)
                elif mv < np.iinfo(np.uint16).max: bdType.append(np.uint16)
                else: bdType.append(np.uint32)
        _dtype = [("datetime", "U26")]
        for i, k in enumerate(dLst.keys()): _dtype.append((k, bdType[i]))
        rD = np.zeros(cols, dtype=_dtype) # initiate array
        #rD1 = np.zeros(cols, dtype=_dtype) # initiate array
        rD["datetime"] = np.array([str(x) for x in bD_dt])
        #rD1["datetime"] = rD["datetime"].copy()
        for k in dLst.keys():
            rD[k] = np.array(dLst[k], dtype=bdType[i])
            #rD1[k] = np.array(sdLst[k], dtype=bdType[i])

        rawData = dict(dLst=dLst)

        return img, cpdImg, rawData 

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

        inputFP, tmpILst, bDataLst, bD_dt, \
          cvFont, fThck, fScale, txtW, txtH, txtBl = args

        fn = "heatmap_%s.avi"%(get_time_stamp())
        _fp0, _fp1 = path.split(inputFP)
        if _fp1 == "": _fp0 = path.split(_fp0)[0]
        fPath = path.join(_fp0, fn)
        if path.isfile(fPath): remove(fPath)
        print(f"\n{fPath}\n")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #rad = int(round(img.shape[0] * 0.002))
        #col = (255,255,0)
        videoFPS = 30
        '''
        ### cut a part of data to make the video
        _i0 = 25000; _i1 = 25300
        bDataLst = bDataLst[_i0:_i1] 
        bD_dt = bD_dt[_i0:_i1] 
        '''
        
        fImg = cv2.imread(tmpILst[0])
        fSh = fImg.shape
        snapshot = np.zeros(fSh, dtype=np.uint8)
        grey = np.zeros(tuple(fSh[:2]), dtype=np.int16) 
        if flag == "gradual":
            frame = np.zeros((fSh[0], int(fSh[1]*2), fSh[2]), np.uint8)
            inc = 85 # increase of pixel value per each motion
            dec = 5 # 1/5/17 decrease of all pixel values per each frame
            mPtRad = 2 # when '0', it means single pixel value change
        elif flag == "flash":
            frame = np.zeros((fSh[0], int(fSh[1]*2), fSh[2]), np.uint8)
            mPtRad = 0
        elif flag == "ani":
            frame = np.zeros((fSh[0], fSh[1], 4), np.uint8)
            inc = 85
            dec = 5
            mPtRad = 5
            _pBD = None
        # init video writer
        video_rec = cv2.VideoWriter(fPath, fourcc=fourcc, fps=videoFPS, 
                            frameSize=(frame.shape[1], frame.shape[0]), 
                            isColor=True)
        
        _ts = path.basename(tmpILst[0]).split("_cam-")[0]
        nextSnapshotTS = get_datetime(_ts)
        #ssROI = [550, 0, 900, 1440]
        if flag in ["gradual", "ani"] and mPtRad > 0:
            tmpGrey = np.zeros(tuple(fSh[:2]), dtype=np.uint8)
        for i, _bD in enumerate(bDataLst):
            if flag != "ani" and i%10 == 0:
                msg = "writing video %i/ %i"%(i+1, len(bDataLst))
                print("\r", msg, end="          ", flush=True)

            ### get snapshot image
            if nextSnapshotTS != None and bD_dt[i] >= nextSnapshotTS:
                snapshot = cv2.imread(tmpILst[0]) # load snapshot image
                #snapshot = snapshot[ssROI[1]:ssROI[1]+ssROI[3], 
                #                    ssROI[0]:ssROI[0]+ssROI[2]]
                tmpILst.pop(0)
                if len(tmpILst) > 0:
                    _ts = path.basename(tmpILst[0]).split("_cam-")[0]
                    nextSnapshotTS = get_datetime(_ts)
                else:
                    nextSnapshotTS = None

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
                frame[0:fSh[0],0:fSh[1]] = cv2.cvtColor(
                                                grey.astype(np.uint8), 
                                                cv2.COLOR_GRAY2BGR
                                                ) 
            elif flag == "flash":
                frame[:,:,:] = 0
                for x, y in _bD:
                    if mPtRad == 0:
                        frame[y,x] = (255, 255, 255)
                    else:
                        cv2.circle(frame, (x, y), mPtRad, (255,255,255), -1)
            elif flag == "ani":
                grey -= dec
                if mPtRad > 0: tmpGrey[:,:] = 0
                for _bDi, (x, y) in enumerate(_bD):
                    if mPtRad == 0:
                        grey[y,x] += inc
                    else:
                        cv2.circle(tmpGrey, (x, y), mPtRad, inc, -1)
                    '''
                    ### draw line to close point from previous frame
                    if _pBD is None: continue
                    _dists = []
                    for _pBDi, (px, py) in enumerate(_pBD):
                        msg = "writing video"
                        msg += " [%i/%i]"%(i+1, len(bDataLst))
                        msg += " [%i/%i]"%(_bDi+1, len(_bD))
                        msg += " [%i/%i]"%(_pBDi+1, len(_pBD))
                        print("\r", msg, end="          ", flush=True)
                        _dists.append([np.sqrt((px-x)**2 + (py-y)**2), _pBDi])
                    if len(_dists) > 0:
                        _dist, _idx = sorted(_dists)[0]
                        if _dist < self.antLen*10:
                            px, py = _pBD[_idx]
                            cv2.line(tmpGrey, (px, py), (x, y), 255, 1)
                    '''
                #_pBD = copy(_bD)
                if mPtRad > 0: grey += tmpGrey
                grey[grey>255] = 255
                grey[grey<0] = 0
                frame = cv2.cvtColor(grey.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            if flag != "ani":
                # draw snapshot image
                #frame[0:ssROI[3], (fSh[1]-ssROI[2]):fSh[1]] = snapshot
                frame[0:fSh[0], fSh[1]:int(fSh[1]*2)] = snapshot
            
                tx = int(frame.shape[1]/2) - int(txtW*19/2)
                ty = txtH + txtBl + 5
                # write datetime
                cv2.putText(frame, str(bD_dt[i]), (tx, ty), cvFont, 
                    fontScale=fScale, color=(255,255,255), thickness=fThck)
            
            # write this frame into video
            video_rec.write(frame)

            '''
            ### [temporary]
            if len(_bD) > 0:
                _fp = "/home/joh/work/CremerLab/doc/000000/AntOS/tmp/"
                _fp += "%05i.jpg"%(i)
                cv2.imwrite(_fp, frame)
            '''

        video_rec.release()

    #---------------------------------------------------------------------------

    def drawMVec(self, args):
        """ draw movement image
        
        Args:
            args (tuple): Input arguments.
         
        Returns:
            (tuple): Return values. 
        """ 
        if DEBUG: MyLogger.info(str(locals()))
            
        proc2run, bDataLst, bD_dt, img, q2m = args

        main = self.mainFrame 

        ### set degree range to emphasize with different color
        ''' [not implemented]
        w = wx.FindWindowByName("saMVecEmph_txt", main.panel["ml"])
        val = widgetValue(w).split(",")
        for v in val:
            try: _deg.append(int(v))
            except: pass
        w = wx.FindWindowByName("saMVecEmphMargin_txt", main.panel["ml"])
        try: m = int(widgetValue(w))
        except: m = 0
        '''
        deg2emph = [
                    dict(min=90, max=91), dict(min=270, max=271),
                    dict(min=0, max=1), dict(min=180, max=181)
                    ]
        deg2emph = []

        ### [begin] drawing movement image -----
        img = img.astype(np.int32)
        tmpImg = np.zeros(img.shape, np.uint8)
        color = dict(
                        b=(10,50,10), # base color for drawing arrow
                        e=(10,10,255), # emphasizing color 
                        )
        _ppos = None # previous position
        dLen = len(bDataLst["direction"])
        for di in range(dLen):
            if di % 100 == 0:
                msg = f'drawing data... {di}/ {dLen}'
                q2m.put(("displayMsg", msg,), True, None)
            _dir = bDataLst["direction"][di]
            if _dir == -1: continue
            _dist = bDataLst["dist"][di]
            _pos = (bDataLst["posX"][di], bDataLst["posY"][di])
            if _ppos is not None and _ppos != (-1, -1):
                tmpImg[:,:,:] = 0
                ### determine color
                flagEmph = False
                for dRng in deg2emph:
                    if dRng["min"] <= _dir < dRng["max"]: flagEmph = True
                if flagEmph:
                    _color = color["e"]
                    _thck = 3 
                else:
                    _color = color["b"]
                    _thck = 1
                # draw line
                cv2.line(tmpImg, _ppos, _pos, _color, _thck)
                img += tmpImg
            _ppos = tuple(_pos)
        img = img / np.max(img) * 255
        img = img.astype(np.uint8)
        _txt = f'thicker red line:'
        for dRng in deg2emph: _txt += f' [{dRng["min"]}-{dRng["max"]})'
        cv2.putText(img, _txt, (5, 20), cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1.0, color=(255,255,255), thickness=1)
        ### [end] drawing movement image -----

        ### get max values of each data column 
        maxVal = []
        for k in bDataLst.keys():
            maxVal.append(np.max(bDataLst[k]))
        ### set data type depending on the max value of each list
        bdType = []
        for mv in maxVal:
            bdType.append(getNumpyDataType(mv, flagIntSign=True)) 
        _dtype = [("datetime", "U26")]
        for mi, mk in enumerate(bDataLst.keys()):
            _dtype.append((mk, bdType[mi]))
        rD = np.zeros(dLen, dtype=_dtype) # initiate array
        #rD["datetime"] = np.array([str(x) for x in bD_dt])
        rD["datetime"] = bD_dt 
        for mi, mk in enumerate(bDataLst.keys()):
            rD[mk] = np.array(bDataLst[mk], dtype=bdType[mi])
        rawData = dict(i=rD)

        return img, rawData

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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 
                               iterations=10) # decrease minor features 
        img = cv2.Canny(img, 10, 30)
        img = img.astype(np.int16)
        img -= 200
        img[img<0] = 0
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return img

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ != "__main__":
    pass


