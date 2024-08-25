# coding: UTF-8
"""
This module is for DataVisualizer app.
It has specific processing algorithms for different type of data, graph, etc. 

Dependency:
    Numpy (1.17)
    SciPy (1.4)
    OpenCV (3.4)

last edited: 2024-06-02
"""

import sys, csv, ctypes
from os import path, remove
from glob import glob
from copy import copy
from time import time
from datetime import datetime, timedelta

import wx, cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter, welch, periodogram
from scipy.spatial.distance import cdist

from initVars import *
from modFFC import *
from modCV import *

DEBUG = False

#csv.field_size_limit(sys.maxsize)
csv.field_size_limit(int(ctypes.c_ulong(-1).value//2))

#===============================================================================

class ProcGraphData:
    """ Class for processing data and generate graph.
    
    Args:
        parent (wx.Frame): Parent frame
    
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    
    def __init__(self, parent):
        if DEBUG: MyLogger.info(str(locals()))

        ##### [begin] setting up attributes on init. -----
        self.parent = parent
        self.colTitles = [] # data column titles
        self.numData = None # Numpy array of numeric data
        self.strData = None # Numpy array of character data
        self.graphImg = {} 
        self.graphImgIdx = -1 
        self.graphImgIdxLst = []
        self.fonts = parent.fonts
        self.interactiveDrawing = {} # for drawing data via user interaction 
        self.subClass = None
        # cases to show some info on certain graphs
        self.c2showInfoOnMouseMove = ["L2020CSV2", "aos", "anVid"]
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

        p = self.parent
        
        if p.eCase == "L2020":
            self.prevGrey = None
            ### store the first frame as the graph image at this initial stage 
            self.storeGraphImg(self.parent.vRW.currFrame.copy(), 
                               self.parent.pi["mp"]["sz"]) 

        elif p.eCase == "L2020CSV1":
            info = {}
            # read heatmap CSV data into NumPy array
            hmArr = np.genfromtxt(p.inputFP, delimiter=',')
            hmArr = hmArr.astype(np.uint16)
            # maximum value of the array 
            maxV = np.max(hmArr) 
            
            ### set number of heatmap level ranges
            flArr = hmArr.flatten()
            flArr = flArr[flArr!=0]
            info["percVal"] = {"min":[], "max":[]}
            for i, perc in enumerate(p.percLst):
            # go through percentiles for seven ranges
                if i == 0: minPercVal = np.min(flArr)
                else: minPercVal = int(np.percentile(flArr, p.percLst[i-1]))
                maxPercVal = int(np.percentile(flArr, perc))
                if i == len(p.percLst)-1: maxPercVal += 1
                info["percVal"]["min"].append(minPercVal)
                info["percVal"]["max"].append(maxPercVal)

            ##### [begin] calculate additional info to display -----
            csvFP1 = p.inputFP.replace("_0.csv", "_1.csv")
            f = open(csvFP1, "r")
            lines = f.readlines()
            f.close()
            ### get average centroid
            centX = [] # centroid x-coordinate
            centY = [] # centroid y-coordinate
            for li in range(1, len(lines)):
                cent = lines[li].split(",")[0].split("/") # centroid
                centX.append(int(cent[0]))
                centY.append(int(cent[1]))
            info["avgCent"] = (int(np.mean(centX)), int(np.mean(centY)))
            ### get where max value is
            tmp = np.where(hmArr==maxV)
            info["maxVPos"] = (tmp[1][0], tmp[0][0])
            ##### [end] calculate additional info to display ----- 
            q2m.put(("finished", hmArr, info,), True, None)

        elif p.eCase == "V2020": 
            ##### [begin] set colors
            # set colors (BGR) for each species
            self.sColor = [(0,205,0), (205,0,0), (0,237,255)]
            # set colors for each population 
            self.pColor = [
                            [[1,255,0],
                             [0,205,0],
                             [34,139,34]],
                            [[255,191,0],
                             [199,149,0],
                             [205,0,0]],
                            [[20,255,255],
                             [0,224,241],
                             [0,204,255]],
                            ]
            ### set colors for each classification
            self.cColor = {}
            cl = "Bunyavirales" 
            self.cColor[cl] = (255,127,255)
            cl = "Mononegavirales"
            self.cColor[cl] = (0,0,0)
            cl = "Narnaviridae"
            self.cColor[cl] = (255,0,127)
            cl = "Nodaviridae"
            self.cColor[cl] = (127,127,255)
            cl = "Permutotetraviridae"
            self.cColor[cl] = (0,0,127)
            cl = "Picornavirales"
            self.cColor[cl] = (0,0,255)
            cl = "Totiviridae"
            self.cColor[cl] = (0,127,255)
            cl = "Unclassified"
            self.cColor[cl] = (100,100,100)
            ##### [end] set colors for each classification
            self.numSpecies = 3
            self.numPopulations = 3
            self.vCR = -1 # radius of virus presence circle
            self.vpPt = {} # dictionary of virus presence points 
              # key will be virus such as 'LHUV-1'
              # value will be list of x,y coordinates where it appears in graph
            self.vlR = {} # dictionary of virus labels' rects (in legend)
            self.clR = {} # virus classifications' rects (in legend)
              # both rects are in form of (x1, y1, x2, y2)
            # column indices of numeric data
            self.numericDataIdx = [2, 3, 4, 5, 6, 7, 8, 9, 10] 
            # column indices of string data
            self.stringDataIdx = [0, 1] 
            ### get CSV data as array
            f = open(p.inputFP, 'r')
            csvTxt = f.read()
            f.close()
            ret = csv2numpyArr(csvTxt, ",", 
                               self.numericDataIdx, self.stringDataIdx, 
                               np.int8)
            self.colTitles, self.numData, self.strData = ret

            ### remove classification labels after semicolon
            ### e.g.: Picornavirales;Dicistroviridae -> Picornavirales
            for ri in range(self.strData.shape[0]):
                for ci in range(self.strData.shape[1]):
                    _str = self.strData[ri,ci]
                    if ";" in _str:
                        self.strData[ri,ci] = _str.split(";")[0]
                
            self.numViruses = self.numData.shape[0] 
            # number of virus presences in inner circle
            self.numVPresence = np.sum(self.numData)
            # degree between two virus presence dots 
            self.vpDeg = -360.0 / self.numVPresence # minus value = clockwise
        return True
   
    #---------------------------------------------------------------------------
  
    def storeGraphImg(self, img, sz, additionalImg=None):
        """ store graph image after resizing, if necessary 

        Args:
            img (numpy.ndarray): graph image 
            sz (tuple): size to display the graph image
            additionalImg (None/numpy.ndarray): additional image to the graph
                                                  such as floating legend

        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        print("[ProcGraphData.storeGraphImg] saving original image..")

        p = self.parent

        if self.graphImgIdxLst == []: gi = 0
        else: gi = max(self.graphImgIdxLst)+1
        self.graphImgIdx = gi
        self.graphImgIdxLst.append(gi)
        fn = "tmp_origImg%i.png"%(gi)
        cv2.imwrite(fn, img) # write original image to a file
        if not additionalImg is None:
            fn = "tmp_origImg%i_a.png"%(gi)
            cv2.imwrite(fn, additionalImg) # write original additional image 
        self.graphImg[gi] = {}

        # store image size (width, height)
        self.graphImg[gi]["imgSz"] = (img.shape[1], img.shape[0])
        
        ### resize (when ratio is not 1.0) & store the image
        print("[ProcGraphData.storeGraphImg] storing image..")
        iSz = (img.shape[1], img.shape[0])
        # set ratio of displayed image to original image
        self.graphImg[gi]["ratDisp2OrigImg"] = calcI2DRatio(iSz, sz)
        # init offset value
        self.graphImg[gi]["offset"] = [0, 0]
        p.zoom_txt.SetValue(str(self.graphImg[gi]["ratDisp2OrigImg"]))
        p.offset_txt.SetValue("0, 0")
        self.zoomNStore(img, additionalImg)

        ### store thumbnail images
        print("[ProcGraphData.storeGraphImg] storing thumbnail image..")
        l = int(self.parent.pi["mr"]["sz"][0]*0.4)
        self.graphImg[gi]["thumbnail"] = self.graphImg[gi]["img"].Scale(l, l)

    #---------------------------------------------------------------------------
  
    def storeGraphData(self, data=[]):
        """ Store data related to graph. 
        Assumes that corresponding graph is already stored in self.graphImg.

        Args:
            data (list): Graph related data. Could be multiple data.

        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        if self.graphImgIdxLst == []: return
        p = self.parent
        eCase = p.eCase
        gi = max(self.graphImgIdxLst) # the latest graph index
        firstKeyType = "data" 
        dLen = len(data)
        if type(data) == dict:
            kLst = list(data.keys())
            if kLst[0].startswith("roi"):
                firstKeyType = "roi" 
                dLen = len(data[kLst[0]])

        def savNPArr(eCase, fn, d):
            header = ""
            for descr in d.dtype.descr: header += descr[0] + ", "
            header = header.rstrip(", ")
            if eCase in ["aos", "anVid"]:
            # cases for saving numpy array as it is
                fn = fn.replace(".csv", ".npy")
                np.save(fn, d, allow_pickle=False)
            else:
                np.savetxt(fn, d, newline="\n", delimiter=",", header=header)

        for di in range(dLen):
            if type(data) == list:
                d = data[di]
                fn = "tmp_data_%i_%i.csv"%(gi, di)

                if eCase == "L2020":
                    colTitles = sorted(d[0].keys())
                    colTitles.remove("bRects")
                    colTitles.append("bRects")
                    fh = open(fn, "w")
                    # write column titles
                    fh.write(str(colTitles).strip("[]").replace("'","")+"\n")
                    for li in range(len(d)): # go though lines (frame data)
                        line = ""
                        for ki, k in enumerate(colTitles):
                            _d = d[li][k]
                            if k == "bRects":
                                for bi in range(len(_d)):
                                # go through rects of ant blob data
                                    line += str(_d[bi]).replace(",","/")
                            else:
                                # other data are tuple of rect or coordinate
                                line += str(_d).strip("()").replace(",","/")
                            line += ", "
                        line = line.rstrip(", ") + "\n"
                        fh.write(line)
                    fh.close()
            
            elif type(data) == dict:
                if firstKeyType == "roi":
                    for roiK in data.keys():
                        _data = data[roiK]
                        key = list(sorted(_data.keys()))[di]
                        d = _data[key]
                        fn = f'tmp_data_{gi}'
                        # skip the basic data (intensity) indicator, 'i' 
                        if key == "i": fn += f'_{roiK}.csv'
                        else: fn += f'_{key}_{roiK}.csv'
                        savNPArr(eCase, fn, d)
                else:
                    key = list(sorted(data.keys()))[di]
                    d = data[key]
                    fn = f'tmp_data_{gi}_{key}.csv'
                    savNPArr(eCase, fn, d)
 
    #---------------------------------------------------------------------------
  
    def removeGraph(self, gi=-1):
        """ remove graph image & data

        Args:
            gi (int): graph image index to remove, -1 means all.

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent

        if gi == -1:
            ### remove all images
            for fp in glob("tmp_origImg*.png"): remove(fp)
            for fp in glob("tmp_data_*.csv"): remove(fp)
            for fp in glob("tmp_data_*.npy"): remove(fp)
            self.graphImg = {} 
            self.graphImgIdxLst = []
        else:
            ### remove graph image with the given index
            fn = "tmp_origImg%i.png"%(gi)
            if path.isfile(fn): remove(fn)
            fn = "tmp_origImg%i_a.png"%(gi)
            if path.isfile(fn): remove(fn)
            exts = ["csv", "npy"]
            for ext in exts:
                for fn in glob("tmp_data_%i_*.%s"%(gi, ext)):
                    if path.isfile(fn): remove(fn)
            del self.graphImg[gi]
            self.graphImgIdxLst.remove(gi)
        self.graphImgIdx = -1

        p.panel["mp"].Refresh()
        p.updateMRWid()

    #---------------------------------------------------------------------------
  
    def graphL2020(self, startFrame, endFrame, q2m):
        """ Draw heatmap of ants' positions in structured/unstructured nest. 
        
        Args:
            startFrame (int): Frame index to start of heatmap data.
            endFrame (int): Frame index to end of heatmap data.
            q2m (queue.Queue): Queue to send data to main thread.
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        #-----------------------------------------------------------------------
        def maskWalls(fImg): 
            for i in range(2):
                obj = wx.FindWindowByName("wallRect%i_txt"%(i), 
                                          p.panel["ml"])
                r = [int(_x) for _x in obj.GetValue().split(",")]
                cv2.rectangle(fImg, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), 
                              (255,200,200), -1)
            return fImg
        #--------------------------------------------------------------------
        def findCols(fImg, tag):
        # function to find colors; tag indicates whether it's for ant or brood
            hsvImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HSV)
            fcRslt = cv2.inRange(hsvImg, 
                                 hsvP["%s0"%(tag)]["Min"],
                                 hsvP["%s0"%(tag)]["Max"])
            if tag == "ant":
                for ci in range(1, 3):
                    fcRslt = cv2.add(
                                fcRslt,
                                cv2.inRange(hsvImg, 
                                            hsvP["ant%i"%(ci)]["Min"], 
                                            hsvP["ant%i"%(ci)]["Max"])
                                )
            ### attempt to remove small erroneous noise 
            fcRslt = cv2.morphologyEx(fcRslt, cv2.MORPH_OPEN, kernel)
            # threshold
            ret, img = cv2.threshold(fcRslt, 50, 255, cv2.THRESH_BINARY)
            return img
        #-----------------------------------------------------------------------
        def calculateAntBlobs(img, broodImg):
        # find ant blobs and calculate some measures with it
            rslt = {} # result dictionary to return
            tmpArr = np.zeros((img.shape[0], img.shape[1]), np.uint8) 
            # get centroid of ants 
            rslt["centroid"] = getCentroid(img)
            # get brood centroid
            rslt["broodCent"] = getCentroid(broodImg)
            ### approximate center of two wall positions
            wallCt = []
            for _i in range(1, 3):
                wallCt.append((p.roi[0] + int(p.roi[2]/2),
                               p.roi[1] + (p.roi[3]/3)*_i))
            # get connected components
            ccOutput = cv2.connectedComponentsWithStats(img,
                                                        connectivity=4)
            nLabels = ccOutput[0] # number of labels (blobs)
            labeledImg = ccOutput[1]
            stats = list(ccOutput[2]) # stat [left, top, width, height, area]
            bRects = [] # blob rects list
            # list of coordinate of large (> single ant) blob
            cxL = []; cyL = []
            # list of coordinate of small (<= single ant) blob
            cxS = []; cyS = []
            for li in range(1, nLabels):
                l, t, w, h, a = stats[li]
                if a < aMinArea: # too small blob
                    ### check whether it's close enough to the tunnels in walls
                    isClose2Tunnel = False
                    cx = l + int(w/2)
                    cy = t + int(h/2)
                    for wpt in wallCt: 
                        _dist = np.sqrt((cx-wpt[0])**2 + (cy-wpt[1])**2)
                        if _dist < antLen*1.5:
                            cv2.circle(fImg, (cx, cy), 2, (100,255,100), -1)
                            isClose2Tunnel = True
                            break
                    if isClose2Tunnel:
                        ### increase blob size to the minimum
                        l = cx - int(gasterLen/2)
                        t = cy - int(gasterLen/2)
                        w = gasterLen 
                        h = gasterLen 
                        a = gasterLen ** 2
                    else:
                        continue # ignore this blob
                bRects.append((l, t, w, h)) # store blob rect
                ### store x1, x2, y1, y2 of large/small blob
                if a > aMinArea*4:
                    cxL += [l, l+w]
                    cyL += [t, t+h]
                else:
                    cxS += [l, l+w]
                    cyS += [t, t+h]
                
            rslt["bRects"] = bRects # store list of all blobs rect;(x, y, w, h)

            ### store bounding rect (x,y,w,h) of all blobs
            if cxL + cxS != []:
                r = [min(cxL+cxS), min(cyL+cyS)]
                r += [max(cxL+cxS)-r[0], max(cyL+cyS)-r[1]]
                rslt["rect"] = tuple(r) 
            else:
                print("WARNING:: [fi-%i] cxL + cxS == empty"%(p.vRW.fi))
                rslt["rect"] = ()
            ### store bounding rect (x,y,w,h) of all large blobs
            if cxL != []:
                r = [min(cxL), min(cyL)]
                r += [max(cxL)-r[0], max(cyL)-r[1]]
                rslt["rectL"] = tuple(r)
            else:
                print("WARNING:: [fi-%i] cxL == empty"%(p.vRW.fi))
                rslt["rectL"] = ()

            ### calculate distance between ant-centroid and brood-centroid
            _c = rslt["centroid"]
            _bc = rslt["broodCent"]
            rslt["dist_c2bc"] = np.sqrt((_c[0]-_bc[0])**2 + (_c[1]-_bc[1])**2)
            ### calculate distance between ant-centroid and entrance
            rslt["dist_c2entrance"] = np.sqrt((_c[0]-entrance[0])**2 + \
                                              (_c[1]-entrance[1])**2) 

            ### calculate mean nearest neighbor distance 
            tNAnts = 0 # approximated total number of ants 
            nearNDistLst = [] # list for each blob's distance to its nearest
                              #  neighbor blob. (multiplied by the blob's area)
            for li1 in range(1, nLabels):
                l1, t1, w1, h1, area = stats[li1]
                if area < aMinArea: continue
                # number of ants in this blob
                nAnts = max(1, int(round(stats[li1][4] / (aMinArea*3))))
                tNAnts += nAnts # increase total number of ants
                tmpArr[:,:] = 0
                tmpArr[labeledImg==li1] = 255
                _ret = cv2.findContours(tmpArr, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
                if cv2.__version__.startswith("3.4"):
                    img_fCont, contours, hierarchy = _ret
                else:
                    contours, hierarchy = _ret
                # contour points of the target blob
                contourPts0 = np.vstack(contours).squeeze()
                # center point of the target blob
                c1 = (l1 + w1/2, t1 + h1/2)
                dists = []
                for li2 in range(1, nLabels):
                    if li1 == li2: continue
                    l2, t2, w2, h2, __ = stats[li2]
                    c2 = (l2 + w2/2, t2 + h2/2)
                    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    dists.append((dist, li2))
                n = sorted(dists)[:2] # three nearest blobs
                nLIs = [item[1] for item in n] # indices of the nearest blobs
                tmpArr[:,:] = 0
                for nLI in nLIs:
                    tmpArr[labeledImg==nLI] = 255
                _ret = cv2.findContours(tmpArr, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
                if cv2.__version__.startswith("3.4"):
                    img_fCont, contours, hierarchy = _ret
                else:
                    contours, hierarchy = _ret
                # contour points of the three blobs near to the target blob
                contourPts1 = np.vstack(contours).squeeze()
                dists = cdist(contourPts0, contourPts1, metric="euclidean")
                nearNDistLst.append(np.min(dists)) # append shortest distance
                nearNDistLst += [0] * (nAnts-1) # append zeros for a number of
                  # ants in this ant blob (if it's more of a single ant)
                """
                #print(li1, (l1, t1), area, "n=%i"%(nAnts), np.min(dists))
                cv2.putText(fImg,
                    "%i"%(li1),
                    (l1, t1), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1.0, 
                    color=(50,50,255),
                    thickness=1)
                """
            rslt["meanNNDist"] = np.mean(nearNDistLst) / antLen

            ### calculate dispersal rate of ant aggregation
            ###   (large blob bounding box, divided by ant clumped area)
            if rslt["rectL"] == ():
                rslt["dispersalRate"] = -1
            else:
                l, t, w, h = rslt["rectL"]
                cellLen = antLen * np.sin(np.deg2rad(45)) # length of square 
                              # cell in which an ant fit into its hypotenuse
                clumpedArea = cellLen**2 * tNAnts  # area when all ants are 
                                                   #   tightly clumped together
                rslt["dispersalRate"] = (w*h)/clumpedArea 

            return rslt 
        #-----------------------------------------------------------------------
        def calcMotion(grey, prev_grey):
        # calculate to sum different pixels (one different pixel = 1) 
            m_thr = 60
            diff = cv2.absdiff(grey, prev_grey)
            ret, diff = cv2.threshold(diff, m_thr, 255, cv2.THRESH_BINARY) 
            #diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
            diff[diff==255] = 1
            return np.sum(diff), diff
        #--------------------------------------------------------------------
        
        p = self.parent
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
         
        ### determine HSV ranges for detecting ant colors
        hsvP = dict(mi=[], ma=[]) 
        hsvP = {}
        for ck in p.defHSVVal.keys(): # go though color ranges
            hsvP[ck] = {}
            for mLbl in ["Min", "Max"]:
                val = []
                for hsvLbl in ["H", "S", "V"]:
                    objName = "c-%s-%s-%s_sld"%(ck, hsvLbl, mLbl)
                    obj = wx.FindWindowByName(objName, p.panel["ml"])
                    val.append(obj.GetValue())
                hsvP[ck][mLbl] = tuple(val) 
        
        ### get ant minimum area
        obj = wx.FindWindowByName("aMinArea_txt", p.panel["ml"])
        aMinAreaTxtVal = obj.GetValue().strip()
        if aMinAreaTxtVal == "": aMinArea = 100; obj.SetValue("100")
        else: aMinArea = int(aMinAreaTxtVal)

        ### approximate length of an ant
        gasterLen = int(np.ceil(np.sqrt(aMinArea)))
        antLen = gasterLen * 3

        ### get entrance position
        obj = wx.FindWindowByName("entrance_txt", p.panel["ml"])
        try:
            entrance = tuple([int(_x) for _x in obj.GetValue().split(",")])
            if len(entrance) != 2: entrance = tuple(p.roi[:2])  
        except:
            pass

        ### whether to mask two walls or not
        obj = wx.FindWindowByName("maskWallArea_chk", p.panel["ml"])
        flagMaskWalls = obj.GetValue()

        ### colors
        color = dict(motion=(255,100,255), brood=(50,150,255), 
                     ant=(50,50,255), antMinAreaSquare=(30,30,30), 
                     meanNNDistTxt=(100,100,0), entrance=(255,255,50), 
                     blobRect=(0,255,0), entireRect=(128,128,128), 
                     lBlob=(0,128,255), brCent=(50,255,255), 
                     aCent=(200,0,0))
        
        if p.debugging["state"]:
        # debugging
            obj = wx.FindWindowByName("debugFrameIntv_txt", p.panel["ml"])
            intvTxt = obj.GetValue().strip()
            try: intv = int(intvTxt)
            except: intv = 1; obj.SetValue("1")
            for __ in range(intv):
                p.vRW.getFrame(-1) # get frame image
            fImg = p.vRW.currFrame
            ### mask out of ROI
            col = tuple([200 for _x in range(fImg.shape[2])])
            fImg = maskImg(fImg, [p.roi], col)
            # pre-process
            fImg = makeImgDull(fImg, 2, 2) 
            if flagMaskWalls:
                fImg = maskWalls(fImg) # mask two walls
            # greyscale image
            grey = cv2.cvtColor(fImg, cv2.COLOR_BGR2GRAY)
            ### motion calculation  
            if self.prevGrey is not None: 
                motion, diff = calcMotion(grey, self.prevGrey)
                if not p.debugging["hideMarkerM"]:
                    fImg[diff==1] = color["motion"]
            self.prevGrey = grey.copy()
            # find brood color
            broodImg = findCols(fImg, "br")
            if not p.debugging["hideMarkerAB"]:
                fImg[broodImg==255] = color["brood"] # display found brood color
            # find ant colors
            img = findCols(fImg, "ant")
            if not p.debugging["hideMarkerAB"]:
                fImg[img==255] = color["ant"] # display found ant colors
            cv2.putText(fImg, "Frame: %i"%(p.vRW.fi), (5, fImg.shape[0]-20),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0,0,0),
                        thickness=1)
            pt1 = (10, 10)
            l = int(np.sqrt(aMinArea))
            pt2 = (pt1[0]+l, pt1[1]+l)
            # draw ant min. area
            cv2.rectangle(fImg, pt1, pt2, color["antMinAreaSquare"], -1) 
            cv2.putText(fImg, "ant min. area", (pt2[0]+10, pt2[1]),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0,0,0),
                        thickness=1)
            # get ant blob info and store in output-data
            oD = calculateAntBlobs(img, broodImg) 
            oD["broodCent"] = getCentroid(broodImg) # store centroid of brood
            ### display mean NN distance & dispersal rate of aggregations
            txt = "meanNN-dist: %.3f"%(oD["meanNNDist"])
            txt += ", aggregation dispersal: %.3f"%(oD["dispersalRate"])
            cv2.putText(fImg,
                        txt, 
                        (5, 50),
                        cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=1.2, 
                        color=color["meanNNDistTxt"],
                        thickness=1)
            ### display entrance position
            cv2.circle(fImg, entrance, 20, color["entrance"], 3)
            ### display blob rects
            for b in oD["bRects"]:
                pt1 = (b[0], b[1])
                pt2 = (b[0]+b[2], b[1]+b[3])
                cv2.rectangle(fImg, pt1, pt2, color["blobRect"], 2)
            ### display entire rect
            if oD["rect"] != ():
                r = oD["rect"]
                cv2.rectangle(fImg, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), 
                              color["entireRect"], 2)
            ### display rect, bounding large blobs
            if oD["rectL"] != ():
                r = oD["rectL"]
                cv2.rectangle(fImg, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), 
                              color["lBlob"], 2)
            # display brood centroid
            cv2.circle(fImg, oD["broodCent"], 12, color["brCent"], -1)
            # display ants' centroid
            cv2.circle(fImg, oD["centroid"], 10, color["aCent"], -1) 
            # store the resultant image
            self.storeGraphImg(fImg, p.pi["mp"]["sz"])
            return
       
        ##### actual analysis starts (not debugging mode)

        # total number of frames
        nFrames = endFrame - startFrame + 1 

        startTime = time()
        ### make array for heatmap
        shape = p.vRW.currFrame.shape
        hmArr = np.zeros((shape[0], shape[1]), 
                         dtype=np.int32) # init array for heatmap 
        oD = [] # output data; ant blob (connected component) 
            # information + brood's centroid
            # Each item is a list, representing info in a frame.
            # Items for a frame are rects (left, top, width, height) of found 
            #   blobs, which fit into the category of minimum area.
        #endSpaces = " "*(50 + len(str(endFrame+1))*2)
        prevGrey = None
        for fi in range(startFrame, endFrame+1):
        # go through all the frames
            if (fi+1)%10 == 0:
                msg = "processing frame-%i/ %i"%(fi+1, endFrame+1)
                _eF = fi - startFrame
                _t4f = (time()-startTime) / (_eF+1)
                _est = (nFrames-1-_eF) * _t4f
                msg += " [Time remaining: %s]"%(
                                    str(timedelta(seconds=_est))[:-3]
                                    )
                q2m.put(("displayMsg", msg), True, None)
                print("\r", msg, end=" ", flush=True)
            if fi == p.vRW.fi:
                fImg = p.vRW.currFrame
            else:
                p.vRW.getFrame(-1) # get frame image
                fImg = p.vRW.currFrame
            ### mask out of ROI
            col = tuple([200 for _x in range(fImg.shape[2])])
            fImg = maskImg(fImg, [p.roi], col)
            # pre-process
            fImg = makeImgDull(fImg, 2, 2)
            if flagMaskWalls:
                fImg = maskWalls(fImg) # mask two walls
            # greyscale image
            grey = cv2.cvtColor(fImg, cv2.COLOR_BGR2GRAY)
            ### motion calculation
            motion = 0 
            if prevGrey is not None: 
                motion, diff = calcMotion(grey, prevGrey)
            prevGrey = grey.copy()
            # find brood color
            broodImg = findCols(fImg, "br")
            # find ant colors
            img = findCols(fImg, "ant")
            # increase where colors are found
            hmArr[img==255] += 1
            # calculate some measures with ant blobs.
            measures = calculateAntBlobs(img, broodImg)
            # add motion value into the dictionary
            measures["motion"] = motion
            oD.append(measures)

        """
        ### set number of heatmap level ranges
        hmLvlRngs = []
        for i in range(len(str(nFrames))-1):
        # through number of frame by order of magnitude  
            rng1 = 10 ** i
            if i > 0: rng1 += 1
            rng2 = 10 ** (i+1)
            hmLvlRngs.append((rng1, rng2))
        hNum = 10**(len(str(nFrames))-1) # number with the highest order of 
                            # magnitude, e.g.: 10000 in 30000 total frames
        for hn in range(hNum, nFrames, hNum):
        # the rest number of frames with the interval of hNum
        # e.g.: [10000, 20000, 30000] in 30000 total frames
            hmLvlRngs.append((hn+1, hn+hNum))
        """
        
        ### make base image to draw heat-map
        gImg = cv2.morphologyEx(
                        fImg.copy(), 
                        cv2.MORPH_OPEN, 
                        kernel, 
                        iterations=5,
                        ) # to decrease noise & minor features
        gImg = cv2.Canny(gImg, 50, 100)
        gImg = gImg.astype(np.int16)
        gImg -= 175 
        gImg[gImg<0] = 0
        bgIdx = np.where(gImg == 0) # indices to draw background color
        rsltImg = cv2.cvtColor(gImg.astype(np.uint8), 
                               cv2.COLOR_GRAY2BGR)
        #rsltImg = np.zeros(fImg.shape, fImg.dtype)
        
        ### set background color of heatmap
        obj = wx.FindWindowByName("bgCol_cPk", p.panel["ml"])
        hmBgCol = tuple(obj.GetColour())
        rsltImg[bgIdx] = [hmBgCol[2], hmBgCol[1], hmBgCol[0]]

        # draw heatmap
        rsltImg = self.drawHeatmapImg(hmArr, hmBgCol, rsltImg)
        
        ### write frame range
        cv2.putText(rsltImg,
                    "Frame: %i - %s"%(startFrame, endFrame),
                    (5, rsltImg.shape[0]-20),
                    cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1.2, 
                    color=(255,255,255),
                    thickness=1)

        q2m.put(("finished", rsltImg, [hmArr, oD], startFrame, endFrame), 
                True, 
                None)
 
    #---------------------------------------------------------------------------
  
    def graphL2020CSV1(self):
        """ Draw heatmap with data from graphL2020 
        
        Args: None
         
        Returns: None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent
        hmArr = self.hmArr 
        ### get heatmap ranges
        hmLvlRngs = []
        for ri in range(7): # seven ranges
            rng = []
            for li, lbl in enumerate(["min", "max"]):
                obj = wx.FindWindowByName("hmRng-%i-%s_txt"%(ri, lbl), 
                                          p.panel["ml"])
                rng.append(int(obj.GetValue()))
            hmLvlRngs.append(tuple(rng))
        print(hmLvlRngs)
        
        # result (heatmap) image 
        rsltImg = np.zeros((hmArr.shape[0], hmArr.shape[1], 3), np.uint8) 
        hmBgCol = (0, 0, 0)
        # draw heatmap
        rsltImg = self.drawHeatmapImg(hmArr, hmBgCol, rsltImg, hmLvlRngs)

        ##### [begin] draw additional info -----
        circRad = int(hmArr.shape[1]*0.01)
        avgX, avgY = self.info["avgCent"]
        maxX, maxY = self.info["maxVPos"]
        # draw position of where max value of the array is
        cv2.circle(rsltImg, (maxX, maxY), circRad, (255,75,75), 5)
        # draw average position of centroid data of all frames
        cv2.circle(rsltImg, (avgX, avgY), circRad, (200,0,0), 5) 
        # draw line between the above two circles
        cv2.line(rsltImg, (maxX,maxY), (avgX,avgY), (255,75,75), 2)
        ### draw label for the circles 
        cv2.putText(rsltImg, "Max.", (maxX, maxY),  cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1.0, color=(50,50,50), thickness=1)
        cv2.putText(rsltImg, "Avg. centroid", (avgX, avgY),  
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, 
                    color=(50,50,50), thickness=1)
        ### draw distance text
        dist = round(np.sqrt((maxX-avgX)**2 + (maxY-avgY)**2))
        distStr = str(int(dist))
        tx = min(maxX, avgX) + int(dist/2)
        ty = min(maxY, avgY) + int(dist/2)
        cv2.putText(rsltImg, distStr, (tx, ty),  cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1.2, color=(150,0,0), thickness=1)
        ### update the distance on ml panel
        obj = wx.FindWindowByName("dist_M2C_txt", p.panel["ml"])
        obj.SetValue(distStr)
        ##### [end] draw additional info -----
        
        self.storeGraphImg(rsltImg, p.pi["mp"]["sz"])

    #---------------------------------------------------------------------------
  
    def graphL2020CSV2(self, q2m):
        """ Draw additional graph for structured/unstructured nest experiment. 
        
        Args:
            q2m (queue.Queue): Queue to send data to main thread.
         
        Returns: None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent

        f = open(p.inputFP, "r")
        lines = f.readlines()
        f.close()

        ### read dispersal-rate and mean-nearest-neighbor-distance
        colTitles = [_x.strip() for _x in lines[0].split(",")]
        print(colTitles)
        idxNND = colTitles.index("meanNNDist")
        idxD = colTitles.index("dispersalRate")
        meanNND = []
        disp = []
        nLines = len(lines)
        for li, line in enumerate(lines):
            if (li+1)%10 == 0:
                msg = "reading data %i/ %i"%(li+1, nLines)
                q2m.put(("displayMsg", msg,), True, None)
            if li == 0: continue
            items = line.split(",")
            meanNND.append(float(items[idxNND]))
            disp.append(float(items[idxD]))
       
        mNND_factor = 200
        disp_factor = 25
        maxV_mNND = int(np.ceil(max(meanNND))) * mNND_factor 
        maxV_disp = int(np.ceil(max(disp))) * disp_factor
        # prepare result graph image array 
        rsltImg = np.zeros((maxV_mNND+maxV_disp, len(lines)-1, 3), np.uint8)
        ### draw line separating mNND and disp
        pt1 = (0, maxV_mNND)
        pt2 = (rsltImg.shape[1], maxV_mNND)
        cv2.line(rsltImg, pt1, pt2, (127, 127, 127), 1)
        ### write graph titles
        cv2.putText(rsltImg, "mean nearest neighbor distance", 
                    (5, 40),  cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2.0, color=(200,200,200), thickness=2)
        cv2.putText(rsltImg, "aggregation dispersal", 
                    (5, maxV_mNND+40),  cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2.0, color=(200,200,200), thickness=2)

        valueData = []
        for li in range(len(meanNND)):
            if (li+1)%10 == 0:
                msg = "drawing data %i/ %i"%(li+1, nLines)
                q2m.put(("displayMsg", msg,), True, None)
            # store value data
            valueData.append(("%.1f"%(meanNND[li]), "%.1f"%(disp[li])))
            if li == 0: continue
            ### draw mean nearest neighbor distance graph
            py_mNND = int(round(meanNND[li-1],1)*mNND_factor)
            y_mNND = int(round(meanNND[li],1)*mNND_factor)
            pt1 = (li-1, maxV_mNND-py_mNND)
            pt2 = (li, maxV_mNND-y_mNND)
            cv2.line(rsltImg, pt1, pt2, (75, 255, 75), 1)
            ### draw dispersal-rate of large blobs
            py_disp = int(round(disp[li-1],1)*disp_factor)
            y_disp = int(round(disp[li],1)*disp_factor)
            pt1 = (li-1, maxV_mNND+maxV_disp-py_disp)
            pt2 = (li, maxV_mNND+maxV_disp-y_disp)
            cv2.line(rsltImg, pt1, pt2, (255,75,75), 1)
            
        q2m.put(("finished", rsltImg, None, valueData), True, None)

    #---------------------------------------------------------------------------
    
    def drawHeatmapImg(self, hmArr, hmBgCol, rsltImg, hmLvlRngs={}, 
                       hmCols={}, titleLbl="", ptRad=-1):
        """ Drawing heatmap image

        Args:
            hmArr (numpy.ndarray): Array for drawing heatmap
            hmBgCol (tuple): Background color of heatmpa image
            rsltImg (numpy.ndarray): Result image to save
            hmLvlRngs (list): List of min and max values
            hmCols (list): List of heatmap colors
            titleLbl (str): Title label
            ptRad (-1): If this is not -1, then draw a circle with this radius,
              instead of changing color of a single pixel.

        Returns:
            rsltImg (numpy.ndarray): Heatmap result image
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent

        if hmLvlRngs == {}:
            ### set number of heatmap level ranges
            flArr = hmArr.flatten()
            flArr = flArr[flArr!=0]
            for i, perc in enumerate(p.percLst):
            # go through percentiles for seven ranges
                if i == 0: minPercVal = np.min(flArr) 
                else: minPercVal = int(np.percentile(flArr, p.percLst[i-1]))
                maxPercVal = int(np.percentile(flArr, perc))
                if i == len(p.percLst)-1: maxPercVal += 1
                if i == 0: key = "0 - %i %%"%(p.percLst[0])
                else: key = "%i - %i %%"%(p.percLst[i-1], p.percLst[i])
                hmLvlRngs[key] = (minPercVal, maxPercVal)

        if hmCols == {}:
            for i, key in enumerate(hmLvlRngs.keys()):
                ### store color for this heatmap range
                if i < 3: # 1st: red
                    c = (i+1) * int(255/3)
                    hmCols[key] = (0, 0, c)
                elif 3 <= i < 6: # 2nd: yellow
                    c = 155 + (i-3) * 50
                    hmCols[key] = (0, c, c)
                else: # 3rd: white
                    hmCols[key] = (255, 255, 255)

        maxV = np.max(hmArr)
        cvFont = cv2.FONT_HERSHEY_PLAIN
        ### get font 
        fThck = 1 
        thP = max(int(min(rsltImg.shape[:2])*0.03), 12)
        fScale, txtW, txtH, txtBl = getFontScale(
                        cvFont, thresholdPixels=thP, thick=fThck
                        )
        # set color square length in legend
        colSqLen = txtH + txtBl 
        legY = colSqLen + 5 
        fCol = getConspicuousCol(hmBgCol)
        fCol = (fCol[2], fCol[1], fCol[0]) # font color
        # write title label
        cv2.putText(rsltImg, titleLbl, (10, legY), cvFont, 
                    fontScale=fScale, color=fCol, thickness=1)
        for i, key in enumerate(hmLvlRngs.keys()):
            ### get indices of pixels for heatmap
            rng1, rng2 = hmLvlRngs[key]
            idx = np.logical_and(np.greater_equal(hmArr, rng1),
                                 np.less(hmArr, rng2))
            
            ### coloring heatmap
            if ptRad == -1:
                rsltImg[idx] = hmCols[key]
            else:
                ys, xs = np.where(idx)
                for pti in range(len(xs)):
                    cv2.circle(rsltImg, (xs[pti], ys[pti]), ptRad, 
                               hmCols[key], -1)
            
            ### draw legend
            legX = copy(colSqLen)
            legY += int(colSqLen * 1.25) 
            cv2.rectangle(rsltImg, 
                          (legX,legY), 
                          (legX+colSqLen,legY+colSqLen), 
                          hmCols[key],
                          -1)
            
            cv2.putText(rsltImg, 
                        key, 
                        (int(legX+colSqLen*1.5), legY+colSqLen),
                        cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=fScale, 
                        color=fCol,
                        thickness=1)
         
        return rsltImg 

    #---------------------------------------------------------------------------
  
    def graphAOSI(self, q2m):
        """ Draw (currently intensity) graph 
        with results data from 'graphAOS'
        
        Args: 
            q2m (queue.Queue): Queue to send data to main thread.
         
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent
        # colors to use draw graph
        color = dict(maxV=(255,100,100), sumV=(100,100,255), 
                     peak=(100,255,100), etc=(255,255,255), 
                     bgGrid=(127,127,127), temperature=(0,255,255)) 
        txt = wx.FindWindowByName("graphSz_txt", p.panel["ml"])
        try:
            graphW, graphH = txt.GetValue().split(",")
            graphW = int(graphW)
            graphH = int(graphH)
        except:
            graphW = 1500
            graphH = 500
        mg = dict(l=100, r=50, t=int(graphH*0.5), b=50) # margin of graph
        cols = mg["l"] + mg["r"] + graphW 
        rows = mg["t"] + mg["b"] + graphH 
        img = np.zeros((rows, cols, 3), dtype=np.uint8) # prepare image to draw
        cvFont = cv2.FONT_HERSHEY_PLAIN
        fThck = 1
        fontThP = int(graphH * 0.03) 
        fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                 thresholdPixels=fontThP,
                                                 thick=fThck)
        fontThP = int(graphH * 0.05)
        lfThck = 2
        lfScale, lf_txtW, lf_txtH, lf_txtBl = getFontScale(cvFont,
                                                 thresholdPixels=fontThP,
                                                 thick=lfThck)
        dRad = int(graphH * 0.005)
       
        dtStr = [] # datetime string of each date
        v = dict(maxV=[], 
                 sumV=[], 
                 #peak=[], 
                 temperature=[]) # values of measures
        for fp in sorted(glob(path.join(p.inputFP, "*.npy"))):
        # go through all numpy files in the input folder
            data = np.load(fp) # load numpy data
            for di in range(len(data)):
                dtStr.append(data[di][0].split(" ")[0])
                v["maxV"].append(data[di][1])
                v["sumV"].append(data[di][2])
                #v["peak"].append(data[di][3])
                v["temperature"].append(data[di][4])
        mi = dict(maxV=np.min(v["maxV"]), 
                  sumV=np.min(v["sumV"]),
                  #peak=np.min(v["peak"]),
                  temperature=np.min(v["temperature"]))
        ma = dict(maxV=np.max(v["maxV"]), 
                  sumV=np.max(v["sumV"]),
                  #peak=np.max(v["peak"]),
                  temperature=np.max(v["temperature"]))
        div = dict(maxV=ma["maxV"]-mi["maxV"], 
                   sumV=ma["sumV"]-mi["sumV"],
                   #peak=ma["peak"]-mi["peak"],
                   temperature=ma["temperature"]-mi["temperature"])
        xInc = int(graphW / len(dtStr))
        bottY = mg["t"] + graphH
        
        ### draw x & y axis line
        pt1 = (mg["l"], 0)
        pt2 = (mg["l"], mg["t"]+graphH)
        pt3 = (cols, pt2[1])
        cv2.line(img, pt1, pt2, color["etc"], 1)
        cv2.line(img, pt2, pt3, color["etc"], 1) 

        ### draw 1.0 and 0.5 line
        for i in range(2):
            num = 0.5 * (i+1)
            y = mg["t"] + int(graphH*(1-num))
            pt1 = (mg["l"], y)
            pt2 = (cols, y) 
            cv2.line(img, pt1, pt2, color["etc"], 1)
            tx = pt1[0] - txtW*3
            cv2.putText(img, "%.1f"%(num), (tx, pt1[1]), cvFont, 
                        fontScale=fScale, color=color["etc"], thickness=fThck)

        ### draw data
        for di in range(len(dtStr)):
            x = mg["l"] + xInc * di
            cv2.line(img, (x,0), (x,mg["t"]+graphH), color["bgGrid"], 1)
            ys = []
            for i, k in enumerate(v.keys()):
                # normalize
                _val = (v[k][di] - mi[k]) / div[k]

                # get y-coord.
                ys.append(bottY - int(_val * graphH))

                if di > 0:
                    ### draw
                    pt1 = (prevX, prevYs[i])
                    pt2 = (x, ys[i])
                    cv2.line(img, pt1, pt2, color[k], 1)
                    cv2.circle(img, pt2, dRad, color[k], -1)
            ### draw date of the data
            _txt = dtStr[di][5:].replace("-","")
            ty = rows - mg["b"] + int((txtH+txtBl)*1.2)
            if di % 2 == 1: ty += int((txtH+txtBl)*1.2)
            cv2.putText(img, _txt, (x, ty), cvFont, fontScale=fScale, 
                        color=color["etc"], thickness=fThck)

            prevX = x
            prevYs = ys

        ### draw label and its max values 
        for i, k in enumerate(v.keys()):
            ty = int((i+1) * (lf_txtH+lf_txtBl))
            if k == "temperature":
                _txt = "%s, %.1f-%.1f"%(k, mi[k], ma[k])
            else:
                _txt = "%s, %i-%i"%(k, mi[k], ma[k])
            cv2.putText(img, _txt, (0, ty), cvFont, fontScale=lfScale,
                        color=color[k], thickness=lfThck)

        q2m.put(("displayMsg", "storing image ..",), True, None)
        q2m.put(("finished", img), True, None)
        
    #---------------------------------------------------------------------------
  
    def graphAOSN1610(self, q2m):
        """ graphs of N1610 experiments in 2022
        
        Args: 
            q2m (queue.Queue): Queue to send data to main thread.
         
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent
        txt = wx.FindWindowByName("graphSz_txt", p.panel["ml"])
        try:
            graphW, graphH = txt.GetValue().split(",")
            graphW = int(graphW)
            graphH = int(graphH)
        except:
            graphW = 1500
            graphH = 500

    #---------------------------------------------------------------------------
  
    def graphV2020(self):
        """ Draw graph for ant-virus study by Cremer and Viljakainen (2020)

        Args: None
        
        Returns: None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent
        '''
        rSz = [-1, -1]
        try:
            objW = wx.FindWindowByName("imgSavResW_txt", p.panel["bm"])
            objH = wx.FindWindowByName("imgSavResH_txt", p.panel["bm"])
            rSz = [int(objW.GetValue()), int(objH.GetValue())]
        except:
            pass
        if -1 in rSz: rSz = [4000, 3000] # default result image size
        '''
        rSz = [3000, 2250]
        # init result image array
        rsltImg = np.zeros((rSz[1], rSz[0], 3), dtype=np.uint8)
        rsltImg[:,:] = (255,255,255)
        ### set font scales
        cvFont = cv2.FONT_HERSHEY_DUPLEX
        fontTh = [2, 2, 3] # font thickness
        fontScale = [0.1, 0.1, 0.2]
        fontHeights = [0, 0, 0]
        lblTargetSz = [rSz[1]*0.011, rSz[1]*0.0125, rSz[1]*0.015]
        lbl = "Test"
        flags = [False, False, False]
        while False in flags:
            for i in range(len(fontScale)):
                if flags[i]: continue
                (lblW, lblH), bl = cv2.getTextSize(lbl, 
                                                   cvFont,
                                                   fontScale[i], 
                                                   1)
                if lblH >= lblTargetSz[i]:
                    flags[i] = True
                    fontHeights[i] = lblH
                else:
                    fontScale[i] += 0.1
        arcType = 1 # 0: type of connecting arc with straight line 
              # 1: simple arc line type
        maxRad = min(rSz) / 2 # maximum radius of circle in this panel
        icRad = int(maxRad * 0.6) # radius of inner circle of circular graph,
          # where virus presence line will be drawn.
        popArcRad = int(maxRad*0.65) # radius of population arc
        spArcRad = int(maxRad*0.8) # radius of species arc
        baseLLen = int(maxRad * 0.01) # base length of straight line 
          # (of virus presence), line length increases stepwise 
          # with this base length
        ct = (int(rSz[0]*0.38), int(rSz[1]/2)) # center point of panel
        cntVP = 0 # for counting overall virus presence circles
        vCR = int(rSz[1]*0.01) # radius of virus presnece circle 
        vpPt1 = {} # This dictionary will have dictionaries with keys of 
          # virus labels (such as 'LHUV-1'),
          # and it'll have two nested lists of points (where center of virus 
          # presence circle is) in each ant species.
          # e.g.) "vpPt1['LHUV-1'][0][1]" has point of the second 
          # virus (LHUV-1 virus) presence circle of the first ant species.
        vpDeg = {} # Similar structure with vpPt1,
          # but it's degree of rotation of vpPt1 points
        for ri in range(self.strData.shape[0]):
            vl = self.strData[ri,0] # virus label 
            vpPt1[vl] = []
            vpDeg[vl] = []
            ### append list for each ant species
            for si in range(self.numSpecies): 
                vpPt1[vl].append([])
                vpDeg[vl].append([])
        
        ##### [begin] drawing graph -----
        ### calculate positions of virus presence circles
        ###   & draw some base parts of graph
        for si in range(self.numSpecies):
            sAng = [-1, -1] # species arc start and end angle
            for pi in range(self.numPopulations):
                pAng = [-1, -1] # population arc start and end angle
                for ri in range(self.numData.shape[0]): # row (= virus)
                    vl = self.strData[ri,0] # virus label 
                    cl = self.strData[ri,1] # classification label  
                    ci = si*self.numPopulations + pi # column index
                      # column = a population of an ant species
                    if self.numData[ri,ci] == 0: # virus is not present
                        continue # to the next virus
                    
                    deg = cntVP * self.vpDeg # degree to rotate for 
                      # drawing virus presence circle
                    deg += 180 # starting from right, instead of left side
                    cntVP += 1 # counting overall virus presences 
                    
                    ### points for virus presence
                    pt1 = rot_pt((ct[0]+icRad, ct[1]), 
                                 ct, 
                                 deg) # center of circle
                    vpPt1[vl][si].append(pt1) # store coordinate
                    vpDeg[vl][si].append(deg)
                    
                    if pAng[0] == -1: pAng[0] = copy(deg)
                    pAng[1] = copy(deg)
                    if sAng[0] == -1: sAng[0] = copy(deg) 
                    sAng[1] = copy(deg) 
                        
                if not -1 in pAng:
                    if pAng[0] == pAng[1]: pAng[1] += 1
                    ### draw population arc
                    pCol = self.pColor[si][pi]
                    _miA = min(pAng) - abs(self.vpDeg)/2
                    _maA = max(pAng) + abs(self.vpDeg)/2
                    drawEllipse(rsltImg, ct, (popArcRad, popArcRad), 0,
                                -_miA, -_maA, pCol, thickness=-1)
                    ### draw line separating each population arc
                    for _ang in [_miA, _maA]:
                        _x, _y = rot_pt((ct[0]+popArcRad,ct[1]), ct, _ang)
                        _th = int(rSz[1]*0.003)
                        cv2.line(rsltImg, ct, (_x,_y), (255,255,255), _th)
                    ### draw text of population label
                    popLabel = self.colTitles[self.numericDataIdx[ci]]
                    popLabel = popLabel.split("[")[1].replace("]","")
                    _x, _y = rot_pt((ct[0]+icRad+int(rSz[1]*0.04), ct[1]), 
                                     ct, 
                                     (pAng[0]+(pAng[1]-pAng[0])/2))
                    (lblW, lblH), bl = cv2.getTextSize(popLabel, 
                                                       cvFont,
                                                       fontScale[1], 
                                                       fontTh[1])
                    if _x <= ct[0]: _x -= int(lblW*0.9) 
                    else: _x -= int(lblW*0.2)
                    if _y > ct[1]: _y += int(lblH*0.5)
                    else: _y -= lblH
                    #_col = tuple(pCol)
                    _col = (0,0,0)
                    '''
                    cv2.putText(rsltImg, popLabel, (_x, _y), cvFont,
                                fontScale[1], color=_col, thickness=fontTh[1])
                    '''
                    #_fontName = "NimbusSans-Italic"
                    _fontName = "LiberationMono-Regular.ttf"
                    rsltImg = drawTxtWithPIL(rsltImg, popLabel, (_x, _y),
                                             _fontName,
                                             fontSz=40, col=_col) 
            if sAng != [-1, -1]:
                ### draw species arc
                _thi = int(rSz[1]*0.005)
                drawEllipse(rsltImg, ct, (spArcRad, spArcRad), 0,
                            -sAng[0], -sAng[1], self.sColor[si], thickness=_thi)
                ### draw text of species label
                spLabel = self.colTitles[self.numericDataIdx[ci]]
                spLabel = spLabel.split("[")[0]
                _x, _y = rot_pt(
                                (ct[0]+spArcRad+int(rSz[1]*0.005), ct[1]), 
                                ct, 
                                (sAng[0]+(sAng[1]-sAng[0])/2)
                                )
                (lblW, lblH), bl = cv2.getTextSize(spLabel, 
                                                   cvFont,
                                                   fontScale[2], 
                                                   fontTh[2])
                if _x <= ct[0]: _x -= int(lblW*0.8)
                else: _x -= int(lblW*0.1)
                if _y > ct[1]: _y += lblH
                else: _y -= int(lblH*2)
                #_col = self.sColor[si]
                _col = (50,50,50)
                '''
                cv2.putText(rsltImg, spLabel, (_x, _y), cvFont,
                            fontScale[2], color=_col, thickness=fontTh[2])
                '''
                rsltImg = drawTxtWithPIL(rsltImg, spLabel, (_x, _y),
                                         "NimbusSans-Italic",
                                         fontSz=55, col=_col)
                

        ### [begin] draw connecting lines & virus presence circles, 
        ###   except viruses occurred in multiple ant speceis
        lLen = []
        for si in range(self.numSpecies):
            lLen.append(baseLLen) # Length of straight part 
              # of connecting line. This will increase as more connecting
              # lines are drawn in each species
        ### dicts for viruses which occurred multiple ant species
        vMSpPt1 = {} # pt1 
        vMSpDeg = {} # degree 
        vMSpCl = {} # classification
        connLinTh = int(rSz[1]*0.002) # connecting line thickness
        connLinTh2 = int(rSz[1]*0.004) # connecting line thickness
          # for viruses present in multiple ant species
        for ri in range(self.strData.shape[0]):
        # go through each virus
            vl = self.strData[ri,0] # virus label
            cl = self.strData[ri,1] # classification label 
            cntLst = [] 
            cntAll = 0
            cntSp = 0
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                cntAll += cnt
                if cnt > 0: cntSp += 1
            
            if cntSp > 1: # if virus occurred in more than one species
                vMSpPt1[vl] = vpPt1[vl]
                vMSpDeg[vl] = vpDeg[vl]
                vMSpCl[vl] = cl
                continue # don't draw it in this section
            
            arcPt1 = None
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vpPt1[vl][si][i]
                    pt2 = rot_pt(
                            (ct[0]+icRad-vCR-lLen[si], ct[1]), 
                            ct, 
                            vpDeg[vl][si][i]) # where connecting line ends
                    if arcType == 0:
                    # graph with arcs + straight lines
                        if cntAll > 1:
                        # there're more than one presences of this virus 
                            ### draw straight line part of connecting line
                            _col = self.sColor[si]
                            cv2.line(rsltImg, pt1, pt2, _col, connLinTh)
                            # store pt2 
                            if arcPt1 == None: arcPt1 = copy(pt2)
                    else:
                    # graph with simple arcs 
                        if i > 0:
                            ### draw arc between virus presence in this species
                            _s = pt1
                            _e = vpPt1[vl][si][i-1]
                            retL = []
                            for sign in [-1, 1]:
                                sag = int(icRad/4)*sign # sagitta
                                ret = convertSagittaArc2AngArc(_s, _e, sag)
                                _x, _y = ret[-1] # pt3 in returned values
                                # measure distance to the graph center
                                dist = np.sqrt((_x-ct[0])**2 + (_y-ct[1])**2) 
                                retL.append((dist, ret))
                            pt1 = retL[0][1][-1]
                            pt2 = retL[1][1][-1]
                            ret = sorted(retL)[0][1] # use arc, which is closer
                              # to the graph center
                            
                            ### to make all arcs inward to the graph center
                            if ret[3] < 0:
                                _sa = convt_180_to_360(ret[2])
                                _ea = convt_180_to_360(ret[3])
                            else:
                                _sa = ret[2]
                                _ea = ret[3]

                            # get the last one; 
                            #   the darkest color among population colors
                            col = copy(self.pColor[si][-1])
                            if si == 0:
                                _adj = -80 # ! reviewer wanted darker line,
                                    # but green (si==0) color was too dark.
                            else: 
                                _adj = -100 
                            col = (min(max(0, col[0]+_adj), 255),
                                   min(max(0, col[1]+_adj), 255),
                                   min(max(0, col[2]+_adj), 255))
                            print(si, col)
                            drawEllipse(rsltImg, ret[0], (ret[1], ret[1]), 0,
                                        _sa, _ea, col, connLinTh)
                if arcType == 0:
                    if cntAll > 1 and cnt > 0:
                        lLen[si] += baseLLen
                        ### draw arc part of the connecting line
                        ret = convertSagittaArc2AngArc(arcPt1, pt2, 
                                                       sagitta=int(icRad/4))
                        drawEllipse(rsltImg, ret[0], (ret[1], ret[1]), 0,
                                    ret[2], ret[3],
                                    color = self.sColor[si],
                                    thickness = connLinTh)
       
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vpPt1[vl][si][i]
                    # draw virus presence circle
                    cv2.circle(rsltImg, pt1, vCR, self.cColor[cl], -1) 
        ### [end] draw connecting lines & virus presence circles
        
        ### [begin] draw connecting line & virus presence circles,
        ###   which present in multiple ant species
        #lLen = max(lLen) + baseLLen*4 # starting line length for these viruses
        len4ArcCt= int(rSz[1]*0.02) # length to calculate center point 
          # for DrawArc
        for vl in vMSpPt1.keys():
            arcPt1 = None
            pts4isa = [] # list of points to draw arc line across ant species 
            for si in range(self.numSpecies):
                cnt = len(vMSpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vMSpPt1[vl][si][i]
                    if i > 0:
                        ### draw arc line between 
                        ###   virus presences in this species
                        ret = convertSagittaArc2AngArc(pt1, pPt1, 
                                                       sagitta=int(icRad/3))
                        drawEllipse(rsltImg, ret[0], (ret[1], ret[1]), 0,
                                    ret[2], ret[3],
                                    color = (0, 0, 0),
                                    thickness = connLinTh2)
                        # store middle point of arc line
                        #   to draw arc line between ant species
                        pts4isa.append(ret[4])
                    # store pt1 for the next loop
                    pPt1 = copy(pt1)
                if cnt == 1: pts4isa.append(pt1)
            ### draw arc line between virus presences across ant species
            for i in range(1, len(pts4isa)):
                pt1 = pts4isa[i-1]
                pt2 = pts4isa[i]
                ret = convertSagittaArc2AngArc(pt1, pt2, 
                                               sagitta=-int(icRad/5))
                drawEllipse(rsltImg, ret[0], (ret[1], ret[1]), 0,
                            ret[2], ret[3],
                            color = (0, 0, 0),
                            thickness = connLinTh2)
            len4ArcCt += baseLLen 
         
        for vl in vMSpPt1.keys():
            for si in range(self.numSpecies):
                for i in range(len(vMSpPt1[vl][si])):
                # virus presences in species
                    pt1 = vMSpPt1[vl][si][i]
                    ### draw virus presence circle
                    _col = self.cColor[vMSpCl[vl]]
                    cv2.circle(rsltImg, pt1, vCR, _col, -1) 
        ### [end] draw connecting line & virus presence circles
        ##### [end] drawing graph ----- 
         
        ##### [begin] drawing legend ----- 
        _x1 = int(rSz[0]*0.74)
        _x2 = int(rSz[0]*0.9)
        _y = int(rSz[1]*0.05)
        _yMargin = int(rSz[1] * 0.01)
        _yInc = fontHeights[1] + _yMargin 
        vlR = {}
        clR = {}
        
        # temporary strData, sorted with virus label
        sD = self.strData[self.strData[:,0].argsort(axis=0)]
        for cl in list(np.unique(sD[:,1])):
        # go through each classification
            ### write classification label
            '''
            cv2.putText(rsltImg, cl, (_x1, _y), cvFont, fontScale[0],
                        color=self.cColor[cl], thickness=fontTh[0])
            '''
            _col = self.cColor[cl]
            _col = (_col[2], _col[1], _col[0])
            rsltImg = drawTxtWithPIL(rsltImg, cl, (_x1, _y),
                                     "NimbusSans-Regular", fontSz=40, col=_col)
            if len(self.clR) == 0: # first time to draw graph
                (lblW, lblH), bl = cv2.getTextSize(cl, 
                                                   cvFont,  
                                                   fontScale[0], 
                                                   1)
                # store rect of text (classification in legend)
                clR[cl] = (_x1, _y-lblH, _x1+lblW, _y)
            ### write virus label
            # set text color for classification of the virus
            for ri in range(sD.shape[0]): # row (= virus)
                if cl == sD[ri,1]:
                    vl = sD[ri,0]
                    # write virus label
                    '''
                    cv2.putText(rsltImg, vl, (_x2, _y), cvFont, fontScale[1],
                                color=self.cColor[cl], thickness=fontTh[1])
                    '''
                    rsltImg = drawTxtWithPIL(rsltImg, vl, (_x2, _y),
                                             "NimbusSans-Regular",
                                             fontSz=40, col=_col)
                    if len(self.vlR) == 0: # first time to draw graph
                        (lblW, lblH), bl = cv2.getTextSize(cl, 
                                                           cvFont, 
                                                           fontScale[0], 
                                                           1)
                        # store rect of text (virus label in legend)
                        vlR[cl] = (_x2, _y-lblH, _x2+lblW, _y)
                    _y += _yInc # increase y-coordinate
        ##### [end] drawing legend -----
        
        if len(self.vpPt) == 0: # first time to draw graph
            self.vCR = vCR # store radius of virus presence circle
            self.vpPt = vpPt1 # store virus presence points
            self.vlR = vlR # store rects of virus labels in legend
            self.clR = clR # store rects of classification labels in legend
        
        self.storeGraphImg(rsltImg, p.pi["mp"]["sz"])

    #---------------------------------------------------------------------------
  
    def zoomNStore(self, img=None, aImg=None):
        """ Zoom in/out and store graph image 

        Args:
            img (None/numpy.ndarray): Graph image to zoom in/out.
            aImg (None/numpy.ndarray): additional image to the graph
                                         such as floating legend
        
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        gi = self.graphImgIdx 
        rat = self.graphImg[gi]["ratDisp2OrigImg"]

        if img is None:
            origImgFP = "tmp_origImg%i.png"%(self.graphImgIdx)
            img = cv2.imread(origImgFP)
        if aImg is None and "additionalImg" in self.graphImg[gi].keys():
            origAImgFP = "tmp_origImg%i_a.png"%(self.graphImgIdx)
            aImg = cv2.imread(origAImgFP)

        if rat != 1.0:
            if rat > 1.0: _interpolation = cv2.INTER_CUBIC
            else: _interpolation = cv2.INTER_AREA
            
            ### resize the graph image
            iSz = (int(img.shape[1]*rat), int(img.shape[0]*rat))
            img = cv2.resize(img, iSz, interpolation=_interpolation)

            if not aImg is None:
            # there's an additional image
                ### resize the additional image
                aISz = (int(aImg.shape[1]*rat), int(aImg.shape[0]*rat))
                aImg = cv2.resize(aImg, aISz, interpolation=_interpolation)
        
        #else:
        #    iSz = (img.shape[1], img.shape[0])

        # store the resized graph image
        self.graphImg[gi]["img"] = convt_cvImg2wxImg(img)
        if not aImg is None:
            # store the resized additional image 
            self.graphImg[gi]["additionalImg"] = convt_cvImg2wxImg(aImg)

    #---------------------------------------------------------------------------
  
    def procUI(self, ui, mp=None):
        """ Process user interaction 

        Args:
            ui (str): User interaction.
            mp (None/tuple): Position of mouse or Slider position (mp[0] 
                             in this case, mp[1] will be -1)
        
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        if self.graphImgIdx == -1: return

        p = self.parent
        x = mp[0]
        y = mp[1]
        pSz = p.pi["mp"]["sz"]
        gImg = self.graphImg[self.graphImgIdx]
        try: proc = gImg["proc"]
        except: proc = ""
        offset = gImg["offset"]
        iSz = gImg["img"].GetSize()

        if ui == "mouseMove":
        # mouse pointer was moved
            if p.eCase in self.c2showInfoOnMouseMove: 

                if p.eCase == "L2020CSV2":
                    if "valueData" in gImg.keys():
                        dataLst = gImg["valueData"]
                    else:
                        dataLst = []
                elif p.eCase in ["aos", "anVid"]:
                    if "intensity" in proc.lower() or \
                      "dist" in proc.lower() or \
                      proc in ["spAGridActivity", "spAGridHeat", 
                               "motionSpread"]:
                        dataLst = gImg["bData"]
                    elif "PSD" in proc:
                        dataLst = gImg["fsPeriod"]
                    else:
                        dataLst = [] # no drawing 

                if dataLst != []:
                    ### line where the current mouse position is
                    pen = wx.Pen((255, 255, 255), 1)
                    self.interactiveDrawing["drawLine"] = dict(
                                                   lines = [(x, 0, x, pSz[1])],
                                                   pens = [pen]
                                                   )
                    x = x - offset[0]
                    y = y - offset[1]
                    if x < 0 or x > iSz[0]:
                        self.interactiveDrawing = {}
                    else: 
                        txt = ""
                        txtCol = wx.Colour(200,0,0)
                        ##### [begin] draw text about data -----
                        if "mg" in gImg.keys(): mg = gImg["mg"]

                        if p.eCase == "L2020CSV2":
                            # data/frame index
                            idx = round(x / iSz[0] * len(dataLst))
                            idx = min(len(dataLst)-1, max(0, idx))
                            self.fIdx = idx # store the data/frame index
                            txt = "Idx:%i, mNND:%s, dispersal:%s"%(idx, 
                                           dataLst[idx][0], dataLst[idx][1]) 
                            txtCol = wx.Colour(200,200,200) 
 
                        if "intensity" in proc.lower() or \
                          "dist" in proc.lower() or \
                          proc in ["spAGridActivity", "spAGridHeat", 
                                   "motionSpread"]:
                            if p.eCase == "aos":
                                nDataInRow = gImg["nDataInRow"]
                                imgH = gImg["imgSz"][1]
                                bD_dt = gImg["bD_dt"]
                                ### calculate index for timestamp
                                idx = int(x / iSz[0] * nDataInRow)
                                days = len(gImg["days"])
                                dayH = imgH / days
                                pDays = int(y / iSz[1] * imgH / dayH)
                                idx += pDays * nDataInRow 
                                idx = min(len(bD_dt)-1, max(0, idx))
                                ### text to display
                                if proc == "localIntensity":
                                    for key in dataLst.keys():
                                        txt += "%.3f/ "%(dataLst[key][idx])
                                else:
                                    txt = "%s, "%(dataLst[idx])
                                txt += "[%s]"%(str(bD_dt[idx]))

                            elif p.eCase == "anVid":
                                roiI = int(y / gImg["roiGraphHght"])
                                roiK = f'roi{roiI:02d}'
                                if roiK in gImg["nDataInRow"].keys():
                                    nDataInRow = gImg["nDataInRow"][roiK]
                                    imgH = gImg["imgSz"][1]
                                    bD_dt = gImg["bD_dt"]
                                    ### calculate index for timestamp
                                    idx = int(x / iSz[0] * nDataInRow)
                                    idx = min(len(bD_dt[roiK])-1, max(0, idx))
                                    ### text to display
                                    txt = "%s, "%(dataLst[roiK][idx])
                                    txt += "[%s]"%(str(bD_dt[roiK][idx]))
                        
                        elif proc in ["spAGridActivity", "spAGridHeat"]:
                            ### get datetime to display
                            bD_dt = gImg["bD_dt"]
                            idx = int(x / iSz[0] * len(bD_dt))
                            idx = min(len(bD_dt)-1, max(0, idx))
                            txt = "[%s]"%(str(bD_dt[idx]))
                            ### get grid index to display
                            spAGrid = gImg["spAGrid"]
                            nCells = spAGrid["rows"] * spAGrid["cols"]
                            idx = int(y / (iSz[1] / nCells))
                            rIdx = int(idx / spAGrid["cols"])
                            cIdx = int(idx % spAGrid["cols"])
                            txt += " [%i][%i]"%(rIdx, cIdx)

                        elif "PSD" in proc:
                            imgW = gImg["imgSz"][0]
                            #idx = round(x/iSz[0]*imgW) - mg["l"]
                            idx = round(x/iSz[0]*imgW) - mg["l"]
                            if idx < 0 or idx >= len(dataLst): txt = ""
                            else: txt = "%s"%(dataLst[idx])

                        dc = wx.ClientDC(p)
                        w, h = dc.GetTextExtent(txt)
                        tx, ty = mp
                        if tx > pSz[0]/2: tx = tx - w - 5
                        else: tx += 10 
                        ty -= h
                        self.interactiveDrawing["drawText"] = dict(
                                                    textList=[txt],
                                                    coords=[(tx,ty)],
                                                    foregrounds=[txtCol]
                                                    )
                        ##### [end] draw text about data -----
            
            if p.eCase == "aos" and len(p.flags["drawingROI"]) > 0:
            # drawing ROI rect in aos 
                ix, iy = p.panel["mp"].mousePressedPt
                if ix != -1 and iy != -1:
                    pen = wx.Pen((255, 0, 0), 2)
                    brush = wx.Brush((0, 0, 0), wx.TRANSPARENT)
                    self.interactiveDrawing["drawRectangle"] = [
                                            dict(x1=ix, y1=iy, x2=x, y2=y, 
                                                 pen=pen, brush=brush)
                                            ]
                    if list(p.flags["drawingROI"].keys())[0] == "spAGrid":
                    # drawing a grid on nest area for spatial-analysis
                        w = wx.FindWindowByName("spAGridRows_cho", 
                                                p.panel["ml"])
                        rows = int(w.GetString(w.GetSelection())) 
                        w = wx.FindWindowByName("spAGridCols_cho", 
                                                p.panel["ml"])
                        cols = int(w.GetString(w.GetSelection())) 
                        _lines = []
                        _pens = []
                        gW = int((max(ix, x)-min(ix, x)) / cols)
                        gH = int((max(iy, y)-min(iy, y)) / rows)
                        for r in range(1, rows):
                            ry = iy + r*gH
                            _lines.append((ix, ry, x, ry))
                            _pens.append(pen)
                        for c in range(1, cols):
                            cx = ix + c*gW
                            _lines.append((cx, iy, cx, y)) 
                            _pens.append(pen)
                        self.interactiveDrawing["drawLine"] = dict(
                                                   lines = _lines,
                                                   pens = _pens
                                                   )
        
        elif ui == "rightClick":
        # mouse right button clicked
            if p.eCase == "L2020CSV2":
                if "drawBitmap" in self.interactiveDrawing.keys():
                    del self.interactiveDrawing["drawBitmap"]
                else:
                    img = p.vRW.currFrame
                    rat = img.shape[0]/img.shape[1]
                    w = int(p.pi["mp"]["sz"][0]/2)
                    h = int(w * rat)
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
                    x = mp[0]
                    y = mp[1]
                    if x > p.pi["mp"]["sz"][0]/2: x -= w
                    bmp = convt_cvImg2wxImg(img, toBMP=True)
                    self.interactiveDrawing["drawBitmap"] = [
                                                    dict(bitmap=bmp, x=x, y=y)
                                                    ]

        elif ui == "navSlider":
        # navigation slider was moved
            if p.eCase == "aos":
                if proc != "heatmap": return
                if x >= len(gImg["bData"]): return
                ### navigation slider moved, 
                ###   draw corresponding motion data points
                bdi = x # here is the slider position
                dLen = len(gImg["bData"])
                self.interactiveDrawing["drawCircle"] = []
                r = max(1, int(min(iSz) * 0.003))
                pen = wx.Pen((0, 0, 0), 0, wx.TRANSPARENT)
                brush = wx.Brush((0, 255, 255))
                for i in range(len(gImg["bData"][bdi])):
                    x, y = gImg["bData"][bdi][i]
                    x = int(x * gImg["ratDisp2OrigImg"])
                    y = int(y * gImg["ratDisp2OrigImg"])
                    self.interactiveDrawing["drawCircle"].append(
                                    dict(x=x, y=y, r=r, pen=pen, brush=brush)
                                    )
                ### timestamp of the current data position
                y = gImg["img"].GetSize()[1]-50
                dt = str(gImg["bD_dt"][bdi])
                col = wx.Colour(100,100,255)
                self.interactiveDrawing["drawText"] = dict(textList=[dt],
                                                           coords=[(5,y)],
                                                           foregrounds=[col])

    #---------------------------------------------------------------------------
    
#===============================================================================

if __name__ != "__main__":
    pass


