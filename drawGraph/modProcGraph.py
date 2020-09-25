# coding: UTF-8
"""
This module is for pyDrawGraph.
It has specific processing algorithms for different type of data, graph, etc. 

Dependency:
    Numpy (1.17), 
"""

import sys
from os import path
from glob import glob
from copy import copy
from time import time
from datetime import timedelta

import wx, cv2
import numpy as np

from modFFC import *
from modCVFunc import *

DEBUG = False

#===============================================================================

class ProcGraphData():
    """ Class for processing data and generate graph.
    
    Args:
        parent (wx.Frame): Parent frame
        inputFP (str): File path of data CSV file
    
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    
    def __init__(self, parent):
        if DEBUG: print("ProcGraphData.__init__()")

        ##### [begin] setting up attributes on init. -----
        self.parent = parent
        self.inputFP = parent.inputFP # input file path
        self.colTitles = [] # data column titles
        self.numData = None # Numpy array of numeric data
        self.strData = None # Numpy array of character data
        self.graphImg = []
        self.graphImgIdx = 0
        self.fonts = parent.fonts
        self.uiTask = {} # task from UI (such as to show one virus, class, etc)
        self.uiTask["showVirusLabel"] = None
        self.uiTask["showThisVirusOnly"] = None 
        self.uiTask["showThisClassOnly"] = None 
        ##### [end] setting up attributes on init. -----
    
    #---------------------------------------------------------------------------
  
    def initOnDataLoading(self, inputFP):
        """ init. process when input file was loaded

        Args:
            inputFP (str): Input data file path

        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.initData()")

        p = self.parent
        
        if p.graphType == "L2020": 
            ### store the first frame as the graph image at this initial stage 
            self.storeGraphImg(self.parent.vRW.currFrame.copy(), 
                               self.parent.pi["mp"]["sz"])

        elif p.graphType == "J2020":
            tmp = glob(path.join(inputFP, "*.csv")) # all file list
            fLst = dict(day=[], night=[])
            for i, fp in enumerate(tmp):
                fn = path.basename(fp)
                if int(fn.split("_")[5]) < 12: fLst["day"].append(fp)
                else: fLst["night"].append(fp)
            
            data = {} # data of each founding queen ants to show on graph
            data["day"] = {}
            data["night"] = {}
            for k in data.keys(): # day & night
                for fp in fLst[k]: # each file in the dat/night file list
                    fileH = open(fp, "r")
                    lines = fileH.readlines()
                    fileH.close()
                    fn = path.basename(fp)
                    tmp = fn.split("_") 
                    mmdd = tmp[3] + tmp[4] # month & date timestamp
                    data[k][mmdd] = dict(a00=[], a01=[], a02=[], a03=[])
                    for line in lines:
                        if line.startswith("frame-index,"): continue
                        items = line.split(",")
                        if len(items) < 13: continue
                        for ai in range(4):
                            ak = "a%02i"%(ai) # ant key
                            val = items[(ai+1)*3].strip().replace("\n","")
                            if val == "None": val = -1
                            else: val = int(val)
                            data[k][mmdd][ak].append(val)
                        if len(data[k][mmdd][ak]) == 1100: break
            self.data = data
            ### color set-up
            self.color = dict(a00 = dict(day=(0,0,255), night=(0,0,128)),
                              a01 = dict(day=(0,255,0), night=(0,128,0)),
                              a02 = dict(day=(255,0,0), night=(128,0,0)),
                              a03 = dict(day=(255,0,255), night=(128,0,128)))
   
    #---------------------------------------------------------------------------
  
    def storeGraphImg(self, img, sz):
        """ store graph image after resizing, if necessary 

        Args:
            img (numpy.ndarray): graph image 
            sz (tuple): size to display the graph image

        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.storeGraphImg()")

        print("[ProcGraphData.storeGraphImg] saving original image..")
        fn = "tmp_origImg%i.png"%(len(self.graphImg))
        cv2.imwrite(fn, img) # store original image as file
        self.graphImg.append({})
        
        ### resize the image
        print("[ProcGraphData.storeGraphImg] resizing image..")
        ratImg2DispImg = calcI2DIRatio(img, sz)
        if ratImg2DispImg != 1.0:
            iSz = (int(img.shape[1]*ratImg2DispImg), 
                   int(img.shape[0]*ratImg2DispImg))
            if ratImg2DispImg > 1.0:
                img = cv2.resize(img, iSz, interpolation=cv2.INTER_CUBIC)
            else:
                img = cv2.resize(img, iSz, interpolation=cv2.INTER_AREA)
        else:
            iSz = (img.shape[1], img.shape[0])
        
        ### store the graph images
        print("[ProcGraphData.storeGraphImg] storing image..")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wxImg = wx.Image(iSz[0], iSz[1])
        wxImg.SetData(img.tostring())
        self.graphImg[-1]["img"] = wxImg

        ### store thumbnail images
        print("[ProcGraphData.storeGraphImg] storing thumbnail image..")
        r = img.shape[0] / img.shape[1]
        w = int(self.parent.pi["mr"]["sz"][0]*0.4)
        h = int(w * r)
        self.graphImg[-1]["thumbnail"] = wxImg.Scale(w, h)

    #---------------------------------------------------------------------------
  
    def storeGraphData(self, data = []):
        """ Store data related to graph. 
        Assumes that corresponding graph is already stored in self.graphImg.

        Args:
            data (list): Graph related data. Could be multiple data.

        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.storeGraphData()")

        gIdx = len(self.graphImg) - 1
        if gIdx == -1: return
        for di in range(len(data)):
            d = data[di]
            fn = "tmp_data_%i_%i.csv"%(gIdx, di)
            if type(d) == np.ndarray:
                np.savetxt(fn, d, newline="\n", delimiter=",")
            elif type(d) == list:
                if self.parent.graphType == "L2020": 
                    colTitles = ['centroid', 'rect', 'rectL', 'bRects']
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
                                    line += str(_d[bi]).strip("[]") \
                                                       .replace(",","/")
                                    line += ", "
                                line = line.rstrip(", ")
                            else:
                                # other data are tuple of rect or coordinate
                                line += str(_d).strip("()").replace(",","/")
                            line += ", "
                        line = line.rstrip(", ") + "\n"
                        fh.write(line)
                    fh.close()
    
    #---------------------------------------------------------------------------
  
    def graphL2020(self, startFrame, endFrame, q2m, flag=""):
        """ Draw heatmap of ants' positions in structured/unstructured nest.
        
        Args:
            startFrame (int): Frame index to start of heatmap data.
            endFrame (int): Frame index to end of heatmap data.
            q2m (queue.Queue): Queue to send data to main thread.
            flag (str): Flag to indicate certain task such as 'save'
         
        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.graphL2020()")

        def findAntCols(fImg):
        # function to find ants by colors
            hsvImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HSV)
            fcRslt = cv2.inRange(hsvImg, hsvP["mi"][0], hsvP["ma"][0])
            for ci in range(1, 3):
                fcRslt = cv2.add(
                            fcRslt,
                            cv2.inRange(hsvImg, hsvP["mi"][ci], hsvP["ma"][ci])
                            )
            ### attempt to remove small erroneous noise 
            kernel = np.ones((3,3),np.uint8)
            fcRslt = cv2.morphologyEx(fcRslt, cv2.MORPH_OPEN, kernel)
            # threshold
            ret, img = cv2.threshold(fcRslt, 50, 255, cv2.THRESH_BINARY)
            return img
        
        def findAntBlobs(img):
        # function to find ant blobs & calculating some measures 
            rslt = {} # result dictionary to return
            rslt["centroid"] = getCentroid(img) # store centroid of image
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
                s = stats[li]
                if s[4] < aMinArea: continue
                # store large blob index
                bRects.append(list(s[:4]))
                if s[4] > aMinArea*4:
                    cxL += [s[0], s[0]+s[2]]
                    cyL += [s[1], s[1]+s[3]]
                else:
                    cxS += [s[0], s[0]+s[2]]
                    cyS += [s[1], s[1]+s[3]]
            rslt["bRects"] = bRects # store list of all blobs rect;(x, y, w, h)
            ### store bounding rect (x,y,w,h) of all blobs
            r = [min(cxL+cxS), min(cyL+cyS)]
            r += [max(cxL+cxS)-r[0], max(cyL+cyS)-r[1]]
            rslt["rect"] = tuple(r) 
            ### store bounding rect (x,y,w,h) of all large blobs
            r = [min(cxL), min(cyL)]
            r += [max(cxL)-r[0], max(cyL)-r[1]]
            rslt["rectL"] = tuple(r)
            return rslt 
        
        p = self.parent
        
        ### determine HSV ranges for detecting ant colors
        hsvP = dict(mi=[], ma=[]) 
        for ci in range(3):
        # 0: ant body color (blackish)
        # 1: color markers other than yellow color 
        # 2: yellow color marker
            for mLbl in ["Min", "Max"]:
                val = []
                for hLbl in ["H", "S", "V"]:
                    objName = "col%i%s%s_sld"%(ci, hLbl, mLbl)
                    obj = wx.FindWindowByName(objName, p.panel["ml"])
                    val.append(obj.GetValue())
                key = mLbl.lower()[:2]
                hsvP[key].append(tuple(val)) # store HSV parameter
        
        ### get ant minimum area
        obj = wx.FindWindowByName("aMinArea_txt", p.panel["ml"])
        aMinAreaTxtVal = obj.GetValue().strip()
        if aMinAreaTxtVal == "": aMinArea = 100; obj.SetValue("100")
        else: aMinArea = int(aMinAreaTxtVal)
        
        if p.debugging:
        # debugging
            obj = wx.FindWindowByName("debugFrameIntv_txt", p.panel["tp"])
            intvTxtVal = obj.GetValue().strip()
            if intvTxtVal == "": intv = 1; obj.SetValue("1")
            else: intv = int(intvTxtVal)
            for i in range(intv):
                p.vRW.getFrame(-1) # get frame image
            fImg = p.vRW.currFrame
            fImg = preProcImg(fImg, 2, 2)
            img = findAntCols(fImg)
            fImg[img==255] = (50, 50, 255)
            cv2.putText(fImg, "Frame: %i"%(p.vRW.fi), (5, fImg.shape[0]-20),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0,0,0),
                        thickness=1)
            pt1 = (10, 10)
            l = int(np.sqrt(aMinArea))
            pt2 = (pt1[0]+l, pt1[1]+l)
            # draw ant min. area
            cv2.rectangle(fImg, pt1, pt2, (30,30,30), -1) 
            cv2.putText(fImg, "ant min. area", (pt2[0]+10, pt2[1]),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=(0,0,0),
                        thickness=1)
            bInfo = findAntBlobs(img) # get ant blob info
            ### display blob rects
            for b in bInfo["bRects"]:
                pt1 = (b[0], b[1])
                pt2 = (b[0]+b[2], b[1]+b[3])
                cv2.rectangle(fImg, pt1, pt2, (0,255,0), 2)
            ### display entire rect
            r = bInfo["rect"]
            cv2.rectangle(fImg, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), 
                          (128,128,128), 2)
            ### display rect, bounding large blobs
            r = bInfo["rectL"]
            cv2.rectangle(fImg, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), 
                          (0,128,255), 2)
            # display centroid
            cv2.circle(fImg, bInfo["centroid"], 10, (200,0,0), -1) 
            # store the resultant image
            self.storeGraphImg(fImg, p.pi["mp"]["sz"])
            return
        
        # total number of frames
        nFrames = endFrame - startFrame + 1
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

        startTime = time()
        ### make array for heatmap
        shape = p.vRW.currFrame.shape
        hmArr = np.zeros((shape[0], shape[1]), 
                         dtype=np.int32) # init array for heatmap 
        bInfo = [] # Blob (connected component) information.
            # Each item is a list, representing info in a frame.
            # Items for a frame are rects (left, top, width, height) of found 
            #   blobs, which fit into the category of minimum area.
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
                q2m.put(("displayMsg", msg,), True, None)
            if fi == p.vRW.fi:
                fImg = p.vRW.currFrame
            else:
                p.vRW.getFrame(-1) # get frame image
                fImg = p.vRW.currFrame
            fImg = preProcImg(fImg, 2, 2)
            img = findAntCols(fImg)
            hmArr[img==255] += 1 # increase where colors are found
            bInfo.append(findAntBlobs(img)) # store ant blob info
        
        ### make base image to draw heat-map
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
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
        
        ##### [begin] coloring for heat-map -----
        maxV = np.max(hmArr)
        legX = 10 # init legend x
        legY = 10 # init legend y
        colSqLen = 20 # color square length in legend
        ### get indices of pixels for heatmap
        hmIdx = []
        nzRngCnt = 0
        for i in range(len(hmLvlRngs)):
            rng1 = hmLvlRngs[i][0]
            rng2 = hmLvlRngs[i][1]
            idx = np.logical_and(np.greater(hmArr, rng1),
                                 np.less_equal(hmArr, rng2))
            if np.sum(idx) == 0:
                hmIdx.append((False, idx))
            else:
                nzRngCnt += 1
                hmIdx.append((True, idx))
        ### set colors for each heatmap range
        hmCols = []
        _cnt = 0
        _cnt1 = np.ceil(nzRngCnt/2)
        _cnt2 = nzRngCnt - _cnt1
        for i in range(len(hmIdx)):
            if hmIdx[i][0]:
                if _cnt < _cnt1: # hmIdx increasing by one order of magnitude
                    c = (_cnt+1) * int(255/_cnt1)
                    hmCols.append((0,0,c))
                else: # numbers in the highest order of magnitude
                    c = 100 + ((_cnt-_cnt1+1) * int(155/_cnt2))
                    hmCols.append((0,c,c))
                _cnt += 1
            else:
                hmCols.append((0,0,0))
        
        for i in range(len(hmIdx)):
            idx = hmIdx[i][1]
            if hmIdx[i][0]: rsltImg[idx] = hmCols[i] # coloring heatmap
            ### draw legend
            legX = 10
            legY = 10 + (i*colSqLen + i*10)
            cv2.rectangle(rsltImg, 
                          (legX,legY), 
                          (legX+colSqLen,legY+colSqLen), 
                          hmCols[i], 
                          -1)
            rng1 = hmLvlRngs[i][0]
            rng2 = hmLvlRngs[i][1]
            msg = "Log %.3f (%i)"%(np.log10(rng1), rng1)
            msg += " - Log %.3f (%i)"%(np.log10(rng2), rng2)
            if hmIdx[i][0]:
                tCol = getConspicuousCol(hmBgCol)
                tCol = (tCol[2], tCol[1], tCol[0])
            else:
                tCol = (127,127,127)
            cv2.putText(rsltImg, 
                        msg, 
                        (int(legX+colSqLen*1.5), legY+colSqLen),
                        cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=1.2, 
                        color=tCol,
                        thickness=1)
        ##### [end] coloring for heat-map -----
        ### write frame range
        cv2.putText(rsltImg,
                    "Frame: %i - %s"%(startFrame, endFrame),
                    (5, rsltImg.shape[0]-20),
                    cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1.2, 
                    color=(255,255,255),
                    thickness=1)
        
        self.storeGraphImg(rsltImg, p.pi["mp"]["sz"])
        self.storeGraphData([hmArr, bInfo])
        ### store start and end frame
        self.graphImg[-1]["startFrame"] = startFrame
        self.graphImg[-1]["endFrame"] = endFrame
        q2m.put(("finished",), True, None)
   
    #---------------------------------------------------------------------------
  
    def graphJ2020(self, ai, q2m, flag=""):
        """ Draw change of founding queen ants' movements 
        
        Args:
            ai (int): Ant index.
            q2m (queue.Queue): Queue to send data to main thread.
            flag (str): Flag to indicate certain task such as 'save'
         
        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.graphJ2020()")
        
        p = self.parent
        data = self.data
        ak = "a%02i"%(ai) # ant key

        ### create array
        rows = [] 
        cols = 0
        for ti, tk in enumerate(data.keys()): # day & night
            _cols = 0
            _rows = 0
            for dk in sorted(data[tk].keys()): # each date
                _cols += len(data[tk][dk][ak])
                maxVal = max(data[tk][dk][ak])
                if maxVal > _rows: _rows = copy(maxVal)
            rows.append(_rows)
            if _cols > cols: cols = copy(_cols)
        msg = "[%s] creating array of size (%i, %i) ..."%(ak, sum(rows), cols)
        q2m.put(("displayMsg", msg,), True, None)
        # init array for graph
        rsltImg = np.zeros((sum(rows), cols, 3), dtype=np.uint8)
        rsltImg[:rows[0],:] = [255,255,255] # bg-color for day
        rsltImg[rows[0]:,:] = [200,200,200] # bg-color for night
        
        dLineThick = int(rsltImg.shape[1] * 0.001)
        ptRad = int(rsltImg.shape[0] * 0.005)
        ptThick = int(rsltImg.shape[0] * 0.0025)
        x = 0
        dKeys = list(sorted(data["day"].keys()))
        for dk in sorted(data["night"].keys()):
            if not dk in dKeys: dKeys.append(dk)
        for dk in dKeys: # each date
            msg = "[%s] processing date - %s.%s ..."%(ak, dk[:2], dk[2:])
            q2m.put(("displayMsg", msg,), True, None)
            xs = [x, x]
            for ti, tk in enumerate(data.keys()): # day & night
                if not dk in data[tk].keys(): continue
                endY = sum(rows[:ti+1])
                color = self.color[ak][tk]
                for i in range(1, len(data[tk][dk][ak])):
                    val = data[tk][dk][ak][i]
                    pVal = data[tk][dk][ak][i-1] # data of previous frame
                    if -1 in [val, pVal]: continue
                    y1 = endY - pVal
                    y2 = endY - val
                    pt1 = (xs[ti]-1, y1)
                    pt2 = (xs[ti], y2)
                    cv2.line(rsltImg, pt1, pt2, color, 1)
                    xs[ti] += 1
                ### display average and median value as empty & filled dot
                medVal = int(np.median(data[tk][dk][ak]))
                avgVal = int(np.average(data[tk][dk][ak]))
                midX = int(x + (xs[ti]-x)/2)
                cv2.circle(rsltImg, (midX, endY-avgVal), ptRad, color, ptThick) 
                cv2.circle(rsltImg, (midX, endY-medVal), ptRad, color, -1) 
                ### write average mov value 
                textY = 500 * (int(dk[2:])%2+1)
                if ti == 1: textY += rows[0]
                cv2.putText(rsltImg,
                            str(avgVal), 
                            (x, textY),
                            cv2.FONT_HERSHEY_PLAIN, 
                            fontScale=30, 
                            color=(0,0,0),
                            thickness=30)
            x = max(xs)
            # line to show end of the date
            cv2.line(rsltImg, (x,0), (x,endY), (128,128,128), dLineThick)
        
        msg = "[%s] storing & resizing the graph image ..."%(ak)
        q2m.put(("displayMsg", msg,), True, None)
        self.storeGraphImg(rsltImg, p.pi["mp"]["sz"])

        q2m.put(("finished",), True, None)
    
    #---------------------------------------------------------------------------
  
    def graphV2020(self, dc, flag):
        """ Draw graph for ant-virus study by Cremer and Viljakainen (2020)  
        
        Args:
            dc (wx.PaintDC): PaintDC to draw graph on.
            flag (str): Flag to indicate certain task such as 'save'
        
        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.graphV2020()")
        
        dcSz = dc.GetSize()
        arcType = 1 # 0: type of connecting arc with straight line 
              # 1: simple arc line type
        maxRad = min(dcSz) / 2 # maximum radius of circle in this DC 
        icRad = int(maxRad * 0.6) # radius of inner circle of circular graph,
          # where virus presence line will be drawn.
        popArcRad = int(dcSz[1]*0.03) # radius of population arc
        spArcRad = int(dcSz[1]*0.12) # radius of species arc
        baseLLen = int(dcSz[1] * 0.007) # base length of straight line 
          # (of virus presence), line length increases stepwise 
          # with this base length
        ct = (int(dcSz[0]*0.38), int(dcSz[1]/2)) # center point of panel
        cntVP = 0 # for counting overall virus presence circles
        fontSz = int(dcSz[1] * 0.01) # font size, depending on DC size
        fSzInc = int(dcSz[1] * 0.005)
        fonts = getWXFonts(fontSz, numFonts=3, fSzInc=fSzInc)
        emphFonts = getWXFonts(fontSz, 
                               numFonts=3,
                               fSzInc=fSzInc,
                               weight=wx.FONTWEIGHT_BOLD,
                               style=wx.FONTSTYLE_ITALIC, 
                               underline=False)
        vCR = int(dcSz[1]*0.01) # radius of virus presnece circle 
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
        dc.SetBackground(wx.Brush(wx.Colour(255,255,255)))
        dc.Clear()
        ### calculate positions of virus presence circles
        ###   & draw some base parts of graph
        for si in range(self.numSpecies):
            sS = (-1, -1) # species arc start point
            sE = (-1, -1) # species arc end point
            for pi in range(self.numPopulations):
                pS = (-1, -1) # population arc start point
                pE = (-1, -1) # population arc end point
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
                    
                    ### store population & species arc points
                    paPt = rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                  ct, 
                                  deg)
                    saPt = rot_pt((ct[0]+icRad+spArcRad, ct[1]), 
                                  ct, 
                                  deg)
                    if pS == (-1, -1):
                        pS = rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                     ct, 
                                     deg-self.vpDeg/2)
                        pSDeg = copy(deg)
                    pE = rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                 ct, 
                                 deg+self.vpDeg/2)
                    pEDeg = copy(deg)
                    if sS == (-1, -1):
                        sS = copy(saPt)
                        sSDeg = copy(deg)
                    sE = copy(saPt)
                    sEDeg = copy(deg)
                        
                if pS != (-1, -1) and pE != (-1, -1):
                    if pS == pE:
                        pEDeg = pSDeg + 1
                        pE= rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                    ct, 
                                    pEDeg)
                    ### draw population arc
                    pCol = self.pColor[si][pi]
                    lighterPCol = tuple(min(x+75, 255) for x in pCol)
                    dc.SetPen(wx.Pen(lighterPCol, 0, wx.TRANSPARENT))
                    dc.SetBrush(wx.Brush(lighterPCol))
                    if self.vpDeg > 0:
                        dc.DrawArc(pS[0], pS[1], pE[0], pE[1], ct[0], ct[1])
                    else:
                        dc.DrawArc(pE[0], pE[1], pS[0], pS[1], ct[0], ct[1])
                    ### draw text of population label
                    dc.SetFont(fonts[2])
                    dc.SetTextForeground(tuple(pCol))
                    popLabel = self.colTitles[self.numericDataIdx[ci]]
                    popLabel = popLabel.split("[")[1].replace("]","")
                    _x, _y = rot_pt((ct[0]+icRad+int(dcSz[1]*0.04), ct[1]), 
                                     ct, 
                                     pSDeg+(pEDeg-pSDeg)/2)
                    w, h = dc.GetTextExtent(popLabel) 
                    if _x <= ct[0]: _x -= w
                    if _y <= ct[1]: _y -= h
                    dc.DrawText(popLabel, _x, _y)
            if sS != (-1, -1) and sE != (-1, -1):
                ### draw species arc
                dc.SetPen(wx.Pen(self.sColor[si], int(dcSz[1]*0.01)))
                dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                if self.vpDeg > 0:
                    dc.DrawArc(sS[0], sS[1], sE[0], sE[1], ct[0], ct[1]) 
                else:
                    dc.DrawArc(sE[0], sE[1], sS[0], sS[1], ct[0], ct[1]) 
                ### draw text of species label
                dc.SetFont(emphFonts[2])
                dc.SetTextForeground(self.sColor[si])
                spLabel = self.colTitles[self.numericDataIdx[ci]]
                spLabel = spLabel.split("[")[0]
                _x, _y = rot_pt(
                        (ct[0]+icRad+spArcRad+int(dcSz[1]*0.005), ct[1]), 
                        ct, 
                        sSDeg+(sEDeg-sSDeg)/2
                        )
                w, h = dc.GetTextExtent(spLabel) 
                if _x <= ct[0]: _x -= w
                if _y <= ct[1]: _y -= h
                dc.DrawText(spLabel, _x, _y)

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
        connLinTh = int(dcSz[1]*0.002) # connecting line thickness
        connLinTh2 = int(dcSz[1]*0.004) # connecting line thickness
          # for viruses present in multiple ant species
        for ri in range(self.strData.shape[0]):
        # go through each virus
            vl = self.strData[ri,0] # virus label
            cl = self.strData[ri,1] # classification label 
            
            ### there's a specfic class to display 
            #   and this virus doesn't belong to it,
            #   continue without drawing
            if self.uiTask["showThisClassOnly"] != None:
                if not self.uiTask["showThisClassOnly"] == cl:
                    continue
            
            ### there's a specfic virus to display 
            #   and this virus is not,
            #   continue without drawing
            if self.uiTask["showThisVirusOnly"] != None:
                if not self.uiTask["showThisVirusOnly"] == vl:
                    continue
                        
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
                    pt2 = rot_pt((ct[0]+icRad-vCR-lLen[si], ct[1]), 
                                  ct, 
                                  vpDeg[vl][si][i]) # where connecting line ends
                    if arcType == 0:
                    # graph with arcs + straight lines
                        if cntAll > 1:
                        # there're more than one presences of this virus 
                            ### draw straight line part of connecting line
                            dc.SetPen(wx.Pen(self.sColor[si], connLinTh))
                            dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                            dc.DrawLine(pt1[0], pt1[1], pt2[0], pt2[1])
                            # store pt2 
                            if arcPt1 == None: arcPt1 = copy(pt2)
                    else:
                    # graph with simple arcs 
                        if i > 0:
                            ### draw arc between virus presence in this species
                            dc.SetPen(wx.Pen(self.sColor[si], connLinTh))
                            dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                            _pt = vpPt1[vl][si][i-1]
                            deg1 = vpDeg[vl][si][i]
                            deg2 = vpDeg[vl][si][i-1]
                            deg = deg1 + (deg2-deg1)/2
                            aCt = rot_pt((ct[0]+icRad, ct[1]),
                                          ct, 
                                          deg)
                            if self.vpDeg > 0: 
                                dc.DrawArc(pt1[0], pt1[1],
                                           _pt[0], _pt[1], 
                                           aCt[0], aCt[1])
                            else:
                                dc.DrawArc(_pt[0], _pt[1],
                                           pt1[0], pt1[1], 
                                           aCt[0], aCt[1])
                if arcType == 0:
                    if cntAll > 1 and cnt > 0:
                        lLen[si] += baseLLen
                        ### draw arc part of the connecting line
                        dc.SetPen(wx.Pen(self.sColor[si], connLinTh))
                        dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                        dc.DrawArc(arcPt1[0], arcPt1[1],
                                   pt2[0], pt2[1], 
                                   ct[0], ct[1])
       
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vpPt1[vl][si][i]
                    ### draw virus presence circle
                    dc.SetPen(wx.Pen('#000000', 1, wx.TRANSPARENT))
                    dc.SetBrush(wx.Brush(self.cColor[cl]))
                    dc.DrawCircle(pt1[0], pt1[1], vCR)
        ### [end] draw connecting lines & virus presence circles
        
        ### [begin] draw connecting line & virus presence circles,
        ###   which present in multiple ant species
        #lLen = max(lLen) + baseLLen*4 # starting line length for these viruses
        len4ArcCt= int(dcSz[1]*0.02) # length to calculate center point 
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
                        deg1 = vMSpDeg[vl][si][i-1]
                        deg2 = vMSpDeg[vl][si][i]
                        deg = deg1 + (deg2-deg1)/2
                        aCt = rot_pt((ct[0]+icRad-len4ArcCt, ct[1]),
                                      ct, 
                                      deg)
                        dc.SetPen(wx.Pen("#000000", connLinTh2))
                        dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
                        if self.vpDeg > 0:
                            dc.DrawArc(pt1[0], pt1[1], 
                                       pPt1[0], pPt1[1], 
                                       aCt[0], aCt[1])
                        else:
                            dc.DrawArc(pPt1[0], pPt1[1], 
                                       pt1[0], pt1[1], 
                                       aCt[0], aCt[1])
                        ### store middle point of arc line
                        ###   to draw arc line between ant species
                        dist = np.sqrt((pt1[0]-pPt1[0])**2+(pt1[1]-pPt1[1])**2)
                        deg = calc_line_angle(aCt, ct)
                        pt4isa = rot_pt((aCt[0]+dist/2, aCt[1]),
                                        aCt,
                                        deg)
                        pts4isa.append(pt4isa)
                    # store pt1 for the next loop
                    pPt1 = copy(pt1)
                    '''
                    ### draw straight line part of connecting line
                    dc.SetPen(wx.Pen("#000000", connLinTh2))
                    dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
                    dc.DrawLine(pt1[0], pt1[1], pt2[0], pt2[1])
                    if arcPt1 == None: arcPt1 = copy(pt2)
                    '''
                if cnt == 1: pts4isa.append(pt1)
            ### draw arc line between virus presences across ant species
            dc.SetPen(wx.Pen("#000000", connLinTh2))
            dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
            for i in range(1, len(pts4isa)):
                pt1 = pts4isa[i-1]
                pt2 = pts4isa[i]
                ptsDist = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
                minX = min(pt1[0], pt2[0]); maxX = max(pt1[0], pt2[0])
                minY = min(pt1[1], pt2[1]); maxY = max(pt1[1], pt2[1])
                ptsCt = (int(minX+(maxX-minX)/2), int(minY+(maxY-minY)/2))
                ptsDeg = calc_line_angle(pt1, pt2)
                ptsDeg += 90
                aCt = calc_pt_w_angle_n_dist(ptsDeg,
                                             int(ptsDist/2), 
                                             ptsCt[0],
                                             ptsCt[1], 
                                             flagScreen=True)
                dc.DrawArc(pt1[0], pt1[1], pt2[0], pt2[1], aCt[0], aCt[1])
            # draw arc part of the connecting line
            #dc.DrawArc(arcPt1[0], arcPt1[1], pt2[0], pt2[1], ct[0], ct[1])
            #lLen += baseLLen
            len4ArcCt += baseLLen 
         
        for vl in vMSpPt1.keys():
            for si in range(self.numSpecies):
                for i in range(len(vMSpPt1[vl][si])):
                # virus presences in species
                    pt1 = vMSpPt1[vl][si][i]
                    ### draw virus presence circle
                    dc.SetPen(wx.Pen('#000000', 1, wx.TRANSPARENT))
                    dc.SetBrush(wx.Brush(self.cColor[vMSpCl[vl]]))
                    dc.DrawCircle(pt1[0], pt1[1], vCR)
        ### [end] draw connecting line & virus presence circles
        ##### [end] drawing graph ----- 
         
        ##### [begin] drawing legend ----- 
        _x1 = int(dcSz[0]*0.76)
        _x2 = int(dcSz[0]*0.92)
        _y = int(dcSz[1]*0.05)
        _yInc = int(fonts[1].GetPixelSize()[1] + dcSz[1]*0.001)
        vlR = {}
        clR = {}
        
        # temporary strData, sorted with virus label
        sD = self.strData[self.strData[:,0].argsort(axis=0)]
        for cl in list(np.unique(sD[:,1])):
        # go through each classification
            onlyCL = self.uiTask["showThisClassOnly"]
            ### write classification label
            if (onlyCL != None and onlyCL == cl):
            # if it's showing only this class
                dc.SetFont(emphFonts[0]) # emphasize this label
            else:
                dc.SetFont(fonts[0])
            dc.SetTextForeground("#999999")
            dc.DrawText(cl, _x1, _y)
            if len(self.clR) == 0: # first time to draw graph
                w, h = dc.GetTextExtent(cl)
                # store rect of text (classification in legend)
                clR[cl] = (_x1, _y, _x1+w, _y+h)
            ### write virus label
            dc.SetFont(fonts[1])
            # set text color for classification of the virus
            dc.SetTextForeground(self.cColor[cl])
            for ri in range(sD.shape[0]): # row (= virus)
                if cl == sD[ri,1]:
                    vl = sD[ri,0]
                    onlyVL = self.uiTask["showThisVirusOnly"]
                    if (onlyCL != None and onlyCL == cl) or \
                      (onlyVL != None and onlyVL == vl):
                    # if it's showing only this classification or viris
                        dc.SetFont(emphFonts[1]) # emphasize this label
                        # write virus label
                        dc.DrawText(vl, _x2, _y)
                        dc.SetFont(fonts[1])
                    else:
                        # write virus label
                        dc.DrawText(vl, _x2, _y)
                    if len(self.vlR) == 0: # first time to draw graph
                        w, h = dc.GetTextExtent(vl)
                        # store rect of text (virus label in legend)
                        vlR[vl] = (_x2, _y, _x2+w, _y+h)
                    _y += _yInc # increase y-coordinate
        ##### [end] drawing legend -----
        
        ##### [begin] drawing virus label user clicked -----
        dc.SetFont(fonts[2])
        if self.uiTask["showVirusLabel"] != None:
            x, y, virus = self.uiTask["showVirusLabel"]
            gpSz = self.parent.pi["gp"]["sz"]
            rat = (dcSz[0]/gpSz[0], dcSz[1]/gpSz[1])
            x = int(x * rat[0])
            y = int(y * rat[1])
            dc.SetTextForeground("#666666")
            dc.DrawText(virus, x, y)
        ##### [end] drawing virus label user clicked -----
        
        if len(self.vpPt) == 0: # first time to draw graph
            self.vCR = vCR # store radius of virus presence circle
            self.vpPt = vpPt1 # store virus presence points
            self.vlR = vlR # store rects of virus labels in legend
            self.clR = clR # store rects of classification labels in legend

    #---------------------------------------------------------------------------
  
    def initUITask(self):
        """ Delete all current UI tasks.
        
        Args: None
        
        Returns: None
        """ 
        if DEBUG: print("ProcGraphData.initUITask()")

        for key in self.uiTask.keys():
            self.uiTask[key] = None

    #---------------------------------------------------------------------------
    
#===============================================================================
