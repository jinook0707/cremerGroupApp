# coding: UTF-8

"""
Open-source software written in Python 
  to simulate ant motion.

This program was coded and tested in Kubuntu 22.04.

Jinook Oh, Cremer group, Institute of Science and Technology Austria 
Last edited: 2024-02-09

Dependency:
    Python (3.9)
    wxPython (4.1)
    NumPy (1.22)
    SciPy (1.8)
    MatPlotLib (3.5)

------------------------------------------------------------------------
Copyright (C) 2022 Jinook Oh & Sylvia Cremer
in Institute of Science and Technology Austria.
- Contact: jinook.oh@ista.ac.at/ jinook0707@gmail.com

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

Changelog -----
v.0.1.202212: Initial development 
v.0.2.202309: Implementing 2D simulation (incomplete yet the moment)
"""

import sys, queue, random
from copy import copy
from os import path
from time import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pareto

sys.path.append("..")
import initVars
initVars.init(__file__)
from initVars import *

from modFFC import *
from modCV import *
from modGraph import *
from ants import AAggregate

DEBUG = False
__version__ = "0.2.1"
ICON_FP = path.join(P_DIR, "image", "icon.png")

#===============================================================================

class AntSimFrame(wx.Frame):
    """ Frame for ant motion simulation 

    Args:
        None
     
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self):
        if DEBUG: MyLogger.info(str(locals()))
        
        ### init
        wg = wx.Display(0).GetGeometry()
        if sys.platform.startswith("win"):
            wPos = (-5, 0)
            wSz = (wg[2], int(wg[3]*0.85))
        else:
            wPos = (0, 20)
            wSz = (wg[2], int(wg[3]*0.9))
        wx.Frame.__init__(
              self,
              None,
              -1,
              "AntSim v.%s"%(__version__), 
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX),
              )
        self.SetBackgroundColour('#333333')
        if __name__ == '__main__' and path.isfile(ICON_FP):
            self.SetIcon(wx.Icon(ICON_FP, wx.BITMAP_TYPE_PNG)) # set app icon
        # frame close event
        self.Bind(wx.EVT_CLOSE, self.onClose)
        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        ### frame resizing
        if sys.platform.startswith("win"): _wSz = wSz
        else: _wSz = (wSz[0], wSz[1]+self.statusbar.GetSize()[1])
        updateFrameSize(self, _wSz)
        
        ##### [begin] setting up attributes -----
        self.wSz = wSz
        self.fonts = getWXFonts()
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread
        self.simTypes = [ 
                "n1", # simulating individuals (n1).
                "n6_s", # simulating aggregates of n6; simple add of motions.
                "n10_s", # simulating aggregates of n10; simple add
                "n6_i", # simulating interacting individuals in a group;N6
                "n10_i", # simulating interacting individuals in a group;N10
                "mComp_s", # simulate N1, N6 & N10 then
                           # compare motion and mean power; simple add
                "mComp_i", # same as mComp, but interacting individuals
                #"sweepThADB", # sweep through a range of thresholds [adb].
                #"sweepThIDB", # sweep through a range of thresholds [idb].
                ]
        self.simTyp = self.simTypes[0]
        self.flags = dict(
                            debug=False,
                            blockUI=False, # block user input
                            dProbDist=False, # draw probability distributions
                            runningThread=False, # a thread is running
                            )
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for status bar message display
        self.mlWid = [] # wx widgets in middle left panel
        self.mrWid = [] # wx widgets in middle right panel
        btnImgDir = path.join(P_DIR, "image")
        self.btnImgDir = btnImgDir
        self.rsltDir = "rslts"
        self.logFP = path.join(FPATH, "log.txt")
        
        ### set probability of N of motions per frame
        # probability of number of motions;
        #   mean probabilities drawn from data of 10 single worker recordings
        origMP = [0.39441436896829063, 0.35808640841940237, 
                  0.16855465236396952, 0.060008456988978096, 
                  0.015003118823786146, 0.003133969709389916, 
                  0.0006250011591267788, 0.00013198590254556605, 
                  3.152824838324997e-05, 1.050941612774999e-05]
        self.mpProb = []
        for _mp in origMP:
            self.mpProb.append(_mp)
            if sum(self.mpProb) > 0.99: break # break if 99% of data included
        # put the rest probability as the last # of motion
        self.mpProb.append(1-sum(self.mpProb))
        # list of number of motion points
        self.mpLst = list(range(1, len(self.mpProb)+1))

        def getParetoDist4DurProb(dL, dataLen, shape, location, scale):
            _p = pareto.pdf(range(dataLen), shape, location, scale)
            _p = _p / np.sum(_p) # make the probability sum to one
            if dL in ["AD", "ID"]:
                ''' Insert zeros between probability numbers.
                Because the probability parameters were calculated 
                without zeros in between duration seconds, while
                data bundling limit was set to 2 seconds in Visualizer.
                '''
                prob = np.zeros(2*len(_p))
                prob[::2] = _p
            else:
                prob = _p
            return prob

        self.prob = {}
        self.pXLst = {}
        # get activity duration probability distribution 
        self.prob["AD"] = getParetoDist4DurProb("AD", 2500, 0.992, -0.8, 1.2)
        # activity duration lists
        self.pXLst["AD"] = list(range(self.prob["AD"].shape[0])) 
        # get inactivity duration probability distribution
        self.prob["ID"] = getParetoDist4DurProb("ID", 1000, 0.244, 0.2, 0.8)
        # inactivity duration lists
        self.pXLst["ID"] = list(range(self.prob["ID"].shape[0])) 
        # get walking distance probability distribution
        self.prob["WD"] = getParetoDist4DurProb("WD", 700, 0.316, 120, 10)
        # walking distance lists
        self.pXLst["WD"] = list(range(self.prob["WD"].shape[0]))

        # set walking probability range
        ''' Non-walking-motion (such as grooming) probabilities (percent)
        from 11 ants from 11 colonies:
        86.958, 88.811, 91.209, 85.611, 90.270, 90.339, 86.124, 86.802, 
        82.456, 86.788, 83.440

        - The following probabilities are from the same 11 colonies, 
        but without larvae
        72.164, 70.726, 82.533, 72.417, 79.365, 63.724, 70.489, 73.965, 
        80.432, 63.67, 75.353
        '''
        self.walkProbRng = [0.08, 0.18]
        
        self.nLbls = ["n01", "n06", "n10"]
        self.n = [int(x.lstrip("n")) for x in self.nLbls]
        # main paramters 
        self.mPa = dict(
                        nDP=-1, # number of data points per output 
                        dPtIntvSec=-1, # data bundling interval 
                        thR={}, # boosting threshold ranges
                        psdMV={}, # max Y values for PSD
                        nOutput=-1, # number of simulated output
                        nLbl="", # N label; n1, n6 or n10
                        nWorkerInG=-1, # number of workers in an aggregate
                        interaction=0, # interact with each other
                        gNCol = 4, # number of columns in graph
                        gNRow = 3, # number of rows in graph
                        bandH = 200, # data band height
                        arenaSz = (350, 350), # arena size 
                        antSz = 30, # ant size 
                        )
        self.mPa["thR"]["WDB"] = [250, 700] # set walking-distance-boost range
        self.rGraph = {} # container for drawn graphs 
        self.gKey = "" # current graph key 
        ''' Current graph keys
        00_mp: Probability of N of motions per data frame
        01_adp: Probability of activity duration
        02_idp: Probability of inactivity duration
        03_intensity: Intensity bar graph
        04_psd_[str]: Power-spectral-density graph; [str] could be n1/ n6/ ..
        05_dur[str]: Duration box plots; [str] could be 'activity'/ ..
        06_mComp_[str]: Comparision of N of motions; [str] could be n6/ n10
        07_heatmap_[str]: Heatmap; [str] could be n1/ n6/ ..
        99_sweep...: [!! Currently not used !!] For sweeping parameter
        '''
        f4g = dict(typ=cv2.FONT_HERSHEY_TRIPLEX, scale=0.5, col=(0,0,0),
                   thck=1) # font info for cv2.putText
        (fW, fH), bl = cv2.getTextSize("T", f4g["typ"], f4g["scale"], 
                                      f4g["thck"])
        f4g["fW"] = fW # font width
        f4g["fH"] = fH # font height
        f4g["bl"] = bl # baseline
        self.cvFont4g = f4g
        # colors for graphs
        self.gCol = setColors()

        ### set arena array for depicting 
        ###   accessible (=0) and inaccessible (=255) area
        ### * The current ant's positon will be 1 during the simulation. 
        '''
        self.arenaArr = np.zeros(self.mPa["arenaSz"], dtype=np.uint8)
        pos = (int(self.mPa["arenaSz"][0]/2), int(self.mPa["arenaSz"][1]/2))
        rad = min(pos)
        cv2.circle(self.arenaArr, pos, rad, 255, -1)
        # invert; black area is where ants can access
        self.arenaArr = cv2.bitwise_not(self.arenaArr)
        '''
        ### read arena images
        arenas = []
        for fp in sorted(glob("arena*.png")):
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            __, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            arenas.append(img)
        ### set arena array
        arenaIdx = 0
        self.arenaArr = arenas[arenaIdx]
        ##### [end] setting up attributes ----- 

        # init log file
        writeFile(self.logFP, str(datetime.now()) + "\n", mode="w")

        btnSz = (35, 35)
        ### create panels and its widgets
        for pk in pi.keys(): 
            w = [] # each itme represents a row in the left panel
            if pk == "tp":
                w.append([
                    {"type":"sTxt", "label":"simulation:", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"cho", "nCol":1, "name":"simTyp",
                     "choices":self.simTypes, "size":(250,-1),
                     "val":self.simTyp},
                    {"type":"btn", "nCol":1, "name":"start", "size":btnSz,
                     "img":path.join(btnImgDir, "start1.png"),
                     "bgColor":"#333333"},
                    {"type":"btn", "nCol":1, "name":"save", "size":btnSz,
                     "img":path.join(btnImgDir, "save.png"),
                     "bgColor":"#333333"},
                    {"type":"sTxt", "label":":", "nCol":1,
                     "fgColor":"#cccccc", "border":20, 
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)},
                    {"type":"sTxt", "label":"Offset:", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"txt", "nCol":1, "name":"offset", "val":"0, 0", 
                     "style":wx.TE_PROCESS_ENTER, "procEnter":True, 
                     "size":(100,-1)}, 
                    ])
            # setup this panel & widgets
            setupPanel(w, self, pk)

        self.offset_txt = wx.FindWindowByName("offset_txt", self.panel["tp"])

        self.initMLWidgets() # set up middle left panel
       
        ### Bind events to the middle panel (graph panel)
        self.panel["mp"].Bind(wx.EVT_PAINT, self.onPaintMP)

    #---------------------------------------------------------------------------
    
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: MyLogger.info(str(locals()))

        wSz = self.wSz 
        pi = {} # information of panels
        if sys.platform.startswith("win"):
            #style = (wx.TAB_TRAVERSAL|wx.SIMPLE_BORDER)
            style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
            bgCol = (120, 120, 120) 
        else:
            style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
            bgCol = (50, 50, 50) 

        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], 50), bgCol=bgCol, style=style)
        tpSz = pi["tp"]["sz"]
        # middle left panel
        pi["ml"] = dict(pos=(0, tpSz[1]), sz=(int(wSz[0]*0.15), wSz[1]-tpSz[1]),
                        bgCol=bgCol, style=style)
        mlSz = pi["ml"]["sz"]
        mpBGC = (240,240,240)
        # middle panel
        pi["mp"] = dict(pos=(mlSz[0], tpSz[1]), bgCol=mpBGC, style=style,
                        sz=(int(wSz[0]*0.7), wSz[1]-tpSz[1]))
        mpSz = pi["mp"]["sz"]
        # right panel 
        pi["mr"] = dict(pos=(mlSz[0]+mpSz[0], tpSz[1]),
                        sz=(wSz[0]-mlSz[0]-mpSz[0], wSz[1]-tpSz[1]),
                        bgCol=bgCol, style=style)
        return pi

    #---------------------------------------------------------------------------
     
    def initMLWidgets(self):
        """ Set up wxPython widgets when input file is loaded. 
        
        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        pk = "ml"
        pSz = self.pi[pk]["sz"]
        hlSz = (int(pSz[0]*0.9), -1)
       
        for i, w in enumerate(self.mlWid): # through widgets in the panel
            try:
                self.gbs[pk].Detach(w) # detach 
                w.Destroy() # destroy
            except:
                pass
        
        ##### [begin] set up middle left panel -----
        w = [] # each item represents a row in the left panel 
        nCol = 2
        bgc = self.pi["ml"]["bgCol"]
        fgc = getConspicuousCol(bgc)

        w.append([
            {"type":"sTxt", "label":"# of days to simulate:", "nCol":1,
             "fgColor":fgc},
            {"type":"txt", "name":"nDays", "nCol":nCol, "numOnly":True,
             "val":"1"}#"6.5"}
            ]) # number of days to simulate 

        w.append([
            {"type":"sTxt", "label":"bundle interval:", "nCol":1,"fgColor":fgc},
            {"type":"txt", "name":"dPtIntvSec", "val":"300",
             "nCol":nCol, "numOnly":True}
            ]) # data bundle interval in seconds 

        if self.simTyp.startswith("sweep"): _lst = [1]
        else: _lst = [1, 6, 10]
        for _n in _lst:
            #if _n == 1: _val = '1000'
            #else: _val = '2500'
            _val = '2500'
            w.append([
                {"type":"sTxt", "label":f'max Y in PSD [{_n:02d}]:', "nCol":1,
                 "fgColor":fgc},
                {"type":"txt", "name":f'psdMVN{_n:02d}', "val":_val,
                 "nCol":nCol, "numOnly":True}
                ]) # max Y value in PSD graph 

        if not self.simTyp.startswith("sweep"):

            w.append([
                {"type":"sTxt", "label":"# of output:", "nCol":1, 
                 "fgColor":fgc},
                {"type":"txt", "name":"nOutput", "nCol":nCol, "numOnly":True,
                 "val":"2"}
                ]) # number of output data 

            _cho = [str(x) for x in range(1,10)]
            w.append([
                {"type":"sTxt", "label":"graph columns:", "nCol":1, 
                 "fgColor":fgc},
                {"type":"cho", "nCol":1, "name":"gNCol", "choices":_cho , 
                 "size":(75,-1), "val":str(self.mPa["gNCol"])}
                ]) # number of columns for graph 
            w.append([
                {"type":"sTxt", "label":"graph rows:", "nCol":1, "fgColor":fgc},
                {"type":"cho", "nCol":1, "name":"gNRow", "choices":_cho , 
                 "size":(75,-1), "val":str(self.mPa["gNRow"])}
                ]) # number of rows for graph 

            _cho = [str(x) for x in range(50,401,50)]
            bandH = str(self.mPa["bandH"])[:-1] + "0"
            self.mPa["bandH"] = int(bandH) 
            w.append([
                {"type":"sTxt", "label":"data band height:", "nCol":1, 
                 "fgColor":fgc},
                {"type":"cho", "nCol":1, "name":"bandH", "choices":_cho , 
                 "size":(75,-1), "val":bandH}
                ]) # data band height 

            for _thr in ["ADB", "IDB"]:
                # initial threshold range
                if _thr == "ADB": _val = "0, 300"
                elif _thr == "IDB": _val = "600, 1200"
                w.append([
                    {"type":"sTxt", "label":"threshold range [%s]:"%(_thr), 
                     "nCol":1, "fgColor":fgc},
                    {"type":"txt", "name":"%sthR"%(_thr), "val":_val,
                     "nCol":nCol}
                    ]) # threshold range; adjusting activity & inactivity dur. 
            w.append([
                {"type":"chk", "nCol":2, "name":"dProbDist",
                 "label":"draw prob. dist", "val":self.flags["dProbDist"],
                 "style":wx.CHK_2STATE, "fgColor":fgc, "bgColor":bgc}
                ]) # draw probability distributions

        else:
        # sweep thresholds 
            for _thr in ["ADB", "IDB"]:
                if _thr == self.simTyp[-3:]: _val = "240, 1200"
                else: _val = "360, 360"
                w.append([
                    {"type":"sTxt", "label":"threshold range [%s]:"%(_thr), 
                     "nCol":1, "fgColor":fgc},
                    {"type":"txt", "name":"%sthR"%(_thr), "val":_val,
                     "nCol":nCol}
                    ]) # threshold range; adjusting activity & inactivity dur. 
        
        self.mlWid = setupPanel(w, self, pk)
   
    #---------------------------------------------------------------------------
   
    def initMRWidgets(self):
        """ init widgets in the middle-right panel
        
        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        for i, w in enumerate(self.mrWid): # widgets in the panel
            try:
                self.gbs["mr"].Detach(w) # detach 
                w.Destroy() # destroy
            except:
                pass

    #---------------------------------------------------------------------------
    
    def updateMRWid(self):
        """ update widget (for thumbnail images) in middle-right panel
        
        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        self.initMRWidgets()
        for gi, k in enumerate(sorted(self.rGraph.keys())):
            img = self.rGraph[k]["img"].copy()
            iSz = (img.shape[1], img.shape[0])
            rat = calcI2DRatio(iSz, self.pi["mr"]["sz"], True, 0.9)
            w = int(iSz[0]*rat)
            h = int(iSz[1]*rat)
            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
            bmp = convt_cvImg2wxImg(img, toBMP=True)
            name = "thumbnail_%s"%(k)
            sBmp = wx.StaticBitmap(self.panel["mr"], -1, bmp, name=name)
            sBmp.Bind(wx.EVT_LEFT_UP, self.onClickThumbnail)
            sBmp.key = k
            add2gbs(self.gbs["mr"], sBmp, (gi,0), (1,1))
            self.mrWid.append(sBmp)
        self.highlightThumbnail(self.gKey)
        self.gbs["mr"].Layout()
        self.panel["mr"].SetupScrolling()

    #---------------------------------------------------------------------------
    
    def highlightThumbnail(self, thK=""):
        """ Highlight the thumbnail image with the given key 

        Args:
            thK (str): String key of thumbnail image

        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        if self.gKey != "":
            obj = wx.FindWindowByName("thumbnail_%s"%(self.gKey), 
                                      self.panel["mr"])
            if obj is not None:
                ### restore (de-highlight) the previously selected thumbnail 
                img = self.rGraph[self.gKey]["img"].copy()
                iSz = (img.shape[1], img.shape[0])
                rat = calcI2DRatio(iSz, self.pi["mr"]["sz"], True, 0.9)
                w = int(iSz[0]*rat)
                h = int(iSz[1]*rat)
                img = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA)
                bmp = convt_cvImg2wxImg(img, toBMP=True)
                obj.SetBitmap(bmp)
            
        ### highlight (drawing yellow border) the given thumbnail image 
        thumbnail = wx.FindWindowByName("thumbnail_%s"%(thK), self.panel["mr"])
        bmp = thumbnail.GetBitmap()
        dc = wx.MemoryDC(bmp)
        w, h = dc.GetSize()
        dc.SetPen(wx.Pen((255,255,0), 5)) 
        dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
        dc.DrawRectangle(0, 0, w, h)
        del dc
        thumbnail.SetBitmap(bmp)

    #---------------------------------------------------------------------------
    
    def onClickThumbnail(self, event):
        """ process mouse-click on thumbnail image (wx.StaticBitmap)

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        if self.flags["blockUI"]: return

        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav")) 

        ### highlight the currently selected thumbnail
        obj = event.GetEventObject()
        self.highlightThumbnail(obj.key)
        
        self.gKey = obj.key # store the current graph key
        self.panel["mp"].Refresh() # re-draw the graph

    #---------------------------------------------------------------------------

    def onButtonPressDown(self, event, objName=""):
        """ wx.Butotn was pressed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "btn")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return
        if self.flags["blockUI"] or not obj.IsEnabled(): return
        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))

        if objName == "start_btn":
            self.onSimStartBtnPressed()

        elif objName == "save_btn":
            self.save()

    #---------------------------------------------------------------------------
    
    def onChoice(self, event, objName=""):
        """ wx.Choice was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "cho")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "simTyp_cho":
            self.simTyp = objVal.split(":")[0].strip() # store simulation type
            self.initMLWidgets() # set up middle left panel

        elif objName == "gNCol_cho":
            self.mPa["gNCol"] = int(objVal) # store number of graph columns
        
        elif objName == "gNRow_cho":
            self.mPa["gNRow"] = int(objVal) # store number of graph rows 
        
        elif objName == "bandH_cho":
            self.mPa["bandH"] = int(objVal) # data band height 

    #---------------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "chk")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        k = objName.rstrip("_chk")
        self.flags[k] = objVal 

    #---------------------------------------------------------------------------
    
    def onTextCtrlChar(self, event, objName="", isNumOnly=False):
        """ Character entered in wx.TextCtrl.
        Currently using to allow entering numbers only.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name.
            isNumOnly (bool): allow number entering only.
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "txt")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if isNumOnly:
            keyCode = event.GetKeyCode()
            ### Allow numbers, backsapce, delete, left, right
            ###   tab (for hopping between TextCtrls), dot (for float number)
            allowed = [ord(str(x)) for x in range(10)]
            allowed += [wx.WXK_BACK, wx.WXK_DELETE, wx.WXK_TAB]
            allowed += [wx.WXK_LEFT, wx.WXK_RIGHT]
            allowed += [ord(".")]
            if keyCode in allowed:
                event.Skip()
                return

    #---------------------------------------------------------------------------
    
    def onEnterInTextCtrl(self, event, objName=""):
        """ Enter-key was pressed in wx.TextCtrl
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        Returns:
            None
        """
        if DEBUG: logging.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "txt")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "offset_txt": 
            self.setGraphOffset(objVal)

    #---------------------------------------------------------------------------

    def onKeyPress(self, event):
        """ Process key-press event
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        event.Skip()
        
        kc = event.GetKeyCode()
        mState = wx.GetMouseState()

        if mState.ControlDown():
        # CTRL modifier key is pressed
            if kc == ord("Q"): # quit the app
                self.onClose(None)
            
            elif kc == ord("H"): # halt the thread
                if self.th is not None: self.q2t.put(("quit",), True, None)
                wx.CallLater(100, self.q2t.clear)

        if kc in [wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_UP, wx.WXK_DOWN]:
            ### adjusting offset
            if mState.AltDown(): move = 10
            elif mState.ControlDown(): move = 100
            elif mState.ShiftDown(): move = 500 
            else: move = None
            if move is not None:
                if kc == wx.WXK_LEFT: self.setGraphOffset("to_left", move) 
                elif kc == wx.WXK_RIGHT: self.setGraphOffset("to_right", move) 
                elif kc == wx.WXK_UP: self.setGraphOffset("to_up", move) 
                elif kc == wx.WXK_DOWN: self.setGraphOffset("to_down", move) 

    #---------------------------------------------------------------------------
    
    def onPaintMP(self, event):
        """ painting graph

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        if self.gKey == "": return

        event.Skip()
        dc = wx.PaintDC(self.panel["mp"]) 
        dc.Clear() 

        ### draw the generated graph image
        img = self.rGraph[self.gKey]["img"].copy()

        x = 0 
        y = 0
        gK = self.gKey
        if self.flags["runningThread"] and gK.startswith("03_intensity"): 
            pH = self.pi["mp"]["sz"][1]
            # parameters for drawing intensity in real-time
            rtdp = self.rtdParams 
            gH = rtdp["nIdx"]*rtdp["aggIntH"]
            gH += (rtdp["outputIdx"]+1) *rtdp["outputIntH"]
            if gH > pH: y = pH-gH
        else:
            '''
            iSz = (img.shape[1],img.shape[0])
            w, h = calcImgSzFitToPSz(iSz, self.pi["mp"]["sz"], 1.0)
            if w > iSz[0] or h > iSz[1]: _intrp = cv2.INTER_CUBIC
            else: _intrp = cv2.INTER_AREA
            img = cv2.resize(img, (w,h), interpolation=_intrp)
            '''
            x, y = self.rGraph[self.gKey]["offset"]
            if gK.startswith("07_heatmap_"):
                if img.dtype != np.uint8:
                    img = self.convtHeatmap2Img(img)
                
        bmp = convt_cvImg2wxImg(img, toBMP=True)
        dc.DrawBitmap(bmp, x, y) 

    #---------------------------------------------------------------------------

    def genDataNdrawInt(self, agg, kw):
        """ Generate simulated data and draw intensity plot.

        Args:
            agg (ants.AAggregate): Instance of AAggregate.
            kw (dict): Arguments

        Returns:
            bData (list): Bundled output data
            rALst (list): List of chosen activity durations 
            rILst (list): List of chosen inactivity durations 
        """
        if DEBUG: MyLogger.info(str(locals()))

        pa = self.mPa
        if not "nw" in kw.keys(): kw["nw"] = pa["nWorkerInG"]

        data = [] # output data (number of motions)
        bData = [] # bundled data
        aPosData = [] # list of ant's position
        for wi in range(kw["nw"]):
            data.append([])
            bData.append([]) 
            aPosData.append([])
        
        agg.initSim() # initialize some variables for the data simulation
        ri = 0
        ci = 0
        #xx1=[]; xx2=[] # [TEMP; debugging] for interaction process
        pbIdx = 0 # index where the previous data bundling was conducted
        cntLIDB = [0]*kw["nw"] # number of boosting of longer inactivity dur.  
                               #   in last bundling
        
        def bundleNMsg2dr(simTyp, nw, bData, d, apd, pbIdx, di, cntLIDB, q2m):
            motions = [] # n of motions of each worker 
                         #   to send for real-time drawing
            aPos = [] # list for position data of each ant
            for wi in range(nw):
                val = sum(d[wi][pbIdx:di])
                bData[wi].append(val) # store bundled data with dPtIntvSec 
                motions.append(val)
                aPos.append(apd[wi][max(0, pbIdx-1):di])
            if not self.simTyp.startswith("sweep"):
                msg = ("drawIntensity", (pbIdx, di-1), motions, cntLIDB,)
                q2m.put(msg, True, None)
                msg = ("drawHeatmap", (pbIdx, di-1), aPos,)
                q2m.put(msg, True, None)
            return bData

        for di in range(pa["nDP"]):
            ret = receiveDataFromQueue(self.q2t)
            if not ret is None and ret[0] == "quit":
                return None, None, None

            if di%100 == 0:
                ### send progress message
                lineMsg = "[%i/%i]"%(kw["simI0"]+1, kw["lLen0"])
                lineMsg += " [%i/%i]"%(kw["simI1"]+1, kw["lLen1"])
                lineMsg += " [%i/%i]"%(di+1, pa["nDP"])
                _p = kw["simI0"] / kw["lLen0"]
                _p += kw["simI1"] / kw["lLen1"] / kw["lLen0"]
                _p += di / pa["nDP"] / kw["lLen1"] / kw["lLen0"]
                lineMsg += ";  %.1f%%"%(_p*100)
                self.q2m.put(("displayMsg", lineMsg,), True, None)

            if di > 0 and di%pa["dPtIntvSec"] == 0:
                bData = bundleNMsg2dr(self.simTyp, kw["nw"], bData, data,
                                      aPosData, pbIdx, di, cntLIDB, self.q2m)
                pbIdx = copy(di)
                cntLIDB = [0]*kw["nw"]

            cntLIDB, data, aPosData = agg.simulate(di, cntLIDB, data, aPosData)

        # bundle the rest data
        bData = bundleNMsg2dr(self.simTyp, kw["nw"], bData, data,
                              aPosData, pbIdx, len(data[wi]), cntLIDB, self.q2m)

        ''' [TEMP; debugging]
        print("\n")
        if xx1 != []:
            xx1 = np.asarray(xx1)
            print("### xx1: ", np.min(xx1), np.max(xx1), np.mean(xx1), 
                  np.median(xx1), np.std(xx1))
        if xx2 != []:
            xx2 = np.asarray(xx2)
            print("### xx2: ", np.min(xx2), np.max(xx2), np.mean(xx2), 
                  np.median(xx2), np.std(xx2))
        print("\n")
        ''' 

        '''
        [OBSOLETE]
        rGraph = None 
        if self.flags["dIntensity"]:
            fig, ax = plt.subplots(self.mPa["gNRow"], self.mPa["gNCol"], 
                                   sharey=True, tight_layout=True)
            for wi in range(kw["nw"]):
                # draw motion bar graph 
                ax[ri, ci].bar(list(range(len(bData[-1]))), height=bData[-1]) 
                ci += 1
                if ci >= self.mPa["gNCol"]: 
                    ci = 0 
                    ri += 1
            fig.suptitle("Simulated data [n%i]"%(kw["nw"]), fontsize=12)
            rGraph = convt_mplFig2npArr(fig)
        '''

        return bData

    #---------------------------------------------------------------------------

    def drawPSDWithSimData(self, kw):
        """ Draw PSD plots with the simulated intensity.

        Args:
            kw (dict): Arguments

        Returns:
            (float): Ratio of [2hr-20min] to [20min-10min] in mean power. 
        """
        if DEBUG: MyLogger.info(str(locals())) 

        pa = self.mPa
        if not "nOutput" in kw.keys(): kw["nOutput"] = pa["nOutput"]
        if not "nw" in kw.keys(): kw["nw"] = pa["nWorkerInG"]

        if not self.simTyp.startswith("sweep"): # non-sweep simulation 
            fig, ax = plt.subplots(pa["gNRow"], pa["gNCol"], 
                                   sharex=True, sharey=True, tight_layout=False)
        ri = 0
        ci = 0
        fsArr = None
        pxxArr = None
        psdMVLst = []
        rat2hrTo20mLst = []
        m20mLst = []
        for oi in range(kw["nOutput"]):
            if self.simTyp.startswith("sweep"): # sweep simulation 
                params = dict(dPtIntvSec=pa["dPtIntvSec"], 
                              maxY=kw["psdMV"], 
                              flagDrawPlot=False) 
                ret = drawPSD(kw["bData"][oi], None, "", params, self.logFP)
            else:
                params = dict(dPtIntvSec=pa["dPtIntvSec"], maxY=kw["psdMV"], 
                              flagDrawPlot=True) 
                ret = drawPSD(kw["bData"][oi], ax[ri, ci], "", params, 
                              self.logFP) 
            rat2hrTo20mLst.append(ret["m2hr"]/ret["m20m"])
            if fsArr is None:
                fsArr = np.zeros((kw["nOutput"], ret["fs"].shape[0])) 
                pxxArr = np.zeros((kw["nOutput"], ret["pxx"].shape[0]))
            fsArr[oi, :] = ret["fs"]
            pxxArr[oi, :] = ret["pxx"]
            psdMVLst.append(ret["psdMV"])
            ci += 1
            if ci >= self.mPa["gNCol"]:
                ci = 0 
                ri += 1 
            
            if self.flags["debug"]: break

        if not self.simTyp.startswith("sweep"): # non-sweep simulation 
            ### set y-ticks
            if kw["psdMV"] == -1: kw["psdMV"] = max(psdMVLst)
            plt.ylim(0, kw["psdMV"])
            dvsrStr = setYFormat(kw["psdMV"], None, "{:,.1f}", False)
            
            plt.gcf().text(0.01, 0.93, f"power (RMS, {dvsrStr})", fontsize=8)

            fig.suptitle("Simulated data [n%i]"%(kw["nw"]), fontsize=8)
            key = "04_psd_n%02i"%(kw["nw"])
            self.rGraph[key] = dict(img=convt_mplFig2npArr(fig),
                                    offset=[0,0])

        return np.mean(rat2hrTo20mLst)

    #---------------------------------------------------------------------------

    def onTimer(self, event, flag):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        ### Receive data from queue
        ### * if it's "displayMsg", keep receiving until getting 
        ###   the last queued message.
        ### * if the received data is other type than a simple message, 
        ###   then process it.
        rData = [] 
        while True:
            ret = receiveDataFromQueue(self.q2m)
            if ret == None:
                break
            else:
                if ret[0] == "displayMsg":
                    rData.append(ret)
                elif ret[0] == "drawIntensity":
                    rData.append(ret)
                else:
                    rData.append(ret)
                    break
        if rData == []: return

        ### display message
        displayMsg = None 
        for i in range(len(rData)):
            if rData[i][0] == "displayMsg":
                displayMsg = rData[i]
                rData[i] = None
        while None in rData: rData.remove(None)
        if not displayMsg is None:
            showStatusBarMsg(self, displayMsg[1], -1) 

        pa = self.mPa # main parameters
        rtdp = self.rtdParams # parameters for drawing intensity in real-time
        c = self.gCol
        f4g = self.cvFont4g 
        flagDrawMP = False

        for _rData in rData:
            if _rData[0] in ["finished", "interrupted"]:
                self.callback(_rData, flag)
                continue

            ### set some parameters for intensity graph
            if _rData[0] == "incOutputIdx":  
                ### increase output index
                if rtdp["outputIdx"]+1 == pa["nOutput"]: return
                rtdp["outputIdx"] += 1
                rtdp["xInt"] = rtdp["marginInt"][0] 
            elif _rData[0] == "incNIdx":
                ### increase aggregate (for n1, 6, 10, ..) index
                if rtdp["nIdx"]+1 == len(self.nLbls): return
                rtdp["nIdx"] += 1
                rtdp["outputIdx"] = 0
                rtdp["xInt"] = rtdp["marginInt"][0]

            if self.simTyp.startswith("mComp"):
                nLbl = self.nLbls[rtdp["nIdx"]]
                nWorkerInG = int(nLbl.lstrip("n"))
            else:
                nLbl = pa["nLbl"]
                nWorkerInG = pa["nWorkerInG"]
            self.gKey = f'07_heatmap_{nLbl}'

            if _rData[0] in ["incOutputIdx", "incNIdx"]:
            # increase index for drawing intensity graph
                img = self.rGraph["03_intensity"]["img"] # graph image

                ### write output info 
                # starting point on y-axis
                y0 = rtdp["nIdx"] * rtdp["aggIntH"] + \
                        rtdp["outputIdx"] * rtdp["outputIntH"]
                ty = y0 + f4g["fH"] 
                tx = rtdp["marginInt"][0] + 5
                txt = "[%s] output-%02i"%(nLbl, rtdp["outputIdx"]+1)
                cv2.putText(img, txt, (tx, ty),
                            f4g["typ"], f4g["scale"], c["cvFont"])
 
            elif _rData[0] == "drawIntensity":
            # draw intensity graph
                # starting point on y-axis
                y0 = rtdp["nIdx"] * rtdp["aggIntH"] + \
                        rtdp["outputIdx"] * rtdp["outputIntH"]

                m = rtdp["marginInt"]

                img = self.rGraph["03_intensity"]["img"] # graph image
                imgH, imgW = img.shape[:2]

                __, mRng, motions, cntLIDB = _rData
                x = rtdp["xInt"]
                for wi in range(nWorkerInG):
                    ### draw real-time intesnity
                    # bottom of the bar
                    y1 = y0 + m[1] + rtdp["bandH"]
                    # bottom of this worker's bar; 
                    #   +1 is for minimum gap between worker bar
                    y1 -= (rtdp["wBarH"]+1) * wi
                    inten = min(int(motions[wi] * rtdp["p4m"]), rtdp["wBarH"])
                    y2 = y1 - inten
                    if y1-y2 > 0:
                        # draw intensity bar
                        cv2.line(img, (x, y1), (x, y2), c["inten"], 1)

                    if cntLIDB[wi] > 0:
                    # there were some long inactivity duration boost
                        val = int(cntLIDB[wi] * rtdp["p4aac"])
                        y1 += 2 
                        y2 = y1 + val 
                        cv2.line(img, (x,y1), (x,y2), c["lidb"], 1)
              
                ### draw output frame lines for an output 
                if mRng[0] == 0:
                    thck = int(m[0]/4)
                    for _i in range(2):
                        if _i == 0:
                            x1 = x - thck*2
                            y1 = y0 + m[1] 
                        else:
                            x1 = rtdp["nBD"] 
                            x1 += m[0] + thck 
                            y1 = y0 + m[1]
                        x2 = x1 + thck
                        y2 = y1 + rtdp["bandH"]
                        y1 += 5 # to give a bit of margin between bands
                        cv2.rectangle(img, (x1, y1), (x2, y2), c["outputFr"], 
                                      -1)
                    x2 = imgW - m[2] 
                    y = y0 + m[1] + rtdp["bandH"] + 1
                    cv2.line(img, (m[0], y), (x2, y), c["outputFr"], 1) 

                ### update
                x += 1
                rtdp["xInt"] = x

                flagDrawMP = False

            elif _rData[0] == "drawHeatmap":
            # draw heatmap
                m = rtdp["marginHM"]
                img = self.rGraph[self.gKey]["img"] # graph image
                tmpImg = np.zeros(img.shape, img.dtype)
                __, (piB, piE), aPos = _rData
                ### starting point of heatmap
                aSz = pa["arenaSz"]
                rad = int(pa["antSz"]/6) # radius for drawing ant's position
                xIdx = rtdp["outputIdx"] % rtdp["nHMInRow"]
                x0 = m[0] + xIdx * (m[0]+aSz[0]+m[2])
                y0 = m[1]
                y0 += int(rtdp["outputIdx"]/rtdp["nHMInRow"])*rtdp["outputHMH"]
                flagDrawMP = False
                for wi in range(nWorkerInG):
                    ### update the ant's position on heatmap
                    for pi in range(1, len(aPos[wi])):
                        x, y = aPos[wi][pi]
                        if piB > 0:
                            px, py = aPos[wi][pi-1]
                            '''
                            # heamap of movements only (not static position)
                            if px == x and py == y: # ant did not move
                                continue # to the next position data 
                            '''
                            px += x0
                            py += y0
                        x += x0
                        y += y0
                        #img[y,x] += 1
                        # draw positional dot
                        cv2.circle(tmpImg, (x,y), rad, 1, -1)
                        '''
                        # draw connecting line
                        if piB > 0: cv2.line(tmpImg, (px,py), (x,y), 1, 1)
                        '''
                        if not flagDrawMP: flagDrawMP = True
                if flagDrawMP:
                    self.rGraph[self.gKey]["img"] = cv2.add(img, tmpImg)

        if flagDrawMP: self.panel["mp"].Refresh() 

    #---------------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread
        
        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        self.flags["runningThread"] = False
        pa = self.mPa

        if flag == "runSimN" and rData[0] != "interrupted":
            bData, acdS, indS = rData[1:]
            kw = dict(
                bData=bData, # bundled data
                psdMV=pa["psdMV"][pa["nLbl"]] # max. value for PSD
                )
            # draw PSD plots with the simulated intensity
            self.drawPSDWithSimData(kw)

            # draw duration distribution graph
            ret = drawDurGraphs(pa["nWorkerInG"], acdS, indS, False)
            for k in ret.keys():
                self.rGraph["05_dur%s"%(k.capitalize())] = dict(img=ret[k],
                                                                offset=[0,0])

        elif flag == "runSimMComp" and rData[0] != "interrupted":
            bData, acdS, indS, l4mComp = rData[1:]

            for ni, nLbl in enumerate(self.nLbls):
                nWorkerInG = int(nLbl.lstrip("n"))
                i1 = ni * pa["nOutput"]
                i2 = (ni+1) * pa["nOutput"]
                kw = dict(
                    bData=bData[i1:i2,:], # bundled data
                    nw=nWorkerInG, # n of workers in aggregate
                    psdMV=pa["psdMV"][nLbl] # max. value for PSD
                    )
                
                # draw PSD plots with the simulated intensity
                self.drawPSDWithSimData(kw)

                # draw duration distribution graph
                ret = drawDurGraphs(nWorkerInG, acdS[nLbl], indS[nLbl], 
                                    False)
                setPLTParams(nRow=pa["gNRow"], nCol=pa["gNCol"])
                for k in ret.keys():
                    _k = "05_dur%s_n%02i"%(k.capitalize(), nWorkerInG)
                    self.rGraph[_k] = dict(img=ret[k], offset=[0,0])
     
                fp = path.join(self.rsltDir, "bData_n%02i.npy"%(nWorkerInG))
                np.save(fp, bData)

            ret = proc_mComp(self, bData, l4mComp, pa["dPtIntvSec"])
            for key in ret:
                self.rGraph["06_mComp_%s"%(key)] = dict(img=ret[key],
                                                        offset=[0,0])

        elif flag == "runSweep" and rData[0] != "interrupted":
            tThLst, mRat2hrTo20mLst, plotTitle = rData[1:]
            fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
            ax.plot(tThLst, mRat2hrTo20mLst, linestyle="dotted", 
                    c=self.gCol["dLn"], linewidth=1)
            
            ### draw regression line
            model = np.poly1d(np.polyfit(tThLst, mRat2hrTo20mLst, deg=3))
            xr = np.linspace(tThLst[0], tThLst[-1], num=len(tThLst))
            ax.plot(xr, model(xr), c=self.gCol["rLn0"], linestyle="-", 
                    linewidth=1)
            
            #plt.xticks(xTLPos, xTLbls, fontsize=8)
            plt.title(plotTitle)
            key = "99_%s"%(self.simTyp)
            self.rGraph[key] = dict(img=convt_mplFig2npArr(fig), offset=[0,0])

        postProcTaskThread(self, flag)

        if rData[0] == "interrupted":
            self.rGraph = {}
            self.gKey = ""
        else:
            ##### [begin] finalize heatmap images -----
            if self.simTyp.startswith("mComp"): nLbls = self.nLbls
            else: nLbls = [pa["nLbl"]]
            for ni, nLbl in enumerate(nLbls):
                hmArr = self.rGraph[f'07_heatmap_{nLbl}']["img"]
                ### draw heatmap backgroud (area inaccesssible to ants)
                rtdp = self.rtdParams
                m = rtdp["marginHM"]
                aSz = pa["arenaSz"]
                hmMax = np.max(hmArr)
                for oi in range(pa["nOutput"]):
                    xIdx = oi % rtdp["nHMInRow"]
                    x0 = m[0] + xIdx * (m[0]+aSz[0]+m[2])
                    y0 = m[1] + int(oi/rtdp["nHMInRow"]) * rtdp["outputHMH"]
                    wallIdx = np.where(self.arenaArr==255)
                    hmArr[y0+wallIdx[0], x0+wallIdx[1]] = hmMax+1
                
                # convert to a color image in uint8 data-type
                img = self.convtHeatmap2Img(hmArr)
                
                ### colorize the heatmap
                nSect = 3 
                step = int(np.ceil(hmMax/nSect))
                def hmCAdj(arr, hci, nSect, b=100, x=155):
                    arr = arr.astype(np.float32)
                    step = int(np.ceil(255/nSect))
                    mi = hci * step 
                    ma = (hci+1) * step 
                    arr = (arr-mi) / (ma-mi) * x + b 
                    return arr.astype(np.uint8)
                for hci in range(nSect):
                    if hci == nSect-1: rng = (hci*step, hmMax+1)
                    else: rng = (max(1, hci*step), (hci+1)*step)
                    # indices of pixels in the range
                    i = np.where(np.logical_and(hmArr >= rng[0], \
                                   hmArr < rng[1]))
                    if hci == 0: # red color
                        img[i[0],i[1],0] = 0
                        img[i[0],i[1],1] = 0
                        img[i[0],i[1],2] = hmCAdj(img[i[0],i[1],2], hci, nSect)
                    elif hci == 1: # yellow color
                        img[i[0],i[1],0] = 0
                        img[i[0],i[1],1] = hmCAdj(img[i[0],i[1],1], hci, nSect)
                        img[i[0],i[1],2] = hmCAdj(img[i[0],i[1],2], hci, nSect)
                    elif hci == 2: # white color
                        img[i[0],i[1],0] = hmCAdj(img[i[0],i[1],0], hci, nSect)
                        img[i[0],i[1],1] = hmCAdj(img[i[0],i[1],1], hci, nSect)
                        img[i[0],i[1],2] = hmCAdj(img[i[0],i[1],2], hci, nSect)

                ### write title & output info 
                title = "Heatmap"
                if pa["interaction"] > 0: title += " [I]"
                imgH, imgW = img.shape[:2]
                self.writeInfoOnGraph(img, imgW, imgH, (0,200,0), title, 
                                      self.rtdParams["marginHM"], nLbl)

                # store the image 
                self.rGraph[f'07_heatmap_{nLbl}']["img"] = img
            ##### [end] finalize heatmap images -----

            ### set the current graph key
            keys = sorted(list(self.rGraph.keys()))
            for k in keys:
                #if k.startswith("03_intensity"):
                if k.startswith("07_heatmap"):
                    self.gKey = k
                    break
            if self.gKey == "": self.gKey = keys[0]

        showStatusBarMsg(self, "", 100)
        if rData[0] == "interrupted":
            msg = "Simulation did not run successfully."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
        else:
            self.updateMRWid()
            self.panel["mr"].Refresh() # display thumbnail images
            self.panel["mp"].Refresh() # display graph 
            self.panel["mr"].SetFocus()

    #---------------------------------------------------------------------------
   
    def onSimStartBtnPressed(self):
        """ Run simulation button pressed

        Args: None

        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        ##### [begin] get parameters -----
        errMsg = ""
        ### get number of data points to simulate
        w = wx.FindWindowByName("nDays_txt", self.panel["ml"])
        try:
            nDays = float(w.GetValue())
            # number of data points 
            # (by default, about one week long data) to generate
            nDP = int(nDays * 24 * 60 * 60)
        except:
            errMsg += "Invalid # of days to simulate.\n"
        ### get data bundle interval 
        w = wx.FindWindowByName("dPtIntvSec_txt", self.panel["ml"])
        dPtIntvSec = int(w.GetValue()) 
        
        ### get threshold range of ADB & IDB
        thR = {} 
        for k in ["ADB", "IDB"]:
            w = wx.FindWindowByName("%sthR_txt"%(k), self.panel["ml"])
            _lst = []
            try: _lst = [int(x) for x in w.GetValue().split(",")]
            except: pass
            if len(_lst) == 2: thR[k] = _lst
            else: errMsg += "Invalid %s threshold range.\n"%(k) 
        ### get max Y values for PSD
        psdMV = {}
        if self.simTyp.startswith("sweep"): _lst = [1]
        else: _lst = [1, 6, 10]
        for i in _lst:
            w = wx.FindWindowByName(f'psdMVN{i:02d}_txt', self.panel["ml"])
            k = f'n{i:02d}'
            psdMV[k] = None
            try: psdMV[k] = int(w.GetValue())
            except: pass
            if psdMV[k] is None: errMsg += "Invalid max Y value for %s\n"%(k)

        if not self.simTyp.startswith("sweep"):
            ### get number of simulated output
            w = wx.FindWindowByName("nOutput_txt", self.panel["ml"])
            nOutput = int(w.GetValue())
        else:
            nOutput = 1
        ##### [end] get parameters -----
        if errMsg != "":
            wx.MessageBox(errMsg, "Error", wx.OK|wx.ICON_ERROR)
            self.panel["mp"].SetFocus()
            return

        pa = self.mPa
        if pa["gNRow"] * pa["gNCol"] < nOutput:
        # too many output, increase number of rows for analysis graphs
            pa["gNRow"] = int(np.ceil(nOutput / pa["gNCol"]))
            w = wx.FindWindowByName("gNRow_cho", self.panel["ml"])
            w.SetSelection(self.gNRow-1)

        self.rGraph = {} # init graph container
        sStr = self.simTyp.split("_")
        if sStr[0].startswith("n"):
            nLbl = sStr[0] # n-label
            nWorkerInG = int(nLbl.lstrip("n")) # number of workers in one group
            pa["nLbl"] = f'n{nWorkerInG:02d}' 
        elif sStr[0].startswith("mComp"):
            nWorkerInG = int(self.nLbls[-1].lstrip("n")) # max # of workers
        elif sStr[0].startswith("sweep"):
            nWorkerInG = 1

        ### get interaction flag
        if len(sStr) > 1 and sStr[1] == "i": interaction = 1.0 
        else: interaction = 0 
 
        pa["nDP"] = nDP
        pa["nWorkerInG"] = nWorkerInG 
        pa["dPtIntvSec"] = dPtIntvSec
        for k in thR.keys(): pa["thR"][k] = thR[k]
        pa["psdMV"] = psdMV
        pa["nOutput"] = nOutput
        pa["interaction"] = interaction 
        
        self.rtdParams = {} 
        # index of nLabel
        self.rtdParams["nIdx"] = 0 
        # index of output currently drawing
        self.rtdParams["outputIdx"] = 0
        pw, ph = self.pi["mp"]["sz"]
        
        ##### [begin] prepare for real-time intensity drawing -----
        nBD = int(np.ceil(nDP / dPtIntvSec)) # number of bundled data points 
        m = [20, 20, 20, 10] # left, top, right, bottom margin
        # set margins
        self.rtdParams["marginInt"] = m 
        # max. number of motions per bundled data for drawing intensity bars
        maxNM = dPtIntvSec * nWorkerInG * self.mpLst[-1]
        # number of bundled data
        self.rtdParams["nBD"] = nBD 
        # height of the band
        self.rtdParams["bandH"] = pa["bandH"]
        # pixel for one motion point
        self.rtdParams["p4m"] = pa["bandH"] / (maxNM + nWorkerInG-1)
        # add nWorkerInG-1 to break bar on each worker by one pixel
        pa["bandH"] += nWorkerInG-1
        wBarH = int(self.rtdParams["p4m"]*(maxNM/nWorkerInG))
        # length of intensity bar for one worker
        self.rtdParams["wBarH"] = wBarH
        # pixel for active ant count line
        self.rtdParams["p4aac"] = wBarH / (nWorkerInG*dPtIntvSec/30)
        # x-coord in mp to start drawing
        self.rtdParams["xInt"] = m[0] 
        # height of an intensity graph output
        self.rtdParams["outputIntH"] = pa["bandH"] + m[1] + m[3]
        # height of an aggregate (n1, n6 or n10)
        self.rtdParams["aggIntH"] = self.rtdParams["outputIntH"] * nOutput 

        imgW = m[0] + nBD + m[2] # image width
        if self.simTyp.startswith("n") or self.simTyp.startswith("sweep"):
            nAggr = 1
        elif self.simTyp.startswith("mComp"):
            nAggr = 3 
        imgH = (pa["bandH"]+m[1]+m[3]) * nOutput * nAggr # image height
        title = "Intensity"
        if interaction > 0: title += " [I]"
        ### array for real-time graph drawing
        img = np.zeros((imgH, imgW, 3), np.uint8)
        img[:,:] = (255,255,255) 
        self.writeInfoOnGraph(img, imgW, imgH, (0,0,0), title, m)
        # store the image
        self.rGraph["03_intensity"] = dict(img=img, offset=[0,0])
        ##### [end] prepare for real-time intensity drawing -----
        
        ##### [begin] prepare for real-time heatmap drawing -----
        # set heatmap drawing color
        self.rtdParams["HMColor"] = [(0,0,255), (0,127,255), 
                                     (0,255,255), (255,255,255)]
        m = [10, 30, 10, 10] # left, top, right, bottom margin
        # set margins
        self.rtdParams["marginHM"] = m 
        ### set a single arena drawing size 
        # height of a heatmap output
        self.rtdParams["outputHMH"] = pa["arenaSz"][1] + m[1] + m[3]
        # number of heatmap images in a column of an aggregate (n1, n6 or n10)
        nHMInColInAggr = int(np.ceil((m[0]+pa["arenaSz"][0]+m[2]) \
                             * nOutput / pw))
        # number of heatmap images in a row (restricted by panel-width)
        nHMInRow = int(np.ceil(nOutput / nHMInColInAggr))
        self.rtdParams["nHMInRow"] = nHMInRow
        # height of an aggregate (n1, n6 or n10)

        imgW = (m[0]+pa["arenaSz"][0]+m[2]) * nHMInRow # image width
        imgH = self.rtdParams["outputHMH"] * nHMInColInAggr # image height 
        if self.simTyp.startswith("mComp"): nLbls = self.nLbls
        else: nLbls = [pa["nLbl"]]
        for nLbl in nLbls:
            img = np.zeros((imgH, imgW), np.uint16) 
            # store the image
            self.rGraph[f'07_heatmap_{nLbl}'] = dict(img=img, offset=[0,0])
        ##### [end] prepare for real-time heatmap drawing -----
        
        # set the graph key to show while generating simulated data 
        self.gKey = f'07_heatmap_{nLbls[0]}'

        # set some general parameters for figures 
        setPLTParams(nRow=pa["gNRow"], nCol=pa["gNCol"])

        if self.flags["dProbDist"]:
            ### draw probability distributions
            kwa = dict(mpl=self.mpLst, mpp=self.mpProb, 
                       acdp=self.prob["AD"], acdpXLim=300, 
                       iadp=self.prob["ID"], iadpXLim=300)
            rGraph = drawProbDistributions(kwa) 
            self.rGraph["00_mp"] = dict(img=rGraph["mp"], offset=[0,0])
            self.rGraph["01_adp"] = dict(img=rGraph["adp"], offset=[0,0])
            self.rGraph["02_idp"] = dict(img=rGraph["idp"], offset=[0,0])

        self.flags["runningThread"] = True
        if self.simTyp.startswith("n"):
        # simulation of n1, n6 or n10
            startTaskThread(self, "runSimN", self.runSimN, wmSz="tiny")
        elif self.simTyp.startswith("mComp"):
        # simulation of n1, n6 and n10, then compare motion values 
            startTaskThread(self, "runSimMComp", self.runSimMComp, wmSz="tiny")
        elif self.simTyp.startswith("sweep"):
            startTaskThread(self, "runSweep", self.runSweep, wmSz="tiny")

        # hide panels except 'mp'
        for pk in ["tp", "ml", "mr"]: self.panel[pk].Hide()

        self.panel["mp"].SetFocus()

    #---------------------------------------------------------------------------
   
    def runSimN(self):
        """ Run simulation

        Args:
            None

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        pa = self.mPa

        # array for storing bundled data
        bData = np.zeros((pa["nOutput"], int(pa["nDP"]/pa["dPtIntvSec"])))
        acdS = [] # activity durations from the simulated data
        indS = [] # inactivity durations from the simulated data
       
        if self.simTyp == "n1":
        # n1 simulation
            
            for oi in range(pa["nOutput"]):
                # ant aggregate instance
                agg = AAggregate(self,
                                 params=dict(nw = 1))
                '''
                kwa = dict(acdp=acdPLst[0], acdpXLim=300, 
                           iadp=iadPLst[0], iadpXLim=300)
                rGraph = drawProbDistributions(kwa) 
                self.rGraph["01_adp"] = dict(img=rGraph["adp"], offset=[0,0])
                self.rGraph["02_idp"] = dict(img=rGraph["idp"], offset=[0,0])
                '''
                print("\n")
                kw = dict(
                        simI0=0, lLen0=1,
                        simI1=oi, lLen1=pa["nOutput"],
                        )
                # generate data and draw intensity plot
                _bD = self.genDataNdrawInt(agg, kw)
                if _bD is None:
                    self.q2m.put(("interrupted",), True, None)
                    return
                bData[oi,:] = np.asarray(_bD) # store the summed data
                acdS.append(agg.rALst)
                indS.append(agg.rILst)
                self.q2m.put(("incOutputIdx",), True, None) 

        else:
        # n6 or n10 simulation

            for oi in range(pa["nOutput"]):

                # ant aggregate instance
                agg = AAggregate(self)
                
                print("\n")
                kw = dict(
                        simI0=0, lLen0=1,
                        simI1=oi, lLen1=pa["nOutput"],
                        ) 
                # generate data and draw intensity plot
                _bD = self.genDataNdrawInt(agg, kw)
                try:
                    _bD = np.asarray(_bD)
                    # sum up the results of all workers into a single data
                    _bD = np.sum(_bD, 0)
                except:
                    self.q2m.put(("interrupted",), True, None)
                    return
                # store the summed data
                bData[oi,:] = _bD
                acdS.append(agg.rALst)
                indS.append(agg.rILst)
                self.q2m.put(("incOutputIdx",), True, None) 

        args = ("finished", bData, acdS, indS,)
        self.q2m.put(args, True, None) 

    #---------------------------------------------------------------------------
   
    def runSimMComp(self):
        """ run simulations of N1, 6 & 10 and compare motoin & mean power

        Args:
            None

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        pa = self.mPa

        rows = pa["nOutput"]*3
        cols = int(pa["nDP"]/pa["dPtIntvSec"])
        bData = np.zeros((rows, cols)) # for bundled data
        acdS = {} # simulated activity durations
        indS = {} # simulated inactivity durations
        l4mComp = [] # labels for motion & mean power comparison
        aggC = [] # aggregate instance for a colony. 
                  # stored to use it for n1, n6 & n10.
                
        for ni, nLbl in enumerate(self.nLbls):
            acdS[nLbl] = []
            indS[nLbl] = []
            
            for oi in range(pa["nOutput"]):
            # output index (oi) = colony index
            
                nWorkerInG = self.n[ni] # number of workers in an aggregate 

                ### ant aggregate instance
                if ni == 0:
                    # 17 (1, 6, 10) workers from the same colony will be used
                    aggC.append(AAggregate(self, dict(nw=np.sum(self.n))))
                else:
                    ### remove already used probability items from the list 
                    for probK in aggC[oi].prob.keys():
                        cutI = self.n[ni-1]
                        aggC[oi].prob[probK] = aggC[oi].prob[probK][cutI:]
                aggC[oi].nw = nWorkerInG # update the N of workers
        
                print("\n")
                kw = dict(
                            nw=nWorkerInG,
                            simI0=ni, lLen0=len(self.nLbls),
                            simI1=oi, lLen1=pa["nOutput"],
                            ) 
                # generate data
                _bD = self.genDataNdrawInt(aggC[oi], kw)
                if _bD is None:
                    self.q2m.put(("interrupted",), True, None)
                    return
                try:
                    if nWorkerInG == 1:
                        _bD = _bD[0]
                    else:
                        # sum up the results of all workers into a single data
                        _bD = np.sum(np.asarray(_bD), 0).tolist()
                except:
                    self.q2m.put(("interrupted",), True, None)
                    return

                _idx = ni*pa["nOutput"] + oi
                # store the data
                bData[_idx,:] = _bD
                # store the N label
                l4mComp.append("%08is_%s"%(oi+1, nLbl))
                ### store durations
                acdS[nLbl].append(aggC[oi].rALst)
                indS[nLbl].append(aggC[oi].rILst)
 
                self.q2m.put(("incOutputIdx",), True, None) 

            self.q2m.put(("incNIdx",), True, None) 

        args = ("finished", bData, acdS, indS, l4mComp)
        self.q2m.put(args, True, None)  
                    
    #---------------------------------------------------------------------------
   
    def runSweep(self):
        """ sweeping thresholds to see resultant mRat2hrTo20mLst

        Args:
            None

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        startTime = time()

        pa = self.mPa

        # list for storing mean ratio of 
        #   mean power value in 2hr-20min to mean power value in <20min
        mRat2hrTo20mLst = [] 

        adbLst = list(range(pa["thR"]["ADB"][0], pa["thR"]["ADB"][1]+1, 2)) 
                # Step is '2'.
                # because the probability data are drawn from intensity data
                # bundled with 2-second-interval @2022-10-10
        indLst = list(range(pa["thR"]["IDB"][0], pa["thR"]["IDB"][1]+1, 2))

        if self.simTyp.endswith("ADB"):
            plotTitle = "Activity duration boost threshold sweep"
            tThLst = adbLst 
        elif self.simTyp.endswith("IDB"):
            plotTitle = "Inactivity duration boost threshold sweep"
            tThLst = indLst 

        '''
        nXTicks = 10
        xLbls = [str(x) for x in _tThLst] 
        xTPStep = int(np.ceil(len(_tThLst)/nXTicks))
        xTLStep = int(np.ceil((_tThLst[-1]-_tThLst[0])/nXTicks))
        xTLbls = [str(x) for x in range(_tThLst[0], _tThLst[-1]+1, xTLStep)]
        xTLPos = list(range(0, len(_tThLst), xTPStep))
        '''

        flagBreak = False
        for i, _adbBS in enumerate(adbLst):
            for j, _indBS in enumerate(indLst):
                _time = time()
                # set threshold range for activity adjustment 
                adb = [_adbBS, _adbBS] 
                # set threshold range for inactivity adjustment 
                idb = [_indBS, _indBS]

                # ant aggregate instance
                agg = AAggregate(self, dict(nw=pa["nWorkerInG"], 
                                            thR=dict(ADB=adb, IDB=idb)))

                print("\n")
                kw = dict(
                    nw=pa["nWorkerInG"], # n of workers in aggregate
                    simI0=i, lLen0=len(adbLst),
                    simI1=j, lLen1=len(indLst),
                    )
                # generate data and draw intensity plot
                bData = self.genDataNdrawInt(agg, kw)
                if bData == []:
                    self.q2m.put(("interrupted",), True, None)
                    return

                kw = dict(
                    bData=bData, # bundled data
                    nw=pa["nWorkerInG"], # n of workers in aggregate
                    nOutput=pa["nOutput"], # n of output PSD
                    psdMV=pa["psdMV"]["n01"] # max. value for PSD
                    )
                # calc mRat2hrTo20m in PSD 
                _mRat2hrTo20m = self.drawPSDWithSimData(kw)

                mRat2hrTo20mLst.append(_mRat2hrTo20m)

                _et = time()-_time
                msg = "Elapsed time for the run: "
                msg += str(timedelta(seconds=_et)) + "\n"
                msg += "mRat2hrTow20m: %.3f\n"%(_mRat2hrTo20m)
                print(msg)

                if len(mRat2hrTo20mLst) > 60:
                    if np.median(mRat2hrTo20mLst[-60:]) > 3.0:
                    # reached enough ratio, stop sweeping
                        if self.simTyp.endswith("ADB"):
                            tThLst = adbLst[:i+1] 
                        elif self.simTyp.endswith("IDB"):
                            tThLst = indLst[:j+1] 
                        flagBreak = True
                if flagBreak: break
            if flagBreak: break

        totalETime = timedelta(seconds=time()-startTime)
        print("Total elapsed time: %s"%(str(totalETime)))
        
        _tag = self.simTyp[-3:]
        np.save("sweep%s_x.npy"%(_tag), tThLst)
        np.save("sweep%s_mRat2hrTo20mLst.npy"%(_tag), mRat2hrTo20mLst) 

        args = ("finished", tThLst, mRat2hrTo20mLst, plotTitle)
        self.q2m.put(args, True, None)   

    #---------------------------------------------------------------------------
    
    def convtHeatmap2Img(self, img):
        """ Convert heatmap array to an image to display 

        Args:
            img (numpy.ndarray): input image array

        Returns:
            img (numpy.ndarray): output image array
        """
        if DEBUG: logging.info(str(locals()))

        if np.max(img) > 0:
            img = (img/np.max(img)) * 255 # normalize to 0-255
        img = img.astype(np.uint8)
        '''
        w, h = np.array([img.shape[1],img.shape[0]]) * self.mPa["arenaM4D"] 
        # resize
        img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
        '''
        # convert to color image 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img 

    #---------------------------------------------------------------------------

    def writeInfoOnGraph(self, img, imgW, imgH, tCol, title, m, nLbl=""):
        """ Write title and some other info on graph 
        
        Args:
            img (numpy.ndarray): Image to write
            imgW (int): Image width
            imgH (int): Image height
            tCol (tuple): Text color
            title (str): Title
            m (tuple): Margins (left, top, right, bottom)
            nLbl (str): n01, 06, ..
        
        Returns:
            None
        """
        if DEBUG: logging.info(str(locals()))

        pa = self.mPa
        rtdp = self.rtdParams
        f4g = self.cvFont4g

        ### write title
        tx = int(imgW/2) - int(len(title)/2*f4g["fW"])
        ty = f4g["fH"] 
        cv2.putText(img, title, (tx, ty), f4g["typ"], f4g["scale"], tCol)

        ### write output info text 
        if title.lower().startswith("heatmap"):
            for oi in range(pa["nOutput"]):
                txt = f'[{nLbl}] output-{oi:02d}'
                xIdx = oi % rtdp["nHMInRow"]
                tx = m[0] + 5
                tx += (oi % rtdp["nHMInRow"]) * (m[0]+pa["arenaSz"][0]+m[2])
                ty = m[1] + f4g["fH"]
                ty += int(oi/rtdp["nHMInRow"])*rtdp["outputHMH"]
                cv2.putText(img, txt, (tx, ty), f4g["typ"], f4g["scale"], tCol)

        else:
            if self.simTyp.startswith("n"): txt = "[%s] output-01"%(pa["nLbl"])
            elif self.simTyp.startswith("mComp"): txt="[n1] output-01"
            tx = m[0]+5
            ty += f4g["fH"] 
            cv2.putText(img, txt, (tx, ty), f4g["typ"], f4g["scale"], tCol)

    #---------------------------------------------------------------------------

    def setGraphOffset(self, v, move=10):
        """ Set offset of the graph 
        
        Args:
            v (list/str): offset values for x & y coordinates
        
        Returns:
            None
        """
        if DEBUG: logging.info(str(locals()))

        if type(v) is not str: return

        prevV = self.rGraph[self.gKey]["offset"]
        prevVStr = str(prevV).strip("[]")

        if v in ["to_left", "to_right"]:
            if v == "to_left": sign = -1
            else: sign = 1
            vItems = v.split("_")
            newX = prevV[0] + move*sign
            v = [newX, prevV[1]]

        elif v in ["to_up", "to_down"]:
            if v == "to_up": sign = -1
            else: sign = 1
            vItems = v.split("_")
            newY = prevV[1] + move*sign
            v = [prevV[0], newY]

        else:
        # this value is string entered in 'self.offset_txt'
            try:
                v = [int(x) for x in v.split(",")] # convert to integer
            except: # failed to convert to integer values
                self.offset_txt.SetValue(prevVStr)
                return
            if len(v) < 2: # not enough values
                self.offset_txt.SetValue(prevVStr)
                return
            elif len(v) > 2: # too many
                v = v[:2] # use the first two values
        
        self.offset_txt.SetValue(str(v).strip("[]"))
        self.rGraph[self.gKey]["offset"] = v 
        self.panel["mp"].Refresh()

    #---------------------------------------------------------------------------
    
    def save(self):
        """ Save graphs 

        Args: None

        Returns: None
        """
        if DEBUG: logging.info(str(locals()))

        if self.rGraph == {}: return

        msg = "Saved -----\n"
        for k in sorted(self.rGraph.keys()):
            ss = k.split("_")
            ss.pop(0)
            fn = ""
            for _ss in ss: fn += _ss + "_"
            fn = fn.rstrip("_") + ".png"
            msg += fn + "\n" 
            # save graph image
            cv2.imwrite(path.join(self.rsltDir, fn), self.rGraph[k]["img"]) 
                            
        wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
        self.panel["mp"].SetFocus()

    #---------------------------------------------------------------------------
    
    def onClose(self, event):
        """ Close this frame. 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))
        
        if self.th is not None: self.q2t.put(("quit",), True, None)
        wx.CallLater(100, stopAllTimers, self.timer)
        # ?? self.Destroy hangs after 'fig.canvas.draw()' in convt_mplFig2npArr
        #wx.CallLater(100, self.Destroy)
        wx.CallLater(200, sys.exit)

    #---------------------------------------------------------------------------


#===============================================================================

class AntSimApp(wx.App):
    def OnInit(self):
        if DEBUG: MyLogger.info(str(locals()))
        self.frame = AntSimFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-w': GNU_notice(1)
        elif sys.argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        app = AntSimApp(redirect = False)
        app.MainLoop()




