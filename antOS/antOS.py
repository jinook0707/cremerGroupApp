# coding: UTF-8
"""
AntOS; Ant Observing System
Developed from CATOS (Oh & Fitch 2016; Behavior research methods 49-1 p.13-23)
Purpose of AntOS is long-term observation of an ant nest.

last edited on 2023-07-14

Dependency:
    Python (3.7)
    wxPython (4.0)
    OpenCV (3.4)
    NumPy (1.18)

----------------------------------------------------------------------
Copyright (C) 2020 Jinook Oh & Sylvia Cremer 
in Institute of Science and Technology Austria. 
- Contact: jinook.oh@ist.ac.at/ jinook0707@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
----------------------------------------------------------------------

Changelog -----
versions below v.0.1: CATOS initial development/ testing/ pilot work
                        for training cats 
v.0.2: Used for experiments with pigeons and common marmoset monkeys. 
v.0.3.202010: Developing a version for observing ants.
"""

import sys, queue, pickle
from threading import Thread
from platform import uname
from os import path, mkdir
from copy import copy
from time import time, sleep
from datetime import datetime, timedelta
from glob import glob
from random import shuffle
from sys import argv
from queue import Queue

import wx, wx.adv, cv2
import wx.lib.scrolledpanel as SPanel 

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is
sys.path.append(FPATH) # add FPATH to path
P_DIR = path.split(FPATH)[0] # parent directory of the FPATH
sys.path.append(P_DIR) # add parent directory 

from modFFC import *
from modCV import *

MyLogger = setMyLogger("AntOS")

##### [begin] import mods -----
FLAG_MOD = dict(
                    videoIn = True,
                    videoOut = False,
                    audioIn = False,
                    audioOut = False,
                    arduino = False,
                    sessionMngr = True,
                    )
FLAG_MOD["raspGPIO"] = False # flag for raspberry pi
FLAG_MOD["raspCSICam"] = False # flag for raspberry pi 
### import for Raspberry pi
if uname().machine.startswith("arm"):
    from modRasp import RaspGPIO
    from modRasp import RaspCSICam
    FLAG_MOD["raspGPIO"] = True
    FLAG_MOD["raspCSICam"] = True
    FLAG_MOD["arduino"] = False

for mn in FLAG_MOD.keys():
    if FLAG_MOD[mn]:
        if mn == "videoIn": from modCV import VideoIn
        elif mn == "videoOut": from modCV import VideoOut
        elif mn == "audioIn": from modAudioIn import AudioIn
        elif mn == "audioOut": from modAudioOut import AudioOut
        elif mn == "arduino": from modArduino import Arduino
        elif mn == "sessionMngr": from modSessionMngr import ESessionManager
##### [end] import mods -----

DEBUG = False 
__version__ = "0.3.202301"

#===============================================================================

class CATOSFrame(wx.Frame):
    """ Frame of AntOS to control experiment

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self):
        if DEBUG: MyLogger.info(str(locals()))

        wPos = (0, 20)
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], int(wg[3]*0.95))
        wx.Frame.__init__(
              self,
              None,
              -1,
              "AntOS v.%s"%(__version__), 
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX),
              )
        self.SetBackgroundColour('#333333')
        iconPath = path.join(FPATH, "icon.png")
        if __name__ == '__main__' and path.isfile(iconPath):
            self.SetIcon(wx.Icon(iconPath)) # set app icon
        # frame close event
        self.Bind(wx.EVT_CLOSE, self.onClose)
        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        # frame resizing
        updateFrameSize(self, (wSz[0], wSz[1]+self.statusbar.GetSize()[1]))

        ##### [begin] setting up attributes -----
        self.expmtTypes = ["antObservation2020"]
        self.expmtType = self.expmtTypes[-1]
        self.btnImgDir = path.join(P_DIR, "image")
        self.cTags = ["ants", "focalAntMarker"] # tags of colors to track
        configV = self.config("load") # load configuration values
        self.configV = configV
        self.classTag = "AntOS"
        ### output folder check
        outputFP = path.join(FPATH, 'output') # output folder path
        if not path.isdir(outputFP): mkdir(outputFP)
        self.outputFP = outputFP
        self.setLogFP() # determine log file path
        self.numROIs = configV["numROIs_cho"] # number of Region Of Interest
        self.updateROIFP() # update folder paths for ROIs
        # audio file to load
        #self.audio_files = ['input/snd_fly.wav', 'input/pos_fb.wav']
        self.wSz = wSz # window size
        self.fonts = getWXFonts()
        self.camIdx = getCamIdx() # get indices of cams
        self.raspCSICamIdx = 99
        if FLAG_MOD["raspCSICam"]: self.camIdx.append(self.raspCSICamIdx)
        if len(self.camIdx) == 0:
            msg = "No cameras found."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            self.Destroy()
            return
        self.chosenCamIdx = []
        self.q2m = queue.Queue() # queue from thread to main
        self.flags = dict(
                            blockUI = False, # block user input
                            raspberryPi = False,
                            tempMeasure = True,
                            detCol = configV["detCol_chk"],
                            showFrame = configV["showFrame_chk"],
                            showCol = configV["showCol_chk"], 
                            )
        self.pressedPipette = "" # pressed pipette button name
        if uname().machine.startswith("arm"): self.flags["raspberryPi"] = True 
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for status bar message display 
        # init dict to contain necessary modules
        self.mods = dict(arduino=None, 
                         raspGPIO=None,
                         sessionMngr=None,
                         videoIn={}, 
                         videoOut=None, 
                         audioIn=None, 
                         audioOut=None)
        self.program_start_time = time()
        self.isChkCamViewOn = False # checking camera view is on/off
        self.mlWid = [] # wx widgets in middle left panel 
        self.vOut = {} # video out (on-screen) info
        self.hsv = {} # current HSV values to find
        for cTag in self.cTags:
            self.hsv[cTag] = {}
            for ck in ["H", "S", "V"]:
                self.hsv[cTag][ck] = {}
                for mm in ["Min", "Max"]:
                    self.hsv[cTag][ck][mm] = configV[f'c-{cTag}-{ck}-{mm}_sld']
        ##### [end] setting up attributes ----- 

        btnSz = (35, 35)
        vlSz = (-1, 25)
        ### create panels and widgets
        for pk in pi.keys():
            self.panel[pk] = SPanel.ScrolledPanel(self, 
                                                  pos=pi[pk]["pos"],
                                                  size=pi[pk]["sz"],
                                                  style=pi[pk]["style"])
            bgColor = pi[pk]["bgColor"]
            fgColor = getConspicuousCol(bgColor)
            self.panel[pk].SetBackgroundColour(bgColor)
            self.panel[pk].mousePressedPt = [-1, -1]
            w = [] # widge list; each item represents a row in the panel 
            if pk == "tp": # top panel
                w.append([
                    {"type":"sTxt", "label":"Experiment type: ", "nCol":1,
                     "fgColor":fgColor},
                    {"type":"cho", "nCol":1, "name":"expmtType", 
                     "choices":self.expmtTypes, "size":(200,-1),
                     "val":self.expmtType, 
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":25}, 
                    {"type":"sTxt", "label":"Cam index: ", "nCol":1,
                     "fgColor":fgColor},
                    {"type":"cho", "nCol":1, "name":"camIdx", "size":(50,-1),
                     "choices":[str(x) for x in self.camIdx], 
                     "val":str(self.camIdx[0])},
                    {"type":"btn", "name":"addCam", "label":"+", "nCol":1,
                     "tooltip":"add camera", "size":btnSz},
                    {"type":"btn", "name":"remCam", "label":"-", "nCol":1,
                     "tooltip":"add camera", "size":btnSz},
                    {"type":"txt", "nCol":1, "name":"chosenCamIdx",
                     "val":str(self.chosenCamIdx), "style":wx.TE_READONLY, 
                     "size":(100,-1)},
                    {"type":"btn", "name":"chkCamView", "label":"", "nCol":1,
                     "img":path.join(P_DIR,"image","camera.png"), 
                     "bgColor":"#333333", "tooltip":"check camera view",
                     "size":btnSz},
                    {"type":"sLn", "size":vlSz, "nCol":3,
                     "style":wx.LI_HORIZONTAL,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":25},
                    {"type":"txt", "nCol":1, "name":"pStart", "size":(100,-1),
                     "val":"0:00:00", "style":wx.TE_READONLY,
                     "fgColor":"#00cc00", "bgColor":"#000000"},
                    {"type":"sTxt", "label":"since program-start", "nCol":1, 
                     "fgColor":"#006600", "font":self.fonts[0],
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":25},
                    {"type":"txt", "nCol":1, "name":"sStart", "size":(100,-1),
                     "val":"0:00:00", "style":wx.TE_READONLY,
                     "fgColor":"#cc9900", "bgColor":"#000000"},
                    {"type":"sTxt", "label":"since session-start", "nCol":1, 
                     "fgColor":"#996600", "font":self.fonts[0],
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":25},
                    ])
            if w != []:
                self.gbs[pk] = wx.GridBagSizer(0,0) 
                addWxWidgets(w, self, pk) 
                self.panel[pk].SetSizer(self.gbs[pk])
                self.gbs[pk].Layout()
                self.panel[pk].SetupScrolling()

        self.initMLWidgets() # init middle left side panel
        self.pTime_txt = wx.FindWindowByName("pStart_txt", self.panel["tp"])
        self.sTime_txt = wx.FindWindowByName("sStart_txt", self.panel["tp"])

        ### keyboard binding
        quitId = wx.NewIdRef(count=1)
        self.Bind(wx.EVT_MENU, self.onClose, id=quitId)
        accel_tbl = wx.AcceleratorTable([ 
                                (wx.ACCEL_CTRL, ord('Q'), quitId)
                                ])
        self.SetAcceleratorTable(accel_tbl)

        ### set timer for processing message and 
        ###   updating the current running time
        self.timer["reg"] = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimer, self.timer["reg"])
        self.timer["reg"].Start(100)

        self.Bind( wx.EVT_CLOSE, self.onClose )

        self.log("Beginning of the program.")

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
        if sys.platform.startswith("win"):
            style = (wx.TAB_TRAVERSAL|wx.SIMPLE_BORDER)
        else:
            style = (wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        pi = {} # information of panels
        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], 60), bgColor=(175,175,175), 
                        style=style)
        tpSz = pi["tp"]["sz"]
        # middle left panel
        mlW = max(300, int(wSz[0]*0.2)) 
        pi["ml"] = dict(pos=(0, tpSz[1]), sz=(mlW, wSz[1]-tpSz[1]),
                        bgColor=(200,200,200), style=style)
        mlSz = pi["ml"]["sz"]
        # middle right panel
        pi["mr"] = dict(pos=(mlSz[0], tpSz[1]),
                        sz=(wSz[0]-mlSz[0]-10, wSz[1]-tpSz[1]-10), 
                        bgColor=(75,75,75),
                        style=style)
        return pi

    #---------------------------------------------------------------------------
   
    def initMLWidgets(self):
        """ Set up wxPython widgets
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        pk = "ml"
        pSz = self.pi[pk]["sz"]
        fgColor = getConspicuousCol(self.pi[pk]["bgColor"])
        
        for i, w in enumerate(self.mlWid): # through widgets in the panel
            try:
                self.gbs[pk].Detach(w) # detach 
                w.Destroy() # destroy
            except:
                pass
        
        configV = self.configV
        w = [] # each item represents a row in the left panel 
        if self.expmtType == "antObservation2020":
            nCol = 2 
            hlSz = (int(pSz[0]*0.75), -1)
            lm = int(pSz[0]*0.05)
            txtW = 60 
            ### delay time before starting initialize session
            w.append([{"type":"sTxt", "nCol":1, "fgColor":fgColor,
                       "label":"Delay (sec) to the start", "border":lm,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)},
                      {"type":"txt", "nCol":1, "size":(txtW,-1),
                       "name":"initSessionDelay", "numOnly":True, "border":10,
                       "val":str(configV["initSessionDelay_txt"]),
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)}]) 
            ### set session duration
            w.append([{"type":"sTxt", "label":"Session-duration (hours)", 
                       "nCol":1, "fgColor":fgColor, "border":lm,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)},
                      {"type":"txt", "nCol":1, "size":(txtW,-1),
                       "name":"sessionDur", "numOnly":True, "border":10,
                       "val":str(configV["sessionDur_txt"]),
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)}])
            ### for starting session manager
            w.append([{"type":"sTxt", "label":"Session start", "nCol":nCol, 
                       "name":"sessionStart", "fgColor":fgColor, "border":lm,
                       "flag":wx.LEFT}])
            w.append([{"type":"btn", "name":"sessionMngrStartStop", "label":"",
                       "tooltip":"start/stop session manager", "size":hlSz,
                       "img":path.join(P_DIR,"image","startStop.png"),
                       "nCol":nCol, "bgColor":"#333333", "border":lm,
                       "flag":wx.LEFT}])
            w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":5}])
            # !! Currently (2023.Jan), only 1 camera is being used. If multiple
            #    cameras are required, ROIs setting in GUI and processing in  
            #    modSessionMngr must be modified for each camera !!
            ### number of ROIs
            w.append([{"type":"sTxt", "label":"number of ROIs",
                       "nCol":1, "fgColor":fgColor, "border":lm,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.LEFT)},
                      {"type":"cho", "nCol":1, "name":"numROIs", 
                       "size":(50,-1), "val":str(self.numROIs),
                       "flag":wx.ALIGN_CENTER_VERTICAL,
                       "choices":[str(x+1) for x in range(3)]}])
            ### ROI (region of interest)
            for roiIdx in range(self.numROIs):
                w.append([{"type":"sTxt", "label":"ROI-%i"%(roiIdx), 
                           "nCol":nCol, "fgColor":fgColor, "border":lm,
                           "flag":wx.LEFT}])
                w.append([{"type":"txt", "name":"roi%i"%(roiIdx), 
                           "val":configV["roi%i_txt"%(roiIdx)], "nCol":nCol,
                           "size":hlSz, "border":lm, "flag":wx.LEFT}])
                w.append([{"type":"sTxt","label":" ","nCol":nCol,"border":5}])

            ### set interval for temperature recording in log file 
            w.append([{"type":"sTxt", "label":"Temp. recording (seconds)", 
                       "nCol":1, "fgColor":fgColor, "border":lm,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)},
                      {"type":"txt", "nCol":1, "size":(txtW,-1),
                       "name":"tempRead", "numOnly":True, "border":10,
                       "val":str(configV["tempRead_txt"]),
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)}])

            ### set interval for saving each ROI images from the frame image
            w.append([{"type":"sTxt", "label":"Image file saving (seconds)", 
                       "nCol":1, "fgColor":fgColor, "border":lm,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)},
                      {"type":"txt", "nCol":1, "size":(txtW,-1),
                       "name":"frameSav", "numOnly":True, "border":10,
                       "val":str(configV["frameSav_txt"]),
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)}])
            w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":5}])
           
            ### checkbox for color detection 
            w.append([{"type":"chk", "nCol":1, "name":"detCol", 
                       "label":"detect color", 
                       "val":configV["detCol_chk"], "style":wx.CHK_2STATE, 
                       "border":lm, "fgColor":fgColor, "flag":wx.LEFT}]) 

            ### color pipette button & slides for color detection 
            for cTag in self.cTags:
                w.append([
                          {"type":"sTxt", "label":f'Color to detect [{cTag}]', 
                           "nCol":1, "fgColor":fgColor, "border":lm,
                           "flag":(wx.ALIGN_CENTER_VERTICAL|wx.ALL)},
                          {"type":"btn", "nCol":1, "size":(35,35), 
                           "name":f'pipette-{cTag}',
                           "img":path.join(self.btnImgDir, "pipette.png")}]) 
                for mm in ["Min", "Max"]:
                    _lbl = f'[{mm}]'
                    w.append([{"type":"sTxt", "nCol":nCol, "fgColor":fgColor,
                               "label":_lbl, "border":lm, "flag":wx.LEFT} 
                               ])
                    for ck in ["H", "S", "V"]:
                        if ck == "H": maxVal = 180
                        else: maxVal = 255
                        wn = "c-%s-%s-%s"%(cTag, ck, mm)
                        w.append([{
                                    "type":"sld", "nCol":nCol, "name":wn,
                                    "val":configV[wn+"_sld"], "size":hlSz,
                                    "style":wx.SL_VALUE_LABEL, "minValue":0, 
                                    "maxValue":maxVal, "border":lm, 
                                    "flag":wx.LEFT,
                                    }])
            w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":5}])
            
            ### checkbox for showing frames 
            w.append([{"type":"chk", "nCol":nCol, "name":"showFrame", 
                     "label":"show frames", "val":self.flags["showFrame"], 
                     "style":wx.CHK_2STATE, "border":lm, "fgColor":fgColor,
                     "flag":wx.LEFT}]) 
            w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":5}])
            
            ### checkbox for showing color detection result
            w.append([{
                        "type":"chk", "nCol":nCol, "name":"showCol", 
                        "label":"show color detection result", 
                        "val":configV["showCol_chk"], "style":wx.CHK_2STATE, 
                        "border":lm, "fgColor":fgColor, "flag":wx.LEFT
                     }]) 
            w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":5}])
            
            ### leaving note in log file 
            w.append([{"type":"sTxt", "nCol":nCol, "fgColor":fgColor,
                       "label":"leave note in LOG", "border":lm, 
                       "flag":wx.LEFT}])
            w.append([{"type":"txt", "val":"", "nCol":nCol, "name":"note",
                       "style":wx.TE_PROCESS_ENTER, "procEnter":True, 
                       "border":lm, "flag":wx.LEFT, "size":hlSz}])
            w.append([{"type":"sTxt", "label":" ", "nCol":nCol, "border":5}])
            if FLAG_MOD["arduino"]:
                ### sending message to Arduino 
                w.append([{"type":"sTxt", "nCol":nCol, "fgColor":fgColor,
                           "label":"message to Arduino", "border":lm,
                           "flag":wx.LEFT}])
                w.append([{"type":"txt", "val":"", "nCol":nCol, 
                           "name":"arduino", "style":wx.TE_PROCESS_ENTER, 
                           "procEnter":True, "size":hlSz, "border":lm,
                           "flag":wx.LEFT}])
                w.append([{"type":"sTxt","label":" ","nCol":nCol,"border":5}])
           
        self.gbs[pk] = wx.GridBagSizer(0,0) 
        self.mlWid, pSz = addWxWidgets(w, self, pk)
        if pSz[0] > self.pi[pk]["sz"][0]:
            self.panel[pk].SetSize(pSz[0], self.pi[pk]["sz"][1])
        self.panel[pk].SetSizer(self.gbs[pk])
        self.gbs[pk].Layout()
        self.panel[pk].SetupScrolling()

        self.sessionStartSTxt = wx.FindWindowByName("sessionStart_sTxt",
                                                    self.panel["ml"])
    
    #---------------------------------------------------------------------------

    def startMods(self, mod='all', modArgs={}):
        """ start modules

        Args:
            mod (str): 'all' to start all possible modules or 
                a name string of a specific module to start
            modArgs (dict): other arguments

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        if mod == "all":
            mods = []
            for mk in self.mods.keys():
                if FLAG_MOD[mk]: mods.append(mk)
        else:
            mods = [mod]

        for mk in mods:
            if mk == "arduino":
                self.mods[mk] = Arduino(self, self.outputFP)
                if self.mods[mk].aConn == None:
                    msg = "Arduino chip is not found.\n"
                    msg = "Please connect it and retry."
                    show_msg(msg, self.panel)
                    return

            elif mk == "raspGPIO":
                self.mods["raspGPIO"] = RaspGPIO(self)

            elif mk == "videoIn":
                nOC = len(self.chosenCamIdx) # number of cams to open
                if nOC == 0: continue 

                ### make temporary panel, showing waiting message
                ###   and covering entire window.
                ###   (starting VideoIn module takes some time)
                msg = "Initiating camera ..."
                makeWaitingMessagePanel(self, self.wSz, msg=msg, bgCol=(0,0,10))

                def delayedExec(nOC, modArgs):
                # code to execute after displaying waiting message
                    # number of images in a row
                    nCol = int(np.ceil(np.sqrt(nOC)))
                    # number of images in a column 
                    nRow = int(np.ceil(nOC/ nCol)) 
                    cw = int(self.pi["mr"]["sz"][0]*0.99/nCol)
                    ch = int(self.pi["mr"]["sz"][1]*0.99/nRow)
                    self.vOut["nCol"] = nCol
                    self.vOut["nRow"] = nRow
                    self.vOut["imgW"] = cw 
                    self.vOut["imgH"] = ch
                    cRow = 0 # row index for cam view
                    cCol = 0 # column index for cam view
                    for cIdx in self.chosenCamIdx: # go through cam indices
                        # init videoIn module
                        if cIdx == self.raspCSICamIdx:
                            viMod = RaspCSICam(self, cIdx) 
                        else:
                            if "videoInFPSLimit" in modArgs.keys():
                                viMod = VideoIn(self, cIdx, 
                                          fpsLimit=modArgs["videoInFPSLimit"]) 
                            else:
                                viMod = VideoIn(self, cIdx) 
                        q2t = queue.Queue() # queue from main to thread
                        args = (self.q2m, q2t, self.outputFP,)
                        # thread of running the module
                        thrd = Thread(target=viMod.run, args=args) 
                        thrd.start()
                        pos = (cCol*cw, cRow*ch) # position for StaticBitmap
                        cCol += 1
                        if cCol == nCol: cCol = 0; cRow += 1
                        # static-bitmap to show the frame image
                        sBmp = wx.StaticBitmap(self.panel["mr"], -1,
                                               size=(cw,ch), pos=pos)
                        sBmp.Bind(wx.EVT_LEFT_UP, self.onMLBUp)
                        ### store objects in self.mods
                        self.mods["videoIn"][cIdx] = viMod
                        self.mods["videoIn"][cIdx].thrd = thrd
                        self.mods["videoIn"][cIdx].q2t = q2t
                        self.mods["videoIn"][cIdx].sBmp = sBmp 
                    self.tmpWaitingMsgPanel.Destroy() # destroy tmp. panel

                # start VideoIn with delay 
                if self.flags["raspberryPi"]: delay = 500
                else: delay = 10
                wx.CallLater(delay, delayedExec, nOC, modArgs) 
               
            elif mk == "videoOut":
                self.mods["videoOut"] = VideoOut(self)
                self.mods["videoOut"].Show(True)

            elif mk == "audioin":
                self.mods["audioIn"] = AudioIn(self)
                self.mods["audioIn"].thrd = Thread(
                                    target=self.mods["audioIn"].run
                                    )
                self.mods["audioIn"].thrd.start()

            elif mk == "audioOut":
                self.mods["audioOut"] = AudioOut(self, self.audio_files)

            elif mk == "sessionMngr":
                self.mods["sessionMngr"] = ESessionManager(self)
        
    #---------------------------------------------------------------------------
        
    def stopMods(self, mod='all', flag =''):
        """ stop modules

        Args:
            mod (str): 'all' to stop all possible modules or 
                a name string of a specific module to stop

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        if mod == "all":
            mods = []
            for mk in self.mods.keys():
                if FLAG_MOD[mk]: mods.append(mk)
        else:
            mods = [mod]

        for mk in mods:
            ### close the mod.
            if mk == "videoIn":
                for viK in self.mods[mk].keys():
                    self.mods[mk][viK].q2t.put("quit", True, None)
                    self.mods[mk][viK].thrd.join()
                    self.mods[mk][viK].sBmp.Destroy()
                    wx.CallLater(10, self.mods[mk][viK].close)
                    #wx.CallLater(1000, self.mods[mk][viK].thrd.join)
                self.mods[mk] = {}
            elif mk == "audioIn":
                if self.mods[mk] != None:
                    self.mods[mk].log_q.put('main/quit/True', True, None)
                    self.mods[mk] = None 
            else:
                if self.mods[mk] != None:
                    self.mods[mk].close()
                    self.mods[mk] = None
    
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

        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))

        if objName in ["addCam_btn", "remCam_btn"]:
            cho = wx.FindWindowByName("camIdx_cho", self.panel["tp"])
            chosenCamIdx = int(cho.GetString(cho.GetSelection()))
            if objName == "addCam_btn":
                if not chosenCamIdx in self.chosenCamIdx:
                    self.chosenCamIdx.append(chosenCamIdx)
            elif objName == "remCam_btn":
                if chosenCamIdx in self.chosenCamIdx:
                    self.chosenCamIdx.remove(chosenCamIdx)
            sTxt = wx.FindWindowByName("chosenCamIdx_txt", self.panel["tp"])
            sTxt.SetValue(str(self.chosenCamIdx))
        
        if objName == "chkCamView_btn":
            ### Turn On/Off webcam to check its views
            if not FLAG_MOD["videoIn"]: return # no videoIn module
            if self.isChkCamViewOn:
                self.stopMods(mod='videoIn')
            else:
                if self.chosenCamIdx == []:
                    msg = "Empty camera index list."
                    wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                    return
                self.startMods(mod='videoIn')
            self.isChkCamViewOn = not self.isChkCamViewOn
        
        elif objName == "sessionMngrStartStop_btn":
            if self.mods["sessionMngr"] == None: 
            # session-manager not found. start a session
                if self.isChkCamViewOn:
                    self.stopMods(mod="videoIn")
                    sleep(0.5)
                self.startMods(mod='sessionMngr') # start session-manager
                if self.mods["sessionMngr"].state == "":
                # if failed to init session-manager
                    self.mods["sessionMngr"] = None 
            else:
            # end the session-manager.
                self.stopMods(mod='sessionMngr')

        elif objName == "trialStart_btn":
            if self.mods["sessionMngr"].state != "inSession": # not in session
                return 
            self.mods["sessionMngr"].init_trial()

        elif objName.startswith("pipette"):
            if self.pressedPipette != "": 
            # pipette button is already pressed, unpress it
                cursor = wx.Cursor(wx.CURSOR_ARROW)
                clickedBtnCol = wx.Colour(51,51,51)
                self.pressedPipette = ""
            else:
                cursor = wx.Cursor(wx.CURSOR_CROSS)
                clickedBtnCol = wx.Colour(50,255,50)
                self.pressedPipette = objName.replace("_btn", "")
            self.panel["mr"].SetCursor(cursor)
            obj.SetBackgroundColour(clickedBtnCol)
  
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
        
        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "expmtType_cho":
            if objVal != self.expmtType:
                self.expmtType = objVal # store experiment type
                self.initMLWidgets() # set up middle left panel

        elif objName == "numROIs_cho":
            self.numROIs = int(objVal) # update number of ROIs
            self.updateROIFP() # update folder paths for ROIs
            self.initMLWidgets() # set up middle left panel

    #---------------------------------------------------------------------------
    
    def onCheckBox(self, event, objName=""):
        """ wx.Checkbox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName in self.configV.keys():
            self.configV[objName] = objVal

        flagKey = objName.rstrip("_chk")
        if flagKey in self.flags.keys():
            self.flags[flagKey] = objVal

        if objName == "detCol_chk":
            if objVal != self.flags["showCol"]:
                widgetValue("showCol_chk", objVal, "set", self.panel["ml"])
                self.onCheckBox(None, "showCol_chk")

    #---------------------------------------------------------------------------
    
    def onSlider(self, event, objName=""):
        """ wx.Slider bar value chnaged 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: logging.info(str(locals()))

        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName.startswith("c-"):
            _, cTag, ck, mm = objName.split("-")
            mm = mm.rstrip("_sld")
            # update the current HSV color value
            self.hsv[cTag][ck][mm] = objVal

    #---------------------------------------------------------------------------

    def onEnterInTextCtrl(self, event, objName=""):
        """ Enter-key was pressed in wx.TextCtrl

        Args:
            event (wx.Event)
            objName (str, optional): objName, when emulating the event
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return
        
        if objName == 'note_txt':
            self.log(objVal)
            showStatusBarMsg(self, "'%s' is written in the log."%(objVal))
            obj.SetValue('')

        elif objName == "arduino_txt":
            if self.mods["arduino"] != None:
                # send a message to Arduino
                self.mods["arduino"].send(objVal.encode()) 
            obj.SetValue("")

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
        
        ret = preProcUIEvt(self, event, objName)
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if isNumOnly:
            keyCode = event.GetKeyCode()
            ### Allow numbers, backsapce, delete, left, right
            ###   and tab (for hopping between TextCtrls)
            allowed = [ord(str(x)) for x in range(10)]
            allowed += [wx.WXK_BACK, wx.WXK_DELETE, wx.WXK_TAB]
            allowed += [wx.WXK_LEFT, wx.WXK_RIGHT]
            if keyCode in allowed:
                event.Skip()
                return

    #---------------------------------------------------------------------------
    
    def onMLBUp(self, event):
        """ Processing when mouse L-buttton clicked 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: logging.info(str(locals()))

        if self.flags["blockUI"] or len(self.mods["videoIn"]) == 0: return

        sBmp = event.GetEventObject()
        mp = event.GetPosition()
        mState = wx.GetMouseState()

        if self.pressedPipette != "":
            cTag = self.pressedPipette.split("-")[1]
            img = convt_wxImg2cvImg(sBmp.Bitmap, fromBMP=True)
            b, g, r = img[mp[1], mp[0]] # get color value of the clicked pixel
            col = {}
            # convert it to HSV value
            col["H"], col["S"], col["V"] = rgb2cvHSV(r, g, b)
            ### set slider bars to HSV value
            for ck in ["H", "S", "V"]:
                if ck == "H":
                    maxVal = 180
                    diff = 10
                else:
                    maxVal = 255
                    diff = 25
                for mm in ["Min", "Max"]: # HSV min & max values
                    if mm == "Min": _col = max(0, col[ck]-diff)
                    elif mm == "Max": _col = min(maxVal, col[ck]+diff)
                    ### set slider value
                    sN = "c-%s-%s-%s_sld"%(cTag, ck, mm)
                    sld = wx.FindWindowByName(sN, self.panel["ml"])
                    sld.SetValue(_col)
                    # store the current HSV values
                    self.hsv[cTag][ck][mm] = _col
            
            ### init
            cursor = wx.Cursor(wx.CURSOR_ARROW)
            self.panel["mr"].SetCursor(cursor)
            btn = wx.FindWindowByName(f'pipette-{cTag}_btn', self.panel["ml"])
            btn.SetBackgroundColour("#333333")
            self.pressedPipette = ""

    #---------------------------------------------------------------------------

    def onTimer(self, event, flag=""):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        #if DEBUG: MyLogger.info(str(locals()))
        
        ### update log file path, if necessary
        d = path.basename(self.logFP).split("_")[-1] # date in log file
        _d = '%.2i'%(datetime.now().day) # current date
        if d != _d: self.setLogFP() 
        
        ### update several running time
        e_time = time() - self.program_start_time
        self.pTime_txt.SetValue(
                str(timedelta(seconds=e_time)).split('.')[0]
                )
        sMMod = self.mods["sessionMngr"]
        if sMMod != None and sMMod.sessionStartTime != -1:
            e_time = time() - sMMod.sessionStartTime
            self.sTime_txt.SetValue(
                    str(timedelta(seconds=e_time)).split('.')[0]
                    )
        
        ##### [begin] processing message -----
        ### Receive data from queue
        ### * if it's "displayMsg", keep receiving until getting 
        ###   the last queued message.
        ### * if the received data is other type than a simple message, 
        ###   then process it.
        rData = None
        while True:
            ret = receiveDataFromQueue(self.q2m)
            if ret == None:
                break
            else:
                rData = ret # store received data
                if ret[0] != "displayMsg": break 
        if rData == None: return

        if rData[0] == "displayMsg":
            showStatusBarMsg(self, rData[1], -1)
            return
        
        elif rData[0].startswith("finished"):
            self.callback(rData, flag)
        
        elif rData[0] == "frameImg":
            cIdx = rData[1]
            frame = rData[2]
            if cIdx in self.mods["videoIn"].keys():
                # store the frame image
                self.mods["videoIn"][cIdx].frame = frame
                if self.isChkCamViewOn: self.displayCamFrame(cIdx, frame)
        ##### [end] processing message ----- 

    #---------------------------------------------------------------------------
    
    def displayCamFrame(self, cIdx, frame):
        """ Display camera frames.

        Args:
            cIdx (int): Camera index.
            frame (numpy.ndarray): Frame image to display.
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        viMod = self.mods["videoIn"][cIdx]
        imgW = self.vOut["imgW"]
        imgH = self.vOut["imgH"]
        frame = cv2.resize(frame, (imgW,imgH)) # resize to display
        ### write info. string on frame
        cv2.putText(frame, # image
                    "Cam-%.2i"%(cIdx), # string
                    (5, 20), # bottom-left
                    cv2.FONT_HERSHEY_PLAIN, # fontFace
                    1.0, # fontScale
                    (0,50,100), # color
                    1) # thickness
        ### display frame image on corresponding StaticBitmap 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = wx.Image(imgW, imgH)
        img.SetData(frame.tobytes())
        viMod.sBmp.SetBitmap(img.ConvertToBitmap())

    #---------------------------------------------------------------------------
    
    def updateROIFP(self):
        """ update ROI output folder paths and make folders if necessary 

        Args: None

        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        self.roiOutputFP = []
        ### output folder for each ROI
        for roiIdx in range(self.numROIs):
            fp = path.join(self.outputFP, 'roi%i'%(roiIdx))
            if not path.isdir(fp): mkdir(fp)
            self.roiOutputFP.append(fp)

    #---------------------------------------------------------------------------
    
    def setLogFP(self):
        """ reset log file path 

        Args: None

        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        ts = get_time_stamp().split("_")
        logFN = "log_%s%s%s.txt"%(ts[0], ts[1], ts[2]) # log_yyyymmdd
        self.logFP = path.join(self.outputFP, logFN)

    #---------------------------------------------------------------------------
    
    def log(self, msg, tag="AntOS"):
        """ leave a log 

        Args:
            msg (str): Message for logging.
            tag (str): Class tag.
        
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        sMngr = self.mods["sessionMngr"]
        if sMngr == None or sMngr.sessionStartTime == -1:
            e_time = "-1"
        else:
            e_time = time() - sMngr.sessionStartTime
            e_time = "%.3f"%(e_time)
        txt = "%s, [%s], [sessionTime:%s]"%(get_time_stamp(), tag, e_time)
        txt += ", %s\n"%(msg)
        writeFile(self.logFP, txt)

    #---------------------------------------------------------------------------

    def config(self, operation):
        """ saving/loading configuration of an experiment 

        Args:
            operation (str): operation; save or load

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))
        
        configFP = path.join(FPATH, "config_%s"%(self.expmtType))

        ### default values of widgets
        defV = {"initSessionDelay_txt":1,
                "numROIs_cho":3, 
                "sessionDur_txt":336,
                "tempRead_txt":600,
                "frameSav_txt":3600,
                "roi0_txt":"110,0,550,1440",
                "roi1_txt":"660,0,550,1440",
                "roi2_txt":"1210,0,550,1440", 
                "detCol_chk":True,
                "showFrame_chk":True,
                "showCol_chk":False}
        ### add default values for the HSV slides of colors to track
        for cTag in self.cTags:
            for ck in ["H", "S", "V"]:
                for mm in ["Min", "Max"]:
                    if mm == "Min":
                        _val = 0
                    elif mm == "Max":
                        if ck == "H": _val = 180
                        elif ck == "S": _val = 255
                        elif ck == "V": _val = 110
                    wn = "c-%s-%s-%s_sld"%(cTag, ck, mm)
                    defV[wn] = _val

        if operation == "save":
            config = {}
            if self.expmtType == "antObservation2020":
                for wn in defV.keys(): 
                    w = wx.FindWindowByName(wn, self.panel["ml"])
                    val = widgetValue(w) # get the widget value 
                    nVal = str2num(val)
                    if nVal is not None: val = nVal
                    config[wn] = val
            fh = open(configFP, "wb")
            pickle.dump(config, fh)
            fh.close()
            return
        
        elif operation == "load":
            if not path.isfile(configFP):
                configV = defV
            else:
                fh = open(configFP, "rb")
                configV = pickle.load(fh)
                fh.close()
                for k in defV.keys():
                    if not k in configV.keys():
                        configV[k] = defV[k]
            return configV

    #---------------------------------------------------------------------------

    def onClose(self, event):
        """ called when the frame is closed 

        Args: event (wx.Event)

        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        stopAllTimers(self.timer)
        self.stopMods(mod='all')
        self.config("save") # save config
        self.log("End of the program.")
        wx.CallLater(1000, self.Destroy)

#===============================================================================

class CATOSApp(wx.App):
    def OnInit(self):
        if DEBUG: MyLogger.info(str(locals()))
        
        self.frame = CATOSFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

if __name__ == '__main__':
    if len(argv) > 1:
        if argv[1] == '-w': GNU_notice(1)
        elif argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        app = CATOSApp(redirect = False)
        app.MainLoop()




