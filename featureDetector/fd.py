# coding: UTF-8

"""
This is an open-source software written in Python for detecting
  ant blobs, motions, and certain behaviors from recorded video.
(App branched from pyABCoder.)

This program was coded and tested in Ubuntu 18.04 

Jinook Oh, Cremer group in Institute of Science and Technology Austria.
2021.May.
last edited: 2023-03-27

Dependency:
    wxPython (4.0)
    NumPy (1.17)
    OpenCV (4.1)

------------------------------------------------------------------------
Copyright (C) 2021 Jinook Oh & Sylvia Cremer. 
- Contact: jinook0707@gmail.com

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

Changelog
------------------------------------------------------------------------
v.0.1; 2021.05
    - Branched from pyABCoder.
v.0.2; 2021.07
    - Added LindaWP21.
    - Adding a feature to run entire folder of video for long time, so that
        a user configure parameters for each video file, store it as file,
        then run the folder for long time without any further input.
        - Added function to store paramerters to a file.
        - Added opening entire folder with video files.
        - Added function of sending notification email.
v.0.2; 2023.01
    - Storing values in left panel to config file.
    - Added sleepDet23; Detecting ant in sleep episode for Rebecca's experiment.
"""

import sys, queue, smtplib, ssl, re, traceback
from threading import Thread 
from os import getcwd, path
from copy import copy
from time import time, sleep
from datetime import timedelta
from glob import glob
from random import shuffle
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import wx, wx.adv, wx.grid
import wx.lib.scrolledpanel as SPanel 
import cv2
import numpy as np

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is
sys.path.append(FPATH) # add FPATH to path
P_DIR = path.split(FPATH)[0] # parent directory of the FPATH
sys.path.append(P_DIR) # add parent directory 

from procCV import ProcCV
from modCV import VideoRW
from modFFC import *
from eL import emailLogin

FLAGS = dict(
                debug = False,
                seqFR2jump = False,
                )
MyLogger = setMyLogger("featureDetector")
__version__ = "0.2.202301"

#===============================================================================

class FeatureDetectorFrame(wx.Frame):
    """ Frame for Ant Video Processor 
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self):
        if FLAGS["debug"]: logging.info(str(locals()))

        ### init frame
        if sys.platform.startswith("win"): wPos = (0,0)
        else: wPos = [0, 25]
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], int(wg[3]*0.85))
        title = "FeatureDetector v.%s"%(__version__)
        style = wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX)
        wx.Frame.__init__(self, None, -1, title, 
                          pos=tuple(wPos), size=tuple(wSz), style=style)
        self.SetBackgroundColour('#333333')
        ### set app icon
        iconPath = path.join(P_DIR, "image", "icon.tif")
        if __name__ == '__main__' and path.isfile(iconPath):
            self.SetIcon(wx.Icon(iconPath)) # set app icon
        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        # frame resizing
        updateFrameSize(self, (wSz[0], wSz[1]+self.statusbar.GetSize()[1]))
        # frame close event
        self.Bind(wx.EVT_CLOSE, self.onClose)
        
        ##### [begin] setting up attributes -----
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread 
        self.wPos = wPos # window position
        self.wSz = wSz # window size
        self.fonts = getWXFonts(initFontSz=8, numFonts=3)
        pi = self.setPanelInfo()
        self.pi = pi # pnael information
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for statusbar
        self.session_start_time = -1
        self.runningDur = 0 # accumulated duration (sec.) of continuous running
        self.logFP = path.join(FPATH, "log.txt")
        self.oData = [] # output data
        self.inputFPLst = [] # input file list (when folder was opened)
        self.inputFP = "" # input file path
        self.flags = {}
        self.flags["blockUI"] = False # block user input 
        self.flags["isRunning"] = False # is video analysis currently running
        self.flags["rsltRecording"] = False # recording video analysis result
        self.flags["notifyEmail"] = False # notify progress to email 
        self.rsltRecParam = {} # parameters for recording video analysis result
        self.rsltRecParam["szRat"] = 0.75 # ratio to the original frame size 
        self.rsltRecParam["fps"] = 60 # FPS for the result video
        self.allowedVideoExt = ["mp4", "mkv", "mov", "avi"]
        ### description of common parameters
        self.paramDesc = {} 
        d = "Number of iterations of morphologyEx (for reducing noise"
        d += " & minor features after absdiff from background image)."
        d += " [cv2.MORPH_OPEN operation]"
        self.paramDesc["preMExOIter"] = d
        d = "Number of iterations of morphologyEx (for reducing noise"
        d += " & minor features after absdiff from background image)."
        d += " [cv2.MORPH_CLOSE operation]"
        self.paramDesc["preMExCIter"] = d
        d = "Parameter for cv2.threshold in pre-processing."
        self.paramDesc["preThres"] = d
        d = "Parameters for hysteresis threshold of Canny function"
        d += " (in edge detection of difference from background image)."
        self.paramDesc["cannyTh"] = d
        d = "Minimum contour size (width + height) in recognizing"
        d += " contours of detected edges."
        self.paramDesc["contourThr"] = d
        d = "Lower and upper threshold for recognizing a motion in"
        d += " a frame. Threshold value is a square root of"
        d += " sum(different_pixel_values)/255."
        self.paramDesc["motionThr"] = d
        d = "Threshold for detecting ant's dark color."
        self.paramDesc["uAColTh"] = d
        # animal experiment cases
        self.animalECaseChoices = [
                "tagless20", # multiple ant position tracking
                "tagless22", # multiple ant position tracking - 2022.05
                "michaela20", # tracking three ants (two are color-marked)
                "aggrMax21", # aggression detection in a pair of ants
                "egoCent21", # ego-centric video from Christoph Sommer
                "lindaWP21", # video of one worker interacting with a pupae
                "motion21", # simple calculation of motion
                "aos21", # for making motion data like data from AntOS 
                "sleepDet23", # sleep detection 
                ]
        self.animalECase = "sleepDet23" # default animal experiment case
        # cases where pre-processing is required
        self.preProcCase = ["aggrMax21", "sleepDet23"] 
        tagless20str = "Attempt to track 10 or more tagless ants."
        tagless22str = "Attempt to track 10 or more tagless ants."
        michaela20str = "Tracking ants in petri dishes."
        michaela20str += "Each petri dish has three ants, two are color-marked"
        michaela20str += "(groomer) and one is non-marked (pathogen treated)."
        michaela20str += "\n* Two color markers should be visible in the"
        michaela20str += " first frame, at least."
        michaela20str += "\n\n[experiment of Michaela Hoenigsberger, 2020]"
        aggrMax21str = "Finding aggression moments in a pair of ants\n"
        aggrMax21str += "[experiment of Max Aubry, 2021]"
        egoCent21str = "Finding whether the focal ant is close to"
        egoCent21str += " another ant(s) and recoring amount of motion."
        lindaWP21str = "Find when the worker ant is close to the pupae."
        motion21str = "Simple calculation of motion in each frame."
        aos21str = "For making motion data like data from AntOS."
        aos21str += "(To use 'j2020aos' option in 'visualizer' program.)"
        sleepDet23str = "For detecting sleep, assuming"
        sleepDet23str += " single ant is in each container (ROI)"
        self.aecHelpStr = dict(
                                    tagless20=tagless20str,
                                    tagless22=tagless22str,
                                    michaela20=michaela20str,
                                    aggrMax21=aggrMax21str,
                                    egoCent21=egoCent21str,
                                    lindaWP21=lindaWP21str,
                                    motion21=motion21str,
                                    aos21=aos21str,
                                    sleepDet23=sleepDet23str,
                               ) 
        # set corresponding parameters
        self.setAECaseParam()
        # set ouput data columns (self.dataCols),
        #   initival values (self.dataInitVal) and column indices 
        self.setDataCols() 
        # display image type choice
        self.dispImgTypeChoices = ["RGB", "Greyscale(debug)"]
        # current display image type
        self.dispImgType = self.dispImgTypeChoices[0]
        self.lpWid = [] # wx widgets in left panel
        self.ratFImgDispImg = None # ratio between frame image and 
          # display image on app
        self.analyzedFrame = None # for storing frame image after CV analysis
        self.cv_proc = ProcCV(self) # computer vision processing module
        self.vRW = VideoRW(self) # for reading/writing video file
        ##### [end] setting up attributes -----

        vlSz = (-1, 20) # vertical line size
        btnSz = (35, 35)
        btnBGCol = "#333333"
        ### create panels
        for pk in pi.keys():
            w = [] # widge list; each item represents a row in the panel 
            if pk == "tp": # top panel
                w.append([ 
                    {"type":"sTxt", "label":"Experiment case: ", "nCol":1,
                     "border":10},
                    {"type":"cho", "nCol":1, "name":"animalECase", 
                     "choices":self.animalECaseChoices, "val":self.animalECase,
                     "border":5},
                    {"type":"btn", "name":"aec_help", "nCol":1,
                     "img":path.join(P_DIR,"image","help.png"),
                     "tooltip":"Description of the chosen experiment case",
                     "size":btnSz, "bgColor":btnBGCol, "border":0},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"btn", "name":"openDir", "nCol":1,
                     "img":path.join(P_DIR,"image","open.png"),
                     "tooltip":"Open a folder with videos", "size":btnSz,
                     "bgColor":btnBGCol, "border":5},
                    {"type":"btn", "name":"open", "nCol":1,
                     "img":path.join(P_DIR,"image","camera.png"),
                     "tooltip":"Open a video file", "size":btnSz,
                     "bgColor":btnBGCol, "border":5},
                    #{"type":"btn", "name":"analyzeImgs", "nCol":1,
                    # "img":path.join(P_DIR,"image","picture.png"),
                    # "tooltip":"Analyze images", "size":btnSz,
                    # "bgColor":btnBGCol},
                    {"type":"txt", "nCol":1, "val":"", "name":"fp", 
                     "style":wx.TE_READONLY, "size":(200,-1), "border":5},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"btn", "name":"contAnalysis", "nCol":1,
                     "img":path.join(P_DIR,"image","startStop.png"),
                     "tooltip":"Start/Stop analysis", "size":btnSz, 
                     "bgColor":btnBGCol,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":5},
                    {"type":"btn", "name":"nextFrame", "nCol":1,
                     "img":path.join(P_DIR,"image","start1.png"),
                     "tooltip":"Process the next frame",
                     "size":btnSz, "bgColor":btnBGCol,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":5},
                    {"type":"txt", "nCol":1, "val":"1", "name":"jump2frame", 
                     "size":(75,-1), "numOnly":True, "border":0},
                    {"type":"btn", "name":"jump2frame", "nCol":1,
                     "img":path.join(P_DIR,"image","jump.png"),
                     "tooltip":"Move to the selected frame",
                     "size":btnSz, "bgColor":btnBGCol, "border":0}, 
                    {"type":"btn", "nCol":1, "name":"save", "size":btnSz,
                     "img":path.join(P_DIR, "image", "save.png"), 
                     "tooltip":"Save data", "bgColor":btnBGCol},
                    {"type":"sTxt", "label":"", "nCol":1, "border":10,
                     "flag":wx.RIGHT},
                    {"type":"btn", "nCol":1, "name":"delData", "size":btnSz,
                     "img":path.join(P_DIR, "image", "deleteAll.png"), 
                     "tooltip":"Delete all data calculated so far", 
                     "bgColor":btnBGCol},
                    {"type":"sTxt", "label":"", "nCol":1, "border":20,
                     "flag":wx.RIGHT},
                    {"type":"chk", "nCol":1, "name":"email", 
                     "label":"notify to email", "style":wx.CHK_2STATE, 
                     "val":self.flags["notifyEmail"]},
                    {"type":"txt", "nCol":1, "val":"", "name":"email", 
                     "size":(200,-1), "border":0, "disable":True},
                    #{"type":"btn", "nCol":1, "name":"sendEmail", "label":">"},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"sTxt", "label":"0:00:00", "nCol":1,
                     "name":"ssTime", "fgColor":"#ccccff", 
                     "bgColor":"#000000"},
                    {"type":"sTxt", "label":"", "nCol":1, "border":25,
                     "flag":wx.RIGHT},
                    {"type":"sTxt", "label":"FPS", "nCol":1, "name":"fps",
                     "fgColor":"#cccccc"}, 
                    ])
            setupPanel(w, self, pk, True)

        lastWidget = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
        x = lastWidget.GetPosition()[0] + lastWidget.GetSize()[0]
        if x > wSz[0]:
        # last widget of the top panel is over the window width
            ### resize panels (for scroll bar in top panel)
            if sys.platform.startswith("win"): inc = 30
            else: inc = 15
            tpSz = self.pi["tp"]["sz"]
            tpSz = (tpSz[0], tpSz[1]+inc)
            self.panel["tp"].SetSize(tpSz)
            for pk in self.pi.keys():
                if pk == "tp": continue
                pos = self.pi[pk]["pos"]
                self.panel[pk].SetPosition((pos[0], pos[1]+inc))
                if pk == "bt": continue
                sz = self.pi[pk]["sz"]
                self.panel[pk].SetSize((sz[0], sz[1]-inc))

        self.initLPWidgets() # left panel widgets

        ### bind events to image panel 
        self.panel["ip"].Bind(wx.EVT_PAINT, self.onPaintIP)
        self.panel["ip"].Bind(wx.EVT_LEFT_DOWN, self.onMLBDown)
        self.panel["ip"].Bind(wx.EVT_LEFT_UP, self.onMLBUp)
        self.panel["ip"].Bind(wx.EVT_RIGHT_UP, self.onMRBUp)
        self.panel["ip"].Bind(wx.EVT_MOTION, self.onMMove)
       
        ### set up menu
        menuBar = wx.MenuBar()
        menu = wx.Menu()
        saveItem = menu.Append(wx.Window.NewControlId(), item="Save\tCTRL+S")
        self.Bind(wx.EVT_MENU, self.save, saveItem)
        quitItem = menu.Append(wx.Window.NewControlId(), item="Quit\tCTRL+Q")
        self.Bind(wx.EVT_MENU, self.onClose, quitItem)
        menuBar.Append(menu, "&FeatureDetector")
        self.SetMenuBar(menuBar) 

        self.log(msg="Program starts")

    #---------------------------------------------------------------------------
   
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
         
        Returns:
            pi (dict): Panel information.
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        wSz = self.wSz # window size
        #style = (wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
        pi = {} # panel information to return
        # top panel for buttons, etc
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], 50), bgCol="#666666", 
                        style=style)
        tpSz = pi["tp"]["sz"]
        # left side panel for setting parameters
        pi["lp"] = dict(pos=(0, tpSz[1]), sz=(300, wSz[1]-tpSz[1]), 
                        bgCol="#333333", style=style)
        lpSz = pi["lp"]["sz"]
        # panel for displaying frame image 
        pi["ip"] = dict(pos=(lpSz[0], tpSz[1]), sz=(wSz[0]-lpSz[0], lpSz[1]), 
                        bgCol="#000000", style=style)
        return pi
    
    #---------------------------------------------------------------------------
    
    def initLPWidgets(self):
        """ initialize wx widgets in left panel

        Args: None

        Returns: None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))

        pk = "lp"
        for i, w in enumerate(self.lpWid):
        # through widgets in left panel
            self.gbs[pk].Detach(w) # detach widget from gridBagSizer
            w.Destroy() # destroy the widget

        ### make a list to produce widgets with parameter keys
        lst = list(self.aecParam.keys()) # list of keys for parameter items 
        uLst = [] # this parameter items will be more likely modified by users.
        for i in range(len(lst)):
            if lst[i][0] == 'u':
                uLst.append(str(lst[i]))
                lst[i] = None
        while None in lst: lst.remove(None)
        lst = sorted(uLst) + ["---"] + sorted(lst) 

        ##### [begin] set up left panel interface -----
        nCol = 3
        lpSz = self.pi[pk]["sz"]
        hlSz = (int(lpSz[0]*0.9), -1) # size of horizontal line
        btnBGCol = "#333333"
        w = [] # list of widgets
        if self.animalECase == "lindaWP21":
            self.dispImgTypeChoices = ["RGB", "(debug) Petri-dish",
                                       "(debug) White", "(debug) Ant",
                                       "(debug) Pupae", "(debug) Motion"]
        if self.animalECase == "sleepDet23":
            self.dispImgTypeChoices = ["RGB", "Greyscale(debug)", 
                                       "(debug) Cotton", "(debug) Motion", 
                                       "(debug) Ant-color"]
        else:
            self.dispImgTypeChoices = ["RGB", "Greyscale(debug)"]
        self.dispImgType = self.dispImgTypeChoices[0]
        w.append([
            {"type":"sTxt", "label":"display-img", "nCol":1, 
             "font":self.fonts[1], "fgColor":"#cccccc"},
            {"type":"cho", "nCol":2, "name":"imgType", 
             "choices":self.dispImgTypeChoices, "size":(160,-1), 
             "val":self.dispImgType}
            ])
        for key in lst:
            if key == "---":
                w.append([{"type":"sLn", "size":hlSz, "nCol":nCol, 
                           "style":wx.LI_HORIZONTAL, "fgColor":"#cccccc"}])
                continue
            param = self.aecParam[key]
            val = str(param["value"]).strip("[]").replace(" ","")
            if "disableUI" in param.keys(): disable = param["disableUI"]
            else: disable = False
            w.append([
                # static-text to show name of the parameter
                {"type":"sTxt", "label":key, "nCol":1, "font":self.fonts[1],
                 "fgColor":"#cccccc"},
                # help button to describe what the parameter is about
                {"type":"btn", "label":"?", "nCol":1, "name":"%s_help"%(key),
                 "size":(25,-1), "bgColor":btnBGCol},
                # text of the parameter value
                {"type":"txt", "nCol":1, "val":val, "name":key, "size":(125,-1),
                 "disable":disable}
                ])
        w.append([{"type":"btn", "nCol":nCol, "name":"applyParam", "size":hlSz,
                   "label":"Apply changed parameters", "bgColor":btnBGCol}])
        tooltipTxt = "Save parameters as a file, for the currently opened video"
        w.append([{"type":"btn", "nCol":nCol, "name":"saveParam", "size":hlSz,
                   "label":"Save parameters", "bgColor":btnBGCol,
                   "tooltip":tooltipTxt}])
        w.append([{"type":"sTxt", "nCol":nCol, "label":" "}])
        
        self.lpWid = setupPanel(w, self, pk)
        ##### [end] set up left panel interface -----

    #---------------------------------------------------------------------------

    def config(self, flag):
        """ saving/loading configuration of the current animal-experiment-case 

        Args:
            flag (str): save or load

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        configFP = path.join(FPATH, f'config_{self.animalECase}')

        if flag == "save":
            fh = open(configFP, "wb")
            pickle.dump(self.aecParam, fh)
            fh.close()

        elif flag == "load":
            if path.isfile(configFP):
            # config file exists
                fh = open(configFP, "rb")
                config = pickle.load(fh)
                fh.close()
                for key in config.keys():
                    self.aecParam[key]["value"] = config[key]["value"]

    #---------------------------------------------------------------------------
    
    def setDataCols(self):
        """ Set output data columns depending on animal experiment case 

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        aec = self.animalECase

        self.dataCols = []
        self.dataInitVal = []
        maxDataLen = [] # maximum length of string for each data column
        
        if aec.startswith("tagless"):
            for ai in range(self.aecParam["uNSubj"]["value"]):
                ### ant position (x)
                self.dataCols.append("a%03iPosX"%(ai))
                self.dataInitVal.append("None")
                maxDataLen.append(4)
                ### ant position (y)
                self.dataCols.append("a%03iPosY"%(ai))
                self.dataInitVal.append("None")
                maxDataLen.append(4)
            ### remarks
            #self.dataCols.append("remarks")
            #self.dataInitVal.append("None")
            #maxDataLen.append(20)
        
        elif aec == "michaela20":
            for pdi in range(self.aecParam["uNPDish"]["value"]):
                ### movements in a petri-dish
                self.dataCols.append("movements%02i"%(pdi))
                self.dataInitVal.append("None")
                maxDataLen.append(4)
                for ai in range(self.aecParam["uNSubj"]["value"]):
                    ### ant position (x)
                    self.dataCols.append("a%02i%02iPosX"%(pdi, ai))
                    self.dataInitVal.append("None")
                    maxDataLen.append(4)
                    ### ant position (y)
                    self.dataCols.append("a%02i%02iPosY"%(pdi, ai))
                    self.dataInitVal.append("None")
                    maxDataLen.append(4)
                    ### ant's movement
                    self.dataCols.append("a%02i%02imov"%(pdi, ai))
                    self.dataInitVal.append("None")
                    maxDataLen.append(4)

        elif aec == "aggrMax21":
            for pi in range(self.aecParam["uNPDish"]["value"]):
                self.dataCols.append("p%03iAggression"%(pi))
                self.dataInitVal.append("0")
                maxDataLen.append(1)

        elif aec == "egoCent21":
            ### overall movements in the scene
            self.dataCols.append("motion")
            self.dataInitVal.append("0")
            maxDataLen.append(6)
            ### whether the focal ant is interacting/touching another ant 
            self.dataCols.append("closeWithAnother")
            self.dataInitVal.append("False")
            maxDataLen.append(5)

        elif aec == "lindaWP21":
            for pdi in range(self.aecParam["uNPDish"]["value"]):
                ### overall movements in the scene
                self.dataCols.append("motion_%02i"%(pdi))
                self.dataInitVal.append("0")
                maxDataLen.append(6)
                ### moment of interest; 
                ###   True when the worker is close to the pupae 
                self.dataCols.append("moi0_%02i"%(pdi))
                self.dataInitVal.append("False")
                maxDataLen.append(5)

        elif aec == "motion21":
            for pdi in range(self.aecParam["uNPDish"]["value"]):
                ### overall movements in the scene
                self.dataCols.append("motion_%02i"%(pdi))
                self.dataInitVal.append("0")
                maxDataLen.append(6)

        elif aec == "aos21":
            self.dataCols.append("Cam-idx")
            self.dataInitVal.append("99")
            maxDataLen.append(2)
            self.dataCols.append("Timestamp")
            self.dataInitVal.append("None")
            maxDataLen.append(26)
            self.dataCols.append("Key")
            self.dataInitVal.append("motionPts")
            maxDataLen.append(9)
            self.dataCols.append("Value")
            self.dataInitVal.append("(-1/-1)")
            #maxDataLen.append(100)
        
        elif aec == "sleepDet23":
            wRows = self.aecParam["uWRows"]["value"]
            wCols = self.aecParam["uWCols"]["value"]
            for ri in range(wRows):
                for ci in range(wCols):
                    pdi = (ri*wCols) + ci
                    self.dataCols.append(f'motion_{pdi:02d}')
                    self.dataInitVal.append("0")
                    maxDataLen.append(7)
                    self.dataCols.append(f'moi0_{pdi:02d}')
                    self.dataInitVal.append("0")
                    maxDataLen.append(1)
         
        ### store indices of each data column
        self.di = {} 
        for key in self.dataCols:
            self.di[key] = self.dataCols.index(key)

        '''
        ### data types for numpy structured array to connect with Grid data 
        self.dataStruct = []
        for ci in range(len(self.dataCols)):
            self.dataStruct.append(
                            (self.dataCols[ci], (np.str_, maxDataLen[ci]))
                            ) 
        '''

    #---------------------------------------------------------------------------
    
    def setAECaseParam(self):
        """ Set up parameters for the current animal experiment case.
        * Parameter key starts with a letter 'u' means that 
          it will probably modified by users more frequently.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.animalECase == "": return
        aec = self.animalECase

        ### set descriptions of some parameters
        d = "Threshold for detecting ant's dark color."
        self.paramDesc["uAColTh"] = d
        d = "Approximate length of an ant. This length is used in detection"
        d += " of ant blob, calculating whether two subjects are close enough"
        d += " and so on."
        self.paramDesc["uAntLength"] = d
        d = "Minimum area (pixels) for an ant blob. Approximate area of" 
        d += " gaster of an ant."
        self.paramDesc["uGasterSize"] = d 
        d = "Number of petri dishes"
        self.paramDesc["uNPDish"] = d
        d = "Number of subjects(ants) to track."
        self.paramDesc["uNSubj"] = d 
        d = "Region of Interest in rect (x, y, w, h)"
        self.paramDesc["uROI"] = d
        d = "Offset values [x, y, radius] for Region Of Interest.\n"
        d += "Offset x & y are offsets from the center of the video.\n"
        d += "Offset radius is the offset from the smaller value between"
        d += " the width and the height of the video."
        for i in range(4): # Currently, max. number of petri-dishes is 4. 
            self.paramDesc["uROI%i_offset"%(i)] = d
        d = "Number of dilation of ant blob image.\n"
        d += "Used when uAColTh is strict."
        self.paramDesc["uDilate"] = d
        d = "HSV color min values to detect the first color marker."
        self.paramDesc["uCol0Min"] = d
        d = "HSV color max values to detect the first color marker."
        self.paramDesc["uCol0Max"] = d
        d = "HSV color min values to detect the second color marker."
        self.paramDesc["uCol1Min"] = d
        d = "HSV color max values to detect the second color marker."
        self.paramDesc["uCol1Max"] = d
        d = "Frame index for each petri-dish to start analysis." 
        self.paramDesc["uASFrame"] = d

        ### common parameters
        self.aecParam = {}
        self.aecParam["preMExOIter"] = dict(value=2) 
        self.aecParam["preMExCIter"] = dict(value=-1) 
        self.aecParam["preThres"] = dict(value=-1)
        self.aecParam["cannyTh"] = dict(value=[50,200]) 
        self.aecParam["contourThr"] = dict(value=10)
        self.aecParam["motionThr"] = dict(value=50)
        ### common user parameters 
        ###   (more likely to be changed by user for different videos)
        self.aecParam["uAntLength"] = dict(value=50) 
        self.aecParam["uNPDish"] = dict(value=1, disableUI=True)
        self.aecParam["uNSubj"] = dict(value=1, disableUI=True)
        
        if aec.startswith("tagless"):
            self.aecParam["preMExOIter"] = dict(value=2) 
            self.aecParam["preMExCIter"] = dict(value=3) 
            self.aecParam["preThres"] = dict(value=90)
            self.aecParam["cannyTh"] = dict(value=[30,50]) 
            self.aecParam["contourThr"] = dict(value=1)
            self.aecParam["uAColTh"] = dict(value=40)
            self.aecParam["uNSubj"] = dict(value=14, disableUI=True)
            self.aecParam["uGasterSize"] = dict(value=300)
            self.aecParam["uROI"] = dict(value=[250,50,900,900])

        elif self.animalECase == "michaela20": 
            self.aecParam["uAColTh"] = dict(value=40)
            self.aecParam["uNPDish"] = dict(value=2, disableUI=True)
            self.aecParam["uNSubj"] = dict(value=3, disableUI=True)
            self.aecParam["uGasterSize"] = dict(value=175)
            self.aecParam["uDilate"] = dict(value=1)
            #self.aecParam["uROI0_offset"] = dict(value=[90,-90,-210])
            self.aecParam["uROI0_offset"] = dict(value=[-340,10,-80])
            self.aecParam["uROI1_offset"] = dict(value=[250,10,-80])
            self.aecParam["uCol0Min"] = {"value":[140,75,100]} # pink min
            self.aecParam["uCol0Max"] = {"value":[170,255,255]} # pink max
            self.aecParam["uCol1Min"] = {"value":[20,150,100]} # yellow min
            self.aecParam["uCol1Max"] = {"value":[40,255,255]} # yellow max
            self.aecParam["uASFrame"] = dict(value=[0, 0])

        elif self.animalECase == "aggrMax21":
        # for detecting aggression in a pair of ants in a petri-dish
            
            self.aecParam = {}
            self.aecParam["preMExOIter"] = dict(value=1)
            self.aecParam["preMExCIter"] = dict(value=1)
            self.aecParam["preThres"] = dict(value=-1)
            self.aecParam["cannyTh"] = dict(value=[50,200])
            self.aecParam["uAColTh"] = dict(value=40)
            self.aecParam["uNPDish"] = dict(value=2, disableUI=True)
            self.aecParam["uAntLength"] = dict(value=20)
            #self.aecParam["uROI0_offset"] = dict(value=[-95,25,-85])
            #self.aecParam["uROI1_offset"] = dict(value=[132,30,-88])
            self.aecParam["uROI0_offset"] = dict(value=[-165,5,-30])
            self.aecParam["uROI1_offset"] = dict(value=[160,-5,-30])

        elif self.animalECase == "egoCent21": # ego-centric video
        # for detecting motion,
        #   plus whether the focal ant is interacting with another
            self.aecParam["preMExOIter"] = dict(value=-1)
            self.aecParam["uAColTh"] = dict(value=40)
            self.aecParam["uNSubj"] = dict(value=2, disableUI=True)
            self.aecParam["uROI0_offset"] = dict(value=[0,0,-10])

        elif self.animalECase == "lindaWP21":
        # video with one worker and a pupae  
            
            self.aecParam["motionThr"] = dict(value=50)
            self.aecParam["contourThr"] = dict(value=8)
            self.aecParam["uAntLength"] = dict(value=25) 
            self.aecParam["uNPDish"] = dict(value=1)
            # including pupae as a subject
            self.aecParam["uNSubj"] = dict(value=4)
            d = "Radius of petri-dish, as a percentage to the frame height." 
            self.paramDesc["uPDishRad"] = d
            self.aecParam["uPDishRad"] = dict(value=50)
            '''
            self.aecParam["uROI0_offset"] = dict(value=[-120,-150,-220])
            self.aecParam["uROI1_offset"] = dict(value=[210,-170,-220])
            self.aecParam["uROI2_offset"] = dict(value=[-80,180,-220])
            self.aecParam["uROI3_offset"] = dict(value=[250,160,-220])
            '''
            d = "HSV color values to detect pupae" 
            self.paramDesc["uColP_min"] = d
            self.paramDesc["uColP_max"] = d
            self.aecParam["uColP_min"] = {"value":[10,80,130]} # pupae color
            self.aecParam["uColP_max"] = {"value":[20,150,180]} # pupae color
            d = "HSV color values to detect ant" 
            self.paramDesc["uColA_min"] = d
            self.paramDesc["uColA_max"] = d
            self.aecParam["uColA_min"] = {"value":[0,0,0]} # ant color
            self.aecParam["uColA_max"] = {"value":[20,150,50]} # ant color
            d = "HSV color values to detect white paper & lighting."
            d += " When only 1 or 2 petri-dishes are detected, this color"
            d += " is used to calculate the center of the entire setup"
            self.paramDesc["uColWh_min"] = d
            self.paramDesc["uColWh_max"] = d
            ### color of white paper background & surrounding light 
            self.aecParam["uColWh_min"] = {"value":[0,0,160]}
            self.aecParam["uColWh_max"] = {"value":[180,100,255]}
            d = "HSV color values to detect petri-dish edge" 
            self.paramDesc["uColD_min"] = d
            self.paramDesc["uColD_max"] = d
            ### color of petri-dish edge
            self.aecParam["uColD_min"] = {"value":[0,50,120]}
            self.aecParam["uColD_max"] = {"value":[10,255,255]} 
            '''
            for sti in range(4):
                d = "It starts detection for the petri-dish-%i"%(sti)
                d += " from this frame index." 
                self.paramDesc["uStartFrame%i"%(sti)] = d
                self.aecParam["uStartFrame%i"%(sti)] = dict(value="0")
            '''
        
        elif self.animalECase == "motion21":
        # calculating motion
            self.aecParam["uNPDish"] = dict(value=2, disableUI=True)
            d = "Region of Interest in rect (x, y, w, h)"
            self.paramDesc["uROI0"] = d
            self.paramDesc["uROI1"] = d
            # here, chambers are square-shape
            self.aecParam["uROI0"] = dict(value=[350,820,1250,1250])
            self.aecParam["uROI1"] = dict(value=[2150,820,1250,1250])

        elif self.animalECase == "aos21":
            self.aecParam["uNPDish"] = dict(value=2, disableUI=True)
            d = "Region of Interest in rect (x, y, w, h)"
            self.paramDesc["uROI0"] = d
            self.paramDesc["uROI1"] = d
            # here, chambers are square-shape
            self.aecParam["uROI0"] = dict(value=[350,820,1250,1250])
            self.aecParam["uROI1"] = dict(value=[2150,820,1250,1250])
        
        elif self.animalECase == "sleepDet23":  
            d = "ant color"
            self.paramDesc["uColA_min"] = d
            self.paramDesc["uColA_max"] = d
            self.aecParam["uColA_min"] = {"value":[0,0,0]} # ant color
            self.aecParam["uColA_max"] = {"value":[180,200,115]} # ant color
            # ant color [180,200,105]?

            self.aecParam["uNSubj"] = dict(value=1, disableUI=True)
            
            wRows = 2
            wCols = 3 
            self.paramDesc["uWRows"] = "rows of wells in the container"
            self.paramDesc["uWCols"] = "columns of wells in the container"
            self.aecParam["uWRows"] = dict(value=wRows, disableUI=True)
            self.aecParam["uWCols"] = dict(value=wCols, disableUI=True)

            # for this case, number of containers is not used.
            # 'uROIRows' & 'uROICols' will be used instead
            self.aecParam["uNPDish"] = dict(value=wRows*wCols, disableUI=True)
            
            d = "Seconds of no-motion before considering sleep episode." 
            d += "(we assume the input video will be 1 FPS.)"
            self.paramDesc["uNoMotionThr"] = d 
            self.aecParam["uNoMotionThr"] = dict(value=5)

            self.paramDesc["uROIOffsetX"] = "x-offset of ROI"
            self.aecParam["uROIOffsetX"] = dict(value=0.13)
            self.paramDesc["uROIOffsetY"] = "y-offset of ROI"
            self.aecParam["uROIOffsetY"] = dict(value=0.01)

            ### set ant length for each well 
            for ri in range(wRows): # rows
                for ci in range(wCols): # columns 
                    key = f'uAntLen{ri}{ci}'
                    d = "Length of ant body (head to gaster) in each well"
                    self.paramDesc[key] = d
                    self.aecParam[key] = dict(value=45)

            ### delete some items, not being used
            for key in ["uAntLength"]:
                del(self.paramDesc[key])
                del(self.aecParam[key])
        
        else:
            msg = "[%s] is not found."%(self.animalECase)
            raise ValueError(msg)
        
        ### add description to the dictionary
        for k in self.aecParam.keys():
            self.aecParam[k]["desc"] = self.paramDesc[k]

        # load and overload the aecParam values (if config exists)
        self.config("load")
   
    #---------------------------------------------------------------------------
    
    def onPaintIP(self, event):
        """ painting image panel 

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.inputFP == "": return

        event.Skip()
        dc = wx.PaintDC(self.panel["ip"])
        dc.SetBackground(wx.Brush((0,0,0,255)))
        dc.Clear()
       
        if not self.analyzedFrame is None:
            ### display analyzed frame image 
            img = cv2.cvtColor(self.analyzedFrame, cv2.COLOR_BGR2RGB)
            wxImg = wx.Image(img.shape[1], img.shape[0])
            wxImg.SetData(img.tobytes())
            dc.DrawBitmap(wxImg.ConvertToBitmap(), 0, 0)
    
    #---------------------------------------------------------------------------

    def onButtonPressDown(self, event, objName=""):
        """ wx.Butotn was pressed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate the button press
              of the button with the given name. 
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"]: return 

        if objName == '':
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
            obj = wx.FindWindowByName(objName, self.panel["tp"])

        if not obj.IsEnabled(): return

        if objName == "open_btn":
            self.openInputData()

        elif objName == "openDir_btn":
            self.openInputData(isOpeningDir=True)

        elif objName == "save_btn":
            self.save(None)

        elif objName == "delData_btn":
            self.delData()
        
        elif objName == "contAnalysis_btn":
            self.onSpace(None)
        
        elif objName == "quit_btn":
            self.onClose(None)
        
        elif objName == "nextFrame_btn":
            self.onRight(True)
        
        elif objName == "jump2frame_btn":
            txt = wx.FindWindowByName("jump2frame_txt", self.panel["tp"])
            self.jumpToFrame(int(txt.GetValue()))
        
        elif objName == "applyParam_btn":
            self.applyChangedParam()

        elif objName == "saveParam_btn":
            self.saveParam()
        
        elif objName.endswith("_help_btn"):
            key = objName.replace("_help_btn", "")
            msg = ""
            if key == "aec":
                if self.animalECase in self.aecHelpStr.keys():
                    msg = self.aecHelpStr[self.animalECase]
            else:
                msg = self.aecParam[key]["desc"]
            self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)  

        elif objName == "sendEmail_btn":
            self.sendNotification("Test 1", "Testing email notification")

        '''
        elif objName == "revise_btn":
            self.dataRevision()
        '''

    #---------------------------------------------------------------------------

    def onChoice(self, event, objName=""):
        """ wx.Choice was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.Choice event 
                with the given name. 
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"]: return 

        if objName == "":
            obj = event.GetEventObject()
            objName = obj.GetName()
            wasFuncCalledViaWxEvent = True
        else:
        # funcion was called by some other function without wx.Event
            obj = wx.FindWindowByName(objName, self.panel["tp"])
            wasFuncCalledViaWxEvent = False 
            
        objVal = obj.GetString(obj.GetSelection()) # text of chosen option
        
        if objName == "animalECase_cho":
        # animal experiment case was chosen
            self.animalECase = objVal
            #if wasFuncCalledViaWxEvent:
            self.setAECaseParam() # set animal experiment case parameters
            self.setDataCols() # set ouput data columns (self.dataCols),
                # initival values (self.dataInitVal) and column indices
            self.loadParam() # load AEC parameters
            self.initLPWidgets() # initialize left panel
            #self.proc_img() # display frame image
        
        if objName == "imgType_cho":
        # display image type changed
            self.dispImgType = objVal
            self.proc_img()

    #---------------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "chk")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "email_chk":
            self.flags["notifyEmail"] = objVal
            txt = wx.FindWindowByName("email_txt", self.panel["tp"])
            if objVal: txt.Enable()
            else: txt.Disable()

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
        if FLAGS["debug"]: logging.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "txt")
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

    def onKeyPress(self, event, kc=None, mState=None):
        """ Process key-press event

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        event.Skip()
        if kc == None: kc = event.GetKeyCode()
        if mState == None: mState = wx.GetMouseState()

        if mState.ControlDown():
        # CTRL modifier key is pressed
            if kc == ord("S"):
                self.save(None)

            elif kc == ord("Q"):
                self.onClose(None)

        else:
        # no modifier key is pressed
            if kc == wx.WXK_SPACE:
                self.onSpace(None)

            elif kc == wx.WXK_RIGHT:
                self.onRight(True)

    #---------------------------------------------------------------------------
       
    def onRight(self, isLoadingOneFrame=False):
        """ Navigate forward 
        
        Args:
            isLoadingOneFrame (bool): whether load only one frame, 
              False by defualt.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.flags["blockUI"]: return
        elif self.inputFP == '': return
        if isLoadingOneFrame: self.flags["isRunning"] = False 

        if self.flags["isRunning"]: # if continuous analysis is running 
            ### FPS update
            self.fps += 1
            if time()-self.last_fps_time >= 1:
                sTxt = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
                sTxt.SetLabel( "FPS: %i"%(self.fps) )
                self.fps = 0
                self.last_fps_time = time()
                self.runningDur += 1
      
        if self.flags["isRunning"] and self.vRW.fi >= self.vRW.nFrames-1:
        # continuous running reached the end of frames
            self.onSpace(None) # stop continuous running

        if self.vRW.fi >= self.vRW.nFrames-1: return
        ret = self.vRW.getFrame(-1) # read one frame
        if not ret and self.flags["isRunning"]: 
            
            if self.vRW.fi >= self.vRW.nFrames*0.9:
            # Read most of frames.
            # Assume that it reached the end of video, 
            # but nFrames from cv2.CAP_PROP_FRAME_COUNT was inaccurate.
                # cut off the rest of output data 
                self.oData = self.oData[:self.vRW.fi] 
                # process saving, etc..
                self.procAtEndOfVideo()

        else:
            self.proc_img() # process the frame
        
    #---------------------------------------------------------------------------

    def onSpace(self, event):
        """ start/stop continuous frame analysis

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"]: return

        if self.inputFP == '': return 
        if self.flags["isRunning"] == False:
            if self.vRW.fi >= self.vRW.nFrames-1: return
            self.flags["isRunning"] = True
            self.fps = 0
            self.last_fps_time = time()
            self.timer["run"] = wx.CallLater(1, self.onRight)
        else:
            try: # stop timer
                self.timer["run"].Stop() 
                self.timer["run"] = None
            except:
                pass
            sTxt = wx.FindWindowByName("fps_sTxt", self.panel["tp"])
            sTxt.SetLabel('')
            self.flags["isRunning"] = False # stop continuous analysis
            
    #---------------------------------------------------------------------------
    
    def onMLBDown(self, event):
        """ Mouse left button is pressed on displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

        if pk != "ip": return

        if self.animalECase in ["Marmoset14", "Macaque19", "Rat15"]:
            ### store the current mouse pointer position 
            if self.ratFImgDispImg != None:
                r = 1.0/self.ratFImgDispImg
                self.panel[pk].mousePressedPt = (int(mp[0]*r), int(mp[1]*r))
            else:
                self.panel[pk].mousePressedPt = (mp[0], mp[1])
        
        #self.tmp_img = self.vRW.currFrame.copy()
    
    #---------------------------------------------------------------------------

    def onMMove(self, event):
        """ Mouse pointer is moving in displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        
        if pk != "ip": return
        
        pos0 = self.panel[pk].mousePressedPt
        if pos0 != (-1, -1): 
            if self.animalECase in ["Marmoset14", "Macaque19", "Rat15"]:
                mp = event.GetPosition()
                if self.ratFImgDispImg != None:
                    r = 1.0/self.ratFImgDispImg
                    pos1 = ( int(mp[0]*r), int(mp[1]*r) )
                else:
                    pos1 = (mp[0], mp[1])
                mInput = dict(hPosX=pos1[0], hPosY=pos1[1], 
                              bPosX=pos0[0], bPosY=pos0[1])
                self.proc_img(mInput)
    
    #---------------------------------------------------------------------------

    def onMLBUp(self, event):
        """ Mouse left button was clicked on displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

        if pk != "ip": return

        ### Set pos1 (mouse position when mouse left button is released).
        ### Adjust it, if the displayed frame is a resized image,
        if self.ratFImgDispImg != None:
            r = 1.0/self.ratFImgDispImg
            pos1 = (int(mp[0]*r), int(mp[1]*r))
        else:
            pos1 = (mp[0], mp[1])

        if self.animalECase == "michaela20":
            # radius of spot
            rad = int(np.sqrt(self.aecParam["uGasterSize"]["value"])/2)
            flagSpotClicked = False # already existing spot was clicked
            for spot in self.cv_proc.spots2ignore:
                dist = np.sqrt((spot[0]-pos1[0])**2 + (spot[1]-pos1[1])**2)
                if dist < rad:
                    self.cv_proc.spots2ignore.remove(spot)
                    flagSpotClicked = True
                    break
            if not flagSpotClicked:
                self.cv_proc.spots2ignore.append(pos1)
            self.proc_img()
        
        self.panel[pk].mousePressedPt = (-1, -1)
        #self.tmp_img = None

    #---------------------------------------------------------------------------
 
    def onMRBUp(self, event):
        """ Mouse right button was clicked on displayed image on UI

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.inputFP == '': return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

        if pk != "ip": return

        ''' !!! CURRRENTLY NOT USED
        ### deleting data of the current frame
        for ci in range(len(self.dataCols)):
            col = str(self.dataCols[ci])
            if col == "remarks": continue # no need to change 'remarks'
            if col.startswith("man"): value = 'True' # manual data marking
            else: value = 'D' # mark as deleted data
            self.oData[self.vRW.fi][ci] = value
        self.proc_img()
        '''

    #---------------------------------------------------------------------------
    
    def applyChangedParam(self):
        """ Apply changed parameters.

        Args: None
        
        Returns: None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.inputFP == "": return
        aecp = self.aecParam
        wasNPDishChanged = False 
        wasNSubjChanged = False 
        for key in sorted(aecp.keys()):
            valWid = wx.FindWindowByName( '%s_txt'%(key), self.panel["lp"] )
            val = valWid.GetValue().strip()
            currVal = aecp[key]["value"]
            if type(currVal) == list:
                vals = val.split(",")
                if len(currVal) != len(vals):
                    msg = "There should be %i items"%(len(aecp[key]))
                    msg += " for %s."%(key)
                    self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
                    return
                else:
                    for i in range(len(vals)):
                        try:
                            vals[i] = int(vals[i])
                        except:
                            msg = "Data type of %s doesn't match."%(key)
                            self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
                            return
                for i in range(len(vals)): aecp[key]["value"][i] = vals[i]
            else:
                try: val = int(val)
                except:
                    try: val = float(val)
                    except:
                        msg = "%s should be a number."%(key)
                        self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
                        return
                if key == "uNSubj": # number of subject
                    if self.aecParam[key]["value"] != val:
                        wasNSubjChanged = True
                elif key == "uNPDish": # number of petri dishes
                    if self.aecParam[key]["value"] != val:
                        wasNPDishChanged = True
                self.aecParam[key]["value"] = val
        msg = " updated."
        
        if wasNSubjChanged or wasNPDishChanged:
        # number of subject or petri-dish changed
            self.setDataCols() # columns should be accordingly changed
            # init output data
            self.oData, __ = self.initOutputData() 
            msg += "\n\nAll data are initialized."
            msg += " Please run analysis from the first frame.\n"

        if hasattr(self.cv_proc, "prevPGImg"):
            self.cv_proc.prevPGImg = None # delete previous grey image

        self.proc_img() # process image
        self.config("save") # save config
        self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)
   
    #---------------------------------------------------------------------------
    
    def loadParam(self):
        """ Read parameters from a file, if it exists, and store it.

        Args: None
        
        Returns: None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))
        if self.inputFP == "": return
        
        ext = "." + self.inputFP.split(".")[-1]
        fp = self.inputFP.replace(ext, "_params.txt")
        if not path.isfile(fp): return

        fh = open(fp, "r")
        lines = fh.readlines()
        fh.close()

        for line in lines:
            items = line.split(",")
            if len(items) < 2: continue
            key = items[0].strip()
            val = items[1].strip()
            if not key in self.aecParam.keys(): continue
            if val[0] == "[": # this is a list
                val = val.strip("[]").split("/")
                for i, v in enumerate(val):
                    nv = str2num(v)
                    val[i] = nv 
            else:
                val = str2num(val)
            self.aecParam[key]["value"] = val

    #---------------------------------------------------------------------------
    
    def saveParam(self):
        """ Save parameters as a file for the currently opened video. 

        Args: None
        
        Returns: None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.inputFP == "":
            msg = "There's no opened video file."
            self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
            return

        ext = "." + self.inputFP.split(".")[-1]
        fp = self.inputFP.replace(ext, "_params.txt")
        fh = open(fp, "w")
        for k in self.aecParam.keys():
            val = str(self.aecParam[k]["value"])
            if val[0] == "[": # this is a list
                val = val.replace(",", "/")
            line = "%s, %s\n"%(str(k), val)
            fh.write(line)
        fh.close()
        msg = "Parameters are stored in '%s'."%(fp)
        self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)
    
    #---------------------------------------------------------------------------
    
    def proc_img(self, mInput=None):
        """ Process image with cv_proc module, update resultant data.
        
        Args:
            mInput (None/dict): Manual user input such as 
              mouse click & drag.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        if self.inputFP == '': return

        aECase = self.animalECase # animal experiment case
        vFI = self.vRW.fi # video frame index
        
        ### set temporary (for processing the current frame) 
        ###   dictionary to store values
        tD = {} # temp. dictionary
        flagMP = False 
        for dIdx, dCol in enumerate(self.dataCols):
            if mInput != None and dCol in mInput.keys():
            # manual input is given
                tD[dCol] = mInput[dCol]
                flagMP = True
            elif self.oData[vFI][dIdx] != 'None':
            # already calculated data available
                tD[dCol] = self.oData[vFI][dIdx]
            else:
                tD[dCol] = self.dataInitVal[dIdx]
            ### convert value to integer if applicable
            try: tD[dCol] = int(tD[dCol])
            except: pass
            ### data from previous frame
            pk = "p_" + dCol
            if vFI == 0:
                tD[pk] = self.dataInitVal[dIdx] 
            else:
                tD[pk] = self.oData[vFI-1][dIdx]
                if not tD[pk] in ['None', 'D', 'True', 'False']:
                    try: tD[pk] = int(tD[pk])
                    except: pass
       
        try:
            # process
            ret, frame_arr = self.cv_proc.proc_img(self.vRW.currFrame.copy(), 
                                                   aECase,
                                                   tD,
                                                   flagMP,
                                                   self.dispImgType)
        except Exception as e:
            _str = "".join(traceback.format_exc())
            print(_str)
            self.showMsg(_str, "ERROR", wx.OK|wx.ICON_ERROR)
            return

        # display the processed frame 
        if self.ratFImgDispImg != 1.0:
            w = int(frame_arr.shape[1] * self.ratFImgDispImg)
            h = int(frame_arr.shape[0] * self.ratFImgDispImg)
            frame_arr = cv2.resize(frame_arr, (w, h))
        self.analyzedFrame = frame_arr
        self.panel["ip"].Refresh()

        if self.flags["rsltRecording"]:
            # write a frame of analysis video recording
            self.vRW.writeFrame(frame_arr)

        ### update oData
        if ret != None:
            for dIdx, dCol in enumerate(self.dataCols):
                self.oData[vFI][dIdx] = str(ret[dCol])

        if self.flags["isRunning"]:
            if vFI < self.vRW.nFrames-1: # there's more frames to run
                # continue to analyse
                self.timer["run"] = wx.CallLater(1, self.onRight)
            else: # reached end of the frames
                self.procAtEndOfVideo() 

    #---------------------------------------------------------------------------
     
    def procAtEndOfVideo(self):
        """ Process when reached end of video data. 

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        self.save(None) # save result data
        if self.inputFPLst != []:
        # not a single video, but a folder was opened
            self.inputFPIdx += 1
            if self.inputFPIdx >= len(self.inputFPLst):
            # reached end of file list
                self.onSpace(None) # stop continuous running
                msg = "All files are analyzed."
                self.showMsg(msg, "Info", wx.OK|wx.ICON_INFORMATION)
            else:
                self.inputFP = self.inputFPLst[self.inputFPIdx]
                self.startAnalysis()
        else:
            self.onSpace(None) # stop continuous running

    #---------------------------------------------------------------------------
     
    def initDataWithLoadedVideo(self):
        """ Init. info of loaded video and init/load result data.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.flags["rsltRecording"]:
            ### start video recorder
            ext = "." + self.inputFP.split(".")[-1]
            r_video_path = self.inputFP.replace(ext, "_FD")
            codec = self.vRW.vRecVideoCodec.lower()
            if codec in ['avc1', 'h264']:
                r_video_path += ".mp4"
            elif codec == 'xvid':
                r_video_path += ".avi"
            szRat = self.rsltRecParam["szRat"]
            rFPS = self.rsltRecParam["fps"]
            fH, fW = self.vRW.currFrame.shape[:2] 
            vSz = (int(np.ceil(fW*szRat)), int(np.ceil(fH*szRat)))
            self.vRW.initWriter(r_video_path, vSz, rFPS) 

        self.oData, endDataIdx = self.initOutputData() # init output data
         
        self.onChoice(None, "animalECase_cho") # to init left panel (parameters)

        self.runningDur = 0
        self.session_start_time = time()
        ### set timer for updating the current session running time
        self.timer["sessionTime"] = wx.Timer(self)
        self.Bind(wx.EVT_TIMER,
                  lambda event: self.onTimer(event, "sessionTime"),
                  self.timer["sessionTime"])
        self.timer["sessionTime"].Start(1000)

        result_csv_file = self.inputFP + '.csv'
        if path.isfile(result_csv_file) and endDataIdx > 0:
        # result CSV file exists & 
        # there's, at least, one data (with head direction) exists 
            self.jumpToFrame(endDataIdx) # move to the 1st None value

        txt = wx.FindWindowByName("fp_txt", self.panel["tp"])
        txt.SetValue('%s'%(path.basename(self.inputFP)))

        msg = "Starting to analyze a file '%s'"%(self.inputFP)
        self.log(msg=msg)

        self.proc_img() # process current frame

        showStatusBarMsg(self, "", -1)

    #---------------------------------------------------------------------------
    
    def initOutputData(self):
        """ initialize output data

        Args: None

        Returns: None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))
        if self.inputFP == "": return

        result_csv_file = self.inputFP + '.csv'
        oData = []
        if path.isfile(result_csv_file):
        # if there's previous result file for this video
            for fi in range(self.vRW.nFrames):
                oData.append(list(self.dataInitVal))
            # load previous CSV data
            oData, endDataIdx = self.loadData(result_csv_file, oData)
        else:
            for fi in range(self.vRW.nFrames):
                oData.append(list(self.dataInitVal))
            endDataIdx = fi
        # return output data as NumPy character array
        #return np.asarray(oData, dtype=self.dataStruct), endDataIdx
        return oData, endDataIdx

    #---------------------------------------------------------------------------
     
    def loadData(self, result_csv_file, oData):
        """ Load data from CSV file

        Args:
            result_csv_file (str): Result CSV file name.
            oData (list): Output data.

        Returns:
            oData (list): Output data with loaded data.
            endDataIdx (int): Index of last data.
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))
        
        endDataIdx = -1 
        ### read CSV file and update oData
        f = open(result_csv_file, 'r')
        lines = f.readlines()
        f.close()
        for li in range(1, len(lines)):
            items = [x.strip() for x in lines[li].split(',')]
            if len(items) <= 1: continue
        
            ### restore parameters
            if items[0] == 'AEC':
                self.animalECase = items[1]
                continue
            if items[0] in self.aecParam.keys():
                val = items[1].strip("[]").split("/")
                for vi in range(len(val)):
                    try: val[vi] = int(val[vi])
                    except:
                        try: val[vi] = float(val[vi])
                        except: pass
                if len(val) == 1: val = val[0]
                self.aecParam[items[0]]['value'] = val 
                continue
            
            ### restore data
            try: fi = int(items[0])
            except: continue
            for ci in range(len(self.dataCols)):
                val = str(items[ci+1])
                oData[fi][ci] = val
                if endDataIdx == -1:
                    if ci == 0 and val == 'None':
                    # store frame-index of 'None' value in the 1st column 
                        endDataIdx = copy(fi) 
            # make row as a tuple for structured numpy array
            #oData[fi] = tuple(oData[fi]) 
            oData[fi] = list(oData[fi]) 

        if endDataIdx == -1: endDataIdx = fi 
        return (oData, endDataIdx)

    #---------------------------------------------------------------------------
    
    def openInputData(self, isOpeningDir=False):
        """ Open input data file/folder. 

        Args:
            isOpeningDir (bool): Opening folder with input videos.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        animalECase_cho = wx.FindWindowByName("animalECase_cho", 
                                              self.panel["tp"])
        if self.session_start_time == -1:
        # not in analysis session. start a session
            if self.inputFP == '':
                ### open input file (folder)
                defDir = path.join(FPATH, "data")
                if isOpeningDir:
                    t = "Choose directory for analysis"
                    style = (wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST) 
                    dlg = wx.DirDialog(self, t, defaultPath=defDir, style=style)
                else:
                    t = "Open video file"
                    _wc = ""
                    for ext in self.allowedVideoExt:
                        _wc += "*.%s;"%(ext.lower())
                        _wc += "*.%s;"%(ext.upper())
                    _wc = _wc.rstrip(";")
                    wc = " (%s)|%s"%(_wc, _wc)
                    style = (wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
                    dlg = wx.FileDialog(self, t, defaultDir=defDir, 
                                        wildcard=wc, style=style)
                if dlg.ShowModal() == wx.ID_CANCEL:
                    dlg.Destroy()
                    return
                ##### [begin] get input video file path(s) -----
                if isOpeningDir:
                    dirPath = dlg.GetPath()
                    self.inputFPLst = []
                    ### add file paths of mp4 or avi files
                    fsLst = []
                    for ext in self.allowedVideoExt:
                        fsLst.append("*.%s"%(ext))
                        fsLst.append("*.%s"%(ext.upper()))
                    for fs in fsLst:
                        _path = path.join(dirPath, fs)
                        for fp in glob(_path):
                            if not fp in self.inputFPLst:
                                self.inputFPLst.append(fp)
                    ### remove video file paths without parameter text file
                    for i, fp in enumerate(self.inputFPLst):
                        ext = "." + fp.split(".")[-1]
                        paramFP = fp.replace(ext, "_params.txt")
                        if not path.isfile(paramFP):
                        # parameter text file doesn't exist
                            self.inputFPLst[i] = None # remove
                    while None in self.inputFPLst: self.inputFPLst.remove(None)
                    self.inputFPLst = sorted(self.inputFPLst)
                    if len(self.inputFPLst) == 0:
                    # no video file to analyze 
                        msg = "No video file with parameter file is found."
                        self.showMsg(msg, "Error", wx.OK|wx.ICON_ERROR)
                        return
                    ### get user confirmation on list of files 
                    msg = "* Selected folder:\n  %s\n\n"%(dirPath)
                    msg += "* List of video (mp4 or avi) files, "
                    msg += " found with parameter text files:\n"
                    for fp in self.inputFPLst:
                        msg += "  %s\n"%(path.basename(fp))
                    msg = msg.rstrip(",")
                    msg += "\n\nPlease confirm the list of files.\n"
                    msg += "After pressing the okay button here and the start"
                    msg += " button on the top panel, the program will go"
                    msg += " through all files. Each time one file is finished,"
                    msg += " a result CSV file will be generated in the same"
                    msg += " folder with the same name to the video file"
                    sz = (int(self.wSz[0]/2), int(self.wSz[1]/2))
                    dlg = PopupDialog(self, title="Query", msg=msg, 
                                      font=self.fonts[1], size=sz,
                                      flagCancelBtn=True) 
                    if dlg.ShowModal() == wx.ID_CANCEL:
                        dlg.Destroy()
                        return
                    ### set the first input file
                    self.inputFPIdx = 0
                    self.inputFP = self.inputFPLst[self.inputFPIdx]
                else:
                    # get video file path
                    self.inputFP = dlg.GetPath()
                ##### [end] get input video file path(s) -----
                dlg.Destroy()
            showStatusBarMsg(self, "Processing... please wait.", -1)
            animalECase_cho.Disable()  
            self.startAnalysis() 
            
        else: # in session. stop it.
            '''
            dlg = PopupDialog(self, 
                              title="Query", 
                              msg="Save data?", 
                              flagCancelBtn=True)
            rslt = dlg.ShowModal()
            dlg.Destroy()
            if rslt == wx.ID_OK:
                self.save(None) # save data
            '''
            if self.flags["isRunning"]:
                self.onSpace(None) # stop continuous running
            if self.flags["rsltRecording"]:
                self.vRW.closeWriter() # stop analysis video recording
            self.cv_proc.bg = None # remove background image
            self.vRW.closeReader() # close video
            ### init
            sTxt = wx.FindWindowByName("ssTime_sTxt", self.panel["tp"])
            sTxt.SetLabel("0:00:00")
            txt = wx.FindWindowByName("fp_txt", self.panel["tp"])
            txt.SetLabel("")
            self.session_start_time = -1
            self.inputFPLst = [] 
            self.inputFP = "" 
            self.oData = []
            self.analyzedFrame = None
            animalECase_cho.Enable()
            self.panel["ip"].Refresh()
            wx.CallLater(10, self.openInputData, isOpeningDir)
 
    #---------------------------------------------------------------------------
    
    def startAnalysis(self):
        """ Start analysis with an input video file. 

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        self.vRW.closeReader()        
        self.vRW.initReader(self.inputFP) # load video file to analyze
        # calculate ratio to resize to display it in UI
        self.ratFImgDispImg = calcI2DIRatio(self.vRW.currFrame, 
                                            self.pi["ip"]["sz"])
        self.cv_proc.initOnLoading() # init variables on loading data
        if self.animalECase in self.preProcCase:
            ext = "." + self.inputFP.split(".")[-1]
            fp = self.inputFP.replace(ext, "_params.txt")
            if path.isfile(fp):
            # parameter file for this input video already exists
                self.initDataWithLoadedVideo()
            else:
            # parameter file does not exist
                ### start pre-processing 
                args = (self.q2m,)
                startTaskThread(self, "preProc", 
                                self.cv_proc.preProcess, args=args)
        else:
            self.initDataWithLoadedVideo()
    
    #---------------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread

        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        """ DEPRECATED;; used for jumping to certain frame index 
                         without using cv2.CAP_PROP_POS_FRAMES
        if flag == "readFrames":
            if self.diff_FI_TFI> 1: # if moving multiple frames
                # update last_motion_frame
                #   to prevent difference goes over motion detection threshold 
                self.cv_proc.last_motion_frame = self.vRW.currFrame.copy()
            self.proc_img() # process loaded image
            self.flags["blockUI"] = False
        """

        if flag == "preProc":
            if self.animalECase == "aggrMax21":
                # store the extracted background image
                self.cv_proc.bg = rData[1]
                self.vRW.initReader(self.inputFP) # init video again
                self.initDataWithLoadedVideo()
            
            elif self.animalECase == "sleepDet23":
                aecp = self.aecParam
                ### update ant length in each well 
                for ri in range(aecp["uWRows"]["value"]): # rows
                    for ci in range(aecp["uWCols"]["value"]): # columns 
                        key = f'uAntLen{ri}{ci}'
                        widgetName = f'{key}_txt'
                        w = wx.FindWindowByName(widgetName, self.panel["lp"])
                        w.SetValue(str(rData[1][key]))
                self.vRW.initReader(self.inputFP) # init video again
                self.initDataWithLoadedVideo()
                self.applyChangedParam()

        postProcTaskThread(self, flag)

    #---------------------------------------------------------------------------
    
    def jumpToFrame(self, targetFI=-1):
        """ leap forward to a frame
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))
        if self.inputFP == "" or targetFI == -1: return
        
        if targetFI >= self.vRW.nFrames: targetFI = self.vRW.nFrames-1
        
        if FLAGS["seqFR2jump"]: # sequential frame reading to jump
            # difference between current frame-index and target-frame-index 
            self.diff_FI_TFI= abs(self.vRW.fi - targetFI)
            ret = self.vRW.getFrame(targetFI, useCAPPROP=False, 
                                    callbackFunc=self.callback)
        else:
            ret = self.vRW.getFrame(targetFI, useCAPPROP=True)

        if ret:
            if hasattr(self.cv_proc, "prevPGImg"): 
                self.cv_proc.prevPGImg = None # delete previous grey image
            self.proc_img() # process current frame
            if FLAGS["seqFR2jump"]: # sequential frame reading to jump
                if path.isfile(self.inputFP): self.flags["blockUI"] = True

    #---------------------------------------------------------------------------
    
    def save(self, event):
        """ Saving anaylsis result to CSV file.

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ext = "." + self.inputFP.split(".")[-1]
        fp = self.inputFP.replace(ext, ".csv")
        fh = open(fp, 'w')

        ### write parameters
        fh.write("Timestamp, %s\n"%(get_time_stamp()))
        fh.write("AEC, %s\n"%(self.animalECase))
        for key in sorted(self.aecParam.keys()):
            val = str(self.aecParam[key]['value'])
            if "," in val: val = val.replace(",", "/")
            line = "%s, %s\n"%(key, val)
            fh.write(line)
        fh.write('-----\n')

        ### write column heads 
        if self.animalECase == "aos21": line = "" 
        else: line = "frame-index, "
        for col in self.dataCols: line += "%s, "%(col)
        line = line.rstrip(", ") + "\n"
        fh.write(line)
        
        ### write data 
        for fi in range(len(self.oData)):
            if self.animalECase == "aos21": line = "" 
            else: line = "%i, "%(fi)
            for ci in range(len(self.dataCols)):
                line += "%s, "%(str(self.oData[fi][ci]))
            line = line.rstrip(", ") + "\n"
            fh.write(line)
        fh.write('-----\n')
        fh.close()

        fn = path.basename(fp)
        msg = "Saved file; '%s'\n"%(fn)
        showStatusBarMsg(self, msg, 5000)
        
        msg0 = "Saved a file '%s'"%(fn)
        if self.inputFPLst != []:
            msg0 += "; Finished"
            msg0 += " the %s file"%(convt_idx_to_ordinal(self.inputFPIdx))
            msg0 += " out of %s files"%(len(self.inputFPLst))
        self.log(msg=msg0)

        ### notify saving as email
        if self.flags["notifyEmail"]:
            msg1 = "FeatureDetector notification ---\n\n"
            ts = get_time_stamp().split("_")
            msg1 += "%s-%s-%s %s:%s:%s\n\n"%(ts[0], ts[1], ts[2], 
                                            ts[3], ts[4], ts[5])
            msg1 += msg0 + "\n"
            if self.inputFPIdx+1 == len(self.inputFPLst): 
                msg1 += "All files are analyzed."
            msg1 += "\n\n"
            self.sendNotification("FeatureDetector notification", msg1)

    #---------------------------------------------------------------------------
    
    def delData(self):
        """ Delete data (init. entire data dict), calculated so far 

        Args: None 

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.inputFP == "": return

        msg = "This will initialize the output data container,"
        msg += " deleting all calculated data so far.\n"
        msg += " Proceed?"
        _sz = (int(self.wSz[0]/3), int(self.wSz[1]/5))
        dlg = PopupDialog(self, title="Query", msg=msg, size=_sz,
                          flagCancelBtn=True) 
        ret = dlg.ShowModal()
        dlg.Destroy()
        if ret == wx.ID_CANCEL: return
        
        for vFI in  range(self.vRW.nFrames):
            for dIdx, dCol in enumerate(self.dataCols):
                self.oData[vFI][dIdx] = self.dataInitVal[dIdx] 
        self.cv_proc.initOnLoading() # init variables
        self.jumpToFrame(targetFI=0)

    #---------------------------------------------------------------------------

    def sendNotification(self, subject, content=""):
        """ Send notification email about analysis progress. 
        
        Args:
            subject (str): Subject of email.
            content (str): Text to send.
        
        Returns: 
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        txt = wx.FindWindowByName("email_txt", self.panel["tp"])
        rAddr = txt.GetValue().strip()
        regex = "^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$"
        if not re.search(regex, rAddr):
            msg = "'%s' is an invalid address."%(rAddr)
            print(msg)
            self.log(msg=msg)
            return
       
        try:
            server = smtplib.SMTP("smtp.gmail.com:587")
            server.ehlo()
            server.starttls()
            server.ehlo()
            msg = MIMEMultipart()
            msg["From"] = emailLogin(server)
            msg["Subject"] = subject 
            msg["To"] = rAddr
            msg_text = MIMEText(content, "plain")
            msg.attach(msg_text)
            server.sendmail(msg["From"], msg["To"], msg.as_string())
            server.quit()
            msg = "Email notification was sent to '%s'"%(rAddr)
        except Exception as e:
            self.log(msg=str(e))
            print(e)
            msg = "Failed to send email notification to '%s'"%(rAddr)

        self.log(msg=msg)
        print(msg)

    #---------------------------------------------------------------------------

    def onTimer(self, event, flag):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        if flag == "sessionTime": 
            ### update session running time
            if self.session_start_time != -1 and self.flags["isRunning"]:
                lbl = str(timedelta(seconds=self.runningDur)).split('.')[0]
                sTxt = wx.FindWindowByName("ssTime_sTxt", self.panel["tp"])
                sTxt.SetLabel(lbl)

        else:
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
            
            elif rData[0].startswith("finished"):
                self.callback(rData, flag)
 
    #---------------------------------------------------------------------------
    
    def showMsg(self, msg, mType, flag):
        """ display simple message 

        Args:
            msg (str): Message for logging.
            mType (str): Message type.
            flag (int): flags for wx.MessageBox
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        wx.MessageBox(msg, mType, flag)
        self.panel["tp"].SetFocus()

    #---------------------------------------------------------------------------
    
    def log(self, msg, tag="FeatureDetectorFrame"):
        """ leave a log 

        Args:
            msg (str): Message for logging.
            tag (str): Class tag.
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        txt = "%s, [%s], %s\n"%(get_time_stamp(), tag, msg)
        writeFile(self.logFP, txt)

    #---------------------------------------------------------------------------

    def onClose(self, event):
        """ Close this frame.
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        self.log(msg="Program closing")
        stopAllTimers(self.timer)
        rslt = True
        if self.flags["isRunning"]: # continuous analysis is running
            msg = "Session is running.\n"
            msg += "Unsaved data will be lost. (Stop analysis or"
            msg += " Cmd+S to save.)\nOkay to proceed to exit?"
            dlg = PopupDialog(self, title="Query", msg=msg, flagCancelBtn=True) 
            rslt = dlg.ShowModal()
            dlg.Destroy()
        if rslt:
            if hasattr(self.vRW, "video_rec") and self.vRW.video_rec != None:
                self.vRW.closeWriter()
            if sys.platform.startswith("win"): wx.CallLater(200, sys.exit)
            else: wx.CallLater(100, self.Destroy)
    
    #---------------------------------------------------------------------------

#===============================================================================

class FeatureDetectorApp(wx.App):
    def OnInit(self):
        if FLAGS["debug"]: logging.info(str(locals()))
        
        self.frame = FeatureDetectorFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-w", "-c"]:
            if sys.argv[1] == '-w': GNU_notice(1)
            elif sys.argv[1] == '-c': GNU_notice(2)
            sys.exit()

        elif sys.argv[1] == "-jumpBySeqRead":
            FLAGS["seqFR2jump"] = True 

    GNU_notice(0)
    CWD = getcwd()
    app = FeatureDetectorApp(redirect=False)
    app.MainLoop()


