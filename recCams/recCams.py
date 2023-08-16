# coding: UTF-8
"""
For recording frame images from multiple webcams. 

Jinook Oh, Cremer group in Institute of Science & Technology.

Dependency:
    Python (3.7)
    wxPython (4.0)
    OpenCV (3.4)

last edited: 2022-05-15

----------------------------------------------------------------------
Copyright (C) 2021 Jinook Oh & Sylvia Cremer 
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
v.0.1.202111: Initial development.

v.0.2.202203: Added functionality to grab images from uEye camera, 
              using pyueye library. The driver for uEye camera should be 
              already installed on the computer beforehand.

v.0.3.202204: Added functinoality to trigger recording only when certain
              color spot newly appeared.
              !!! NOTE; currently there're following assupmtions !!!
              1) A single camera is used.
              2) A single color is monitored.
              3) A single new blob is considered, meaning that.. 
                   a color detection might be missed 
                   if two ants spray acid at the same time, 
                   causing two new color blobs in two separate spots 
                   occur simultaneously.
"""
FLAGS = dict(
                debug = False,
                ueye = False, # start app in uEye camera mode
                ctRec = False, # start app in color-triggered recording mode
                )

import sys, queue, traceback
from threading import Thread
from os import path, mkdir
from copy import copy
from time import time, sleep
from datetime import datetime, timedelta
from glob import glob
from random import shuffle
from queue import Queue
import logging
logging.basicConfig(
    format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
    level=logging.INFO
    )

import wx, wx.adv, cv2
import wx.lib.scrolledpanel as SPanel 

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is
sys.path.append(FPATH) # add FPATH to path
P_DIR = path.split(FPATH)[0] # parent directory of the FPATH
sys.path.append(P_DIR) # add parent directory 
from modFFC import *
from modCV import *

__version__ = "0.3.202204"

#===============================================================================

class RecCamsFrame(wx.Frame):
    """ Frame for recording images from multiple webcams 

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self):
        if FLAGS["debug"]: logging.info(str(locals()))

        wPos = (0, 20)
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], wg[3]-100)
        title = "RecCams v.%s"%(__version__)
        style = wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX)
        wx.Frame.__init__(self, None, -1, title, 
                          pos=tuple(wPos), size=tuple(wSz), style=style)
        self.SetBackgroundColour('#333333')
        iconPath = path.join(P_DIR, "image", "icon.png")
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
        config = self.configuration("load") # load config file
        self.config = config
        self.flags = {}
        self.flags["blockUI"] = False # block user input 
        self.flags["dispFrame"] = True # whether to display frame on panel
        self.flags["prevFramePainted"] = True # whether previous frame is 
                                              #   painted on panel
        self.flags["recording"] = False # recording acquired images
        self.flags["videoRec"] = False # recording acquired images as video
        self.flags["useSKVideo"] = True # using scikit-video for video writing
        self.outputDir = path.join(FPATH, "output")
        if not path.isdir(self.outputDir): mkdir(self.outputDir)
        self.wSz = wSz # window size
        self.fonts = getWXFonts()
        print("[begin] Indexing cam ..")
        self.camIdx = []
        if FLAGS["ueye"]: # using uEye camera
            self.camIdx = getUEyeCamIdx() # get indices of cams
        else:
            self.camIdx = getCamIdx() # get indices of cams
        if len(self.camIdx) == 0:
            msg = "No cameras found."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            self.Destroy()
            return 
        print("[end] Indexing cam ..")
        """
        print("[begin] Getting the largest resolution of each cam ..")
        self.res = {}
        pRes = getPossibleResolutions()
        for ci in self.camIdx:
            resLst = []
            for _res in list(pRes[ci].keys()):
                resLst.append([int(r) for r in _res.split("x")])
            print(resLst)
            self.res[ci] = sorted(resLst)[-1]
        print("[end] Getting the largest resolution of each cam ..")
        """
        self.chosenCamIdx = [] 
        self.q2m = queue.Queue() # queue from thread to main
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for status bar message display
        self.videoIn = {} # videoIn module
        for ci in self.camIdx: self.videoIn[ci] = None
        self.lpWid = [] # wx widgets in left panel
        self.btnImgDir = path.join(P_DIR, "image")
        self.vRecFPS = {"default":10} # video recording FPS
        self.regTimerFPS = 20 # FPS of regular timer 
        self.codec = "XVID" # codec when using OpenCV's video-writer
        # parameters when using scikit-video's FFmpegWriter
        self.skOut = {"-c:v":"ffv1", "-pix_fmt":"yuv420p"}
        # for reading/writing video file
        self.vRW = VideoRW(self,
                           codec=self.codec,
                           useSKVideo=self.flags["useSKVideo"], 
                           skOut=self.skOut)
        shape = (pi["ip"]["sz"][1], pi["ip"]["sz"][0], 3)
        self.frameImg = np.zeros(shape, dtype=np.uint8) # frame array
        # widgets, disabled during recording
        self.wDisabledInRec = ["camIdx_cho", "addCam_btn", "removeCam_btn",
                               "addAllCam_btn", "removeAllCam_btn", "intv_txt"]
        self.colorPick = "" # temporary string for color picking
        self.pipBtnImgPath = path.join(self.btnImgDir, "pipette.png")
        self.pipBtnActImgPath = path.join(self.btnImgDir, "pipette_a.png")

        btnSz = (35, 35)
        vlSz = (-1, 25)
        ### create panels and widgets
        for pk in pi.keys():
            w = [] # widget list; each item represents a row in the panel
            setupPanel(w, self, pk)
        self.initLPWidgets()

        if FLAGS["ctRec"]: # color-triggered recording mode
            self.ctr = CTR(self) # initiate color-triggered recording class
        ##### [end] setting up attributes ----- 
        
        ### binding mouse event
        self.panel["ip"].Bind(wx.EVT_PAINT, self.onPaintIP)
        #self.panel["ip"].Bind(wx.EVT_LEFT_DOWN, self.onMLBDown)
        self.panel["ip"].Bind(wx.EVT_LEFT_UP, self.onMLBUp)
        #self.panel["ip"].Bind(wx.EVT_MOTION, self.onMMove)
        #self.panel["ip"].Bind(wx.EVT_MOUSEWHEEL, self.onMWheel) 

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
        self.timer["reg"].Start(int(1000/self.regTimerFPS))

        self.Bind(wx.EVT_CLOSE, self.onClose)

    #---------------------------------------------------------------------------
    
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        wSz = self.wSz
        if sys.platform.startswith("win"):
            style = (wx.TAB_TRAVERSAL|wx.SIMPLE_BORDER)
        else:
            style = (wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        pi = {} # information of panels
        # top panel for major buttons
        pi["lp"] = dict(pos=(0, 0), sz=(300, wSz[1]), bgCol=(75,75,75), 
                        style=style)
        pi["lp"]["fgCol"] = getConspicuousCol(pi["lp"]["bgCol"])
        lpSz = pi["lp"]["sz"]
        # middle right panel
        pi["ip"] = dict(pos=(lpSz[0], 0),
                        sz=(wSz[0]-lpSz[0], wSz[1]), bgCol=(30,30,30),
                        style=style)
        pi["ip"]["fgCol"] = getConspicuousCol(pi["ip"]["bgCol"])
        return pi

    #---------------------------------------------------------------------------
   
    def initLPWidgets(self):
        """ Set up wxPython widgets in 'lp' panel. 
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        pk = "lp"
        pSz = self.pi[pk]["sz"]
        fullW = int(pSz[0]*0.94)
        btnW = int(pSz[0]*0.49)
        bgCol = self.pi[pk]["bgCol"]
        fgCol = self.pi[pk]["fgCol"]

        for i, w in enumerate(self.lpWid):
        # through widgets in the panel
            try:
                self.gbs[pk].Detach(w) # detach grid from gridBagSizer
                w.Destroy() # destroy the widget
            except:
                pass

        w = [] # widget list; each item represents a row in the panel  
        w.append([
            {
                "type": "sTxt", 
                "nCol": 2, 
                "label": "Cam index: ", 
                "fgColor": fgCol,
                "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                },
            {
                "type": "cho", 
                "name": "camIdx", 
                "nCol": 2, 
                "size": (50,-1),
                "choices": [str(x) for x in self.camIdx], 
                "val": str(self.camIdx[0])
                }
            ])
        w.append([
            {
                "type": "sTxt", 
                "nCol": 2,
                "label": "rec. intv. (sec.): ", 
                "fgColor": fgCol, 
                "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                },
            {
                "type": "txt", 
                "name": "intv", 
                "nCol": 2, 
                "numOnly": True,
                "size": (int(pSz[0]*0.45),-1),
                "val": str(self.config["intv_txt"]) 
                }
            ])
        if FLAGS["ueye"]: fpsMaxVal = 20
        else: fpsMaxVal = 30
        w.append([
            {
                "type": "sTxt", 
                "label": "FPS limit: ", 
                "nCol": 2, 
                "fgColor": fgCol, 
                "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                },
            {
                "type": "sld", 
                "name": "fpsLimit",
                "nCol": 2, 
                "size": (int(pSz[0]*0.45),-1),
                "style": wx.SL_VALUE_LABEL,
                "minValue": 1, 
                "maxValue": fpsMaxVal, 
                "val": self.config["fpsLimit_sld"],
                }
            ])

        if FLAGS["ueye"]: # using uEye camera
            w.append([
                {
                    "type": "sTxt", 
                    "nCol": 2, 
                    "label": "Pixel-clock (MHz): ", 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    },
                {
                    "type": "sld", 
                    "name": "pixelClock",
                    "nCol": 2, 
                    "size": (int(pSz[0]*0.45),-1), 
                    "style": wx.SL_VALUE_LABEL,
                    "minValue": 5, 
                    "maxValue": 40, 
                    "val": self.config["pixelClock_sld"], 
                    "disable": True
                    }
                ])

            w.append([{"type":"sTxt", "nCol":4, "label":"", "border":5, 
                       "flag":wx.BOTTOM}]) 
        w.append([
            {
                "type": "btn", 
                "name": "addCam", 
                "nCol": 2, 
                "size": (btnW,-1), 
                "label": "+", 
                "bgColor": bgCol, 
                "fgColor": fgCol,
                "border": 0
                },
            {
                "type": "btn", 
                "name": "removeCam", 
                "nCol": 2, 
                "size": (btnW,-1), 
                "label": "-", 
                "bgColor": bgCol, 
                "fgColor": fgCol, 
                "border": 0
                }
            ])
        w.append([
            {
                "type": "btn", 
                "name": "addAllCam", 
                "nCol": 2, 
                "size": (btnW,-1), 
                "label": "Add all", 
                "bgColor": bgCol, 
                "fgColor": fgCol, 
                "border": 0
                },
            {
                "type": "btn", 
                "name": "removeAllCam", 
                "nCol": 2, 
                "size": (btnW,-1), 
                "label": "Remove all", 
                "bgColor": bgCol, 
                "fgColor": fgCol, 
                "border": 0
                }
            ])
        w.append([
            {
                "type": "txt", 
                "nCol": 4, 
                "name": "chosenCamIdx",
                "val": str(self.chosenCamIdx), 
                "style": wx.TE_READONLY, 
                "size": (fullW,-1)
                }
            ])
        w.append([{"type":"sTxt", "nCol":4, "label":"", "border":5, 
                       "flag":wx.BOTTOM}]) 

        if FLAGS["ueye"]:
        # using uEye camera
            ### add several image adjustment parameters
            w.append([
                {
                    "type": "sTxt", 
                    "label": "Exposure-time (ns): ", 
                    "nCol": 2, 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    },
                {
                    "type": "sld", 
                    "name": "exposureTime",
                    "nCol": 2, 
                    "style": wx.SL_VALUE_LABEL,
                    "size": (int(pSz[0]*0.45),-1),
                    "minValue": 50, 
                    "maxValue": 53000,
                    "val": self.config["exposureTime_sld"] 
                    }
                ])
            w.append([
                {
                    "type": "sTxt", 
                    "label": "Gamma: ", 
                    "nCol": 2, 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    },
                {
                    "type": "sld", 
                    "name": "gamma",
                    "nCol": 2,
                    "size": (int(pSz[0]*0.45),-1),
                    "style": wx.SL_VALUE_LABEL,
                    "minValue": 100, 
                    "maxValue": 220, 
                    "val": self.config["gamma_sld"] 
                    }
                ])
            w.append([
                {
                    "type": "sTxt", 
                    "nCol": 2, 
                    "label": "Color-U: ", 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    },
                {
                    "type": "sld", 
                    "name": "uEyeColU",
                    "nCol": 2, 
                    "size": (int(pSz[0]*0.45),-1),
                    "style": wx.SL_VALUE_LABEL,
                    "minValue": 0, 
                    "maxValue": 200, 
                    "val": self.config["uEyeColU_sld"]
                    }
                ])
            w.append([
                {
                    "type": "sTxt", 
                    "nCol": 2, 
                    "label": "Color-V: ", 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    },
                {
                    "type": "sld", 
                    "name": "uEyeColV",
                    "nCol": 2, 
                    "size": (int(pSz[0]*0.45),-1),
                    "style": wx.SL_VALUE_LABEL,
                    "minValue": 0, 
                    "maxValue": 200, 
                    "val": self.config["uEyeColV_sld"]
                    }
                ])
            w.append([{"type":"sTxt", "nCol":4, "label":"", "border":5, 
                       "flag":wx.BOTTOM}]) 

        if FLAGS["ctRec"]:
        # color-triggered recording 
            w.append([
                {
                    "type": "sTxt", 
                    "nCol": 1, 
                    "label": "Color to detect", 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    },
                {
                    "type": "btn", 
                    "name": "pipette-0", 
                    "nCol": 1, 
                    "size": (35,35),
                    "img": self.pipBtnImgPath,
                    "bgColor": bgCol,
                    "flag": (wx.ALIGN_CENTER|wx.ALIGN_CENTER_VERTICAL)
                    },
                ])
            for mLbl in ["min", "max"]: # HSV min & max values
                tmp = []
                tmp.append({
                    "type": "sTxt", 
                    "nCol": 1, 
                    "label": "      HSV-%s"%(mLbl), 
                    "fgColor": fgCol, 
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
                    })
                for k in ["H", "S", "V"]:
                    if k == "H": maxVal = 180
                    else: maxVal = 255
                    wn = "c-0-%s-%s"%(k, mLbl.capitalize())
                    tmp.append({
                        "type": "sld", 
                        "name": wn, 
                        "nCol": 1, 
                        "size": (int(pSz[0]*0.23), -1), 
                        "style": wx.SL_VALUE_LABEL,
                        "border": 1,
                        "minValue": 0, 
                        "maxValue": maxVal, 
                        "val": self.config[wn+"_sld"]
                        })
                w.append(tmp)
            w.append([{"type":"sTxt", "nCol":4, "label":"", "border":5, 
                       "flag":wx.BOTTOM}]) 
            w.append([
                {
                    "type": "sTxt", 
                    "label": "blob radius range to find:", 
                    "nCol": 2, 
                    "fgColor": fgCol,
                    "flag": (wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL),
                    "tooltip": "Minimum & maximum radius to find a color blob"
                    },
                {
                    "type": "txt", 
                    "name": "bMinRad", 
                    "nCol": 1, 
                    "size": (40,-1),
                    "numOnly": True,
                    "val": str(self.config["bMinRad_txt"]),
                    },
                {
                    "type": "txt", 
                    "name": "bMaxRad", 
                    "nCol": 1,
                    "size": (40,-1),
                    "numOnly": True,
                    "val": str(self.config["bMaxRad_txt"]),
                    }
                ])
            w.append([{"type":"sTxt", "nCol":4, "label":"", "border":5, 
                       "flag":wx.BOTTOM}]) 

        w.append([
            {
                "type": "btn", 
                "name": "startRec", 
                "nCol": 4, 
                "size": (fullW,-1), 
                "label": "Start recording", 
                "bgColor": bgCol, 
                "fgColor": fgCol
                }
            ])


        self.lpWid = setupPanel(w, self, pk)

    #---------------------------------------------------------------------------
    
    def onPaintIP(self, event):
        """ painting event 

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        # use BufferedPaintDC for counteracting flicker
        dc = wx.BufferedPaintDC(self.panel["ip"])
        dc.Clear()

        ### draw the frame
        sz = self.pi["ip"]["sz"]
        frame = cv2.cvtColor(self.frameImg, cv2.COLOR_BGR2RGB)
        img = wx.Image(sz[0], sz[1])
        img.SetData(frame.tobytes())
        dc.DrawBitmap(img.ConvertToBitmap(), 0, 0) 

        self.flags["prevFramePainted"] = True
        event.Skip()

    #---------------------------------------------------------------------------

    def startVideoIn(self, cIdx=-1):
        """ start videoIn 

        Args:
            cIdx (int): Camera index to start videoIn 

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        desiredRes = (-1, -1) # tuple(self.res[ci])
        outputFormat = "image"
        txt = wx.FindWindowByName("intv_txt", self.panel["lp"])
        intv = int(txt.GetValue()) # recording interval
        if cIdx == -1: idx = self.camIdx # start all cams
        else: idx = [cIdx]
        w = wx.FindWindowByName("fpsLimit_sld", self.panel["lp"])
        fpsLimit = int(w.GetValue())
        ### set VideoIn class to use
        if FLAGS["ueye"]: # using uEye camera
            viClass = VideoInUEye
            w = wx.FindWindowByName("pixelClock_sld", self.panel["lp"])
            pixelClock = int(w.GetValue())
            w = wx.FindWindowByName("exposureTime_sld", self.panel["lp"])
            expTime = (float(w.GetValue()) / 1000.0)
            w = wx.FindWindowByName("gamma_sld", self.panel["lp"])
            gamma = int(w.GetValue())
            params = dict(pixelClock=pixelClock, 
                          exposureTime=expTime, 
                          gamma=gamma)
        else:
            viClass = VideoIn
            params = dict(imgExt="jpg", jpegQuality=99) 
        for ci in idx:
            ### init videoIn module
            q2t = queue.Queue() # queue from main to thread
            self.videoIn[ci] = viClass(self, 
                                       ci, 
                                       desiredRes, 
                                       fpsLimit, 
                                       outputFormat, 
                                       intv,
                                       params)
            ### call 'run' as a thread
            args = (self.q2m, q2t, self.outputDir, True,)
            thrd = Thread(target=self.videoIn[ci].run, args=args) 
            thrd.start()
            self.videoIn[ci].thrd = thrd
            self.videoIn[ci].q2t = q2t

        if FLAGS["ctRec"]:
            self.ctr.initVars()

    #---------------------------------------------------------------------------
        
    def stopVideoIn(self, cIdx=-1):
        """ stop videoIn 

        Args:
            cIdx (int): Camera index to start videoIn 

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if cIdx == -1: idx = self.camIdx # stop all cams
        else: idx = [cIdx]
        for ci in idx:
            if self.videoIn[ci] == None: continue
            self.videoIn[ci].q2t.put("quit", True, None)
            self.videoIn[ci].thrd.join()
            self.timer["viClose"] = wx.CallLater(10, self.videoIn[ci].close)
            self.videoIn[ci] = None
            sleep(0.01)
        if self.chosenCamIdx == []:
            self.timer["frameInit"] = wx.CallLater(100, self.eraseFrameImg) 
            if FLAGS["ctRec"]: self.ctr.fBuffer = []
   
    #---------------------------------------------------------------------------

    def eraseFrameImg(self):
        """ Put 0 to frameImg array. 
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        self.frameImg[:,:,:] = 0
        self.panel["ip"].Refresh()

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
        if FLAGS["debug"]: logging.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "btn")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))

        if objName == "startRec_btn":
            self.flags["recording"] = not self.flags["recording"]
            if self.flags["recording"]:
                # start recording
                self.timer["startRec"] = wx.CallLater(10, self.startRecording)
                obj.SetLabel("Stop recording")
                
            else:
                # stop recording
                self.timer["endRec"] = wx.CallLater(10, self.stopRecording)
                obj.SetLabel("Start recording")

        elif objName.endswith("Cam_btn"):
            cho = wx.FindWindowByName("camIdx_cho", self.panel["lp"])
            cIdx = int(cho.GetString(cho.GetSelection()))
            if objName == "addCam_btn":
                if not cIdx in self.chosenCamIdx:
                    self.chosenCamIdx.append(cIdx)
                    self.timer["bvi"] = wx.CallLater(10, 
                                                     self.startVideoIn, cIdx)
            elif objName == "removeCam_btn":
                if cIdx in self.chosenCamIdx:
                    self.chosenCamIdx.remove(cIdx)
                    self.timer["evi"] = wx.CallLater(10, self.stopVideoIn, cIdx)

            elif objName == "addAllCam_btn":
                self.chosenCamIdx = copy(self.camIdx)
                self.timer["bvi"] = wx.CallLater(10, self.startVideoIn, -1)
            
            elif objName == "removeAllCam_btn":
                self.chosenCamIdx = [] 
                self.timer["evi"] = wx.CallLater(10, self.stopVideoIn, -1)

            txt = wx.FindWindowByName("chosenCamIdx_txt", self.panel["lp"])
            txt.SetValue(str(self.chosenCamIdx))

            self.frameImg[:,:,:] = 0 # init frame
            ### determine camera frame size and position
            nCam = len(self.chosenCamIdx)
            self.chosenCamPos = {} # top-left position of each cam frame
            ipSz = self.pi["ip"]["sz"]
            if nCam == 0: return
            if nCam == 1:
                w, h = ipSz 
            elif nCam == 2:
                w = int(ipSz[0]/2)
                h = ipSz[1]
                div = 2
            elif nCam > 2: 
                div = int(nCam/2)
                if div % 2 > 0: div += 1
                w = int(ipSz[0]/div)
                h = int(ipSz[1]/div)
            self.cFrameSz = [w, h] # store frame size of each cam
            pos = [0, 0]
            for i, ci in enumerate(self.chosenCamIdx):
                if i > 0:
                    if i % div == 0:
                        pos[0] = 0
                        pos[1] += h
                    else:
                        pos[0] += w
                self.chosenCamPos[ci] = tuple(pos) # store frame position
        
        if objName.startswith("pipette"): 
            if self.chosenCamIdx == []: return
            if self.colorPick == "":
            # no previously clicked color picking pipette
                pLbl = objName.replace("pipette-","").replace("_btn","") 
                self.colorPick = pLbl
                cursor = wx.Cursor(wx.CURSOR_CROSS)
                imgPath = self.pipBtnActImgPath
            else:
                # previously clicked button name
                bN = "pipette-%s_btn"%(self.colorPick) 
                if objName == bN:
                # the same pipette button is clicked
                    self.colorPick = ""
                    cursor = wx.Cursor(wx.CURSOR_ARROW)
                    imgPath = self.pipBtnImgPath
                else:
                # another pipette button is clicked 
                    pLbl = objName.replace("pipette-","").replace("_btn","") 
                    self.colorPick = pLbl
                    cursor = wx.Cursor(wx.CURSOR_CROSS) 
                    imgPath = self.pipBtnActImgPath
                    _btn = wx.FindWindowByName(bN, self.panel["ml"])
                    # set previously pressed button image back to inactive
                    set_img_for_btn(self.pipBtnImgPath, _btn)
            self.panel["ip"].SetCursor(cursor)
            set_img_for_btn(imgPath, obj)

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

        if objName == "dispFrame_chk":
            self.flags["dispFrame"] = objVal

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
        if FLAGS["debug"]: logging.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "cho")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return 

    #---------------------------------------------------------------------------
    
    def onSlider(self, event, objName=""):
        """ wx.Slider was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "sld")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName.startswith("exposureTime_sld"):
            ### send message to change exposure time 
            for ci in self.chosenCamIdx:
                msg = "expTime-%s"%(str(float(objVal) / 1000.0))
                self.videoIn[ci].q2t.put(msg, True, None)

        elif objName.startswith("gamma_sld"):
            ### send message to change gamma value 
            for ci in self.chosenCamIdx:
                msg = "gamma-%s"%(objVal)
                self.videoIn[ci].q2t.put(msg, True, None)

        elif objName.startswith("uEyeCol"):
            cs = objName.rstrip("_sld")[-1] # u or v
            self.config["uEyeCol%s_sld"%(cs)] = objVal
            ### send message to change U or V (YUV color mode)
            for ci in self.chosenCamIdx:
                msg = "color%s-%i"%(cs, objVal)
                self.videoIn[ci].q2t.put(msg, True, None)

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
        if FLAGS["debug"]: logging.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "txt")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

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

    def onTimer(self, event, flag=""):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """

        ##### [begin] process queued message -----
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
        
        elif rData[0] == "frameImg": 
            cIdx, frame, avgFPS = rData[1:]
            if self.videoIn[cIdx] == None: return
            self.videoIn[cIdx].frame = frame # store the frame image 

            if self.flags["dispFrame"] and \
                self.flags["prevFramePainted"] and \
                self.chosenCamIdx != []:      
                try:
                    ### put this camera frame image into self.frameImg 
                    fSz = tuple(self.cFrameSz)
                    l, t = self.chosenCamPos[cIdx]
                    if fSz[0] != frame.shape[1] or fSz[1] != frame.shape[0]:
                        fImg = cv2.resize(frame, 
                                          fSz, 
                                          interpolation=cv2.INTER_AREA)
                        self.frameImg[t:t+fSz[1], l:l+fSz[0]] = fImg
                    else:
                        self.frameImg = fImg
                    ### color-triggered recording
                    if FLAGS["ctRec"]:
                        self.frameImg = self.ctr.procFrame(
                                                            self.frameImg, 
                                                            avgFPS
                                                            )
                    ### draw
                    self.flags["prevFramePainted"] = False
                    self.panel["ip"].Refresh() 
                except Exception as e:
                    em = ''.join(traceback.format_exception(None, 
                                                        e, 
                                                        e.__traceback__))
                    print(em)
        '''
        elif rData[0] == "ERROR":
            showStatusBarMsg(self, rData[1], 10000, "#ff5555")
            return
        
        elif rData[0].startswith("finished"):
            self.callback(rData, flag)
        '''
        ##### [end] process queued message -----

    #---------------------------------------------------------------------------
    
    def onMLBDown(self, event):
        """ Processing when mouse L-buttton pressed down 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoIn == None: return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

    #---------------------------------------------------------------------------
    
    def onMLBUp(self, event):
        """ Processing when mouse L-buttton clicked 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"]: return

        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))
        
        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition()
        mState = wx.GetMouseState()

        if pk == "ip":
            if self.colorPick != "" and not mState.ControlDown():
            # color pick button was pressed and ctrl key is not pressed
                ### get color value of the clicked pixel
                img = self.frameImg
                b, g, r = self.frameImg[mp[1], mp[0]]
                col = {}
                # convert it to HSV value
                col["H"], col["S"], col["V"] = rgb2cvHSV(r, g, b)
                ck = self.colorPick
                ### set slider bars to HSV value
                for mLbl in ["Min", "Max"]: # HSV min & max values
                    for k in col.keys():
                        ### determine color value
                        if mLbl == "Min":
                            _col = max(0, col[k]-10) 
                        else:
                            if k == "H": maxVal = 180
                            else: maxVal = 255
                            _col = min(maxVal, col[k]+10)
                        ### set slider value
                        sN = "c-%s-%s-%s_sld"%(ck, k, mLbl.capitalize())
                        sld = wx.FindWindowByName(sN, self.panel["lp"])
                        sld.SetValue(_col)
                ### init
                cursor = wx.Cursor(wx.CURSOR_ARROW)
                self.panel["ip"].SetCursor(cursor)
                bN = "pipette-%s_btn"%(self.colorPick)
                btn = wx.FindWindowByName(bN, self.panel["lp"])
                set_img_for_btn(self.pipBtnImgPath, btn)
                self.colorPick = ""

    #---------------------------------------------------------------------------
    
    def onMMove(self, event):
        """ Mouse pointer moving 
        Show some info

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        #if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoIn == None: return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition()
        mState = wx.GetMouseState()

    #---------------------------------------------------------------------------
    
    def onMWheel(self, event):
        """ Mouse wheel event

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        #if FLAGS["debug"]: logging.info(str(locals()))
        
        if self.flags["blockUI"] or self.videoIn == None: return

        pk = event.GetEventObject().panelKey # panel key
        mWhRot = event.GetWheelRotation()
        mState = wx.GetMouseState()

    #---------------------------------------------------------------------------

    def startRecording(self):
        """ start recording 
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.chosenCamIdx == []: return

        ### disable widgets
        for wn in self.wDisabledInRec:
            w = wx.FindWindowByName(wn, self.panel["lp"])
            w.Disable() 

        if self.flags["videoRec"]:
        # video recording
            pass
            '''
            ### start video recorder
            vfp = path.join(self.outputDir, get_time_stamp())
            if self.flags["useSKVideo"]:
                codec = self.skOut["-c:v"].lower()
                if codec == "ffv1": vfp += ".mkv"
            else:
                codec = self.code.lower()
                if codec in ['avc1', 'h264']: vfp += ".mp4"
                elif codec in ['xvid', "mjpg"]: vfp += ".avi"
            w = self.videoIn.frame.shape[1]
            h = self.videoIn.frame.shape[0]
            vRecFPS = str(self.vRecFPS["default"])
            # init video writer
            self.vRW.initWriter(vfp, (w,h), vRecFPS=vRecFPS)
            self.recStartTime = time()
            '''

        else:
        # image recording
            for ci in self.chosenCamIdx:
                self.videoIn[ci].q2t.put("rec_init", True, None) 

    #---------------------------------------------------------------------------

    def stopRecording(self):
        """ stop recording 
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ### enable widgets
        for wn in self.wDisabledInRec:
            w = wx.FindWindowByName(wn, self.panel["lp"])
            w.Enable() 

        for ci in self.chosenCamIdx:
            self.videoIn[ci].q2t.put("rec_stop", True, None) 

    #---------------------------------------------------------------------------
    
    def log(self, msg, classTag):
        """ log 

        Args:
            msg (str): Message to log
            classTag (str): Class that issued the message

        Returns:
            None
        """ 
        if FLAGS["debug"]: logging.info(str(locals()))
    
    #---------------------------------------------------------------------------

    def configuration(self, flag):
        """ saving/loading configuration of the app 

        Args:
            flag (str): save or load

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))
        
        configFP = path.join(FPATH, "recCamsConfig")
       
        ### widget names and default values to be saved/loaded
        wNames = []
        dv = {} 
        wNames.append("fpsLimit_sld")
        dv["fpsLimit_sld"] = 20 
        wNames.append("intv_txt")
        dv["intv_txt"] = 300 
        if FLAGS["ueye"]:
            dv["fpsLimit_sld"] = 1
            wNames.append("pixelClock_sld")
            dv["pixelClock_sld"] = 5 
            wNames.append("exposureTime_sld")
            dv["exposureTime_sld"] = 40000 
            wNames.append("gamma_sld")
            dv["gamma_sld"] = 170 
            wNames.append("uEyeColU_sld")
            dv["uEyeColU_sld"] = 100 
            wNames.append("uEyeColV_sld")
            dv["uEyeColV_sld"] = 100 
        if FLAGS["ctRec"]:
            for mLbl in ["min", "max"]: # HSV min & max values
                for k in ["H", "S", "V"]:
                    wn = "c-0-%s-%s_sld"%(k, mLbl.capitalize())
                    wNames.append(wn)
                    dv[wn] = 0 
            wNames.append("bMinRad_txt")
            dv["bMinRad_txt"] = 10 
            wNames.append("bMaxRad_txt")
            dv["bMaxRad_txt"] = 30 

        if flag == "save":
        # saving config
            config = self.config 
            for wn in wNames:
                w = wx.FindWindowByName(wn, self.panel["lp"])
                if wn.endswith("_cho"):
                    config[wn] = w.GetString(w.GetSelection())
                elif wn.endswith("_spin") or wn.endswith("_sld"):
                    config[wn] = w.GetValue()
                elif wn.endswith("_txt"):
                    config[wn] = str2num(w.GetValue())
            fh = open(configFP, "wb")
            pickle.dump(config, fh)
            fh.close()
            return
        
        elif flag == "load":
        # loading config
            if path.isfile(configFP):
            # config file exists
                fh = open(configFP, "rb")
                config = pickle.load(fh)
                fh.close()
                for wn in wNames:
                    if not wn in config.keys():
                        config[wn] = dv[wn] # get default value
            else:
            # no config file found
                config = dv # use the default values
            return config
    
    #---------------------------------------------------------------------------

    def onClose(self, event):
        """ called when the frame is closed 

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        stopAllTimers(self.timer)
        if self.videoIn != None: self.stopVideoIn()
        self.configuration("save") # save configuration of the app 
        wx.CallLater(500, self.Destroy)

#===============================================================================

class RecCamsApp(wx.App):
    def OnInit(self):
        if FLAGS["debug"]: logging.info(str(locals()))
        
        self.frame = RecCamsFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

if __name__ == '__main__':
    GNU_notice(0) 
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '-w': GNU_notice(1)
            elif arg == '-c': GNU_notice(2)
            
            elif arg == '-ueye':
            # using uEye camera from IDS
                if path.isfile(path.join(P_DIR, "modUEye.py")):
                    from modUEye import *
                    if not FLAG_PYUEYE: # no 'pyueye' library is found
                        print("'pyueye' library is NOT found.")
                        sys.exit()
                    else:
                        FLAGS["ueye"] = True
                else:
                    print("'modUEye.py' file is NOT found.")
                    sys.exit()

            '''
            !!! this mode is not complete !!!
            elif arg == '-ctr':
            # color-triggered recording 
                FLAGS["ctRec"] = True # set color-triggered recording
                from procCTR import ColorTriggeredRecording as CTR  
            '''
    
    app = RecCamsApp(redirect=False)
    app.MainLoop()


