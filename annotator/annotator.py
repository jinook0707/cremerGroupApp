# coding: UTF-8

"""
An open-source software written in Python for annotating ant's behavior
  further with result data from FeatureDetector.
When a CSV file produced by FeatureDetector is available, the app will
  display moments of interest and other interesting points on video timeline.

This program was developed on Ubuntu Linux 18.04
& tested on Ubuntu & Windows 10.

Jinook Oh, Institute of Science and Technology Austria; Cremer group
2020.Jun.
last edited: 2023-03-27

Dependency:
    wxPython (4.0)
    NumPy (1.17)
    OpenCV (4.1)
    Decord (0.6)

------------------------------------------------------------------------
Copyright (C) 2021 Jinook Oh, Sylvia Cremer
- Contact: jinook.oh@ist.ac.at/ jinook0707@gmail.com

This program is released under GNU GPL v3.

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

v.0.1.202008: Initial development
v.0.1.202008: Minor fixes
v.0.2.202010: Major changes for Michaela's experiment
              (3 ants with 2 color marked ants)
v.0.3.202106:
  - Flowing video bar.
  - Adding view menu.
  - Opening video file,
  - Optionally, opening video file without CSV data file
    from 'featureDetector' is possible.
  - Concept of 'Behavior set' such as selfGroom, alloGroom, etc..,
  - Listing 22 behaviors in 'behaviorList.txt' file
      to make this more general annotator for ant's behavior.

v.0.3.202109:
  - Annotation buttons are changed from wx.Button
      to wx.lib.agw.gradientbutton for more noticeable state change
      of the button pressed/unpressed.

v.0.3.202110:
  - Removed 'michaela20' option.
  - Added a feature of magnifying annotation section by Ctrl+MouseWheel
      in the section. The magnification factor is stored in 'self.aRatP2F'.
  - Some variables such as behavior-set, names, colors, default subject number,
      etc are now read from a file, annoV.txt, to have users to be able to
      change them easily.

v.0.4.202111:
  - Changed video reading from OpenCV function to decord library
      for faster reading.

v.0.4.202202:
  - Behavior-set; 'all' was added. 'all' has currently one behavior 'deleteAll'.
  'deleteAll' is not a real behavior, but just for deleting any annotation of
  the selected performer.

v.0.4.202204:
  - logging; logging to a text file for debugging purpose was added.

v.0.4.202206:
  - Added '-opencv' argument option. If this argument is attached,
    'VideoRW' class will be used for reading video, instead of 'decord' library.
    This is due to too large amount of RAM usage increase of 'decord' with
    larger resolution videos. See 'test_decord_RAM_usage' folder.
    RAM usage increases from 5 GB to 25 GB as resolution increases from
    1 Megapixels to 6 Megapixels.

v.0.5.202212:
  - Changed the default video reading library to opencv (as a mistake in
    reading frames using OpenCV library was fixed in modCV.py). 'decord' library
    requires huge RAM capacity when reading 4k video.
  - Added 'Settings' menu, giving a mean to change the contents of annoV.txt
    in GUI. This could be helpful especially for picking up annotation colors
    for behaviors as it provides the color-picker interface.
  - View menu is now changed to turn on/off each behavior instead of
    behavior-set, giving a user more fine control over which behavior
    annotation button one wants to see. Especially for when user wants to
    annotate only a small number of behaviors.
"""

FLAGS = {}
# printing message to console at the beginning of each function
# for simple debugging
FLAGS["debug"] = False
# make a log file for debugging
FLAGS["log2file"] = False
# pyaudio library for audio play. (make it obsolete?)
FLAGS["pyaudio"] = False
# using 'decord' library to obtain video frames
FLAGS["decord"] = False

import sys, queue, itertools, pickle, traceback
from os import getcwd, path, remove
from glob import glob
from random import randint
from copy import copy, deepcopy
from datetime import timedelta

#import faulthandler; faulthandler.enable()
#python -Xfaulthandler my_program.py

import wx, wx.adv, cv2, decord
import numpy as np
#from wx.lib.wordwrap import wordwrap
import wx.lib.scrolledpanel as SPanel
### test whether PyAudio pacakage is installed in the system
try:
    import pyaudio
    FLAGS["pyaudio"] = True
except:
    pass

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is
sys.path.append(FPATH) # add FPATH to path
P_DIR = path.split(FPATH)[0] # parent directory of the FPATH
sys.path.append(P_DIR) # add parent directory

from modFFC import *
from modCV import *
if FLAGS["pyaudio"]:
    from modAudio import AudioOut

MyLogger = setMyLogger("annotator")
__version__ = "0.5.202212"

#===============================================================================

class AnnotatorFrame(wx.Frame):
    """ For navigating and annotating CSV data, produced by FeatureDetector

    Args:
        None

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self):
        if FLAGS["debug"]: logging.info(str(locals()))

        ### init
        nDP = wx.Display.GetCount() # number of screens
        if nDP == 1:
            screenIdx = 0
        else:
            ### get the index of the widest screen
            screenW = [int(wx.Display(i).GetGeometry()[2]) for i in range(nDP)]
            screenIdx = screenW.index(max(screenW))
        wg = wx.Display(screenIdx).GetGeometry()
        if sys.platform.startswith("win"):
            wPos = (wg[0], wg[1])
            wSz = (wg[2], int(wg[3]*0.83))
        else:
            wPos = (wg[0], wg[1]+25)
            wSz = (wg[2], int(wg[3]*0.85))
        title = "Annotator"
        title += " v.%s "%(__version__)
        style = wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX)
        wx.Frame.__init__(self, None, -1, title,
                          pos=tuple(wPos), size=tuple(wSz), style=style)
        self.frameBGC = "#333333"
        self.SetBackgroundColour(self.frameBGC)
        iconPath = path.join(P_DIR, "image", "icon.tif")
        if __name__ == '__main__' and path.isfile(iconPath):
            self.SetIcon(wx.Icon(iconPath)) # set app icon
        self.Bind(wx.EVT_CLOSE, self.onClose)
        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        # frame resizing
        updateFrameSize(self, (wSz[0], wSz[1]+self.statusbar.GetSize()[1]))

        ##### [begin] setting up attributes -----
        self.wSz = wSz
        self.fonts = getWXFonts()
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread
        self.flags = {}
        self.flags["isVPlaying"] = False # is video playing
        self.flags["blockUI"] = False # block user input
        self.flags["hideMarksOnFrame"] = False # whether to hide markings
                                               #   on frame-image
        self.flags["csvFile"] = False # CSV data file for the opened video file
                                      #   is available
        self.flags["annoBtnNum"] = False # 'True' means user is typing
                                         # number of annotation button.
        # container for storing menu items
        #   to disable/enable when annotation starts/stops
        self.menuItem = dict(view={}, settings={})
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for status-bar message display
        self.mrWid = [] # wx widgets in middle right panel
        self.csvFP = "" # CSV file path
        self.videoFP = "" # video for the analysis;
                          #   either a folder which has extracted frame images
                          #   or a video file
        self.inputVideoType = "" # 'frames' or 'videoFile'
        self.fi = 0 # current frame-index
        self.drawnFI = 0 # frame-index drawn on ML panel
        # x, y, w, h of bar, representing entire video, in "bt" panel
        self.vBarRect = (0, 10, pi["bt"]["sz"][0], 30)
        # margins in MM (MOI & Movements) bar
        self.mmM = dict(topMM=15, botMM=15, topVB=5, botVB=15)
        # height of sub-bar in MM bar
        self.mmH = dict(moi=2, mov=4)
        # Navigator bar under MM bar. This is used for jumping around
        # points on MM Bar in the current screen (while vBar is used for
        # jumping around entire video)
        self.mmNavH = 15
        self.annoH = 20 # height of annotation bar (for each subject)
        # margin between each annotation bar
        self.annoM = dict(top=20, btwn=5, bottom=10)
        self.fiVB = -1 # frame index, mouse pointer is on the video bar in "bt"
        self.fiMM = -1 # frame index, mouse pointer is on the MOI & Movement
        self.mMStartIdx = 0 # starting index of MM (MOI & Movements) bar
        # beginning and end of visible frame indices (in videoBar)
        #self.visibleIdxOnVideoBar = [0, self.vBarRect[2]]
        self.aRatP2FChoices = [str(x) for x in range(50, 0, -5)] + ["1"]
        self.aRatP2F = 1 # Ratio of pixel(width)-to-frame in annotation bar.
                         # If 1, a single frame is drawn in one pixel width.
                         # If 2, it'd be drawn in two pixel width and so on.
        self.defSPtCol = wx.Colour(127,127,127,50) # default color
          # for subject points
        # mouse wheel sensitivity choices
        self.wheelSensChoices = ["3.0", "2.5", "2.0", "1.5", "1.0", "0.5"]
        self.wheelSensChoices += ["0.1", "0.05", "0.04", "0.03", "0.02", "0.01"]
        self.mWheel = {}
        self.mWheel["val"] = [] # container list for wheel rotation values
                                #   of the recent mouse wheel event
        self.mWheel["thrSec"] = 0.1 # Seconds to recognize a single wheel event
            # Gather wheel values in self.mWheel["val"] as long as
            # last wheel event occurred less than self.mWheel["thrSec"].
        self.mWheel["lastEvtTime"] = -1 # time of the last mouse wheel event
        self.playSpdCho = ["fast+++", "fast++", "fast+", "fast",
                           "normal", "slow", "slow-", "slow--", "slow---"]
        self.playSpd = "normal"
        self.regTimerIntv = 50 # regular timer interval (when playSpd is normal)
        self.rTDefVal = 30 # default value of regular timer interval
        # adjustment value, used when 'playSpd' is changed
        # 'slow' is for adjusting the interval of regular timer
        # 'fast' is for adjusting the frame index in 'moveFrame'
        self.playSpdAdj = dict(slow=50, fast=5)
        self.vReader = None # for video reading when file opened
        self.rData = {} # data read from CSV file
        self.animalECaseChoices = ["default", "lindaWP21", "4PDish3Subj", 
                                   "sleepDet23"]
        '''
        - 'default' was egoCent21; annotating Christoph's video (20211018).
            A single ant in a video (one more ant in the arena, 
            but the video was cropped around a single ant. 
            Checks the amount of motion in video and 
            whether it's close to another ant(s). 
        - 'sleepDet23'; Results of FeatureDetector has amount of motion 
            in each arena and whether it seems to be sleeping.
            There's a single ant in an arena (total six arenas 
            in the video).
        '''
        self.aecWithMultiplePDishes = ["lindaWP21", "4PDish3Subj", "sleepDet23"]
        # read annoV.txt for behavior settings
        self.behSettings("read")
        self.defBehavior = copy(self.behavior)
        ### load config file
        config = self.configuration("load")
        config["view_all"] = True # set-'all' should be always True
        # animal experiment case
        self.animalECase = config["animalECase_cho"]
        # FPS of the opened video; this could be manually set,
        #   when the input video is a folder with frame images.
        self.videoFPS = float(config["fps_spin"])
        # adjustment value for mouse wheel sensitivity
        self.wheelSensAdj = float(config["wheelSensitivity_cho"])
        self.setAECaseParam() # set animal experiment case parameters
        self.dColTitle = [] # column titles of CSV data
        self.endDataIdx = -1 # row index where all data is 'None',
          # or simply end row index of data
        self.dur2chk4moi = 5 # duration in seconds to check for marking as
          # moment of interest
        self.dist4prox = 2.0 # distance threshold to determine whether two
          # individuals are close or not.
          # (e.g.: 2.0 = twice of approx. body length of an ant)
        self.numFD = 100 # how many frame data to show on frame-image
        self.sLen = 0.03 # approximate length of a single subject
                         # (in terms of ratio of frame height)
        self.vBBMP = None # for storing BMP of videoBar
        self.mmBarBMPLst = [] # list for storing BMPs of Moment of interest &
                              # Motion bar
        self.maxBMPWidth = 32767 # max. width of a wx.Bitmap
        # color of annotation button background when activated
        self.btnActCol = (0,0,0)
        # color of annotation button background when deactivated
        self.btnDeactCol = (127,127,127)
        self.clickedAnnoBar = [] # clicked annotation bar; [bKey, annoI]
        self.annoBtnNumStr = "" # entered annotation button number string
        self.activeBtnName = "" # currently active annotation button name
        # init some variables
        self.initOnInputFileLoading()
        ##### [end] setting up attributes -----

        btnSz = (35, 35)
        vlSz = (-1, 20)
        bw = 2
        imgPath = path.join(P_DIR, "image")
        defFG = "#cccccc"
        ### create panels
        for pk in pi.keys():
            w = [] # widge list; each item represents a row in the panel
            if pk == "tp": # top panel
                if FLAGS["decord"]: _decordLbl = "[-decord]"
                else: _decordLbl = "[-opencv]"
                w.append([
                    {"type":"cho", "nCol":1, "name":"animalECase",
                     "choices":self.animalECaseChoices, "val":self.animalECase,
                     "border":5},
                    {"type":"btn", "name":"open", "nCol":1,
                     "img":path.join(imgPath, "open.tif"),
                     "bgColor":self.frameBGC,
                     "tooltip":"Open CSV file", "size":btnSz, "border":bw},
                    {"type":"txt", "nCol":1, "val":"name of opened file",
                     "name":"inputFP", "size":(200,-1)},
                    {"type":"btn", "name":"closeFile", "nCol":1,
                     "img":path.join(imgPath, "delete.tif"),
                     "bgColor":self.frameBGC,
                     "tooltip":"Close file", "size":btnSz, "border":bw},
                    {"type":"sTxt", "label":"", "nCol":1, "fgColor":defFG,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":10},
                    {"type":"btn", "name":"save", "nCol":1,
                     "img":path.join(imgPath, "save.tif"),
                     "bgColor":self.frameBGC,
                     "tooltip":"Save data", "size":btnSz, "border":bw},
                    {"type":"sTxt", "label":"", "nCol":1, "fgColor":defFG,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":10},
                    #{"type":"sTxt", "label":"FPS ", "nCol":1,
                    # "fgColor":defFG, "border":bw},
                    {"type":"spin", "nCol":1, "name":"fps", "min":1, "max":60,
                     "inc":0.01, "initial":self.videoFPS, "double":True,
                     "style":(wx.SP_ARROW_KEYS|wx.SP_WRAP), "border":bw,
                     "size":(100,-1)},
                    {"type":"sTxt", "label":"", "nCol":1, "fgColor":defFG,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":10},
                    {"type":"sTxt", "label":"Wheel sensitivity", "nCol":1,
                     "fgColor":defFG},
                    {"type":"cho", "nCol":1, "name":"wheelSensitivity",
                     "choices":self.wheelSensChoices, "size":(80,-1),
                     "val":str(self.wheelSensAdj), "border":bw},
                    {"type":"sTxt", "label":"", "nCol":1, "fgColor":defFG,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":10},
                    {"type":"sTxt", "label":"PlaySpeed", "nCol":1,
                     "fgColor":defFG},
                    {"type":"cho", "nCol":1, "name":"playSpd",
                     "choices":self.playSpdCho, "size":(100,-1),
                     "val":self.playSpd, "border":bw},
                    {"type":"sTxt", "label":"", "nCol":1, "fgColor":defFG,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":10},
                    {"type":"sTxt", "nCol":1, "fgColor":defFG,
                     "label":"Annotation P2F"},
                    {"type":"cho", "nCol":1, "name":"aRatP2F",
                     "choices":self.aRatP2FChoices, "size":(75,-1),
                     "val":str(self.aRatP2F), "border":bw},
                    {"type":"sTxt", "label":"", "nCol":1,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":20},
                    {"type":"chk", "nCol":1, "name":"debugLogging2File",
                     "label":"logging", "style":wx.CHK_2STATE,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":20,
                     "val":DEBUG, "fgColor":"#cccccc"},
                    {"type":"sTxt", "label":_decordLbl, "nCol":1,
                     "fgColor":defFG},
                    {"type":"sTxt", "label":"", "nCol":1,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT), "border":100},
                    {"type":"btn", "name":"help", "nCol":1,
                     "img":path.join(imgPath, "help.tif"),
                     "bgColor":self.frameBGC,
                     "tooltip":"Help", "size":btnSz, "border":bw},
                    ])
            if pk in ["tp", "mr"]: useSPanel = True
            else: useSPanel = False
            # set up panel and wxPython widgets
            setupPanel(w, self, pk, useSPanel)

        ### Hide FPS SpinCtrl.
        ### This widget was for setting FPS for when the input is a folder
        ###   with frame images, however, currently we're using
        ###   only video files as input.
        ### Maybe used later? Currently just hiding.
        w = wx.FindWindowByName("fps_spin", self.panel["tp"])
        w.Hide()

        '''
        ### for counteracting flicker
        def disableEvt(*args, **kwargs): pass
        self.Bind(wx.EVT_ERASE_BACKGROUND, disableEvt)
        '''

        lastWidget = wx.FindWindowByName("help_btn", self.panel["tp"])
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

        ### bind events to panels
        self.panel["bt"].Bind(wx.EVT_PAINT, self.onPaintBT)
        self.panel["bt"].Bind(wx.EVT_LEFT_UP, self.onMLBUp)
        self.panel["bt"].Bind(wx.EVT_MOTION, self.onMMove)
        self.panel["bt"].Bind(wx.EVT_MOUSEWHEEL, self.onMWheel)
        self.panel["ml"].Bind(wx.EVT_PAINT, self.onPaintML)
        self.panel["ml"].Bind(wx.EVT_LEFT_DOWN, self.onMLBDown)
        self.panel["ml"].Bind(wx.EVT_LEFT_UP, self.onMLBUp)
        self.panel["ml"].Bind(wx.EVT_MOTION, self.onMMove)
        self.panel["ml"].Bind(wx.EVT_MOUSEWHEEL, self.onMWheel)
        #self.panel["ml"].Bind(wx.EVT_RIGHT_UP, self.onMRBUp)
        self.panel["mr"].Bind(wx.EVT_MOUSEWHEEL, self.onMWheel)

        ### set up audio module & loading click sounds
        self.audio = None
        if not sys.platform.startswith("win"):
            if FLAGS["pyaudio"]:
                if sys.platform.startswith("linux"): devKey = ["default"]
                #elif sys.platform.startswith("win"): devKey = ["Speaker"]
                elif sys.platform.startswith("darwin"): devKey = ["built-in"]
                self.audio = AudioOut(parent=self, devKeywords=devKey)
                self.audio.initSnds()
                self.sndFiles = [path.join(P_DIR, "sound", "snd_click.wav"),
                                 path.join(P_DIR, "sound", "snd_rClick.wav")]
                self.audio.loadSnds(self.sndFiles)

        ### set regular timer
        self.timer["reg"] = wx.Timer(self)
        self.Bind(wx.EVT_TIMER,
                  lambda event: self.onTimer(event, "reg"),
                  self.timer["reg"])
        self.timer["reg"].Start(self.regTimerIntv)

        # set up the menu bar
        self.setUpMenuBar(config)

        # update behaviors with the checked behavior sets
        self.onViewMenuItemChanged(None)

    #---------------------------------------------------------------------------

    def setUpMenuBar(self, config):
        """ set up the menu bar

        Args:
            config (dict/None): Configuration (which has info whether certain
                                menu item in View is checked or not)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        menuBar = wx.MenuBar()
        ### Annotator menu
        menu = wx.Menu()
        openItem = menu.Append(wx.Window.NewControlId(), item="Open\tCTRL+O")
        self.Bind(wx.EVT_MENU, self.openFile, openItem)
        closeItem = menu.Append(wx.Window.NewControlId(), item="Close\tCTRL+C")
        self.Bind(wx.EVT_MENU,
                  lambda event: self.onButtonPressDown(event, "closeFile_btn"),
                  closeItem)
        saveItem = menu.Append(wx.Window.NewControlId(), item="Save\tCTRL+S")
        self.Bind(wx.EVT_MENU, self.save, saveItem)
        quitItem = menu.Append(wx.Window.NewControlId(), item="Quit\tCTRL+Q")
        self.Bind(wx.EVT_MENU, self.onClose, quitItem)
        menuBar.Append(menu, "&Annotator")
        ### View menu
        menuV = wx.Menu()
        if config == None:
            ### store the current values
            origVMIVals = {}
            for k in self.menuItem["view"].keys():
                origVMIVals[k] = self.menuItem["view"][k].IsChecked()
        self.menuItem["view"] = {}
        for behSet in self.behSets:
            if behSet == "all": continue # all cateogry is not adjusted by user
            ### add behavior-set item
            cId = wx.Window.NewControlId()
            self.menuItem["view"][behSet] = menuV.Append(cId, item="* "+behSet,
                                                     kind=wx.ITEM_CHECK)
            menuV.Check(cId, False) # used only for turning on/off all
                                # behaviors which belong to this behavior-set
            self.Bind(wx.EVT_MENU, self.onViewMenuItemChanged,
                      self.menuItem["view"][behSet])
            for beh in self.behavior:
                if beh.split("-")[0] != behSet: continue
                ### add behavior item
                if config == None:
                    if key in origVMIVals.keys(): checked = origVMIVals[key]
                    else: checked = True
                else:
                    _k = "view_%s"%(beh)
                    if _k in config.keys(): checked = config[_k]
                    else: checked = True
                cId = wx.Window.NewControlId()
                lbl = "      " + beh
                self.menuItem["view"][beh] = menuV.Append(cId, item=lbl,
                                                      kind=wx.ITEM_CHECK)
                menuV.Check(cId, checked)
                self.Bind(wx.EVT_MENU, self.onViewMenuItemChanged,
                          self.menuItem["view"][beh])
        menuBar.Append(menuV, "&View")
        ### Settings menu
        menuS = wx.Menu()
        ms = "settings"
        self.menuItem[ms] = {}
        self.menuItem[ms]["general"] = menuS.Append(wx.Window.NewControlId(),
                                                    item="Settings\tF10")
        self.Bind(wx.EVT_MENU, lambda event: self.onKeyPress(event, wx.WXK_F10),
                  self.menuItem[ms]["general"])
        self.menuItem[ms]["color"] = menuS.Append(wx.Window.NewControlId(),
                                                  item="Color-settings\tF11")
        self.Bind(wx.EVT_MENU, lambda event: self.onKeyPress(event, wx.WXK_F11),
                  self.menuItem[ms]["color"])
        menuBar.Append(menuS, "&Settings")
        self.SetMenuBar(menuBar)

    #---------------------------------------------------------------------------

    def initOnInputFileLoading(self):
        """ Initializations/ re-init. when input data was loaded.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.videoFP == "": return

        self.oData = {} # output annotation data to save
        self.offset4disp = [0, 0] # Original image might be larger than
          # the frame image displaying area. This list shows x, y offset of
          # the frame image to show it.
        self.fImgZoomRate = 1.0
        self.currPDI = 0 # currently chosen petri-dish-index
        uNSubj = self.aecParam["uNSubj"]

        if self.animalECase in ["default"]:
        # originally annotating ego-centric video from Christoph
            self.moiCol = [0, 255, 0] # color for MOI bar
            self.movBCol = [200, 200, 0] # base color for marking motion
            subjIdx = list(range(uNSubj))
            self.perfLst = copy(subjIdx) # performer index list
            self.recvLst = copy(subjIdx) # receiver index list
            lst = list(itertools.product(subjIdx, repeat=2))
            self.behPairKey = []
            for p, r in lst: self.behPairKey.append("%i_%i"%(p, r))
            ### init dictionary for temporarily storing annotation-related
            ###   info. (before storing it in self.oData)
            self.aOp = {}
            for bpk in self.behPairKey:
                for beh in self.behavior:
                    bKey = "%s_%s"%(bpk, beh)
                    self.aOp[bKey] = {}
                    self.aOp[bKey]["state"] = "" # 'anno' or 'delete'
                      # annotating of behavior or deleting
                    self.aOp[bKey]["idx"] = -1 # beginning index for
                      # annotation or deletion

            ### init output data dictionary
            self.oData = {}
            for pdi in range(self.aecParam["uNPDish"]):
                self.oData[pdi] = {}
                for bpk in self.behPairKey:
                    for beh in self.behavior:
                        bKey = "%s_%s"%(bpk, beh)
                        self.oData[pdi][bKey] = []

        elif self.animalECase in self.aecWithMultiplePDishes:
            self.moiCol = [0, 255, 0] # color for MOI bar
            self.movBCol = [127, 127, 0] # base color for marking motion

            ### Set performer and reciever list.
            if self.animalECase == "lindaWP21":
                # half of subjects are ants and the other half are larvae.
                self.perfLst = list(range(int(uNSubj/2))) # performer indices
            else:
                self.perfLst = list(range(uNSubj)) # performer indices
            self.recvLst = list(range(uNSubj)) # receiver indices
            self.behPairKey = []
            for p in self.perfLst:
                for r in self.recvLst:
                    self.behPairKey.append("%i_%i"%(p, r))
            ### init dictionary for temporarily storing annotation-related
            ###   info. (before storing it in self.oData)
            self.aOp = {}
            for bpk in self.behPairKey:
                for beh in self.behavior:
                    bKey = "%s_%s"%(bpk, beh)
                    self.aOp[bKey] = {}
                    self.aOp[bKey]["state"] = "" # 'anno' or 'delete'
                      # annotating of behavior or deleting
                    self.aOp[bKey]["idx"] = -1 # beginning index for
                      # annotation or deletion

            ### init output data dictionary
            self.oData = {}
            for pdi in range(self.aecParam["uNPDish"]):
                self.oData[pdi] = {}
                for bpk in self.behPairKey:
                    for beh in self.behavior:
                        bKey = "%s_%s"%(bpk, beh)
                        self.oData[pdi][bKey] = []


        if self.csvFP != "":
            # pre-draw Moment of interest & Movements
            self.drawMMBar()

        annoH = self.annoH # height of annotation bar
        annoM = self.annoM # margin between each bar
        y = self.vBarRect[1] + self.vBarRect[3]
        if self.csvFP == "": self.mmBarHeight = 0
        else: self.mmBarHeight = self.mmBarBMPLst[0].GetSize()[1]
        y += self.mmBarHeight + self.mmNavH + annoM["top"]
        # set y-coordinates of annotation bars
        self.yABar = [y + x*(annoH+annoM["btwn"]) for x in range(uNSubj)]

        ### reset panels' heights
        vbr = self.vBarRect
        btH = vbr[1] + vbr[3] + self.mmBarHeight + self.mmNavH
        mg = self.annoM
        btH += mg["top"]
        btH += uNSubj * (self.annoH+mg["btwn"])
        btH += mg["bottom"]
        btSz = (self.pi["bt"]["sz"][0], btH)
        self.pi["bt"]["sz"] = btSz
        self.panel["bt"].SetSize(btSz)
        tpSz = self.pi["tp"]["sz"]
        mlH = self.wSz[1] - tpSz[1] - btSz[1]
        mlSz = (self.pi["ml"]["sz"][0], mlH)
        mlY = tpSz[1] + btSz[1]
        mlPos = (self.pi["ml"]["pos"][0], mlY)
        self.pi["ml"]["sz"] = mlSz
        self.panel["ml"].SetSize(mlSz)
        self.panel["ml"].SetPosition(mlPos)
        mrSz = (self.pi["mr"]["sz"][0], mlH)
        mrPos = (self.pi["mr"]["pos"][0], mlY)
        self.pi["mr"]["sz"] = mrSz
        self.panel["mr"].SetSize(mrSz)
        self.panel["mr"].SetPosition(mrPos)
        #wx.YieldIfNeeded() # update

        ### set frame image zoom rate to fit the image to the ml-panel
        fitSz = calcImgSzFitToPSz(self.fImgSz, self.pi["ml"]["sz"], 1.0)
        self.fImgZoomRate = (fitSz[0]/self.fImgSz[0])

        # init wxPython widgets in the middle-right panel
        self.initMRWidgets()

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
        #style = (wx.TAB_TRAVERSAL|wx.BORDER_SUNKEN)
        style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
        pi = {} # information of panels
        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], 50), bgCol="#333333",
                        style=style)
        tpSz = pi["tp"]["sz"]
        # bottom part of top (UI) panel
        pi["bt"] = dict(pos=(0, tpSz[1]), sz=(wSz[0], 300),
                        bgCol="#000000", style=style)
        btSz = pi["bt"]["sz"]
        # middle-left panel for showing frame image and other information
        pi["ml"] = dict(pos=(0, tpSz[1]+btSz[1]), sz=(int(tpSz[0]*0.7),
                        (wSz[1]-tpSz[1]-btSz[1])), bgCol="#161616",
                        style=style)
        mlSz = pi["ml"]["sz"]
        # middle-right panel
        pi["mr"] = dict(pos=(mlSz[0]+1, tpSz[1]+btSz[1]),
                        sz=(wSz[0]-mlSz[0], (wSz[1]-tpSz[1]-btSz[1])),
                        bgCol="#333333", style=style)
        return pi

    #---------------------------------------------------------------------------

    def initMRWidgets(self):
        """ Set up wxPython widgets when CSV is loaded.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        pk = "mr"
        mrP = self.panel[pk]
        mrSz = self.pi[pk]["sz"]
        hlSz = (int(mrSz[0]*0.95), -1)

        for i, w in enumerate(self.mrWid):
        # through widgets in middle right panel
            try:
                self.gbs[pk].Detach(w) # detach grid from gridBagSizer
                w.Destroy() # destroy the widget
            except:
                pass

        ##### [begin] set up middle right panel -----
        w = [] # widget list; each item represents a row in the panel
        sTxtFG = "#cccccc"
        if self.animalECase != "":
            nCol = 3
            uNSubj = self.aecParam["uNSubj"]
            perfLst = [str(_x) for _x in self.perfLst]
            recvLst = [str(_x) for _x in self.recvLst]
            # button to change petri-dish
            w.append([{"type":"btn", "name":"changePDish", "nCol":nCol,
                   "label":"Current Petri-dish-index: [%i]"%(self.currPDI),
                   "size":(int(mrSz[0]*0.9),-1), "bgColor":"#000000",
                   "fgColor":"#ffffff", "border":10}])
            w.append([{"type":"sLn", "size":hlSz, "nCol":nCol,
                       "style":wx.LI_HORIZONTAL}])
            w.append([{"type":"sTxt", "label":"Performer", "nCol":1,
                       "fgColor":sTxtFG},
                      {"type":"cho", "nCol":1, "name":"performer",
                       "choices":perfLst, "size":(100,-1), "val":"0"}])
            w.append([{"type":"sTxt", "label":"Receiver", "nCol":1,
                       "fgColor":sTxtFG},
                      {"type":"cho", "nCol":1, "name":"receiver",
                       "choices":recvLst, "size":(100,-1), "val":"0"}])
            w.append([{"type":"sLn", "size":hlSz, "nCol":nCol,
                       "style":wx.LI_HORIZONTAL}])
            w.append([{"type":"sTxt", "label":"Annotate behavior",
                       "nCol":2, "fgColor":sTxtFG},
                      {"type":"sTxt", "label":"Delete anno.", "nCol":1,
                       "fgColor":sTxtFG}])
            for bi, beh in enumerate(self.behavior):
                btnCol = self.annoCol[beh]
                if beh == "all-delAll":
                # only for delete all annotations button
                    w.append([{"type":"gBtn", "nCol":nCol,
                               "name":"%sDel"%(beh),
                               "label":"%i Delete all annotations"%(bi),
                               "size":(int(mrSz[0]*0.85),-1),
                               "bgColor":self.btnDeactCol, "fgColor":btnCol,
                               "border":10, "style":wx.NO_BORDER}])
                else:
                # other annotation buttons
                    w.append([{"type":"gBtn", "nCol":2, "name":beh,
                               "label":"%i %s"%(bi, beh),
                               "size":(int(mrSz[0]*0.6),-1), "border":10,
                               "bgColor":self.btnDeactCol, "fgColor":btnCol,
                               "style":wx.NO_BORDER},
                              {"type":"gBtn", "nCol":1, "name":"%sDel"%(beh),
                               "label":"Del.", "size":(int(mrSz[0]*0.2),-1),
                               "bgColor":self.btnDeactCol, "fgColor":btnCol,
                               "border":10, "style":wx.NO_BORDER}])
            w.append([{"type":"sTxt", "label":"", "nCol":nCol}])

        self.mrWid = setupPanel(w, self, pk, True)
        ##### [end] set up middle right panel -----

    #---------------------------------------------------------------------------

    def setAECaseParam(self):
        """ Set up parameters for the animal experiment case.

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        self.aecParam = {}

        if self.animalECase == "default":
            self.aecParam["uNPDish"] = self.defV["uNPDish"]
            self.aecParam["uNSubj"] = self.defV["uNSubj"]

        elif self.animalECase == "4PDish3Subj":
            self.aecParam["uNPDish"] = 4
            self.aecParam["uNSubj"] = 3

        elif self.animalECase == "lindaWP21":
            self.aecParam["uNPDish"] = 4
            self.aecParam["uNSubj"] = 4

        elif self.animalECase == "sleepDet23":
            self.aecParam["uNPDish"] = 6
            self.aecParam["uNSubj"] = 1 

    #---------------------------------------------------------------------------

    def onButtonPressDown(self, event, objName="", flag=""):
        """ wx.Butotn was pressed.

        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate the button press
              of the button with the given name.
            flag (str)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "btn")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        if objName == "save_btn":
            if sys.platform.startswith("win"):
                wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))
            else:
                if self.audio != None:
                    # sound_idx(0:click, 1:rClick), stream_idx, interupt
                    self.audio.playSnd(0, 0, True)
            self.save(None)
            return

        if self.flags["blockUI"] or not obj.IsEnabled(): return
        if sys.platform.startswith("win"):
            wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))
        else:
            if self.audio != None:
                # sound_idx(0:click, 1:rClick), stream_idx, interupt
                self.audio.playSnd(0, 0, True)

        if objName == "help_btn":
            '''
            if self.videoFP == "":
                msg = "Please load a file first."
                wx.MessageBox(msg, "Information", wx.OK|wx.ICON_INFORMATION)
            else:
                msg = HELPSTR
            '''
            msg = HELPSTR
            sz = (int(self.wSz[0]*0.5), int(self.wSz[1]*0.8))
            dlg = PopupDialog(self, -1, "Help string", msg,
                              size=sz, font=self.fonts[2], flagDefOK=True)
            dlg.ShowModal()
            dlg.Destroy()

        elif objName == "quit_btn":
            self.onClose(None)

        elif objName == "open_btn":
            self.openFile()

        elif objName.startswith("closeFile_btn"):
            '''
            if not flag == "initException":
                dlg = PopupDialog(self, title="Query", msg="Save data?",
                                  flagCancelBtn=True)
                rslt = dlg.ShowModal()
                dlg.Destroy()
                if rslt == wx.ID_OK: self.save(None)
            '''
            ### init
            self.csvFP = ""
            self.videoFP = ""
            self.aOp = {}
            self.oData = {}
            self.rData = {}
            self.dColTitle = []
            self.endDataIdx = -1
            self.panel["bt"].Refresh()
            self.panel["ml"].Refresh()
            self.initMRWidgets()
            txt = wx.FindWindowByName("inputFP_txt", self.panel["tp"])
            txt.SetValue("")
            cho = wx.FindWindowByName("animalECase_cho", self.panel["tp"])
            cho.Enable() # enable experiment case choice
            self.animalECase = cho.GetString(cho.GetSelection())
            self.setAECaseParam()
            self.ableMenuItems(True) # enable menu items

        elif objName == "changePDish_btn":
        # button to change the petri-dish annotating on
            #if self.animalECase in ["default", "lindaWP21", "4PDish3Subj"]:

            ### switch the current petri-dish-index
            ###   in the range of number of petri-dishes
            self.currPDI += 1
            if self.currPDI >= self.aecParam["uNPDish"]:
                self.currPDI = 0
            self.fi = 0 # init frame-index
            self.mMStartIdx = 0 # init MM bar start index
            self.clickedAnnoBar = []
            ### update static-text to display the current petri-dish-index
            btn = wx.FindWindowByName("changePDish_btn", self.panel["mr"])
            btn.SetLabel("Current Petri-dish-index: [%i]"%(self.currPDI))
            ### init annotation operation info
            for bpk in self.behPairKey:
                for beh in self.behavior:
                    bKey = "%s_%s"%(bpk, beh)
                    self.aOp[bKey]["state"] = ""
                    self.aOp[bKey]["idx"] = -1
            ### init annotation button color
            for beh in self.behavior:
                btnCol = self.annoCol[beh]
                for addTxt in ["", "Del"]:
                    name = "%s%s_gBtn"%(beh, addTxt)
                    btn = wx.FindWindowByName(name, self.panel["mr"])
                    if not btn is None:
                        btn.SetForegroundColour(btnCol)
            if self.csvFP != "":
                self.drawMMBar() # pre-draw Moment of interest & Movement
            self.panel["bt"].Refresh() # re-draw 'bt' (bar graph) panel
            self.panel["ml"].Refresh() # re-draw 'ml' (frame image) panel

        beh = objName.replace("_gBtn","").replace("Del","")
        behSet = beh.split("-")[0]

        # END of this funciton when the clicked button is -----
        #   not an annotation (or its deletion) button
        if not beh in self.behavior: return
        # -----------------------------------------------------

        ##### [begin] processing annotation button press -----
        cho = wx.FindWindowByName("performer_cho", self.panel["mr"])
        perf = cho.GetString(cho.GetSelection())
        cho = wx.FindWindowByName("receiver_cho", self.panel["mr"])
        rcvr = cho.GetString(cho.GetSelection())
        if behSet in self.monadicB: # this is monadic behavior
            if perf != rcvr:
                ### changed the receiver index as performer index
                rcvr = perf
                cho.SetSelection(int(perf))
        if behSet in self.dyadicB: # this is dyadic behavior
            if perf == rcvr:
                msg = "Performer & receiver should be different when"
                msg += " the behavior is dyadic."
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                return
        bPair = "%s_%s"%(perf, rcvr)
        bKey = "%s_%s"%(bPair, beh)
        flag = ""
        if "Del" in objName:
        # delete annotation button clicked
            if not self.aOp[bKey]["state"] == "delete":
                flag = "delete_on"
        else:
        # annotation button clicked
            if not self.aOp[bKey]["state"] == "anno":
                flag = "anno_on"

        # deactivate the currently active button/ finalizing current annotation
        self.deactivateCurrActiveBtn(perf)

        ### turning on
        if flag.endswith("_on"):
            if flag == "delete_on":
                self.aOp[bKey]["state"] = "delete"
            elif flag == "anno_on":
                self.aOp[bKey]["state"] = "anno"
            self.aOp[bKey]["idx"] = self.fi
            # change color of the button
            self.gBtnSolidCol(obj, self.btnActCol)
            # store the active button name
            self.activeBtnName = copy(objName) # activated
        else:
            self.activeBtnName = "" # no button is activated

        self.panel["bt"].Refresh()
        ##### [end] processing annotation button press -----

    #---------------------------------------------------------------------------

    def deactivateCurrActiveBtn(self, perf):
        """ deactivate currently active button.

        Args:
            perf (str): currently selected behavior performer

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        flagSave = False
        pdi = self.currPDI
        # behavior key for deleting all annotations
        bk4delAll = "%s_%s_all-delAll"%(perf, perf)
        sbjIdxLst = [str(_x) for _x in list(range(self.aecParam["uNSubj"]))]
        for rcvr in sbjIdxLst:
        # go through all receivers
            bPair = "%s_%s"%(perf, rcvr) # behavior pair
            for _beh in self.behavior:
            # go through each behavior
                _bKey = "%s_%s"%(bPair, _beh)
                if self.aOp[_bKey]["state"] == "delete" or \
                  self.aOp[bk4delAll]["state"] == "delete":
                # deletion was being conducted
                    if self.aOp[bk4delAll]["state"] == "delete":
                        _idx = self.aOp[bk4delAll]["idx"]
                    else:
                        _idx = self.aOp[_bKey]["idx"]
                    rng = [min(_idx, self.fi), max(_idx, self.fi)] # delete
                      # operation range
                    for annoI in range(len(self.oData[pdi][_bKey])):
                    # go through stored annotation index ranges
                        fi1, fi2 = self.oData[pdi][_bKey][annoI]
                        if fi1<=rng[0]<=fi2 or fi1<=rng[1]<=fi2 or \
                          rng[0]<=fi1<=rng[1] or rng[0]<=fi2<=rng[1]:
                        # at least, part of the range is overlapping
                        # with delete range
                            if rng[0] > fi1:
                                self.oData[pdi][_bKey].append([copy(fi1),
                                                               rng[0]-1])
                            if rng[1] < fi2:
                                self.oData[pdi][_bKey].append([rng[1]+1,
                                                               copy(fi2)])
                            self.oData[pdi][_bKey][annoI] = None
                    while None in self.oData[pdi][_bKey]:
                        self.oData[pdi][_bKey].remove(None)
                    flagSave = True
                elif self.aOp[_bKey]["state"] == "anno":
                # annotation was being conducted
                    ### store the range of behavior to output data
                    _startIdx = self.aOp[_bKey]["idx"]
                    if _startIdx != self.fi:
                        rngMin = min(_startIdx, self.fi)
                        rngMax = max(_startIdx, self.fi)
                        ### remove previous overlapping data
                        ### (new range overwrites previous data)
                        self.oData[pdi], nRng = self.removeOverlapData(
                                perf, self.oData[pdi], _bKey, [rngMin, rngMax]
                                )
                        if nRng != [-1, -1]:
                            # store the new range
                            self.oData[pdi][_bKey].append(nRng)
                    flagSave = True

                if not self.aOp[bk4delAll]["state"] == "delete":
                # not conducting 'deleting-all annotations'
                    ### init the behavior key's annotation state
                    self.aOp[_bKey]["state"] = ""
                    self.aOp[_bKey]["idx"] = -1
                ### change color of button (and deletion button)
                ###   to the color of deactivation
                for addTxt in ["", "Del"]:
                    _btnName = "%s%s_gBtn"%(_beh, addTxt)
                    _btn = wx.FindWindowByName(_btnName, self.panel["mr"])
                    # change color of the button
                    self.gBtnSolidCol(_btn, self.btnDeactCol)

        if self.aOp[bk4delAll]["state"] == "delete":
        # was deleting all annotations
            ### init its state
            self.aOp[bk4delAll]["state"] = ""
            self.aOp[bk4delAll]["idx"] = -1

        if flagSave:
            self.save(None, get_time_stamp()) # save backup CSV file

    #---------------------------------------------------------------------------

    def removeOverlapData(self, perf, oData, nBK, nRng):
        """ Remove annotation ranges where it overlaps with
        the new annotation range.

        Args:
            perf (str): currently selected behavior performer
            oData (dict): Output data with the current petri-dish index.
            nBK (str): Behavior key of new annotation.
            nRng (list): Min. and Max. index of new annotation.

        Returns:
            oData (dict): Output data with the current petri-dish index.
            nRng (list): Min. and Max. index of new annotation.
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        sbjIdxLst = [str(_x) for _x in list(range(self.aecParam["uNSubj"]))]
        for rcvr in sbjIdxLst:
        # go through all receivers
            bPair = "%s_%s"%(perf, rcvr) # behavior pair
            for beh in self.behavior:
            # go through each behavior
                _bk = "%s_%s"%(bPair, beh) # behavior key
                for i, _rng in enumerate(oData[_bk]):
                # go though already recorded old ranges (_rng)
                    _rng = oData[_bk][i]
                    if _rng == None: continue
                    if (_rng[0] < nRng[0] <= _rng[1]) and \
                      (nRng[1] >= _rng[1]):
                    # new range overlaps on the right side
                        if _bk == nBK: # behavior key is same as new one
                            oData[_bk][i] = None
                            nRng = [_rng[0], nRng[1]]
                        else:
                            oData[_bk][i] = [_rng[0], nRng[0]-1]
                    elif (_rng[0] <= nRng[1] < _rng[1]) and \
                      (nRng[0] <= _rng[0]):
                    # new range overlaps on the left side
                        if _bk == nBK: # behavior key is same as new one
                            oData[_bk][i] = None
                            nRng = [nRng[0], _rng[1]]
                        else:
                            oData[_bk][i] = [nRng[1]+1, _rng[1]]
                    elif nRng[0] <= _rng[0] and nRng[1] >= _rng[1]:
                    # new range covers the entire old range
                        # remove the old range
                        oData[_bk][i] = None
                    elif nRng[0] > _rng[0] and nRng[1] < _rng[1]:
                    # new range is in the middle of the old range
                        if _bk == nBK: # behavior key is same as new one
                            nRng = [-1, -1]
                        else:
                            oData[_bk][i] = [_rng[0], nRng[0]-1]
                            oData[_bk].append([nRng[1]+1, _rng[1]])
                    _rng = oData[_bk][i]
                    if _rng != None and _rng[0] == _rng[1]:
                    # start & end frame can't be same; at least
                    # one frame difference is required
                        oData[_bk][i] = None
                while None in oData[_bk]: oData[_bk].remove(None)
        return oData, nRng

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

        ret = preProcUIEvt(self, event, objName, "cho")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        if objName == "animalECase_cho":
        # animal experiment case was chosen
            self.animalECase = objVal
            self.setAECaseParam() # set animal experiment case parameters

        elif objName in ["performer_cho", "receiver_cho"]:
            cho = wx.FindWindowByName("performer_cho", self.panel["mr"])
            perf = cho.GetString(cho.GetSelection())
            cho = wx.FindWindowByName("receiver_cho", self.panel["mr"])
            receiver = cho.GetString(cho.GetSelection())
            bPair = "%s_%s"%(perf, receiver)
            for beh in self.behavior:
            # go through each behavior
                bKey = "%s_%s"%(bPair, beh)
                for addTxt in ["", "Del"]:
                    behSet = beh.split("-")[0]
                    btnName = "%s%s_gBtn"%(beh, addTxt)
                    ### change color of button
                    bgCol = self.btnDeactCol
                    if addTxt == "Del" and self.aOp[bKey]["state"] == "delete":
                    # deleting annotation is ongoing
                        bgCol = self.btnActCol
                    elif addTxt == "" and self.aOp[bKey]["state"] == "anno":
                    # annotation is ongoing
                        bgCol = self.btnActCol
                    btn = wx.FindWindowByName(btnName, self.panel["mr"])
                    # change color of the button
                    self.gBtnSolidCol(btn, bgCol)

        elif objName == "wheelSensitivity_cho":
        # change mouse wheel sensitivity adjustment value
            self.wheelSensAdj = float(objVal)
            #msg = "Mouse wheel rotation values will be adjusted by the"
            #msg += " factor of %.2f"%(self.wheelSensAdj)
            #wx.MessageBox(msg, "Information", wx.OK|wx.ICON_INFORMATION)

        elif objName == "playSpd_cho":
            self.playSpd = objVal
            self.timer["reg"].Stop()
            if objVal.startswith("slow"):
            # take care of slower play speed
            # (fast play speed will be processed in 'moveFrame' function)
                slowF = objVal.count("-") + 1
                intv = self.rTDefVal + self.playSpdAdj["slow"]*slowF
            else:
                intv = self.rTDefVal
            wx.CallLater(10, self.timer["reg"].Start, intv)

        elif objName == "aRatP2F_cho":
        # change annotation bar pixel-to-frame ratio
            self.aRatP2F = float(objVal)
            # update MM bar start index
            self.mMStartIdx = max(0,
                              self.fi-int(self.vBarRect[2]/self.aRatP2F/2))
            # re-draw bt panel (sections of MM bar + annotation bar)
            self.panel["bt"].Refresh()

    #---------------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.

        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event
                                     with the given name.

        Returns: None
        """
        global FLAGS
        if FLAGS["debug"]: logging.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "chk")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        if objName == "debugLogging2File_chk":
            FLAGS["debug"] = objVal
            FLAGS["log2file"] = objVal
            for h in logging.root.handlers[:]: logging.root.removeHandler(h)
            _format = "%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s"
            if FLAGS["log2file"]:
                logging.basicConfig(
                    filename=path.join(FPATH, "log_annotator.log"),
                    format=_format,
                    level=logging.DEBUG
                    )
                obj.SetForegroundColour("#cc0000")
            else:
                logging.basicConfig(
                    format=_format,
                    level=logging.DEBUG
                    )
                obj.SetForegroundColour("#cccccc")

    #---------------------------------------------------------------------------

    def onSpinCtrl(self, event, objName=""):
        """ wx.SpinCtrl value has changed.

        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate the button press
              of the button with the given name.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "spin")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        if objName == "fps_spin":
            self.videoFPS = objVal

        event.Skip()

    #---------------------------------------------------------------------------

    def onEnterInTextCtrl(self, event, objName=""):
        """ Enter-key was pressed in wx.TextCtrl (or SpinCtrl)

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

        event.Skip()

    #---------------------------------------------------------------------------

    def onKeyPress(self, event, kc=None, mState=None):
        """ Process key-press event

        Args: event (wx.Event)

        Returns: None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if kc == None: kc = event.GetKeyCode()
        if mState == None: mState = wx.GetMouseState()

        if mState.ControlDown():
        # CTRL modifier key is pressed
            if kc == ord("O"):
                self.openFile(None)

            elif kc == ord("C"):
                self.onButtonPressDown(None, "closeFile_btn")

            elif kc == ord("S"):
                self.save(None)

            elif kc == ord("Q"):
                self.onClose(None)

            elif kc == ord("I"):
                # increase annotation bar pixel-to-frame
                self.adjChoiceItem("aRatP2F_cho",
                                   self.aRatP2FChoices, "increase")
            elif kc == ord("K"):
                # decrease annotation bar pixel-to-frame
                self.adjChoiceItem("aRatP2F_cho",
                                   self.aRatP2FChoices, "decrease")

            elif kc in [wx.WXK_LEFT, wx.WXK_RIGHT, ord("J"), ord("L")]:
                ### jump to the closest begin/end of previous/next annotation
                if kc in [wx.WXK_LEFT, ord("J")]: flag = "dec"
                elif kc in [wx.WXK_RIGHT, ord("L")]: flag = "inc"
                pdi = self.currPDI
                idx = []
                cho = wx.FindWindowByName("performer_cho", self.panel["mr"])
                # get performer index string
                perf = cho.GetString(cho.GetSelection())
                for bKey in self.oData[pdi].keys():
                    # ignore when the performer doesn't match
                    if bKey.split("_")[0] != perf: continue
                    for annoI, (fi1, fi2) in enumerate(self.oData[pdi][bKey]):
                    # go through stored annotation index ranges
                        if flag == "dec":
                            if fi1 < self.fi: idx.append((fi1, bKey, annoI))
                            if fi2 < self.fi: idx.append((fi2, bKey, annoI))
                        elif flag == "inc":
                            if fi1 > self.fi: idx.append((fi1, bKey, annoI))
                            if fi2 > self.fi: idx.append((fi2, bKey, annoI))
                if idx != []:
                    if flag == "dec":
                        idx = sorted(idx, reverse=True)
                        self.moveFrame(None, "fIdx:%i"%(idx[0][0]))
                    elif flag == "inc":
                        idx = sorted(idx)
                        self.moveFrame(None, "fIdx:%i"%(idx[0][0]))
                    # store bKey & annotation index to display name
                    # of this annotation bar
                    self.clickedAnnoBar = list(idx[0][1:])

        elif mState.ShiftDown():
        # SHIFT modifier key is pressed

            if kc in [wx.WXK_LEFT, ord("J")]:
                self.moveFrame(None, "backBegin")

            elif kc in [wx.WXK_RIGHT, ord("L")]:
                self.moveFrame(None, "forEnd")

        elif mState.AltDown():
        # ALT modifier key is pressed

            if kc == ord("I"):
                # increase wheel sensitivity
                self.adjChoiceItem("wheelSensitivity_cho",
                                   self.wheelSensChoices, "increase")
            elif kc == ord("K"):
                # decrease wheel sensitivity
                self.adjChoiceItem("wheelSensitivity_cho",
                                   self.wheelSensChoices, "decrease")

            elif kc in [ord("J"), ord("L")]:
                ### move frame index according to the wheelSensitivity value.
                idx = self.fi
                if self.wheelSensAdj == self.wheelSensChoices[0]: mWh = 1
                else: mWh = round(120 * self.wheelSensAdj)
                if kc == ord("J"): idx = max(0, idx-mWh)
                else: idx = min(self.endDataIdx, idx+mWh)
                self.moveFrame(None, "fIdx:%i"%(idx))

        else:
        # No modifier key is pressed

            if kc == wx.WXK_SPACE:
                self.onSpace()

            elif kc in [wx.WXK_F10, wx.WXK_F11]:
                ### if settings menu is disabled, don't process further
                for k in self.menuItem["settings"]:
                    if not self.menuItem["settings"][k].IsEnabled():
                        return

                ### show popup dialog for settings
                if kc == wx.WXK_F10:
                    flag = "getWidgetInfo"
                    sz = (int(self.wSz[0]*0.2), int(self.wSz[1]*0.9))
                elif kc == wx.WXK_F11:
                    flag = "getWidgetInfo_color"
                    sz = (int(self.wSz[0]*0.2), int(self.wSz[1]*0.9))
                w, wNames = self.behSettings(flag, dlgSz=sz)
                dlg = PopupDialog(self, -1, "Settings", msg="", size=sz,
                                  font=self.fonts[2], flagCancelBtn=True,
                                  addW=w)
                if dlg.ShowModal() == wx.ID_OK:
                    # get values of settings
                    values = dlg.getValues(wNames)
                    # update & write the changed values
                    self.behSettings("write", v=values)
                dlg.Destroy()

            elif kc == ord("H"):
                self.hideMarkingsOnFrameImg()

            elif kc == ord("I"):
                # increase play speed
                self.adjChoiceItem("playSpd_cho", self.playSpdCho, "increase")

            elif kc == ord("K"):
                # decrease play speed
                self.adjChoiceItem("playSpd_cho", self.playSpdCho, "decrease")

            elif kc in [wx.WXK_LEFT, ord("J")]:
                self.moveFrame(None, "backward")

            elif kc in [wx.WXK_RIGHT, ord("L")]:
                self.moveFrame(None, "forward")

            elif kc == ord("P"):
                choices = [str(x) for x in self.perfLst]
                # change performer
                self.adjChoiceItem("performer_cho", choices, "change", "mr")

            elif kc == ord("R"):
                choices = [str(x) for x in self.recvLst]
                # change receiver
                self.adjChoiceItem("receiver_cho", choices, "change", "mr")

            elif kc == ord("C"):
                self.onButtonPressDown(None, "changePDish_btn")

            elif kc == wx.WXK_BACK:
                if self.clickedAnnoBar != []:
                    # clear displayed annotation behavior info on 'bt' panel
                    self.clickedAnnoBar = []
                    self.panel["bt"].Refresh()

            ##### [begin] accept behavior button number input -----
            def initABNVars():
                self.annoBtnNumStr = ""
                self.flags["annoBtnNum"] = False
                showStatusBarMsg(self, "", delTime=-1)

            numKey = str2num(chr(kc))

            if not self.flags["annoBtnNum"]:
                if numKey != None: # number was pressed
                    # turn on annotation button number press mode
                    self.flags["annoBtnNum"] = True
                    # store the pressed number
                    self.annoBtnNumStr = str(numKey)
                else:
                    if kc == wx.WXK_ESCAPE:
                        if self.activeBtnName != "": # there's active button
                            # deactivate the currently active button
                            self.onButtonPressDown(None, self.activeBtnName)
                showStatusBarMsg(self, str(self.annoBtnNumStr), delTime=-1)

            else:
            # annotation button number has been being pressed
                if kc == wx.WXK_ESCAPE:
                    # cancel annotation button number entering
                    initABNVars()

                elif kc == wx.WXK_BACK:
                    # delete the last character
                    self.annoBtnNumStr = self.annoBtnNumStr[:-1]

                elif kc in [wx.WXK_RETURN, ord("D")]:
                    if self.annoBtnNumStr != "":
                        btnNum = int(self.annoBtnNumStr)
                        if 0 <= btnNum <= len(self.behavior):
                            ### press the annotation button
                            objName = self.behavior[btnNum]
                            if kc == ord("D") or objName == "all-delAll":
                            # D was pressed or behavior is 'all-delAll'
                            # 'all-delAll' is only for delete all annotations
                                objName += "Del"
                            objName += "_gBtn"
                            self.onButtonPressDown(None, objName)
                    initABNVars()

                else:
                    if numKey != None: # number was pressed
                        # store the pressed number
                        self.annoBtnNumStr += str(numKey)

                showStatusBarMsg(self, str(self.annoBtnNumStr), delTime=-1)
            ##### [end] accept behavior button number input -----

    #---------------------------------------------------------------------------

    def onViewMenuItemChanged(self, event):
        """ Item in 'view' menu was (un)checked.

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if event != None:
        # a view menu item is clicked
            menu = event.GetEventObject()
            viKey = menu.GetLabel(event.Id).strip().lstrip("* ")
            sK = viKey.split("-")
            if len(sK) == 1:
            # behavior-set item is clicked
                bLst = []
                isAllChecked = True
                for beh in self.defBehavior:
                    if beh.split("-")[0] == viKey:
                    # this behavior belongs to the clicked behavior-set
                        bLst.append(beh)
                        if isAllChecked and \
                          not self.menuItem["view"][beh].IsChecked():
                            isAllChecked = False
                for beh in bLst:
                    if isAllChecked:
                        if beh in self.behavior:
                            self.behavior.remove(beh)
                            self.menuItem["view"][beh].Check(check=False)
                    else:
                        if not beh in self.behavior:
                            self.behavior.append(beh)
                            self.menuItem["view"][beh].Check(check=True)
                # behavior-set itme is always unchecked, and used just for
                # turning on/off all its child behaviors
                self.menuItem["view"][viKey].Check(check=False)
            else:
            # behavior item is clicked
                if self.menuItem["view"][viKey].IsChecked():
                    if not viKey in self.behavior:
                        self.behavior.append(viKey)
                else:
                    if viKey in self.behavior:
                        self.behavior.remove(viKey)

        elif event == None:
        # called for updating the behavior list
            # start from full behavior list
            self.behavior = copy(self.defBehavior)
            for key in self.menuItem["view"].keys():
                sK = key.split("-")
                if len(sK) == 1: continue # ignore behavior-set item
                behSet = sK[0]
                if not self.menuItem["view"][key].IsChecked():
                # this item is unchecked
                    if key in self.behavior:
                        self.behavior.remove(key)

        self.sortBehLst()
        self.behavior.remove("all-delAll")
        self.behavior.insert(0, "all-delAll")

    #---------------------------------------------------------------------------

    def onSpace(self):
        """ start/stop continuous play

        Args:
            None

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"]: return
        self.flags["isVPlaying"] = not self.flags["isVPlaying"]

    #---------------------------------------------------------------------------

    def makeModal(self, modal=True):
        """ Function to make the current frame to be modal.
        Recommended way in the document,
        https://wxpython.org/Phoenix/docs/html/MigrationGuide.html#makemodal

        Args:
            modal (bool): To make modal or not.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if modal and not hasattr(self, '_disabler'):
            self._disabler = wx.WindowDisabler(self)
        if not modal and hasattr(self, '_disabler'):
            del self._disabler

    #---------------------------------------------------------------------------

    def onClose(self, event):
        """ Close this frame.

        Args:
            event (wx.Event/str)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        stopAllTimers(self.timer)
        '''
        if self.inputVideoType == "videoFile":
            self.vRW.closeReader()
        if hasattr(self.vRW, "video_rec") and self.vRW.video_rec != None:
            self.vRW.closeWriter()
        '''
        if hasattr(self, 'audio') and self.audio != None:
            self.audio.stopAllSnds()
            self.audio.closeStreams()
        if not event == "failedInit":
            self.configuration("save") # save configuration of the app
        wx.CallLater(500, self.Destroy)

    #---------------------------------------------------------------------------

    def openFile(self, event=None):
        """ Open video file.

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ### choose video file to open
        _dir = path.join(FPATH, "data")
        if path.isdir(_dir): defDir = _dir
        else: defDir = FPATH
        t = "Open video file"
        wc = " (*.mp4;*.mov;*.avi;*.mkv)|*.mp4;*.mov;*.avi;*.mkv"
        style = (wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        dlg = wx.FileDialog(self, t, defaultDir=defDir, wildcard=wc,
                            style=style)
        '''
        ### choose result CSV file
        wc = 'CSV files (*.csv)|*.csv'
        dlg = wx.FileDialog(self,
                            "Open CSV file",
                            defaultDir=defDir,
                            wildcard=wc,
                            style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
        '''
        if dlg.ShowModal() == wx.ID_CANCEL: return
        videoFP = dlg.GetPath()

        #if path.isdir(videoFP):
        #    self.inputVideoType = "frames"
        self.inputVideoType = "videoFile"

        ### get CSV file path
        vExt = "." + videoFP.split(".")[-1]
        dlg.Destroy()
        csvFP = videoFP.replace(vExt, ".csv")

        ##### [begin] validate CSV file ---
        if path.isfile(csvFP):
            self.flags["csvFile"] = True
            fh = open(csvFP, 'r')
            lines = fh.readlines()
            fh.close()
            ###
            flagFI = False; fIdx = None
            for line in lines:
                items = [x.strip() for x in line.split(',')]
                if line.startswith("frame-index"):
                    flagFI = True
                    continue
                if flagFI:
                # line with column titles already found
                    # frame-index of 1st data line
                    fIdx = str2num(items[0], 'int')
                    break
            if flagFI == False or fIdx == None:
            # there should be column title line and
            #   there was, at least, one integer value for frame-index column
                self.flags["csvFile"] = False
                msg = "The CSV file is not a valid result file."
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                csvFP = ""
        else:
            msg = "No CSV file, %s, is found."%(path.basename(csvFP))
            wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)
            self.flags["csvFile"] = False
            csvFP = ""
        ##### [end] validate CSV file ---

        ### empty variables
        self.aecParam = {}
        self.dColTitle = []
        self.rData = {}
        self.endDataIdx = -1
        self.aecsData = {}
        self.vBBMP = None
        self.mmBarBMPLst = [] 

        self.videoFP = videoFP
        self.csvFP = csvFP
        txt = wx.FindWindowByName("inputFP_txt", self.panel["tp"])
        # display input video file name
        txt.SetValue(path.basename(videoFP))
        cho = wx.FindWindowByName("animalECase_cho", self.panel["tp"])
        # disable experiment case choice
        cho.Disable()

        self.fi = 0 # current frame-index
        self.drawnFI = 0 # frame-index drawn on ML panel
        if self.inputVideoType == "frames":
            # number of frames
            self.nFrames = len(glob(path.join(videoFP, "*.jpg")))
            img = cv2.imread(path.join(videoFP, "f0000000.jpg"))
            self.fImgSz = (img.shape[1], img.shape[0]) # frame image size
            ### enable FPS widget
            spin = wx.FindWindowByName("fps_spin", self.panel["tp"])
            spin.Enable(True)

        elif self.inputVideoType == "videoFile":
            if FLAGS["decord"]:
                self.vReader = decord.VideoReader(videoFP, ctx=decord.cpu(0))
                self.nFrames = len(self.vReader) # number of frames
                sh = self.vReader[0].shape
                self.fImgSz = (sh[1], sh[0]) # frame size
                self.videoFPS = self.vReader.get_avg_fps() # get FPS info
            else:
                self.vReader = VideoRW(self)
                self.vReader.initReader(videoFP)
                self.nFrames = self.vReader.nFrames
                self.fImgSz = self.vReader.vCapFSz
                self.videoFPS = 30
            ### re-start regular timer
            self.timer["reg"].Stop()
            intv = int(1/self.videoFPS*1000)
            wx.CallLater(10, self.timer["reg"].Start, intv)
            ### disable FPS widget
            spin = wx.FindWindowByName("fps_spin", self.panel["tp"])
            spin.SetValue(self.videoFPS)
            spin.Enable(False)

        if csvFP != "":
            args = (csvFP, self.q2m,)
            # start reading CSV data (from FeatureDetector)
            startTaskThread(self, "readData", self.readData, args)
        else:
            self.endDataIdx = self.nFrames-1
            self.setAECaseParam()
            # init class variables, relevant to loaded data
            self.initOnInputFileLoading()
            annoCSVFP = self.videoFP.replace(vExt, "_anno.csv")
            if path.isfile(annoCSVFP):
            # there is already file with annotation info
                self.readAnnotationData(annoCSVFP)
            self.ableMenuItems(False) # Disable view menu items
            self.panel["bt"].SetFocus()

    #---------------------------------------------------------------------------

    def readData(self, csvFP, q2m):
        """ Reading data from CSV file & calculation.

        Args:
            csvFP (str): File path of result CSV.
            q2m (queue.Queue): Queue to send data to main thread.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        rCSV = {} # raw data read from CSV lines
        aecsData = {} # AEC-specific data
        dColTitle = [] # columns of data
        aecParam = {} # experimental parameter, read from CSV file
        endDataIdx = None # frame-index where no data was recorded
        f2chk4moi = int(self.dur2chk4moi * self.videoFPS) # frames to check
          # for MOI

        ### read CSV file and update rData
        f = open(csvFP, 'r')
        lines = f.readlines()
        f.close()
        lineCnt = len(lines)
        for li in range(lineCnt):
            if li%100 == 0:
                msg = "Reading CSV file.. line-%i/ %i"%(li+1, lineCnt)
                q2m.put(("displayMsg", msg,), True, None)

            items = [x.strip() for x in lines[li].split(',')]
            if len(items) <= 1: continue

            ### store data columns
            if lines[li].startswith("frame-index"):
                for ii in range(len(items)):
                    dColTitle.append(items[ii])
                ### init some variables depending on animal experiment case
                if aecParam["AEC"] == "default":
                # for annotating egocentric ant video
                #   (for training in Christoph's program)
                # * In default (egoCent21), it's the moment of interest,
                #   when 'closeWithAnother' is True.
                    motionLst = [] # motion value around the focal ant

                elif aecParam["AEC"] in self.aecWithMultiplePDishes: 
                    nPDish = aecParam["uNPDish"] # number of petri-dishes
                    motionLst = [[] for x in range(nPDish)]

                continue

            ### store parameters
            if dColTitle == []:
            # has not reached the line of data column titles yet
            # (meaning the line is a parameter)
                pKey = items[0] # parameter key
                pVal = items[1].strip("[]").split("/") # parameter value
                for vi in range(len(pVal)): # parameter value might be a list
                    ### convert value to a number(int or float), if applicable
                    numVal = str2num(pVal[vi])
                    if numVal != None: # this is a number value
                        pVal[vi] = numVal
                    else:
                        ### convert value to boolean, if applicable
                        if pVal[vi].lower() == "true": pVal[vi] = True
                        elif pVal[vi].lower() == "false": pVal[vi] = False
                if len(pVal) == 1: pVal = pVal[0]
                aecParam[pKey] = pVal # store this parameter
                if pKey == "AEC" and pVal == "egoCent21":
                    aecParam[pKey] = "default"
                '''
                if pKey == "uAntMinArea":
                    aBodyLen = int(np.sqrt(pVal)*3) # length of ant body
                elif pKey == "uAntLength":
                    aBodyLen = pVal
                '''
                continue

            ### store data
            try: fi = int(items[0])
            except: continue
            flagAllNone = True # whether data in all columns were 'None'
            fi = str2num(items[0], 'int') # frame-index
            if fi == None:
                msg = "ERROR:: Invalid frame-index.\n%s"%(lines[li])
                print(msg)
                return
            rCSV[fi] = {}
            for ci in range(len(dColTitle)):
                cKey = dColTitle[ci]
                val = str2num(items[ci]) # try to convert value to number
                if val == None: # this is string value
                    ### convert value to boolean, if applicable
                    if items[ci].lower() == "true": val = True
                    elif items[ci].lower() == "false": val = False
                    else: val = items[ci] # store string value
                rCSV[fi][cKey] = val # store the value
                if val != "None": flagAllNone = False
                if aecParam["AEC"] == "default":
                    if cKey == "motion":
                        motionLst.append(rCSV[fi][cKey])
                elif aecParam["AEC"] in self.aecWithMultiplePDishes: 
                    if cKey.startswith("motion"):
                        _pdi = int(cKey.split("_")[1])
                        if _pdi < nPDish:
                            motionLst[_pdi].append(rCSV[fi][cKey])

            if flagAllNone: # the row was all None data; end of data
                del rCSV[fi]
                endDataIdx = fi-1
                break
        if endDataIdx == None: endDataIdx = fi
        endDataIdx = min(endDataIdx, self.nFrames-1) # endDataIdx can't be
          # larger than number of frames

        ### function to change list of frame-index to
        ###   frame ranges ([beginning-index, end-index])
        def lst2rng(inputLst, f2chk4moi, endDataIdx):
            outputLst = [[inputLst[0]]]
            consecIdx = outputLst[-1][0]
            for i in range(1, len(inputLst)):
                currIdx = inputLst[i]
                consecIdx += 1
                if consecIdx == currIdx:
                    continue
                else:
                    outputLst[-1].append(consecIdx+f2chk4moi-1)
                    outputLst[-1][0] = max(0, outputLst[-1][0])
                    outputLst.append([currIdx])
                    consecIdx = outputLst[-1][0]
            if len(outputLst[-1]) == 1:
                idx = min(consecIdx+f2chk4moi-1, endDataIdx)
                outputLst[-1].append(consecIdx+f2chk4moi-1)
            return outputLst

        ##### [begin] prepare AEC-specific data -----
        if aecParam["AEC"] == "default":
            aecsData["mMed"] = np.median(motionLst)
            aecsData["mStd"] = np.std(motionLst)

        elif aecParam["AEC"] in self.aecWithMultiplePDishes:
            aecsData["mMed"] = []
            aecsData["mStd"] = []
            for pdi in range(nPDish): # through petri-dishes
                aecsData["mMed"].append(np.median(motionLst[pdi]))
                aecsData["mStd"].append(np.std(motionLst[pdi]))
        ##### [end] prepare AEC-specific data -----

        ### send resulting data
        outputData = ("outputOfCSVLoading",
                      aecParam,
                      dColTitle,
                      rCSV,
                      endDataIdx,
                      aecsData)
        q2m.put(outputData, True, None)

    #---------------------------------------------------------------------------

    def readAnnotationData(self, annoCSVFP):
        """ Read annotation data.

        Args:
            annoCSVFP (str): Path to CSV file of annotation information.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        fh = open(annoCSVFP, "r")
        lines = fh.readlines()
        fh.close()
        dColTitle = []
        for line in lines:
            items = line.split(",")
            if dColTitle != []:
                ci = dColTitle.index("petri-dish-index")
                pdi = str2num(items[ci]) # petri-dish-index
                if pdi == None: continue
                ci = dColTitle.index("behavior-performer")
                bP = str2num(items[ci]) # behavior performer
                ci = dColTitle.index("receiver")
                bR = str2num(items[ci]) # behavior receiver
                ci = dColTitle.index("behavior")
                beh = items[ci].strip() # behavior
                ci = dColTitle.index("beginning-frame")
                fi1 = str2num(items[ci]) # behavior beginning index
                ci = dColTitle.index("end-frame")
                fi2 = str2num(items[ci]) # behavior end index
                if None in [bP, bR, fi1, fi2]: continue
                bKey = "%s_%s_%s"%(bP, bR, beh)
                self.oData[pdi][bKey].append([fi1, fi2])
            if line.startswith("Petri-dish-index"):
                ### store data column titles
                for item in items:
                    dColTitle.append(item.strip().lower())

    #---------------------------------------------------------------------------

    def ableMenuItems(self, enable=True):
        """ enable/disable stored menu items

        Args:
            enable (bool): Enable/disable menu items

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        for menuK in self.menuItem.keys():
            for k in self.menuItem[menuK].keys():
                self.menuItem[menuK][k].Enable(enable)

    #---------------------------------------------------------------------------

    def onPaintML(self, event):
        """ painting panel on mid-left side

        Args:
            event (wx.Event)

        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        if self.videoFP == "": return

        event.Skip()
        # use BufferedPaintDC for counteracting flicker
        dc = wx.BufferedPaintDC(self.panel['ml'])
        dc.SetBackground(wx.Brush(self.pi["ml"]["bgCol"]))
        dc.Clear()

        aecp = self.aecParam
        fzr = self.fImgZoomRate # frame image zoom rate
        offset = self.offset4disp # frame image offset
        fontCol = "#ffffff"
        if self.animalECase != "default":
            # color value to draw normal (not current or selected) data points
            nc = 30
            # change value of 'nc'
            ncChange = int((255-nc)/self.numFD)
            dDRng = [] # range of position data to display
            _hfRng = int(self.numFD/2)
            # beginning frame index
            dDRng.append(max(0, self.fi-_hfRng))
            # end frame index+1
            dDRng.append(min(self.endDataIdx+1, self.fi+_hfRng))
            # color for chosen petri-dish
            chosenPDishCol = "#ffffff"

        dc.SetFont(self.fonts[2])
        dc.SetTextForeground(fontCol)

        ### get frame image
        if self.inputVideoType == "frames":
            img = cv2.imread(path.join(self.videoFP, "f%07i.jpg"%self.fi))
        elif self.inputVideoType == "videoFile":
            if FLAGS["decord"]:
                img = self.vReader[self.fi].asnumpy()
                img = np.ascontiguousarray(img[:,:,::-1], dtype=np.uint8)
            else:
                self.vReader.getFrame(self.fi)
                self.fi = self.vReader.fi
                img = self.vReader.currFrame

        imgSz = (int(img.shape[1]*fzr), int(img.shape[0]*fzr))
        if fzr != 1.0:
            ### zoom frame image
            if fzr < 1.0: interpM = cv2.INTER_AREA
            else: interpM = cv2.INTER_CUBIC
            img = cv2.resize(img, imgSz, interpolation=interpM)
        ### crop image if it's out of panel
        pSz = self.pi["ml"]["sz"]
        x1 = 0
        if offset[0] < 0: x1 = abs(offset[0])
        y1 = 0
        if offset[1] < 0: y1 = abs(offset[1])
        x2 = imgSz[0]
        if offset[0] + imgSz[0] > pSz[0]: x2 = pSz[0] - offset[0]
        y2 = imgSz[1]
        if offset[1] + imgSz[1] > pSz[1]: y2 = pSz[1] - offset[1]
        img = img[y1:y2, x1:x2]
        ### display frame image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wxImg = wx.Image(img.shape[1], img.shape[0])
        wxImg.SetData(img.tobytes())
        _offX = 0
        _offY = 0
        if offset[0] > 0: _offX = offset[0]
        if offset[1] > 0: _offY = offset[1]
        dc.DrawBitmap(wxImg.ConvertToBitmap(), x=_offX, y=_offY)

        if self.animalECase != "default":
            offsetKey = "uROI%i_offset"%(self.currPDI)
            if offsetKey in aecp.keys():
                ### draw a circle around the chosen petri-dish
                roiOffset = aecp[offsetKey]
                x = int(imgSz[0]/2 + roiOffset[0]*fzr + offset[0])
                y = int(imgSz[1]/2 + roiOffset[1]*fzr + offset[1])
                r = int(roiOffset[2]*fzr + imgSz[1]/2)
                dc.SetPen(wx.Pen(chosenPDishCol, 5))
                dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
                dc.DrawCircle(x, y, r)

        ### draw status message for the current frame
        status_msg = "[frame: %i/ %i]"%(self.fi, self.endDataIdx)
        status_msg += ", [zoom: %i %%]"%(int(fzr*100))
        dc.DrawText(status_msg, 5, 5)

        if self.flags["hideMarksOnFrame"]:
            if self.animalECase == "michaela20" and self.csvFP != "":
                ### draw some (past and future frames') position data
                cfCoord = {} # coordinate of each ant in the current frame
                for pdi in range(aecp["uNPDish"]): cfCoord[pdi] = {}
                for fi in range(dDRng[0], dDRng[1]):
                # go through frames to display data points
                    dc.SetPen(wx.Pen(wx.Colour(nc,nc,nc), 0))
                    dc.SetBrush(wx.Brush(wx.Colour(nc,nc,nc)))
                    for pdi in range(aecp["uNPDish"]):
                        for ai in range(aecp["uNSubj"]):
                            ### draw data point of the subject
                            xCKey = "a%02i%02iPosX"%(pdi, ai)
                            yCKey = "a%02i%02iPosY"%(pdi, ai)
                            x =  self.rData[fi][xCKey]
                            y =  self.rData[fi][yCKey]
                            if x == "None" or y == "None": continue
                            x = int(x * fzr) + offset[0]
                            y = int(y * fzr) + offset[1]
                            dc.DrawCircle(x, y, 1) # draw point
                            if fi == self.fi:
                            # it's current video frame
                                cfCoord[pdi][ai] = (x,y)
                    nc += ncChange # increase color value

                ### draw the current frame position indicator
                dc.SetPen(wx.Pen("#ffffff", 1))
                for pdi in range(aecp["uNPDish"]):
                    for ai in range(aecp["uNSubj"]):
                        if not ai in cfCoord[pdi].keys(): continue
                        x, y = cfCoord[pdi][ai]
                        col = self.moiCol[str(ai)]
                        dc.SetBrush(wx.Brush(col))
                        dc.DrawCircle(x, y, 5) # draw the indicator
                        dc.DrawText(str(ai), x, y) # draw current subject index

        # refresh the panel below the top panel for video bar
        self.panel["bt"].Refresh()

        # update frame-index drawn on the panel
        self.drawnFI = self.fi

    #---------------------------------------------------------------------------

    def onPaintBT(self, event):
        """ Painting panel below the top panel

        Args:
            event (wx.Event)

        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        if self.videoFP == "": return

        event.Skip()
        # use BufferedPaintDC for counteracting flicker
        dc = wx.BufferedPaintDC(self.panel["bt"])
        dc.SetBackground(wx.Brush(self.pi["bt"]["bgCol"]))
        dc.Clear()

        pSz = self.pi["bt"]["sz"] # panel size

        dataLen = self.endDataIdx + 1
        if dataLen == 0: return # return, if no data

        vbr = self.vBarRect
        currFICol = "#ffffff" # current frame index color
        mpFICol = "#aaaaaa" # frame index color where mouse pointer is on
        # color for baseline of annotation bar
        annoBarBaseLineCol = (127,127,127)
        mMSIdx = self.mMStartIdx # MM bar starting index
        aRatP2F = self.aRatP2F # ratio of pixel of annotation to a frame
        dc.SetFont(self.fonts[1])
        fontW, fontH = dc.GetTextExtent("0")
        dc.SetTextForeground((255,255,255))

        dc.SetPen(wx.Pen((0,0,0), 0))
        dc.SetBrush(wx.Brush((100,100,100)))
        ### draw bar representing entire video
        if self.vBBMP != None:
            # draw video bar
            dc.DrawBitmap(self.vBBMP, vbr[0], vbr[1])
        else:
            dc.DrawRectangle(vbr[0], vbr[1], vbr[2], vbr[3])
        ### draw MM (MOI and Movements) bar 
        y = vbr[1] + vbr[3]
        if self.mmBarBMPLst != []:
            def getMMBarPart(bmp, sIdx, bmpW, resizedW): 
                img = bmp.ConvertToImage()
                mmSz = img.GetSize()
                rect = [sIdx, 0, bmpW, mmSz[1]]
                rect[2] = min(rect[2], mmSz[0]-sIdx)
                img = img.GetSubImage(tuple(rect))
                return img.Rescale(resizedW, mmSz[1]).ConvertToBitmap()
            if len(self.mmBarBMPLst) == 1: # there's only one BMP
                _w = min(dataLen-mMSIdx, int(vbr[2]/aRatP2F))
                _reW = int(_w * aRatP2F)
                # get BMP part to draw
                bmps = [getMMBarPart(self.mmBarBMPLst[0], mMSIdx, _w, _reW)]
                x4bmp = [vbr[0]] # x-coordinate to draw BMP
            else: # there're multiple BMPs 
                _idx = int(mMSIdx / self.maxBMPWidth)
                _sIdx = mMSIdx % self.maxBMPWidth
                _w = min(dataLen - mMSIdx,
                         (_idx+1)*self.maxBMPWidth - mMSIdx, 
                         int(vbr[2] / aRatP2F))
                _reW = int(_w * aRatP2F)
                # get BMP part to draw
                bmps = [getMMBarPart(self.mmBarBMPLst[_idx], _sIdx, _w, _reW)]
                x4bmp = [vbr[0]] # x-coordinate to draw BMP
                if _reW < vbr[2] and _idx < len(self.mmBarBMPLst)-1:
                # width of left length of this BMP is shorter than vbr width
                # & there's more mm-bar to display
                    x4bmp.append(vbr[0]+_reW)
                    _w = min(dataLen - mMSIdx,
                             int((vbr[2]-_reW) / aRatP2F))
                    _reW = int(_w * aRatP2F)
                    # get another BMP part
                    bmps.append(getMMBarPart(self.mmBarBMPLst[_idx+1], 
                                             0, _w, _reW))
            for _bi, bmp in enumerate(bmps): 
                dc.DrawBitmap(bmp, x4bmp[_bi], y) # draw BMP
            y += self.mmBarHeight
        ### draw navigator bar area (for navigating frames in screen)
        dc.SetBrush(wx.Brush((150,150,150)))
        dc.DrawRectangle(vbr[0], y, vbr[2], self.mmNavH)

        y += self.mmNavH

        ##### [begin] draw annotation data -----
        annoH = self.annoH # height of annotation bar
        annoM = self.annoM # margin between each bar
        currA = {} # currently drawing annotation bar info
        drawClickedAnnoBarStr = []
        nSubj = self.aecParam["uNSubj"]
        yA = self.yABar # y-coordinates for annotation bars
        pdi = self.currPDI
        for ai in range(nSubj):
            #dc.SetBrush(wx.Brush(c))
            ### draw base line for this behavior
            y = yA[ai] + int(annoH/2)
            dc.SetPen(wx.Pen(annoBarBaseLineCol, 1))
            dc.DrawLine(vbr[0], y, vbr[0]+vbr[2], y)
            for bi, beh in enumerate(self.behavior):
                annoCol = self.annoCol[beh]
                dc.SetPen(wx.Pen("#000000", 0, wx.TRANSPARENT))
                dc.SetBrush(wx.Brush(annoCol))
                ### draw already stored annotated behavior
                for bpk in self.behPairKey:
                    _bpk = bpk.split("_")
                    if _bpk[0] != str(ai): continue
                    bKey = "%s_%s"%(bpk, beh)
                    for annoI in range(len(self.oData[pdi][bKey])):
                        x1, x2 = self.oData[pdi][bKey][annoI]
                        x1 = vbr[0] + int((x1-mMSIdx)*aRatP2F)
                        x2 = vbr[0] + int((x2-mMSIdx)*aRatP2F)
                        if x2 < vbr[0] or x1 > vbr[0]+vbr[2]:
                            continue
                        x1 = max(vbr[0], x1)
                        x2 = min(x2, vbr[0]+vbr[2])
                        dc.DrawRectangle(x1, yA[ai], x2-x1+1, annoH)

                        if self.clickedAnnoBar != []:
                        # if there's clcked annotation bar
                            cab_bKey, cab_annoI = self.clickedAnnoBar
                            if cab_bKey == bKey and cab_annoI == annoI:
                            # bKey and annoI matches
                                drawClickedAnnoBarStr = [x1, yA[ai],
                                                         bKey, annoCol]

                ### store currently working annotation, if available
                for bpk in self.behPairKey:
                    _bpk = bpk.split("_")
                    if _bpk[0] != str(ai): continue
                    bKey = "%s_%s"%(bpk, beh)
                    _idx = self.aOp[bKey]["idx"]
                    if self.aOp[bKey]["state"] != "" and _idx != self.fi:
                        x1 = min(_idx, self.fi)
                        x2 = max(_idx, self.fi)
                        x1 = vbr[0] + int((x1-mMSIdx)*aRatP2F)
                        x2 = vbr[0] + int((x2-mMSIdx)*aRatP2F)
                        if x2 < vbr[0] or x1 > vbr[0]+vbr[2]: continue
                        x1 = max(vbr[0], x1)
                        x2 = min(x2, vbr[0]+vbr[2])
                        currA["x1"] = x1
                        currA["x2"] = x2-x1+1
                        currA["y"] = yA[ai]
                        if self.aOp[bKey]["state"] == "delete":
                            currA["col"] = "#000000"
                        else:
                            currA["col"] = annoCol

        if currA != {}:
            dc.SetBrush(wx.Brush(currA["col"]))
            # draw currently working annotation
            dc.DrawRectangle(currA["x1"], currA["y"], currA["x2"], annoH)

        if drawClickedAnnoBarStr != []:
        # there was clicked annotation bar
            x, y, bKey, annoCol = drawClickedAnnoBarStr
            tw, th = dc.GetTextExtent(bKey)
            dc.SetPen(wx.Pen("#ffffff", 1))
            dc.SetBrush(wx.Brush(annoCol))
            # draw background for writing bkey string
            dc.DrawRectangle(x, y, tw, th)
            tCol = getConspicuousCol(annoCol)
            dc.SetTextForeground(tCol)
            # draw text (bKey)
            dc.DrawText(bKey, x, y)
        ##### [end] draw annotation data -----

        ### draw the current frame position on video bar
        x = vbr[0] + int(vbr[2] * (self.fi/dataLen))
        dc.SetPen(wx.Pen(wx.Colour(currFICol), 1))
        dc.DrawLine(x, 0, x, vbr[1]+vbr[3])
        ### write text of the current frame-index and also in time-format
        if x > vbr[0]+vbr[2]/2:
            x -= int(fontW*(len(str(self.fi))+8)) # 8: for writing time-stamp
        else:
            x += 3
        dc.SetTextForeground(currFICol)
        ts = str(timedelta(seconds=int(self.fi/self.videoFPS))) # time-stamp
        dc.DrawText("%i [%s]"%(self.fi, ts), x, vbr[1]+vbr[3]-fontH)
        ### draw the current frame position on MOI & Movements part
        x = vbr[0] + int((self.fi-mMSIdx)*aRatP2F)
        dc.DrawLine(x, vbr[1]+vbr[3], x, pSz[1])

        if self.fiVB != -1: # mouse position is on video bar
            ### draw frame-index and line where the mouse pointer is hovering
            x, y = self.panel["bt"].ScreenToClient(wx.GetMousePosition())
            dc.SetPen(wx.Pen(wx.Colour(mpFICol), 1))
            dc.DrawLine(x, 0, x, vbr[1]+vbr[3])
            dc.SetTextForeground(mpFICol)
            frameIdxStr = str(self.fiVB)
            if x > vbr[0]+vbr[2]/2: x -= int(fontW * len(frameIdxStr) + 3)
            else: x += 3
            dc.DrawText(frameIdxStr, x+3, 2)
            ### draw line on MOI & Movements bar, corresponding to 'self.fiVB'
            x = vbr[0] + int((self.fiVB-mMSIdx)*aRatP2F)
            if x >= vbr[0] and x <= vbr[0]+vbr[2]:
                dc.DrawLine(x, vbr[1]+vbr[3], x, pSz[1])

        if self.fiMM != -1: # mouse position is on MOI & Movements
            ### draw frame-index and line where the mouse pointer is hovering
            x, y = self.panel["bt"].ScreenToClient(wx.GetMousePosition())
            frameIdxStr = str(self.fiMM)
            if x > vbr[0]+vbr[2]/2: x -= (fontW * len(frameIdxStr))
            dc.DrawText(frameIdxStr, x, y-fontH-2)

    #---------------------------------------------------------------------------

    def drawMMBar(self):
        """ Pre-draw Moment of interest and Movements as a BMP image,
        this part involves operations in long for-loop (for each frame),
        plus, it doesn't have to be updated frequently on mouse-move
         , mouse-click or mouse-wheel event on graph.

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        #-----------------------------------------------------------------------
        def detColor(baseCol, val, med, std):
        # function for determining line color, representing movements
            col = []
            for colI in range(3): # r, g, b
                c = baseCol[colI]
                if c != 0:
                    c = int(c * (val-med)/(std*3))
                    c = min(255, max(0, c))
                col.append(c)
            return wx.Colour(tuple(col))
        #-----------------------------------------------------------------------

        nPDish = self.aecParam["uNPDish"] # number of petri-dishes
        vbr = self.vBarRect # video bar rect
        dataLen = self.endDataIdx + 1
        pdi = self.currPDI # current index of petri-dish

        ### get median and standard-deviation of movementes
        if self.animalECase == "default":
            if "mMed" in self.aecsData.keys():
                mMed = self.aecsData["mMed"]
            if "mStd" in self.aecsData.keys():
                mStd = self.aecsData["mStd"]
        else:
            mMed = self.aecsData["mMed"][pdi]
            mStd = self.aecsData["mStd"][pdi]
        m = self.mmM # margins
        ht = self.mmH # height of each sub-bar
        ### set top y-coordinate for each bar
        yMOI_vb = m["topVB"]
        yMov_vb = m["topVB"] + ht["moi"]
        yMOI_mm = m["topMM"]
        yMov_mm = m["topMM"] + ht["moi"]

        ### set DC for VideoBar
        ###   (video bar will also have the same M&M info,
        ###    but entire video info will fit into one screen)
        vBHeight = m["topVB"] + ht["moi"] + ht["mov"] + m["botVB"]
        bmpVB = wx.Bitmap(vbr[2], vBHeight, depth=24)
        dcVB = wx.MemoryDC()
        dcVB.SelectObject(bmpVB)
        dcVB.SetBackground(wx.Brush((0,0,0)))
        dcVB.Clear()
        
        bmpH = m["topMM"] + ht["moi"] + ht["mov"]
        nBMPs = int(np.ceil(dataLen/self.maxBMPWidth))
        bFI = 0 # beginning frame index
        self.mmBarBMPLst = []
        for bi in range(nBMPs):
            if bi == nBMPs-1: bmpW = dataLen % self.maxBMPWidth
            else: bmpW = self.maxBMPWidth
            bmp = wx.Bitmap(bmpW, bmpH, depth=24)
            ### set DC for MM Bar
            dc = wx.MemoryDC()
            dc.SelectObject(bmp)
            dc.SetBackground(wx.Brush((0,0,0)))
            dc.Clear()
            ### draw movement bars
            for fi in range(bFI, dataLen):
                if fi-bFI >= self.maxBMPWidth:
                    bFI = copy(fi)
                    break
                if fi % 100 == 0:
                    msg = "Drawing data bars for frame-%i"%(fi)
                    showStatusBarMsg(self, msg, delTime=-1)
                    wx.YieldIfNeeded() # update
                if self.animalECase == "default":
                    motion = self.rData[fi]["motion"]
                    isClose = self.rData[fi]["closeWithAnother"]
                elif self.animalECase in ["lindaWP21", "4PDish3Subj", 
                                          "sleepDet23"]:
                    motion = self.rData[fi]["motion_%02i"%(pdi)]
                    isClose = self.rData[fi]["moi0_%02i"%(pdi)]
                vbx = int(fi / dataLen * vbr[2])
                mmx = fi - bFI
                if isClose:
                    ### draw MOI on MM bar
                    dc.SetPen(wx.Pen(self.moiCol, 1))
                    dc.DrawLine(mmx, yMOI_mm, mmx, yMOI_mm+ht["moi"])
                    ### draw MOI on Video-bar
                    dcVB.SetPen(wx.Pen(self.moiCol, 1))
                    dcVB.DrawLine(vbx, yMOI_vb, vbx, yMOI_vb+ht["moi"])
                col = detColor(self.movBCol, motion, mMed, mStd)
                ### draw movement on MM bar
                dc.SetPen(wx.Pen(col, 1))
                dc.DrawLine(mmx, yMov_mm, mmx, yMov_mm+ht["mov"])
                ### draw movement on video-bar
                dcVB.SetPen(wx.Pen(col, 1))
                dcVB.DrawLine(vbx, yMov_vb, vbx, yMov_vb+ht["mov"])

            dc.SelectObject(wx.NullBitmap)
            img = bmp.ConvertToImage()
            img.SaveFile(f"x{bi}.jpg")
            self.mmBarBMPLst.append(bmp) # store the drawn BMP
            
        # update video bar rect (height)
        self.vBarRect = (vbr[0], vbr[1], vbr[2], vBHeight)
        ### draw separating line between VB and MM
        dcVB.SetPen(wx.Pen(wx.Colour(100,100,100), m["botVB"]))
        y = vBHeight-int(m["botVB"]/2)
        dcVB.DrawLine(vbr[0], y, vbr[2], y)
        dcVB.SelectObject(wx.NullBitmap)
        # store the drawn VideoBar
        self.vBBMP = bmpVB

        showStatusBarMsg(self, "", delTime=-1)
        wx.YieldIfNeeded() # update

    #---------------------------------------------------------------------------

    def hideMarkingsOnFrameImg(self):
        """ Set variable whether to hide markings on frame image

        Args:
            None

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoFP == "": return
        self.flags["hideMarksOnFrame"] = not self.flags["hideMarksOnFrame"]
        self.panel["ml"].Refresh() # re-draw

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

        if flag == "finalizeSavingVideo":
            msg = 'Saved.\n'
            msg += self.savVidFP
            wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)
            self.fi = 0

        elif flag == "readData":
            try:
                self.aecParam = rData[1] # loaded parameters
                self.animalECase = self.aecParam["AEC"] # animal experiment case
                cho = wx.FindWindowByName("animalECase_cho", self.panel["tp"])
                selectionIdx = self.animalECaseChoices.index(self.animalECase)
                cho.SetSelection(selectionIdx)
                self.dColTitle = rData[2] # columns of loaded CSV data
                self.rData = rData[3] # data
                self.endDataIdx = rData[4] # frame-index of end of data;
                  # length data might be shorter than video length
                self.aecsData = rData[5] # AEC-specific data
                # init class variables, relevant to loaded data
                self.initOnInputFileLoading()
                annoCSVFP = self.csvFP.replace(".csv", "_anno.csv")
                if path.isfile(annoCSVFP):
                # there is already file with annotation info
                    self.readAnnotationData(annoCSVFP)
                self.ableMenuItems(False) # Disable view menu items
                ### show finish loading message
                fn = path.basename(self.csvFP)
                msg = "Loading [%s] is finished.\n\n"%(fn)
                wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)
                self.panel["bt"].SetFocus()
            except Exception as e:
                self.animalECase = "default"
                self.csvFP = ""
                self.setAECaseParam()
                self.initOnInputFileLoading()
                em = ''.join(traceback.format_exception(None,
                                                        e,
                                                        e.__traceback__))
                wx.MessageBox(em, "ERROR", wx.OK|wx.ICON_ERROR)
                wx.CallLater(100, self.onButtonPressDown, None,
                             "closeFile_btn", "initException")

        self.flags["blockUI"] = False
        if self.th != None:
            try: self.th.join()
            except: pass
        if flag in self.timer.keys():
            try: self.timer[flag].Stop()
            except: pass
        if hasattr(self, "tmpWaitingMsgPanel"):
            self.tmpWaitingMsgPanel.Destroy()
        for pk in self.panel.keys(): self.panel[pk].Show() # show other panels
        self.panel["ml"].Refresh() # draw frame-image panel
        showStatusBarMsg(self, "", delTime=-1)

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

        if flag == "reg":
        # regular timer
            if self.drawnFI != self.fi: return
            if self.flags["isVPlaying"]:
                self.moveFrame(None, "forward") # next frame
                if self.fi == 0 or self.fi == self.endDataIdx:
                # reached end of available data
                    self.onSpace() # stop
            self.onMWheel(None)

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
                showStatusBarMsg(self, rData[1], delTime=-1)
                return

            if flag == "readData": # reading CSV data
                if rData[0] == "outputOfCSVLoading":
                    self.timer[flag].Stop()
                    showStatusBarMsg(self, "", delTime=-1)
                    self.callback(rData, flag=flag)

    #---------------------------------------------------------------------------

    def onMLBDown(self, event):
        """ Left mouse button was pressed

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

        if pk == "bt": # mouse pressed in "bt" panel
            pass

        elif pk == "ml": # mouse pressed in "ml" panel
            if mState.ControlDown():
                self.prevOffset4disp = copy(self.offset4disp)
                self.panel[pk].mousePressedPt = mp
            #elif mState.ShiftDown():

    #---------------------------------------------------------------------------

    def onMLBUp(self, event):
        """ Left mouse button was released

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoFP == "": return
        flag_graphRedraw = False

        if sys.platform.startswith("win"):
            wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))
        else:
            if self.audio != None:
                self.audio.playSnd(0, 0, True)

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse position
        mState = wx.GetMouseState()

        if pk == "bt": # mouse clicked in "bt" panel
            vbr = self.vBarRect

            if vbr[0] <= mp[0] <= vbr[0]+vbr[2]:
                vbrY2 = vbr[1]+vbr[3]
                if vbr[1] <= mp[1] <= vbrY2:
                # mouse pointer is in video bar rect
                    mpRat = (mp[0]-vbr[0]) / vbr[2]
                    self.fiVB = int((self.endDataIdx+1) * mpRat)
                    self.moveFrame(None, "fIdx:%i"%(self.fiVB))
                elif vbrY2 < mp[1] <= vbrY2+self.mmBarHeight+self.mmNavH:
                # mouse pointer is in MOI & Movement area
                    self.fiMM = int((mp[0]-vbr[0])/self.aRatP2F) + \
                                  self.mMStartIdx
                    self.moveFrame(None, "fIdx:%i"%(self.fiMM))

            ##### [begin] storing clicked behavior tag -----
            ### if user clicked one of the annotation bars,
            ###   show its behavior tag.
            self.clickedAnnoBar = []
            yA = self.yABar # y-coordinates for annotation bars
            flagY = False
            for ai, y1 in enumerate(yA):
                y2 = y1 + self.annoH
                if y1 <= mp[1] <= y2:
                    if ai in self.perfLst:
                    # this index (ai) could be one of the performers
                        flagY = True
                        break
            # mouse pointer-y is not on one of the annotation bars, return
            if not flagY:
                self.panel["bt"].Refresh()
                return
            pdi = self.currPDI
            mMSIdx = self.mMStartIdx # video-bar starting index
            aRatP2F = self.aRatP2F # ratio of width of annotation to a frame
            bKeyLst = []
            for beh in self.behavior:
                for _ai in range(self.aecParam["uNSubj"]):
                    bKeyLst.append("%i_%i_%s"%(ai, _ai, beh))
            for bKey in bKeyLst:
                for annoI in range(len(self.oData[pdi][bKey])):
                    x1, x2 = self.oData[pdi][bKey][annoI]
                    x1 = vbr[0] + int((x1-mMSIdx)*aRatP2F)
                    x2 = vbr[0] + int((x2-mMSIdx)*aRatP2F)
                    if x2 < vbr[0] or x1 > vbr[0]+vbr[2]: continue
                    if x1 <= mp[0] <= x2:
                        # store the behavior of clicked annotation bar
                        self.clickedAnnoBar = [bKey, annoI]
            ##### [end] storing clicked behavior tag -----
            self.panel["bt"].Refresh()

        elif pk == "ml": # mouse clicked in "ml" panel
            pass

        self.panel[pk].mousePressedPt = [-1, -1] # init mouse pressed point

    #---------------------------------------------------------------------------

    def onMMove(self, event):
        """ Mouse pointer moving

        Args:
            event (wx.Event)

        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position

        if not hasattr(self, "lastMMoveTime"):
            self.lastMMoveTime = time()
        # no more than 25 FPS update
        if time() - self.lastMMoveTime < 0.04: return

        if pk == "bt": # mouse moved in "bt" panel
            vbr = self.vBarRect # rect of bar, representing entire video
            self.fiVB = -1
            self.fiMM = -1
            if vbr[0] <= mp[0] <= vbr[0]+vbr[2]:

                if vbr[1] <= mp[1] <= vbr[1]+vbr[3]:
                # mouse pointer is in video bar rect
                    mpRat = (mp[0]-vbr[0]) / vbr[2]
                    self.fiVB = int((self.endDataIdx+1) * mpRat)
                    self.panel[pk].Refresh()
                elif mp[1] > vbr[1]+vbr[3]:
                # mouse pointer is in MOI & Movement area
                    self.fiMM = int((mp[0]-vbr[0])/self.aRatP2F) + \
                                  self.mMStartIdx
                    self.panel[pk].Refresh()

        elif pk == "ml": # mouse moved in "ml" panel
            if self.panel[pk].mousePressedPt != [-1, -1]:
                offset = list(self.prevOffset4disp)
                pPt = self.panel[pk].mousePressedPt
                offset[0] += (mp[0] - pPt[0])
                offset[1] += (mp[1] - pPt[1])
                self.offset4disp = offset
                self.panel[pk].Refresh()

        self.lastMMoveTime = time()

    #---------------------------------------------------------------------------

    def onMRBUp(self, event):
        """ Right mouse button was released

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoFP == "": return

        if sys.platform.startswith("win"):
            wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))
        else:
            if self.audio != None:
                self.audio.playSnd(1, 0, True)
        self.panel["ml"].Refresh() # re-draw

    #---------------------------------------------------------------------------

    def onMWheel(self, event):
        """ Mouse wheel rotating

        Args:
            event (wx.Event)

        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        if self.flags["blockUI"] or self.videoFP == "": return

        if event != None:
            ### collect mouse wheel event data
            self.mWheel["val"].append(event.GetWheelRotation())
            self.mWheel["lastEvtTime"] = time()
            pk = event.GetEventObject().panelKey # panel key
            self.mWheel["pk"] = event.GetEventObject().panelKey
            self.mWheel["mState"] = wx.GetMouseState()
            event.Skip()
            return
        else:
        # event==None; called by onTimer function
            if self.mWheel["lastEvtTime"] != -1 and \
              time()-self.mWheel["lastEvtTime"] >= self.mWheel["thrSec"]:
            # single mouse wheel stroke is finished
                ### process collected mouse wheel rotation data
                if np.median(self.mWheel["val"]) > 0:
                    mWhRot = max(self.mWheel["val"])
                else:
                    mWhRot = min(self.mWheel["val"])
                ### init
                self.mWheel["val"] = []
                self.mWheel["lastEvtTime"] = -1
            else:
                return

        if self.mWheel["mState"].ControlDown(): # Ctrl + mouse-wheel event
            if self.mWheel["pk"] == "ml": # event in "ml" panel
                ### change frame-image zoom rate
                ratChange = 0.5 * (mWhRot/abs(mWhRot))
                ratChange *= self.wheelSensAdj
                self.fImgZoomRate += ratChange
                self.panel["ml"].Refresh()

            elif self.mWheel["pk"] == "bt": # event in "bt" panel
                choices = self.aRatP2FChoices
                if mWhRot > 0:
                    self.adjChoiceItem("aRatP2F_cho", choices, "increase")
                else:
                    self.adjChoiceItem("aRatP2F_cho", choices, "decrease")

        else: # mouse-wheel event without modifier key
            if self.mWheel["pk"] != "mr":
            # wheel event other than 'mr' panel
                ### move frames according to wheel rotation direction and speed
                idx = self.fi
                if self.wheelSensAdj == self.wheelSensChoices[0]:
                    mWh = int(mWhRot / abs(mWhRot)) # +1 or -1
                else:
                    mWh = round(mWhRot * self.wheelSensAdj)
                if mWh > 0: idx = min(self.endDataIdx, idx+mWh)
                elif mWh < 0: idx = max(0, idx+mWh)
                else: return
                self.moveFrame(None, "fIdx:%i"%(idx))

    #---------------------------------------------------------------------------

    def moveFrame(self, event, flag):
        """ Move to certain frame in the graph, depending on 'flag'.

        Args:
            event (wx.Event)
            flag (str): Indicator for moving direction and distance.

        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        ### set frame index increment value
        if self.playSpd.startswith("fast"):
            fastF = self.playSpd.count("+") + 1
            inc = self.playSpdAdj["fast"] * fastF
        else:
            inc = 1

        if flag == "backBegin":
            idx = 0
        elif flag == "forEnd":
            idx = self.endDataIdx
        elif flag == "forward":
            idx = min(self.endDataIdx, self.fi+inc)
        elif flag == "backward":
            idx = max(0, self.fi-inc)
        elif flag.startswith("fIdx:"):
            idx = str2num(flag.replace("fIdx:", ""))

        if idx != None:
            self.fi = idx # set frame-index
            # update MM bar start index
            self.mMStartIdx = max(0,
                                  self.fi-int(self.vBarRect[2]/self.aRatP2F/2))
            self.panel["ml"].Refresh() # re-draw

    #---------------------------------------------------------------------------

    def adjChoiceItem(self, objName, choices, flag, pk="tp"):
        """ Increase/decrease item index of wx.Choice widget.

        Args:
            objName (str): Name of the widget.
            choices (list): List of choices.
            flag (str): 'increase', 'decrease' or 'change'
            pk (str): Panel key.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        w = wx.FindWindowByName(objName, self.panel[pk])
        currS = w.GetSelection()
        if flag == "increase":
            if currS == 0: return
            w.SetSelection(currS-1)
        elif flag == "decrease":
            if currS == len(choices)-1: return
            w.SetSelection(currS+1)
        elif flag == "change":
            s = currS + 1
            if s >= len(choices): s = 0
            w.SetSelection(s)
        self.onChoice(None, objName=objName)

    #---------------------------------------------------------------------------

    def gBtnSolidCol(self, gBtn, col):
        """ Change the color of GradientButton to a solid color.

        Args:
            gBtn (wx.lib.agw.gradientbutton.GradientButton)
            col (tuple): Color of the gBtn to be changed.

        Returns:
            None
        """
        #if FLAGS["debug"]: logging.info(str(locals()))

        if gBtn == None: return
        col = wx.Colour(col)
        gBtn.SetTopStartColour(col)
        gBtn.SetTopEndColour(col)
        gBtn.SetBottomStartColour(col)
        gBtn.SetBottomEndColour(col)

    #---------------------------------------------------------------------------

    def save(self, event, ts=""):
        """ Save revised CSV result as another CSV file

        Args:
            event (wx.Event)
            ts (str): Timestamp when saving.
                      If this is given, it means saving backup annotation file.

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        if self.videoFP == "": return

        vExt = "." + self.videoFP.split(".")[-1]
        fp = self.videoFP.replace(vExt, "_anno.csv") # set output CSV file path
        ### set output CSV file path
        if ts != "":
        # saving a backup annotation result file
            ts = ts.replace("_", "")
            backupFNstr1 = self.videoFP.replace(vExt, "_anno_backup")
            fp = "%s%s.csv"%(backupFNstr1, ts) # add timestamp and extension
            # backup file list, sorted by backup file stamp in ascending order
            bfLst = sorted(glob(backupFNstr1 + "*"))
            nBackupR = 3 # number of backup result files to store
            if len(bfLst) >= nBackupR: # there are too many backup files
                ### delete the old backup files
                n2remove = len(bfLst)+1 - nBackupR
                for i in range(n2remove): remove(bfLst[i])
        else:
            fp = self.videoFP.replace(vExt, "_anno.csv")
        fh = open(fp, 'w') # open new file to write

        ### write parameters
        line = "Timestamp, %s\n"%(get_time_stamp())
        fh.write(line) # write saving timestamp
        line = "Video file, %s\n"%(path.basename(self.videoFP))
        fh.write(line) # write video file name
        if self.inputVideoType == "frames":
            line = "FPS, %.2f\n"%(self.videoFPS)
            fh.write(line) # write video FPS (used for calculating time)
        fh.write("-----\n")

        ### write column title
        columnTitle = ["Petri-dish-index", "Behavior-performer", "Receiver"
                       , "Behavior", "Beginning-frame", "End-frame"
                       , "Dur. (frames)", "Beginning-time", "End-time"
                       , "Dur. (seconds)"]
        line = ""
        for c in columnTitle: line += "%s, "%(c)
        line = line.rstrip(", ") + "\n"
        fh.write(line)
        ### write annotation data
        for pdi in range(self.aecParam["uNPDish"]):
            for bpk in sorted(self.behPairKey):
                for beh in self.behavior:
                    bKey = "%s_%s"%(bpk, beh)
                    if self.aOp[bKey]["state"] == "anno":
                    # it there's ongoing annotation
                        btnName = "%s_gBtn"%(bKey)
                        # finalize it
                        self.onButtonPressDown(None, btnName)
                    # petri-dish-index, performer, receiver and behavior
                    _bpk = bpk.split("_")
                    _txt = "%i, %s, %s, %s"%(pdi, _bpk[0], _bpk[1], beh)
                    rngLst = sorted(self.oData[pdi][bKey])
                    for ri, (fi1, fi2) in enumerate(rngLst):
                        if ri > 0:
                            ### fix (or ignore) annotation data range
                            ###   if its overlaps with the previous data
                            pfi1, pfi2 = rngLst[ri-1] # frame indices of
                              # the previous data range
                            if fi1 <= pfi2:
                                if fi2 <= pfi2: continue
                                else: fi1 = pfi2 + 1
                        sec1 = fi1 / self.videoFPS
                        sec2 = fi2 / self.videoFPS
                        ts1 = timedelta(seconds=sec1)
                        ts2 = timedelta(seconds=sec2)
                        # beginning, end frame
                        line = _txt + ", %i, %i"%(fi1, fi2)
                        # duration (frames)
                        line += ", %i"%(fi2-fi1)
                        # beginning, end time
                        line += ", %s, %s"%(str(ts1), str(ts2))
                        # duration (time)
                        line += ", %.3f\n"%(sec2-sec1)
                        fh.write(line) # write data
        fh.write("-----\n")
        fh.close()

        if ts == "":
        # this was not backup file saving
            msg = 'Saved.\n'
            msg += fp
            wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)

    #---------------------------------------------------------------------------

    def configuration(self, flag):
        """ saving/loading configuration of the app

        Args:
            flag (str): save or load

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        configFP = path.join(FPATH, "config")

        objNames = ["animalECase_cho", "wheelSensitivity_cho",
                    "fps_spin"]
        if flag == "save":
            config = {}
            for on in objNames:
                obj = wx.FindWindowByName(on, self.panel["tp"])
                if on.endswith("_cho"):
                    config[on] = obj.GetString(obj.GetSelection())
                elif on.endswith("_spin"):
                    config[on] = obj.GetValue()
            for viKey in self.menuItem["view"].keys():
                if len(viKey.split("-")) == 1: continue # ignore behavior-set
                if self.menuItem["view"][viKey].IsChecked():
                    config["view_%s"%(viKey)] =  True
                else:
                    config["view_%s"%(viKey)] = False
            fh = open(configFP, "wb")
            pickle.dump(config, fh)
            fh.close()
            return

        elif flag == "load":
            ### set default values of config
            dv = dict(animalECase_cho = self.animalECaseChoices[0],
                      wheelSensitivity_cho = 0.05,
                      fps_spin = 30)
            for beh in self.defBehavior:
                dv["view_%s"%(beh)] = True

            if path.isfile(configFP):
            # config file exists
                fh = open(configFP, "rb")
                config = pickle.load(fh)
                fh.close()
                for on in objNames:
                    if not on in config.keys():
                        config[on] = dv[on] # get default value
                for beh in self.defBehavior:
                    k = "view_%s"%(beh)
                    if not k in config.keys():
                        config[k] = dv[k]
            else:
            # no config file found
                config = dv # use the default values
            return config

    #---------------------------------------------------------------------------

    def behSettings(self, flag, v={}, dlgSz=(-1,-1)):
        """ read/write behavior settings in annoV.txt to/from class variables
        * Additionally, it returns widget info list for settings pop-up.

        Args:
            flag (str): read, write or getWidgetInfo
            v (dict): values to store (only for 'write' flag)
            dlgSz (tuple): dialog window size (only for 'getWidgetInfo' flag)

        Returns:
            ret (None/list)
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        ret = None

        fp = path.join(FPATH, "annoV.txt")
        if not path.isfile(fp):
            msg = "%s is not found."%(fp)
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return

        if flag == "read":
        # read settings from annoV.txt
            fh = open(fp, "r")
            lines = fh.readlines()
            self.defV = {}
            self.behSets = [] # behavior-set names
            self.monadicB = []
            self.dyadicB = []
            self.behavior = []
            self.annoCol = {}
            flag = ""
            for line in lines:
                if flag == "":
                    if "###BEGIN" in line:
                        flag = line.split(":")[1].strip()
                    continue
                else:
                    if "###END" in line:
                        flag = ""
                        continue
                line = line.strip()
                line = line.split("#")[0].strip() # cut off comments
                if flag == "defaultV":
                    vn, val = line.split(":")
                    vn = vn.strip()
                    try:
                        self.defV[vn] = int(val)
                    except Exception as e:
                        msg = "Failed to convert defaultV value to an integer."
                        procExceptionWX(self, e, msg, "failedInit")
                        return
                elif flag == "behavior-sets":
                    self.behSets.append(line)
                elif flag == "monadic-behavior":
                    self.monadicB.append(line)
                elif flag == "dyadic-behavior":
                    self.dyadicB.append(line)
                elif flag == "behavior-list":
                    s = line.split("-")
                    if s[0] in self.behSets:
                        self.behavior.append(line)
                elif flag == "annotation-color":
                    beh, colVal = line.split(":")
                    beh = beh.strip()
                    colVal = [int(x) for x in colVal.split(",")]
                    self.annoCol[beh] = tuple(colVal)

        elif flag == "write":
        # write settings into annoV.txt (also store it in class variables)
            ##### [begin] process values & check validity -----
            errMsg = ""
            if "behavior" in v.keys():
            # general settings (not the color setting)
                ### store # of petri-dishes and subjects (in each pDish)
                for k in ["uNPDish", "uNSubj"]:
                    try: v[k] = int(v[k])
                    except:
                        errMsg += "Failed to convert"
                        errMsg += " %s to an integer.\n"%(k)
                ### store behavior-set list
                _lst = []
                for bs in v["behSets"].split("\n"):
                    bs = bs.replace(" ","")
                    if bs != "": _lst.append(bs)
                # 'all' set should be in the list
                if not "all" in _lst: _lst.append("all")
                v["behSets"] = copy(_lst)
                ### store list of monadic and dyadic behavior-sets
                for k in ["monadicB", "dyadicB"]:
                    _lst = []
                    for b in v[k].split("\n"):
                        b = b.replace(" ","")
                        if b == "": continue
                        if b in v["behSets"]:
                            _lst.append(b)
                        else:
                            errMsg += "%s is NOT "%(b)
                            errMsg += "in the behavior set list.\n"
                    v[k] = copy(_lst)
                if len(v["monadicB"])+len(v["dyadicB"]) != len(v["behSets"]):
                # all behavior-sets should be either monadic or dyadic
                    errMsg += "Sum of monadic and dyadic behaviors doesn't"
                    errMsg += " match with the number of behavior sets.\n"
                ### store list of all behaviors for annotation
                _lst = []
                for b in v["behavior"].split("\n"):
                    b = b.replace(" ","")
                    if b == "": continue
                    bS = b.split("-")[0]
                    if not bS in v["behSets"]:
                        errMsg += "%s doesn't belong to any behavior set.\n"%(b)
                    _lst.append(b)
                v["behavior"] = _lst

                if errMsg != "":
                # if failed in validity check
                    errMsg += "\nError(s) occurred.\n"
                    errMsg += "Updating the setting is aborted.\n\n"
                    wx.MessageBox(errMsg, "ERROR", wx.OK|wx.ICON_ERROR)
                    return

                ### update new settings to class variables
                self.defV = dict(uNPDish = v["uNPDish"],
                                 uNSubj = v["uNSubj"])
                self.behSets = v["behSets"]
                if self.behSets != v["behSets"]:
                    self.setUpMenuBar(None) # reset the View menu-bar
                self.monadicB = v["monadicB"]
                self.dyadicB = v["dyadicB"]
                self.defBehavior = v["behavior"]
                self.behavior = v["behavior"]
                # update self.behavior according to the currently checked
                # view items
                self.onViewMenuItemChanged(None)

                ### add the new behavior into 'annoCol' dict.
                oldCols = copy(self.annoCol)
                self.annoCol = {}
                for k in self.defBehavior:
                    if k in oldCols.keys(): # old behavior
                        self.annoCol[k] = oldCols[k]
                    else: # new behavior
                        self.annoCol[k] = (255,255,255) # default color

            else:
            # color setting
                ### store new color settings to class variables
                for b in self.defBehavior:
                    self.annoCol[b] = v[b]
            ##### [end] process values & check validity -----

            ### write settings to the annoV.txt file
            fh = open(fp, "w")
            fh.write("###BEGIN:defaultV\n")
            for k in self.defV.keys():
                fh.write("%s: %i\n"%(k, self.defV[k]))
            fh.write("###END:defaultV\n\n")
            fh.write("###BEGIN:behavior-sets\n")
            for item in self.behSets: fh.write(item+"\n")
            fh.write("###END:behavior-sets\n\n")
            fh.write("###BEGIN:monadic-behavior\n")
            for item in self.monadicB: fh.write(item+"\n")
            fh.write("###END:monadic-behavior\n\n")
            fh.write("###BEGIN:dyadic-behavior\n")
            for item in self.dyadicB: fh.write(item+"\n")
            fh.write("###END:dyadic-behavior\n\n")
            fh.write("###BEGIN:behavior-list\n")
            for item in self.defBehavior: fh.write(item+"\n")
            fh.write("###END:behavior-list\n\n")
            fh.write("###BEGIN:annotation-color\n")
            for k in self.annoCol.keys():
                color = self.annoCol[k]
                fh.write("%s: %i,%i,%i\n"%(k, color[0], color[1], color[2]))
            fh.write("###END:annotation-color\n\n")
            fh.close()

        elif flag == "getWidgetInfo":
        # for returning list of widget info, used in settings popup window
            bw = 5
            defColor = "#ffffff"
            ww = int(dlgSz[0]*0.9) # word-wrap width
            w = []
            wNames = [] # widget names which have values of settings
            w.append([{"type":"sTxt", "label":" ", "nCol":2, "border":bw}])
            for k in self.defV.keys():
                wNames.append(k+"_cho")
                _cho = [str(x+1) for x in range(6)] # !!! max number of choies
                                        # is set to a specific number here.
                                        # (2023-02-18) Change?
                w.append([
                        {"type":"sTxt", "label":k, "nCol":1,
                         "fgColor":defColor},
                        {"type":"cho", "nCol":1, "name":k, "choices":_cho,
                         "val":str(self.defV[k]), "border":bw},
                        ])
            w.append([{"type":"sTxt", "label":" ", "nCol":2, "border":bw}])
            sLst = [
                    ("behSets", "Behavior sets", self.behSets),
                    ("monadicB", "Monadic behavior", self.monadicB),
                    ("dyadicB", "Dyadic behavior", self.dyadicB),
                    ("behavior", "Behavior list", self.defBehavior)
                    ]
            for k, lbl, lst in sLst:
                _lbl = " (* one line = one behavior item)"
                w.append([
                        {"type":"sTxt", "label":lbl+_lbl, "nCol":2,
                         "fgColor":defColor, "border":bw, "wrapWidth":ww}
                        ])
                if k in ["behSets", "behavior"]:
                    if k == "behSets":
                        _lbl = "* If you change behavior-set list, all"
                        _lbl += " behavior-set items in View menu will be"
                        _lbl += " turned on. Check off items that you"
                        _lbl += " will not annotate in View menu."
                        _lbl += "  * Make sure that the newly added behavior"
                        _lbl += " set should also be either in monadic or"
                        _lbl += " dyadic behavior."
                    elif k == "behavior":
                        _lbl = "* Do NOT use space ' ', hypen '-' or "
                        _lbl += "asterik '*'. "
                        _lbl += "* If you add a new behavior, please change its"
                        _lbl += " annotation color in Color-settings (F11)."
                    w.append([
                        {"type":"sTxt", "label":_lbl, "nCol":2, "border":bw,
                         "fgColor":defColor, "wrapWidth":ww}
                        ])
                _val = ""
                for item in lst: _val += item + "\n"
                w.append([
                        {"type":"txt", "nCol":2, "val":_val, "name":k,
                         "size":(int(dlgSz[0]*0.5),150),
                         "style":wx.TE_MULTILINE}
                        ])
                w.append([{"type":"sTxt", "label":" ", "nCol":2, "border":bw}])
                wNames.append("%s_txt"%(k))
            ret = [w, wNames]

        elif flag == "getWidgetInfo_color":
        # for returning list of widget info, used in settings popup window
            bw = 5
            defColor = "#ffffff"
            w = []
            wNames = [] # widget names which have values of settings
            w.append([
                        {"type":"sTxt", "label":"Annotation color",
                         "nCol":2, "fgColor":defColor, "border":bw}
                        ])
            for k in self.annoCol.keys():
                cC = self.annoCol[k] # current color
                w.append([
                        {"type":"sTxt", "label":k, "nCol":1,
                         "fgColor":defColor},
                        {"type":"cPk", "nCol":1, "name":k,
                         "color":wx.Colour(cC[0],cC[1],cC[2])}
                        ])
                wNames.append("%s_cPk"%(k))
            ret = [w, wNames]

        return ret

    #---------------------------------------------------------------------------

    def sortBehLst(self):
        """ sort the current behavior list (self.behavior)
        in the order of what user entered in the setting

        Args:
            None

        Returns:
            None
        """
        if FLAGS["debug"]: logging.info(str(locals()))

        behLst = []
        for bs in self.behSets:
            for beh in self.defBehavior:
                if beh.split("-")[0] == bs and beh in self.behavior:
                    behLst.append(beh)
        self.behavior = behLst

    #---------------------------------------------------------------------------

#===============================================================================

class AnnotatorApp(wx.App):
    def OnInit(self):
        if FLAGS["debug"]: logging.info(str(locals()))
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        self.frame = AnnotatorFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

##### [begin] help string -----
HELPSTR = """
#-------------------------------------------------------------------------------
# Navigation using mouse
#-------------------------------------------------------------------------------
- User can click any point in the video bar, grey (darker) color bar at the top, (frame index will be displayed when user hovers mouse pointer on it) to jump to that frame where mouse point is on.
This 1st grey bar represents the entire video.

- User can also click any point in the 2nd grey (lighter) color bar on the 2nd position at the top, to jump to that frame.
In this 2nd grey bar, a single pixel represents a single frame, at the least, while one pixel might represents several or more frames in the 1st grey bar.

- Mouse-wheel can be rotated to navigate forward/backward.

- Mouse-wheel sensitivity can be adjusted with the drop-down box in the top toolbar to set the proper speed of navigation.

- Play speed can be adjusted with the drop-down box in the top bar.


#-------------------------------------------------------------------------------
# Annotation using mouse
#-------------------------------------------------------------------------------
Annotation can be done in the right-side panel.

- User should choose indices of 'Performer' and 'Receiver' for the behavior annotation.
If the behavior is monadic, the performer and reciever index should be same,
and it should be different if the behavior is dyadic.

- If one of the annotation button is clicked, the button color changes & annotation bar will start to occur as user moves frames.
When user clicks the same button again (or clicks different annotation button), the button label color changes back to its normal state & the annotation data will be stored as output data.

- User can also use deletiong [Del.] button in the same way with annotating button to delete already stored annotated data.


#-------------------------------------------------------------------------------
# For facilitating annotation
#-------------------------------------------------------------------------------
- User can move the frame image left/right/up/down by click-and-dragging the frame image while 'Ctrl' key is pressed.

- User can zoom in/out the frame image with mouse-wheel while 'Ctrl' key is pressed.

- User can hide/show markers on the frame image by pressing 'h' key.

- Graph bars on top represents followings
-- The first bar section represents video timeline for navigation.
-- The second bar section represents moments of interest.
-- The third bar section represents annotated data.


#-------------------------------------------------------------------------------
# Saving annotated data
#-------------------------------------------------------------------------------
- Ctrl+S or pressing save button on top toolbar panel will save the annotated data to a CSV file.


#-------------------------------------------------------------------------------
# Keyboard shortcuts
#-------------------------------------------------------------------------------
(With Ctrl key)
- Ctrl + O: open file.
- Ctrl + C: close file.
- Ctrl + S: save annotation data.
- Ctrl + Q: quit program.

- Ctrl + I: increase annotation bar pixel-to-frame ratio.
- Ctrl + K: decrease annotation bar pixel-to-frame ratio.
- Ctrl + J: go backward to the closest annotation begin/end point.
- Ctrl + L: go forward to the closest annotation begin/end point.

(With Shift key)
- Shift + J: go back to the beginning of the video.
- Shift + L: go forward to the end of the video.

(With Alt key)
- Alt + I: increase wheel sensitivity.
- Alt + K: decrease wheel sensitivity.
- Alt + J: go backward based on the wheel sensitivity value.
- Alt + L: go forward based on the wheel sensitivity value.

(Without any modifier key)
- I: increase play-speed.
- K: decrease play-speed.
- J: go backward for one frame.
- L: go forward for one frame.

- SPACEBAR: play/pause.
- P: change performer index.
- R: change receiver index.
- C: change petri-dish index.
- H: hide/unhide markings on the frame image.
- BACKSPACE: Clear displayed annotation segment info., when one is displayed.

- F10: Settings.
- F11: Annotation color settings.


* Annotation button activation using keyboard input -----

- {Any number}: entering into the mode for typing annotation button number.
This number will be displayed on the status-bar at the bottom of the window.

(When the mode of annotation button number typing is on)
- BACKSPACE: delete one digit of the typed annotation button number.
- ESC: escape the mode of annotation button number entering.
- ENTER: annotation button with the typed number is activated/deactivated.
- D: deletion [Del.] button of the annotation button with the typed number is activated/deactivated.

(when any annotation or deletion button is activated)
- ESC: deactivate the currently active annotation/ deletion button.
"""
##### [end] help string -----


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-w':
            GNU_notice(1)
            sys.exit()
        elif sys.argv[1] == '-c':
            GNU_notice(2)
            sys.exit()

    if len(sys.argv) > 1:
        if sys.argv[1] == "-opencv":
            FLAGS["decord"] = False
        elif sys.argv[1] == "-decord":
            FLAGS["decord"] = True

    GNU_notice(0)
    CWD = getcwd()
    app = AnnotatorApp(redirect=False)
    app.MainLoop()
