# coding: UTF-8

"""
Open-source software written in Python 
  to explore experimental data and produce (interactive) graphs.

This program was coded and tested in Ubuntu 18.04.

Jinook Oh, Cremer group, Institute of Science and Technology Austria
Last edited: 2024-06-02

Dependency:
    Python (3.9)
    wxPython (4.0)
    OpenCV (3.4)
    NumPy (1.18)

------------------------------------------------------------------------
Copyright (C) 2022 Jinook Oh & Sylvia Cremer
in Institute of Science and Technology Austria.
- Contact: jinook.oh@ist.ac.at/ jinook0707@gmail.com

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
v.0.1.1: Initial development (for viruses in ants of Lumi Viljakainen).
v.0.1.20200818: Refactoring + heatmap graph for Linda Sartoris's work
v.0.1.20200925: Updating heatmap for Linda to save raw data, including
    heatmap matrix & some ant blob information in CSV format.
    The ant blob information is also displayed in debug mode.
v.0.2.202011:
    J2020fq was added in 2020.09 to show some preliminary graph for showing
    positions of four founding queens.
    V2020 for Lumi's paper about viruses in three ant species was restored.
    Another option (L2020CSV) for reading and showing heatmap array,
    generated from L2020.
    'aos' was added deal with data from AntOS.
v.0.2.202102:
    Changed name from drawGraph to DataVisualizer.
    Changed L2020CSV to L2020CSV1, which is for heatmap range adjustment
      and drawing additional info.
    Added L2020CSV2 for drawing different graphs using information CSV (rects,
      centroid, distances, etc..), resulted from L2020
v.0.2.202108:
    Minor adjustments for heatmap graph in 'aos'.
v.0.3.202204:
    Added motion spatial analysis in AntOS.
v.0.3.202212:
    Added 'proxMCluster' analysis in AntOS.
v.0.4.202303:
    Added 'aosSec'; the secondary processing with numpy arrays produced by 'aos'
v.0.5.202402:
    Removed J2020fq.
    Added 'anVid'; drawing graphs with CSV data from AnVid.
v.0.6.202405:
    Added 'cremerGroupApp'; drawing visualization of overall functionalities 
    of all files in CremerGroupApp.
"""

import sys, queue, traceback
from os import path, mkdir
from shutil import copyfile
from glob import glob
from copy import copy

import cv2, wx, wx.adv, wx.stc
import wx.lib.scrolledpanel as SPanel 
import numpy as np

sys.path.append("..")
import initVars
initVars.init(__file__)
from initVars import *

from modFFC import *
from modCV import * 
from procGraph import ProcGraphData 

FLAGS = dict(debug = False)
__version__ = "0.6.202405"
ICON_FP = path.join(P_DIR, "image", "icon.png")

#===============================================================================

class DataVisualizerFrame(wx.Frame):
    """ Frame for drawing a graph
    for saving it as a image file or interactive graph on screen.

    Args:
        None
     
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
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
              "DataVisualizer v.%s"%(__version__), 
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX)
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
        self.flags = dict(blockUI=False, # block user input 
                          drawingROI={},  # for drawing region of interest on
                                          # certain area, such as nest
                          selectPt={}) # for selecting certain point in frame
                                       # image, such as center of brood
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for status bar message display
        self.mlWid = [] # wx widgets in middle left panel
        self.mrWid = [] # wx widgets in middle right panel
        self.inputFP = ""
        self.colTitles = [] # column titles of CSV data
        self.numData = None # numpy array, containing numeric data
        self.strData = None # numpy array, containing string data
        self.frameImgFP = [] # list of file paths of 
          # frame images (graph which will become a frame of a video file)
        self.pgd = None # instance of ProcGraphData class
        self.debugging = dict(state=False)
        self.eCaseChoices = [
                "anVid: data from AnVid (analysis of recorded video)",
                "aos: data from AntOS [Jinook]",
                "aosSec: secondary data processing with data from 'aos'",
                #"aosI: intensities from multiple data of aos [Jinook]",
                "L2020: structured nest [Sartoris et al]",
                #"L2020CSV1: structured nest (add. info) [Sartoris et al]",
                #"L2020CSV2: structured nest (add. info) [Sartoris et al]",
                "V2020: viruses in ant species [Viljakainen et al]",
                "cremerGroupApp: visualization of cremerGroupApp functionality",
                ]
        self.inputType = dict(
                                anVid=("file", "csv"),
                                aos=("dir", ""),
                                aosSec=("dir", ""),
                                #aosI=("dir", ""),
                                L2020=("file", "mp4"),
                                V2020=("file", "csv"),
                                )
        self.eCase = self.eCaseChoices[0].split(":")[0].strip()
        self.eCase_noRawData = ["V2020", "L2020", "cremerGroupApp"]
        self.colorPick = "" # temporary string for color picking
        self.percLst = [60, 70, 80, 90, 95, 99, 100] # list of percentile 
        btnImgDir = path.join(P_DIR, "image")
        self.btnImgDir = btnImgDir
        self.targetBtnImgPath = path.join(self.btnImgDir, "target.png")
        self.targetBtnActImgPath = path.join(self.btnImgDir, "target_a.png") 
        ##### [end] setting up attributes -----
       
        btnSz = (35, 35)
        vlSz = (-1, 20) # size of vertical line separator
        ### create panels and its widgets
        for pk in pi.keys(): 
            w = [] # each itme represents a row in the left panel
            if pk == "tp":
                w.append([
                    {"type":"sTxt", "label":"experiment-case:", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"cho", "nCol":1, "name":"eCase",
                     "choices":self.eCaseChoices, "size":(250,-1),
                     "val":self.eCaseChoices[0]},
                    {"type":"btn", "nCol":1, "name":"help", "size":btnSz,
                     "img":path.join(btnImgDir, "help.png"),
                     "bgColor":"#333333"},
                    {"type":"btn", "nCol":1, "name":"open", "size":btnSz,
                     "img":path.join(btnImgDir, "open.png"),
                     "bgColor":"#333333"},
                    {"type":"txt", "nCol":1, "name":"inputFP", 
                     "val":"[Opened input file]", "style":wx.TE_READONLY,
                     "size":(300,-1), "fgColor":"#cccccc"},
                    {"type":"sLn", "nCol":1, "size":vlSz, 
                     "style":wx.LI_VERTICAL},
                    {"type":"chk", "nCol":1, "name":"debug", "label":"debug", 
                     "val":self.debugging["state"], "style":wx.CHK_2STATE, 
                     "fgColor":"#cccccc", "border":50,
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)},
                    {"type":"sTxt", "label":"Zoom:", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"txt", "nCol":1, "name":"zoom", "val":"1.000", 
                     "style":wx.TE_PROCESS_ENTER, "procEnter":True, 
                     "border":20, "size":(100,-1), 
                     "flag":(wx.ALIGN_CENTER_VERTICAL|wx.RIGHT)},
                    {"type":"sTxt", "label":"Offset:", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"txt", "nCol":1, "name":"offset", "val":"0, 0", 
                     "style":wx.TE_PROCESS_ENTER, "procEnter":True, 
                     "size":(100,-1)}, 
                    ])
            elif pk == "bm":
                w.append([
                    {"type":"sTxt", "label":"Saving resolution :", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"txt", "nCol":1, "name":"imgSavResW", 
                     "val":str(int(pi["mp"]["sz"][1]*1.3333)),
                     "size":(100,-1), "numOnly":True},
                    {"type":"txt", "nCol":1, "name":"imgSavResH", 
                     "val":str(pi["mp"]["sz"][1]),
                     "size":(100,-1), "numOnly":True},
                    {"type":"btn", "nCol":1, "name":"save", "size":btnSz,
                     "img":path.join(btnImgDir, "save.png"), 
                     "tooltip":"Save graph", "bgColor":"#333333"},
                    {"type":"sLn", "nCol":1, "size":vlSz, 
                     "style":wx.LI_VERTICAL},
                    {"type":"btn", "nCol":1, "name":"saveRawData", "size":btnSz,
                     "img":path.join(btnImgDir, "saveR.png"),
                     "tooltip":"Save raw data as CSV", "bgColor":"#333333"}
                    ])
            elif pk == "br":
                w.append([
                    {"type":"btn", "nCol":1, "name":"saveAll", "size":btnSz,
                     "img":path.join(btnImgDir, "save.png"),
                     "tooltip":"Save all graphs", "bgColor":"#333333"},  
                    {"type":"btn", "nCol":1, "name":"saveAllRawData", 
                     "size":btnSz, "bgColor":"#333333",
                     "img":path.join(btnImgDir, "saveR.png"),
                     "tooltip":"Save all raw data as CSV"},
                    {"type":"btn", "nCol":1, "name":"deleteGImg", 
                     "size":btnSz, "bgColor":"#333333",
                     "img":path.join(btnImgDir, "delete.png"),
                     "tooltip":"Delete selected graph image"},
                    {"type":"btn", "nCol":1, "name":"deleteAllGImg", 
                     "size":btnSz, "bgColor":"#333333",
                     "img":path.join(btnImgDir, "deleteAll.png"),
                     "tooltip":"Delete all graph images"}
                    ])
            # setup this panel & widgets
            setupPanel(w, self, pk)
        
        ### Bind events to the middle panel (graph panel)
        self.panel["mp"].Bind(wx.EVT_PAINT, self.onPaintMP)
        self.panel["mp"].Bind(wx.EVT_LEFT_DOWN, self.onMLBDown)
        self.panel["mp"].Bind(wx.EVT_LEFT_UP, self.onMLBUp)
        self.panel["mp"].Bind(wx.EVT_RIGHT_UP, self.onMRBUp)
        self.panel["mp"].Bind(wx.EVT_MOTION, self.onMMove)
        self.panel["mp"].Bind(wx.EVT_MOUSEWHEEL, self.onMWheel)

        self.zoom_txt = wx.FindWindowByName("zoom_txt", self.panel["tp"])
        self.offset_txt = wx.FindWindowByName("offset_txt", self.panel["tp"])

        self.config("load") # load configs

        # set up the menu bar
        self.setUpMenuBar()
     
    #---------------------------------------------------------------------------

    def setUpMenuBar(self):
        """ set up the menu bar

        Args: None

        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        menuBar = wx.MenuBar()
        
        ### Visualizer menu
        menu = wx.Menu()
        openItem = menu.Append(wx.Window.NewControlId(), item="Open\tCTRL+O")
        self.Bind(wx.EVT_MENU, self.openInputF, openItem)
        quitItem = menu.Append(wx.Window.NewControlId(), item="Quit\tCTRL+Q")
        self.Bind(wx.EVT_MENU, self.onClose, quitItem)
        menuBar.Append(menu, "&Visualizer")
        
        ### help menu
        menuH = wx.Menu()
        helpStrItem= menuH.Append(wx.Window.NewControlId(), 
                                  item="Help string\tF1")
        self.Bind(wx.EVT_MENU,
                  lambda event: self.onButtonPressDown(event, "help_btn"),
                  helpStrItem)
        keymapItem = menuH.Append(wx.Window.NewControlId(), 
                                  item="Key mapping\tF2")
        self.Bind(wx.EVT_MENU, self.displayKeyMapping, keymapItem)
        menuBar.Append(menuH, "&Help")

        self.SetMenuBar(menuBar)

    #---------------------------------------------------------------------------
    
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        wSz = self.wSz 
        pi = {} # information of panels
        if sys.platform.startswith("win"):
            #style = (wx.TAB_TRAVERSAL|wx.SIMPLE_BORDER)
            style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
            bgCol = "#333333"
        else:
            style = (wx.TAB_TRAVERSAL|wx.BORDER_NONE)
            bgCol = "#333333"

        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], 60), bgCol=bgCol, style=style)
        tpSz = pi["tp"]["sz"]
        # middle left panel
        pi["ml"] = dict(pos=(0, tpSz[1]), sz=(int(wSz[0]*0.2), wSz[1]-tpSz[1]),
                        bgCol=bgCol, style=style)
        mlSz = pi["ml"]["sz"] 
        savH = 50 # height for saving interface for graph image/video
        gph = wSz[1]-tpSz[1]-savH # graph height
        gpw = int(gph * (4/3)) # graph width
        # middle panel
        pi["mp"] = dict(pos=(mlSz[0], tpSz[1]), sz=(gpw, gph), 
                        bgCol="#111111", style=style)
        mpSz = pi["mp"]["sz"]
        # bottom panel (for showing graph image saving interface)
        pi["bm"] = dict(pos=(mlSz[0], wSz[1]-savH), sz=(mpSz[0], savH),
                        bgCol=bgCol, style=style)
        bmSz = pi["bm"]["sz"]
        # right panel (for showing frame images)
        pi["mr"] = dict(pos=(mlSz[0]+mpSz[0], tpSz[1]),
                        sz=(wSz[0]-mlSz[0]-mpSz[0], wSz[1]-tpSz[1]-savH),
                        bgCol=bgCol, style=style)
        # panel for showing graph video saving interface 
        pi["br"] = dict(pos=(pi["mr"]["pos"][0], wSz[1]-savH),
                        sz=(pi["mr"]["sz"][0], savH),
                        bgCol=bgCol, style=style) 
        return pi

    #---------------------------------------------------------------------------
     
    def initMLWidgets(self):
        """ Set up wxPython widgets when input file is loaded. 
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
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
        # list of button names to draw ROI or pick position, etc.
        self.drawBtns = [] 
        w = [] # each item represents a row in the left panel 
        fgC = "#cccccc" 
        bgC = self.pi["ml"]["bgCol"] 
        if self.eCase == "L2020":
            self.debugging["hideMarkerAB"] = False
            self.debugging["hideMarkerM"] = False
            ##### [begin] set widgets for HSV colors to track -----
            self.defHSVVal = {}
            self.defHSVVal["ant0"] = {} # ant body color
            self.defHSVVal["ant0"]["min"] = (0, 0, 0)
            self.defHSVVal["ant0"]["max"] = (180, 150, 100)
            self.defHSVVal["ant1"] = {} # color markers
            self.defHSVVal["ant1"]["min"] = (50, 80, 80)
            self.defHSVVal["ant1"]["max"] = (180, 255, 200)
            self.defHSVVal["ant2"] = {} # yellow color marker
            self.defHSVVal["ant2"]["min"] = (10, 200, 150)
            self.defHSVVal["ant2"]["max"] = (30, 255, 250)
            self.defHSVVal["br0"] = {} # yellowish color of brood 
            self.defHSVVal["br0"]["min"] = (10, 100, 100)
            self.defHSVVal["br0"]["max"] = (30, 180, 180)
            for ck in self.defHSVVal.keys(): # go though color ranges
                for mLbl in ["min", "max"]: # HSV min & max values
                    col = dict(H=0, S=0, V=0)
                    col["H"], col["S"], col["V"] = self.defHSVVal[ck][mLbl]
                    rgbVal = cvHSV2RGB(col["H"], col["S"], col["V"])
                    w.append([
                      {"type":"sTxt", "label":"%s [HSV-%s.]"%(ck, mLbl), 
                       "nCol":1},
                      {"type":"panel", "bgColor":rgbVal, "size":(20,20),
                       "nCol":1, "name":"c-%s-%s"%(ck, mLbl.capitalize())},
                      {"type":"btn", "nCol":1, "size":(35,35),
                       "name":"pipette-%s-%s"%(ck, mLbl), 
                       "img":path.join(self.btnImgDir, "pipette.png"),
                       "bgColor":"#333333"},
                      {"type":"btn", "nCol":1, "size":(35,35),
                       "name":"reset-%s-%s"%(ck, mLbl), 
                       "img":path.join(self.btnImgDir, "undo.png"),
                       "bgColor":"#333333"},
                      ])
                    tmp = []
                    for k in ["H", "S", "V"]:
                        if k == "H": maxVal = 180
                        else: maxVal = 255
                        if k == "V": _nC = 2
                        else: _nC = 1
                        tmp.append(
                            {"type":"sld", "nCol":_nC, "val":col[k],
                             "name":"c-%s-%s-%s"%(ck, k, mLbl.capitalize()), 
                             "size":(int(pSz[0]*0.3), -1), "border":1,
                             "minValue":0, "maxValue":maxVal, 
                             "style":wx.SL_VALUE_LABEL}
                            )
                    w.append(tmp) 
            ##### [end] set widgets for HSV colors to track -----
            
            ### for button to set HSV colors back to default 
            w.append([{"type":"btn", "size":(int(pSz[0]*0.85),-1), "nCol":4,
                       "name":"resetHSV", 
                       "label":"Reset all HSV values to its default"}])
            w.append([{"type":"sLn", "size":(int(pSz[0]*0.85),-1), "nCol":4,
                       "style":wx.LI_HORIZONTAL}])
            
            ### for changing color scheme
            w.append([{"type":"sTxt", "label":"Background color", "nCol":2},
                      {"type":"cPk", "nCol":2, "color":(0,0,0), 
                       "name":"bgCol", "size":(60,-1)}])
            
            ### for frame range to calculate heatmap
            w.append([{"type":"sTxt", "label":"start-frame", "nCol":2},
                      {"type":"txt", "name":"startFrame", "val":"0",
                       "nCol":2, "numOnly":True, "size":(60,-1)}])
            w.append([{"type":"sTxt", "label":"end-frame", "nCol":2},
                      {"type":"txt", "name":"endFrame", "val":"1",
                       "nCol":2, "numOnly":True, "size":(60,-1)}])
            ### for frame intervals for heatmap generation
            w.append([{"type":"sTxt", "label":"frame-interval", "nCol":2},
                      {"type":"txt", "name":"frameIntv", "val":"1",
                       "nCol":2, "numOnly":True, "size":(60,-1)}])
            ### for minimum area of an ant
            w.append([{"type":"sTxt", "label":"ant min. area", "nCol":2},
                      {"type":"txt", "name":"aMinArea", "val":"100",
                       "nCol":2, "numOnly":True, "size":(60,-1)}])
            ### for region of interest
            w.append([{"type":"sTxt", "label":"Region of interest", "nCol":2},
                      {"type":"txt", "name":"roi", "val":"675, 0, 300, 1024",
                       "nCol":2, "size":(100,-1)}])
            ### for entrance position 
            w.append([{"type":"sTxt", "label":"Entrance position", "nCol":2},
                      {"type":"txt", "name":"entrance", "val":"675, 0",
                       "nCol":2, "size":(100,-1)}])
            ### to set and mask wall area 
            w.append([{"type":"chk", "nCol":4, "name":"maskWallArea", 
                     "label":"Mask two walls", "style":wx.CHK_2STATE,
                     "val":True, "fgColor":fgC,
                     "tooltip":"Mask two walls for unstructured nest setup"}])
            w.append([{"type":"sTxt", "label":"Wall-rect-0", "nCol":2},
                      {"type":"txt", "name":"wallRect0", 
                       "val":"675, 305, 300, 50", "nCol":2, "size":(100,-1)}])
            w.append([{"type":"sTxt", "label":"Wall-rect-1", "nCol":2},
                      {"type":"txt", "name":"wallRect1",
                       "val":"675, 660, 300, 50", "nCol":2, "size":(100,-1)}])
            ### to set frame to jump for debugging
            w.append([{"type":"sTxt", "nCol":2, "fgColor":fgC,
                       "label":"frames to jump [debug]:"},
                      {"type":"txt", "nCol":2, "name":"debugFrameIntv", 
                       "val":"1", "numOnly":True, "size":(50,-1)}])
            ### to hide some markings in debugging mode
            w.append([{"type":"chk", "nCol":4, "name":"hideMarkerAB", 
                     "label":"Hide ant, brood coloring [debug]", 
                     "style":wx.CHK_2STATE,
                     "val":self.debugging["hideMarkerAB"], "fgColor":fgC,
                     "tooltip":"Hide ant & brood coloring in debug mode"}])
            ### to hide some markings in debugging mode
            w.append([{"type":"chk", "nCol":4, "name":"hideMarkerM", 
                     "label":"Hide motion coloring [debug]", 
                     "style":wx.CHK_2STATE,
                     "val":self.debugging["hideMarkerM"], "fgColor":fgC,
                     "tooltip":"Hide motion coloring in debug mode"}])
            ### button widget to draw the graph
            w.append([{"type":"btn", "name":"draw", "label":"Draw heatmap",
                       "size":(int(pSz[0]*0.85),-1), "nCol":4}])
        elif self.eCase == "L2020CSV1":
            nCol = 3
            w.append([{"type":"sTxt", "label":"Heatmap ranges", "nCol":nCol}])
            txtSz = (int(pSz[0]*0.4), -1)
            for i in range(7):
                if i == 0: lbl = "0 - %i %%"%(self.percLst[0])
                else: lbl = "%i - %i %%"%(self.percLst[i-1], self.percLst[i])
                w.append([{"type":"sTxt", "label":lbl, "nCol":nCol}])
                w.append([{"type":"txt", "name":"hmRng-%i-min"%(i), "val":"0",
                           "nCol":1, "size":txtSz, "style":wx.TE_READONLY},
                          {"type":"sTxt", "label":"-", "nCol":1},
                          {"type":"txt", "name":"hmRng-%i-max"%(i), "val":"0",
                           "nCol":1, "size":txtSz, "style":wx.TE_READONLY}])
            w.append([{"type":"sTxt", "nCol":nCol,
                       "label":"Distance between avg. centroid & max. point",
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.TOP), "border":25}])
            w.append([{"type":"txt", "name":"dist_M2C", "val":"0",
                       "nCol":nCol, "numOnly":True, "border":25, 
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM),
                       "style":wx.TE_READONLY}])
            '''
            ### button widget to draw the graph
            w.append([{"type":"btn", "name":"draw", "label":"Draw heatmap",
                       "nCol":nCol}])
            '''
        elif self.eCase == "aos":
            nCol = 3 # max number of coloumns
            ### draw button
            w.append([{"type":"btn", "name":"draw", "label":"Draw", 
                       "nCol":nCol, "size":hlSz}])
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])
            if self.eCase == "aos":
                ### cam index to process
                w.append([{"type":"sTxt", "label":"cam-index: ", "nCol":1, 
                           "fgColor":fgC, "bgColor":bgC},
                          {"type":"cho", "nCol":2, "name":"camIdx",
                           "choices":["99", "1"], "val":"99"}])
            ### process to run
            ###   Key should NOT include '_' underbar symbol.
            ###     Underbar is used to indicate multiple measures 
            ###     such as 'intensity_reach' (deprecated in 2022-05).
            proc2runCho = ["initImg", "sanityChk"]
            proc2runCho += sorted([
                    "intensity",
                    "intensityPSD", # power spectral density
                    "intensityL", # local intensity (brood, food,...)
                    "dist2b", # distance to approx. brood position 
                              # (max. distance with multiple motion points)
                    "dist2c", # distance to center of ROI0
                              # (max. distance with multiple motion points)
                    "dist2ca", # distance to center of ROI0
                               # (record distances of each motion point,
                               #  regardless of data-bundle-interval)
                    "saMVec", # single ant's movement vector
                    "spAHeatmap",
                    "spAGridDenseActivity",
                    "spAGridHeatMean",
                    "spAGridHeatMax",
                    #"spAGridOFlow", # not being used
                    "motionSpread",
                    "proxMCluster",
                    #"distCent", #[deprecated]
                    ]) 
            
            """
            - intensity: number of motions; When raw-data are saved, 
                5 raw numpy files will be saved. original intensity data, 
                smoothed intensity data, summed daily data, motion bout
                durations (in sec), inactivity durations (in seconds)
            - intensityPSD: Power spectral density of intensity
                * PSD is calculated with Welch's method (scipy.signal.welch)
            - intensityL: calculated motion-value depending on
                              three locations; brood, food and garbage.
                pt_m = currently calculating motion-point
                pt_l0 = currently calculating location point
                pt_l1 = another location point, closest from pt_l0
                thr = threshold to determine location boundary 
                d_l = distance from pt_l0 to pt_l1 
                d_m = distance from pt_l0 to pt_m
                d_th = d_l * thr 
                ptW = max(0, (d_th-d_m)/d_th)
                local-intesnity of the location in a frame is
                  sum of ptW of all motion-points in the frame.
            - saMVec: Direction & distance of moving (walking/running).
                It assumes a single ant is in the scene.
                The distance is relative value to the ant's body length 
                (user-defined).
            - spAHeatmap: (spA; spatial-analysis) heatmap.
            - spAGridHeatMean / Max: Draw a heatmap during a given short time 
                (with data bundle interval), then record the mean (or max) 
                value in each grid-cell.
            - spAGridDenseActivity: Measing 'dense' activity only. 'dense' here
                means activity occurring in spatially concentrated manner.
                Measurement of motion blobs which occurred only in a spatially
                close area in a given time (with data bundle interval)
                This produces a graph similar to 'spAGridHeatMean'.
            - motionSpread: Number of motion points, one individual produces,
                during a single frame (0.4-0.6 sec.) is usually 1-3. 
                When 1-3 are considered, it's about 90% of data,
                over 97% when '1-4' is considered,
                over 99% when '1-5' is considered.
                'Spread-value' is an approximate number of individual ants,
                    moved around an individual's motion in the previous frame.
                'Spread-value' is calculated as following:
                    * r = a half of single ant's body length
                    Go through each data frame (i is frame index) ...
                    0) Initially, spread-value of s[i] is 0.
                    if i > 0:
                      1) if len(m[i-1]) > 0:
                      # there was motion-point in the previous frame
                        1-1) Calculate spread-value of j-th center, s[i][j] = 
                            Number of motion groups around the j-th center 
                              with the radius of r*3 disregarding the points 
                              in the radius of r.
                            Motion points grouping is done with the radius of r.
                        1-2) Remove all the considered motions points 
                            in the r*3, to prevent duplicate counting.
                        1-3) sum(s[i]) is the spread-value of all centers 
                            in [i]-th frame.
                    2) Group motion points in [i]-th frame.
                        Motion points in i-frame is grouped using 'r';
                        Up to 4 points are ground as a single' ants movements
                        within the radius of r.
                    3) Temporarily store the center of each group and 
                        its number of motion pooints in m[i-1].
            - proxMCluster: Consecutive (3 data-frames;currently 3 seconds) 
                moving motion clusters that contains more than 
                5 motion points. 
            - distCent: avg. distance between brood position and ants' centroid 
            """
            w.append([{"type":"sTxt", "label":"process: ", "nCol":1, 
                       "fgColor":fgC},
                      {"type":"cho", "nCol":2, "name":"process",
                       "choices":proc2runCho, "val":proc2runCho[0]}])
            ### interval to bundle data
            dPtIntvCho = ["1", "2", "3", "5", "10", "30", "60", "300", 
                        "600", "1800", "3600", "43200", "86400"]
            w.append([{"type":"sTxt", "label":"data-point per", "nCol":1, 
                       "fgColor":fgC},
                      {"type":"cho", "nCol":1, "name":"dataPtIntv",
                       "choices":dPtIntvCho, "val":"300"},
                      {"type":"sTxt", "label":"sec.", "nCol":1, 
                       "fgColor":fgC}])
            ### maximum number of motion in a frame;
            ###   threshold to cut motion data 
            tt = "Maximum number of motion in a single frame;"
            tt += " threshold to cut off frame data where there're too many"
            tt += " motions"
            w.append([{"type":"sTxt", "nCol":1, "label":"Max. motions",
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"maxMotionThr", "val":"100",
                       "nCol":2, "size":(100,-1), "numOnly":True}])
            ### default graph size 
            tt = "width & height of graphs for some graphs."
            w.append([{"type":"sTxt", "label":"def. graph size", "nCol":1, 
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"graphSz", "val":"1500, 500",
                       "nCol":2, "size":(100,-1)}])
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "fgColor":fgC,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])
            
            ### filename prefix in saving
            w.append([{"type":"sTxt", "nCol":1, "label":"Filename prefix",
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"fnPrefix", "val":"",
                       "nCol":2, "size":(100,-1)}])
            ### filename suffix in saving
            w.append([{"type":"sTxt", "nCol":1, "label":"Filename suffix",
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"fnSuffix", "val":"",
                       "nCol":2, "size":(100,-1)}])

            ### set hours to process
            _w = [
                    ("h2ignore", int(24*2.5)), # first hours to ignore
                    ("h2proc", int(24*4)), # hours to process
                    ]
            for _n, _v in _w:
                w.append([{"type":"sTxt", "label":_n, "nCol":1, "fgColor":fgC},
                          {"type":"txt", "nCol":1, "name":_n, "size":(100,-1), 
                           "val":str(_v), "numOnly":True}])

            ### buttons for drawing point or ROI
            self.drawBtns = ["broodPt_btn", "foodPt_btn", "garbagePt_btn", 
                             "roi0ROI_btn"]

            ### brood position
            brPos = "206, 728" 
            w.append([{"type":"sTxt", "label":"brood position", "nCol":1,
                       "fgColor":fgC},
                      {"type":"txt", "name":"broodPt", "val":brPos, 
                       "nCol":1, "size":(100,-1)},
                      {"type":"btn", "name":"broodPt", "nCol":1,
                       "img":self.targetBtnImgPath,
                       "bgColor":"#333333", "size":(35,35),
                       "tooltip":"Choose brood position by mouse-click"}])
            ### food position
            gdPos = "208, 236" 
            tt = "Choose food position by mouse-click"
            w.append([{"type":"sTxt", "label":"food position", "nCol":1,
                       "fgColor":fgC},
                      {"type":"txt", "name":"foodPt", "val":gdPos, 
                       "nCol":1, "size":(100,-1)},
                      {"type":"btn", "name":"foodPt", "nCol":1,
                       "img":self.targetBtnImgPath,
                       "bgColor":"#333333", "size":(35,35),
                       "tooltip":tt}])
            ### garbage dump position
            gdPos = "100, 596" 
            tt = "Choose garbage dump position by mouse-click"
            w.append([{"type":"sTxt", "label":"garbage position", "nCol":1,
                       "fgColor":fgC},
                      {"type":"txt", "name":"garbagePt", "val":gdPos, 
                       "nCol":1, "size":(100,-1)},
                      {"type":"btn", "name":"garbagePt", "nCol":1,
                       "img":self.targetBtnImgPath,
                       "bgColor":"#333333", "size":(35,35),
                       "tooltip":tt}])
            ### Threshold to determine location boundary 
            tt = "Thresholds to determine a location; [brood, food, garbage]"
            tt += " If this is 0.5, it means that weight on a motion-point"
            tt += " becomes zero when it is located further than 50% of"
            tt += " distance between the location and another nearest location."
            w.append([{"type":"sTxt", "nCol":1, "label":"Distance threshold",
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"locThr", "val":"0.9, 0.3, 0.3",
                       "nCol":2, "size":(100,-1)}])
            ### nest rect
            _r = "0, 0, 600, 1440"
            w.append([{"type":"sTxt", "label":"ROI-0", "nCol":1,
                       "fgColor":fgC},
                      {"type":"txt", "name":"roi0ROI", "val":_r, 
                       "nCol":1, "size":(150,-1)},
                      {"type":"btn", "name":"roi0ROI", "nCol":1, 
                       "img":self.targetBtnImgPath,
                       "bgColor":"#333333", "size":(35,35),
                       "tooltip":"Select ROI by mouse click-and-drag"}])
            
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "fgColor":fgC,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])

            # Grid setting on nest area for spatial-analysis
            _choices = [str(x) for x in range(2, 10)]
            w.append([{"type":"sTxt", "nCol":1, "fgColor":fgC,
                       "label":"spAGrid row, col."},
                      {"type":"cho", "nCol":1, "name":"spAGridRows",
                       "choices":_choices, "val":"4"},
                      {"type":"cho", "nCol":1, "name":"spAGridCols",
                       "choices":_choices, "val":"3"}])
            _choices = [str(x) for x in range(0, 6)]
            w.append([{"type":"sTxt", "nCol":1, "fgColor":fgC,
                       "label":"spAGrid denoise"},
                      {"type":"cho", "nCol":2, "name":"spAGridMExIter",
                       "choices":_choices, "val":"5"}])
            _r = "117, 442, 371, 672"
            _tt = "Select spAGrid ROI by mouse click-and-drag"
            w.append([{"type":"sTxt", "label":"spAGrid rect", "nCol":1,
                       "fgColor":fgC},
                      {"type":"txt", "name":"spAGridROI", "val":_r, 
                       "nCol":1, "size":(150,-1)},
                      {"type":"btn", "name":"spAGridROI", "nCol":1, 
                       "img":self.targetBtnImgPath,
                       "bgColor":"#333333", "size":(35,35), "tooltip":_tt}])
            
            ### ant body length 
            tt = "Length of a single ant (in pixels) in this video." 
            tt += " Currently used for 'motionSpread'."
            initAL = "35" # inital ant body length (in pixels)
            w.append([{"type":"sTxt", "label":"Ant length", "nCol":1,
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"antLen", "val":initAL, "nCol":2,
                       "size":(100,-1), "numOnly":True}])
             
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "fgColor":fgC,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])

            w.append([{"type":"sTxt", "label":"Intensity graph details", 
                       "nCol":3, "fgColor":fgC}])
            ### whether to draw smoothed line on intensity graph
            w.append([{"type":"chk", "nCol":3, "name":"smoothLine", 
                     "label":"draw smooth line", "style":wx.CHK_2STATE,
                     "val":True, "fgColor":fgC, "bgColor":bgC}]) 
            ### whether to draw peak points on intensity graph
            w.append([{"type":"chk", "nCol":3, "name":"peak", 
                     "label":"draw peak points", "style":wx.CHK_2STATE,
                     "val":True, "fgColor":fgC, "bgColor":bgC}]) 
            ### whether to draw outlier on intensity graph
            w.append([{"type":"chk", "nCol":3, "name":"outlier", 
                     "label":"Mark outlier data-lines", "style":wx.CHK_2STATE,
                     "val":False, "fgColor":fgC, "bgColor":bgC}]) 
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])

            ### interval for heatmap 
            w.append([{"type":"sTxt", "label":"heatmap per", "nCol":1,
                       "fgColor":fgC},
                      {"type":"cho", "nCol":1, "name":"heatMapIntv",
                       "choices":["-1", "30", "60", "120", "180", "240",
                                  "360", "720", "1440", "10080"], 
                       "val":"-1"},
                      {"type":"sTxt", "label":"min.", "nCol":1,
                       "fgColor":fgC}])
            ### heatmap point radius
            w.append([{"type":"sTxt", "label":"heatmap point rad.", "nCol":1,
                       "fgColor":fgC,
                       "tooltip":"Radius of each data point in heatmap"},
                      {"type":"txt", "name":"hmPt", "val":"-1",
                       "nCol":2, "size":(100,-1), "numOnly":True}])
            ### save motion video in heatmap
            w.append([{"type":"chk", "nCol":3, "name":"saveHMVideo", 
                     "label":"Save motion video on heatmap", 
                     "style":wx.CHK_2STATE,
                     "val":False, "fgColor":fgC, "bgColor":bgC}]) 
            ### slider bar for heatmap
            w.append([{"type":"sTxt", "label":"heatmap navigation with time", 
                       "fgColor":fgC, "nCol":nCol}])
            w.append([{"type":"sld", "nCol":nCol, "val":0, "name":"nav", 
                       "size":hlSz, "style":wx.SL_VALUE_LABEL,
                       "minValue":0, "maxValue":100}])
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "fgColor":fgC,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])

            ### power spectral density; signal length
            w.append([{"type":"sTxt", "nCol":1, "label":"PSD-length",
                       "fgColor":fgC},
                      {"type":"cho", "nCol":2, "name":"psdLen",
                       "choices":["3 h", "6 h", "12 h", "24 h", "72 h",
                                  "entire input data"],
                       "val":"24 h"}])
            ### power spectral density; number of data per segment 
            _choices = [str("%i"%(2**x)) for x in range(6,17)]
            w.append([{"type":"sTxt", "nCol":1, "label":"PSD-segment",
                       "fgColor":fgC},
                      {"type":"cho", "nCol":2, "name":"psdNPerSeg",
                       "choices":_choices, "val":"512"}])
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "fgColor":fgC,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])

            '''
            ### degrees to emphasize in saMVec image 
            w.append([{"type":"sTxt", "nCol":1, "label":"saMVec emph. deg.",
                       "fgColor":fgC},
                      {"type":"txt", "name":"saMVecEmph", "nCol":2, 
                       "val":"90, 270, 0, 180", "size":(200,-1)}])
            w.append([{"type":"sTxt", "nCol":1, "label":"saMVec emph. margin",
                       "fgColor":fgC},
                      {"type":"txt", "name":"saMVecEmphMargin", "val":"0",
                       "nCol":2, "size":(75,-1), "numOnly":True}])
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "fgColor":fgC,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])
            '''

            ### whether to save cam-index when files are saved
            w.append([{"type":"chk", "nCol":3, "name":"savCamIdx", 
                     "label":"Save cam-index in filenames", 
                     "style":wx.CHK_2STATE, "val":False, 
                     "fgColor":fgC, "bgColor":bgC}]) 
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])

        elif self.eCase == "anVid":
            nCol = 3 # max number of coloumns
            ### draw button
            w.append([{"type":"btn", "name":"draw", "label":"Draw", 
                       "nCol":nCol, "size":hlSz}])
            
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10}])
            
            ### process to run
            proc2runCho = ["initImg"]
            proc2runCho += [
                    "intensity",
                    "intensityT", # intensity around tunnel-area
                    "intensityP", # intensity around pupae 
                    "intensityPABR", # intensity with ABR around pupae 
                    #"presenceT",
                    "intensityPSD", # power spectral density
                    "dist2EO", # distance to each other (ant)
                    #"dist2EOCh",
                    "distP2T", # distance pupae to tunnel
                    "distP2A", # distance pupae to ants 
                    "distA2T", # distance ant to tunnel
                    "spAHeatmap", # heatmpa with motion data
                    "spAHeatmapP", # heatmap around pupae
                    "spAHeatmapABR", # heatmap with ant-blob-rects
                    "spAHeatmapPABR", # heatmap with ABR around pupae
                    ]
            """
            - intensity: number of motions; When raw-data are saved, 
                5 raw numpy files will be saved. original intensity data, 
                smoothed intensity data, summed daily data, motion bout
                durations (in sec), inactivity durations (in seconds)
            - intensityT: intensity around the tunnel,
                the closest accessible position to the stimulus (sham/fungus)
            - intensityP: intensity-sum around pupae,
            - presenceT: presence of ants around the tunnel
            - intensityPSD: Power spectral density of intensity
                * PSD is calculated with Welch's method (scipy.signal.welch)
            - dist2EO: sum of distances between two closest blobs. 
            - dist2EOCh: change of dist2EO from the previous data-bundle
            - distP2T: Mean distance of pupae to tunnel-center.
            - distP2A: Mean distance of pupae to the (closest) ant.
            - distA2T: Mean distance of ants to tunnel-center.
            - spAHeatmap: (spA; spatial-analysis) heatmap with motion.
            - spAHeatmapP: heatmap with motion around pupae.
            - spAHeatmapABR: heatmap with ant-blob-rects (detected by color).
            - spAHeatmapPABR: heatmap with ant-blob-rects around pupae
            """
            w.append([{"type":"sTxt", "label":"process: ", "nCol":1, 
                       "fgColor":fgC},
                      {"type":"cho", "nCol":nCol-1, "name":"process",
                       "choices":proc2runCho, "val":proc2runCho[0]}])
            ''' ###
            - interval (sec) to bundle data
            - default graph size
            
            - first hours to ignore
            - hours to process
            - y-axis max. value for bar-graph 
            - y-axis max. value for heatmap
            - ant body length
            - tunnel positions

            - power-spectral-density; signal length
            - power-spectral-density; number of data per segment 
            
            - whether to draw smoothed line on bar graph
            - whether to draw peak points on bar graph
            - whether to draw outlier data-lines on bar graph
            '''
            dPtIntvCho = ["1", "2", "3", "5", "10", "30", "60", "300", 
                        "600", "1800", "3600", "43200", "86400"]
            psdLenCho = ["3 h", "6 h", "12 h", "24 h", "72 h", 
                         "entire input data"]
            psdNPerSegCho = [str("%i"%(2**x)) for x in range(6,17)]
            _w = [
                    ("cho", "dataPtIntv", dPtIntvCho, "bundle interval"),
                    ("txt", "graphSz", "1500, 500", 
                        "guiding width & height for bar graphs"), 
                    #("txt", "fnPrefix", "", "Filename prefix"), 
                    #("txt", "fnSuffix", "", "Filename suffix"), 
                    ("txt", "h2ignore", "0", "hours to ignore"), 
                    ("txt", "h2proc", "0", "hours to process"), 
                    ("txt", "barGMax", "-1", "y max. for bar graph"), 
                    ("txt", "heatmapMax", "-1", "y max. for heatmpa"),
                    ("txt", "antLen", "40", "ant's body length"),
                    ]
            
            ### buttons for tunnel position in two ROIs
            for bA in ["Before", "After"]: # before/after the trigger stimulus
                for i in range(2):
                    _pos = f'{i*200}, 200'
                    _w.append(("txt4position", f'tunnel{bA}{i}Pt', _pos, 
                               f'tunnel ROI-{i:02d} ({bA.lower()})'))
            
            _w += [
                    ("sTxt", "", " ", ""), # space
                    ("cho", "psdLen", psdLenCho, "PSD-length"),
                    ("cho", "psdNPerSeg", psdNPerSegCho, "PSD-segment"),
                    ("sTxt", "", " ", ""), # space
                    ("sTxt", "", "Bar graph details", ""),
                    ("chk", "smoothLine", True, "Draw smooth line"),
                    ("chk", "peak", True, "Draw peak points"),
                    ("chk", "outlier", False, "Mark outlier data-lines"),
                    ("sTxt", "", " ", ""), # space
                    ]
            w = self.makeDictLst4widgets(_w, w, nCol, fgC, bgC)

            ### interval for heatmap
            _cho = ["-1", "5", "10", "20", "30"]
            for _intv in range(60, 60*24+1, 60): _cho.append(str(_intv))
            _cho.append(str(60*24*7))
            w.append([{"type":"sTxt", "label":"heatmap per", "nCol":1,
                       "fgColor":fgC},
                      {"type":"cho", "nCol":1, "name":"heatMapIntv",
                       "choices":_cho, "val":"-1"},
                      {"type":"sTxt", "label":"min.", "nCol":1,
                       "fgColor":fgC}])
            '''
            - heatmap point radius
            - save video of heatmap
            '''
            fpsCho = ["1", "10", "30", "60"]
            _w = [
                    ("txt", "hmPt", "-1", 
                        "Radius of each data point in heatmap"), 
                    ("chk", "saveHMVideo", False, "Save video of heatmap"),
                    ("cho", "heatMapVideoFPS", fpsCho, "FPS of heatmap video"),
                    ]
            w = self.makeDictLst4widgets(_w, w, nCol, fgC, bgC)
            
        elif self.eCase == "aosI":
            tt = "width & height of graphs for some graphs."
            w.append([{"type":"sTxt", "label":"Graph size", "nCol":1, 
                       "fgColor":fgC, "tooltip":tt},
                      {"type":"txt", "name":"graphSz", "val":"1500, 500",
                       "nCol":2, "size":(100,-1)}])
        
        elif self.eCase == "aosSec":
            nCol = 2 # max number of coloumns
            ### draw button
            w.append([{"type":"btn", "name":"draw", "label":"Draw", 
                       "nCol":nCol, "size":hlSz}])
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10,
                       "flag":(wx.ALIGN_CENTER_VERTICAL|wx.BOTTOM)}])
            ### analysis type
            _cho = ["n1610;intensity", "n1610;dist2b", "n1610;proxMCluster",
                    "n1610;actInact", "n1;saMVec", "n1;saDist2c", "dist2ca"]
            w.append([{"type":"sTxt", "label":"analysis", "nCol":1, 
                       "fgColor":fgC},
                      {"type":"cho", "nCol":1, "name":"aType",
                       "choices":_cho, "size":(250,-1), "val":_cho[0]}])

            _w = [
                    ("txt", "dPtIntvSec", "300", "bundle-interval"),
                    ("txt", "h2ignore", "0", "first hours to ignore"),
                    ("txt", "h2proc", str(int(24*4)), ""),
                    ("txt", "h2ignoreDev", "0", 
                        "for deviation from mean graph"),
                    ("txt", "h2procDev", "-1", "for deviation from mean graph"),
                    ("txt", "nCol", "4", 
                        "columns in the resultant graph image"),
                    ("txt", "xlimDist2c", "-1", 
                        "max. distance in dist2c/dist2ca"),
                    ("cho", "decPlDev", ["3", "6"], "decimal places"),
                    ("txt", "maxYDev", "-1", "Max. Y in deviation graph."),
                    ]
            w = self.makeDictLst4widgets(_w, w, nCol, fgC, bgC)

        elif self.eCase == "cremerGroupApp":
            nCol = 3 # max number of coloumns
            ### draw button
            w.append([{"type":"btn", "name":"draw", "label":"Draw", 
                       "nCol":nCol, "size":hlSz}])
            
            w.append([{"type":"sTxt", "nCol":nCol, "label":"", "border": 10}])
            
            ### process to run
            proc2runCho = ["all"]
            w.append([{"type":"sTxt", "label":"process: ", "nCol":1, 
                       "fgColor":fgC},
                      {"type":"cho", "nCol":nCol-1, "name":"process",
                       "choices":proc2runCho, "val":proc2runCho[0]}]) 

        self.mlWid = setupPanel(w, self, pk)
        
        ##### [end] set up middle left panel ----- 

    #---------------------------------------------------------------------------
   
    def initMRWidgets(self):
        """ init widgets in the middle-right panel
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        for i, w in enumerate(self.mrWid): # widgets in the panel
            try:
                self.gbs["mr"].Detach(w) # detach 
                w.Destroy() # destroy
            except:
                pass

    #---------------------------------------------------------------------------
   
    def makeDictLst4widgets(self, inputLst, w, nCol, fgC, bgC):
        """ utility function to reduce line of codes to generate wxWidgets.
        
        Args:
            inputLst (list): List of some info to make dicts
            w (list): List of info to generate widgets
            nCol (int): Number of columns in a row
            fgC (str): foreground color
            bgC (str): background color
        
        Returns:
            w (list): List of info to generate widgets
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        for typ, wN, wV, wDesc in inputLst:
            if typ == "sTxt":
                w.append([{"type":"sTxt", "label":wV, "nCol":nCol, 
                           "fgColor":fgC}])
            
            elif typ == "txt":
                w.append([{"type":"sTxt", "label":wN, "nCol":1,
                           "fgColor":fgC, "tooltip":wDesc},
                          {"type":"txt", "nCol":nCol-1, "name":wN,
                           "size":(120,-1), "val":wV, "numOnly":True}])
            
            elif typ == "chk":
                w.append([{"type":"chk", "nCol":nCol, "name":wN,
                           "label":wDesc, "style":wx.CHK_2STATE,
                           "val":wV, "fgColor":fgC, "bgColor":bgC}]) 
            
            elif typ == "cho":
                w.append([{"type":"sTxt", "nCol":1, "label":wDesc,
                           "fgColor":fgC},
                          {"type":"cho", "nCol":nCol-1, "name":wN,
                           "choices":wV, "val":wV[0]}])

            elif typ == "txt4position":
                w.append([{"type":"sTxt", "label":wDesc, "nCol":1, 
                           "fgColor":fgC},
                          {"type":"txt", "name":wN, "val":wV, 
                           "nCol":1, "size":(100,-1)},
                          {"type":"btn", "name":wN, "nCol":1,
                           "img":self.targetBtnImgPath,
                           "bgColor":"#333333", "size":(35,35),
                           "tooltip":"Choose position by mouse-click"}])
                self.drawBtns.append(f'{wN}_btn')

        return w

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
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "btn")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return
        if self.flags["blockUI"] or not obj.IsEnabled(): return
        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))

        if objName == "help_btn":
            if self.eCase not in HELPSTR.keys(): return 
            msg = HELPSTR[self.eCase]
            sz = (int(self.wSz[0]*0.5), int(self.wSz[1]*0.8))
            dlg = PopupDialog(self, -1, "Help string", msg, 
                              size=sz, flagDefOK=True)
            dlg.ShowModal()
            dlg.Destroy()

        elif objName == "open_btn": self.openInputF(None)

        elif objName == "save_btn": self.save()

        elif objName == "saveAll_btn": self.save(isSavingAll=True) 

        elif objName == "saveRawData_btn": self.save(savType="raw")
        
        elif objName == "saveAllRawData_btn": self.save("raw", True)

        elif objName == "deleteGImg_btn": 
            if self.pgd != None: self.pgd.removeGraph(self.pgd.graphImgIdx)

        elif objName == "deleteAllGImg_btn": 
            if self.pgd != None: self.pgd.removeGraph()

        elif objName.startswith("pipette"): 
            if self.inputFP == "": return
            if self.colorPick == "":
            # no previously clicked color picking pipette
                pLbl = objName.replace("pipette-","").replace("_btn","") 
                self.colorPick = pLbl
                cursor = wx.Cursor(wx.CURSOR_CROSS)
                clickedBtnCol = wx.Colour(50,255,50)
            else:
                # previously clicked button name
                bN = "pipette-%s_btn"%(self.colorPick) 
                if objName == bN:
                # the same pipette button is clicked
                    self.colorPick = ""
                    cursor = wx.Cursor(wx.CURSOR_ARROW)
                    clickedBtnCol = wx.Colour(51,51,51)
                else:
                # another pipette button is clicked
                    _btn = wx.FindWindowByName(bN, self.panel["ml"])
                    _btn.SetBackgroundColour("#333333")
                    pLbl = objName.replace("pipette-","").replace("_btn","") 
                    self.colorPick = pLbl
                    cursor = wx.Cursor(wx.CURSOR_CROSS) 
                    clickedBtnCol = wx.Colour(50,255,50)
            self.panel["mp"].SetCursor(cursor)
            obj.SetBackgroundColour(clickedBtnCol)

        elif objName.startswith("reset-"): # reset HSV color button
            rBLbl = objName.replace("reset-","").replace("_btn","")
            ck, mLbl = rBLbl.split("-")
            col = dict(H=0, S=0, V=0)
            col["H"], col["S"], col["V"] = self.defHSVVal[ck][mLbl]
            for k in ["H", "S", "V"]:
                sldN = "c-%s-%s-%s_sld"%(ck, k, mLbl.capitalize())
                sld = wx.FindWindowByName(sldN, self.panel["ml"])
                sld.SetValue(col[k])
            pN = "c-%s-%s_panel"%(ck, mLbl.capitalize()) 
            rgbVal = cvHSV2RGB(col["H"], col["S"], col["V"])
            obj = wx.FindWindowByName(pN, self.panel["ml"])
            obj.SetBackgroundColour(rgbVal)
            obj.Refresh()

        elif objName == "resetHSV_btn":
            for ck in self.defHSVVal.keys(): # go though color ranges
                for mLbl in ["min", "max"]: # HSV min & max values
                    col = dict(H=0, S=0, V=0)
                    col["H"], col["S"], col["V"] = self.defHSVVal[ck][mLbl]
                    for k in ["H", "S", "V"]:
                        sldN = "c-%s-%s-%s_sld"%(ck, k, mLbl.capitalize())
                        sld = wx.FindWindowByName(sldN, self.panel["ml"])
                        sld.SetValue(col[k])
                    pN = "c-%s-%s_panel"%(ck, mLbl.capitalize()) 
                    rgbVal = cvHSV2RGB(col["H"], col["S"], col["V"])
                    obj = wx.FindWindowByName(pN, self.panel["ml"])
                    obj.SetBackgroundColour(rgbVal)
                    obj.Refresh()

        elif objName.endswith("ROI_btn"): # ROI drawing button is pressed
            self.flags["selectPt"] = {} # make sure other selection is off
            ### set default image for the all drawing PT and ROI buttons
            for btnName in self.drawBtns:
                w = wx.FindWindowByName(btnName, self.panel["ml"])
                set_img_for_btn(self.targetBtnImgPath, w)
            k = objName.replace("ROI_btn", "") # nest, ..
            if k in self.flags["drawingROI"]: # already on 
                self.flags["drawingROI"] = {} # remove; turning off drawing 
            else:
                self.flags["drawingROI"] = {k: True} # drawing mode on
                set_img_for_btn(self.targetBtnActImgPath, obj)

        elif objName.endswith("Pt_btn"): # point selection
            self.flags["drawingROI"] = {} # make sure other selection is off
            ### set default image for the all drawing PT and ROI buttons
            for btnName in self.drawBtns:
                w = wx.FindWindowByName(btnName, self.panel["ml"])
                set_img_for_btn(self.targetBtnImgPath, w)
            k = objName.replace("Pt_btn", "") # brood, food, garbage ..
            if k in self.flags["selectPt"]: # already on 
                self.flags["selectPt"] = {} # turn off
            else:
                self.flags["selectPt"] = {k: True} # mode on
                set_img_for_btn(self.targetBtnActImgPath, obj)

        elif objName == "draw_btn":

            if self.eCase == "cremerGroupApp":
                if self.pgd is None: self.pgd = ProcGraphData(self)
                if not 'ProcCremerGroupApp' in sys.modules:
                    from procCremerGroupApp import ProcCremerGroupApp
                self.pgd.subClass = ProcCremerGroupApp(self, self.pgd) 
                self.flags["blockUI"] = True
                showStatusBarMsg(self, "Processing data ... ", -1)
                wx.CallLater(100, self.pgd.subClass.drawGraph)
                return

            if self.pgd is None:
                msg = "Select the input folder/file first."
                wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                return

            self.pgd.interactiveDrawing = {} # clear interactive drawing
            if self.eCase == "L2020":
            # Heatmap for Linda (2020)
                ### update ROI
                obj = wx.FindWindowByName("roi_txt", self.panel["ml"])
                try:
                    roi = [int(x) for x in obj.GetValue().split(",")]
                    if len(roi) == 4: self.roi = roi
                except:
                    pass

                self.pgd.removeGraph() # remove all previous graphs
                if self.debugging["state"]:
                    self.pgd.graphL2020(-1, -1, self.q2m)
                    self.panel["mp"].Refresh() # display graph 
                else:
                    self.initMRWidgets() 
                    ### determine start frame
                    obj = wx.FindWindowByName("startFrame_txt", 
                                              self.panel["ml"])
                    startFrame = int(obj.GetValue())
                    if startFrame < 0 or startFrame >= self.vRW.nFrames:
                        startFrame = 0
                        obj.SetValue('0')
                    self.startFI = startFrame
                    ### determine end frame
                    obj = wx.FindWindowByName("endFrame_txt", self.panel["ml"])
                    endFrame = int(obj.GetValue())
                    if endFrame <= startFrame or endFrame >= self.vRW.nFrames:
                        endFrame = self.vRW.nFrames-1
                        obj.SetValue(str(endFrame))
                    nFrames = endFrame - startFrame + 1
                    self.endFI = endFrame
                    ### determine frame interval
                    obj = wx.FindWindowByName("frameIntv_txt", self.panel["ml"])
                    frameIntv = int(obj.GetValue())
                    if frameIntv < 1 or frameIntv > nFrames:
                        frameIntv = nFrames
                        obj.SetValue(str(frameIntv))
                    self.frameIntv = frameIntv
                    nextEndFrame = min(startFrame+frameIntv-1, endFrame)
                    ### start drawing graph
                    if startFrame == self.vRW.fi:
                        args = (startFrame, nextEndFrame, self.q2m,)
                        startTaskThread(self, "drawGraph", self.pgd.graphL2020, 
                                       args=args)
                    else:
                        self.vRW.getFrame(startFrame, 
                                          callbackFunc=self.callback)

            elif self.eCase == "L2020CSV1":
            # Heatmap for Linda (2020); result CSV file loaded
                """
                startTaskThread(self, 
                                "drawGraph", 
                                self.pgd.graphL2020CSV1,
                                args=(self.q2m,))
                """
                self.pgd.graphL2020CSV1()

            elif self.eCase == "L2020CSV2":
            # Heatmap for Linda (2020); result CSV file loaded
                """
                startTaskThread(self, 
                                "drawGraph", 
                                self.pgd.graphL2020CSV2,
                                args=(self.q2m,))
                """
                self.pgd.graphL2020CSV2()

            elif self.eCase in ["aos", "anVid"]: 
                startTaskThread(self, 
                                "drawGraph", 
                                self.pgd.subClass.drawGraph,
                                args=(self.q2m, 0,))

            elif self.eCase == "aosSec":
                self.flags["blockUI"] = True
                showStatusBarMsg(self, "Processing data ... ", -1)
                wx.CallLater(100, self.pgd.subClass.drawGraph)
     
    #---------------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate event 
                                     with the given name. 
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ret = preProcUIEvt(self, event, objName, "chk")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "debug_chk":
            self.debugging["state"] = objVal
            if self.debugging["state"]:
                if hasattr(self, "vRW") and self.inputFP != "":
                    self.vRW.initReader(self.inputFP) # init video reader

        elif objName.startswith("hideMarker"):
            key = objName.replace("_chk", "")
            self.debugging[key] = objVal
            self.onButtonPressDown(None, objName="draw_btn") # re-draw
        
        elif objName == "maskWallArea_chk":
            for i in range(2): 
                txt = wx.FindWindowByName("wallRect%i_txt"%(i), 
                                          self.panel["ml"])
                txt.Enable(objVal)

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
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "cho")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "eCase_cho":
            self.config("save_eCase") # save config of this eCase
            ## store the chosen experiment-case 
            self.eCase = objVal.split(":")[0].strip()
            self.initMLWidgets() # set up middle left panel
            if self.pgd != None:
                self.pgd.interactiveDrawing = {} # delete interactive drawing
            self.config("load_eCase") # load configs for this eCase

        elif objName == "process_cho":
        # process choice was made 

            # this part is only for AOS for now (2024-03-28)
            if self.eCase != "aos": return

            w = wx.FindWindowByName("dataPtIntv_cho", self.panel["ml"])
            if objVal in ["intensity", "proxMCluster"]:
            # default bundling interval is 300 
                widgetValue(w, "300", "set")
            elif objVal in ["saMVec", "dist2c", "dist2b", "dist2ca"]:
            # it's a process to set data-bundle-interval to one
                widgetValue(w, "1", "set")
            else: # otherwise
                # the data-bundle-interval is better to be higher than one
                #   (especially for 'intensity', one second bundling creates 
                #    an artifact of a periodical (20 min. or so) activity 
                #    pattern, probably due to a slight time lag at each data 
                #    recording in Raspbbery Pi.)
                val = widgetValue(w)
                if val == "1":
                    widgetValue(w, "2", "set")
            self.panel["ml"].Refresh()

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
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "sld")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName.startswith("c-"): # HSV color slider
            ### update chosen HSV color on the corresponding panel
            __, ck, hsvL, mLbl = objName.rstrip("_sld").split("-")
            val = []
            for hsvLbl in ["H", "S", "V"]:
                objName = "c-%s-%s-%s_sld"%(ck, hsvLbl, mLbl)
                obj = wx.FindWindowByName(objName, self.panel["ml"])
                val.append(obj.GetValue())
            rgbVal = cvHSV2RGB(val[0], val[1], val[2])
            objName = "c-%s-%s_panel"%(ck, mLbl)
            obj = wx.FindWindowByName(objName, self.panel["ml"])
            obj.SetBackgroundColour(rgbVal)
            obj.Refresh()

        elif objName.startswith("nav_sld"): # navigation slider
            if self.pgd == None: return
            self.pgd.procUI("navSlider", (objVal, -1))
            self.panel["mp"].Refresh()

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
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
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
            allowed += [ord("."), ord("-")]
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
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        ret = preProcUIEvt(self, event, objName, "txt")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "zoom_txt": 
            ### update ratio of displayed image to original image
            gi = self.pgd.graphImgIdx
            try:
                rat = float(objVal) # new ratio
            except:
                _rat = self.pgd.graphImg[gi]["ratDisp2OrigImg"]
                # set text value back to the current zoom ratio 
                self.zoom_txt.SetValue(str(_rat))
                return
            self.pgd.graphImg[gi]["ratDisp2OrigImg"] = rat 
            self.pgd.zoomNStore() # zoom the graph image and store it
            self.panel["mp"].Refresh()

        elif objName == "offset_txt": 
            self.setGraphOffset(objVal)

    #---------------------------------------------------------------------------
    
    def onClose(self, event):
        """ Close this frame. 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        #self.makeModal(False)
        self.config("save") # save config
        stopAllTimers(self.timer)
        # delete original graph images saved as temporary files
        if self.pgd != None: self.pgd.removeGraph()
        self.Destroy()
    
    #---------------------------------------------------------------------------

    def openInputF(self, event):
        """ Open input file or folder
        
        Args:
            event (wx.Event)
        
        Returns:
            None 
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.eCase == "cremerGroupApp": return

        fType, ext = self.inputType[self.eCase] # input file/folder type

        _dir = path.join(FPATH, "data")
        if path.isdir(_dir): defDir = _dir
        else: defDir = FPATH

        ### choose input file or directory 
        if fType == "file":
            t = "Open %s file"%(ext.upper())
            wc = "%s files (*.%s)|*.%s"%(ext.upper(), ext, ext)
            style = (wx.FD_OPEN|wx.FD_FILE_MUST_EXIST) 
            dlg = wx.FileDialog(self, t, defDir, wildcard=wc, style=style)
        elif fType == "dir":
            t = "Choose directory for analysis"
            style = (wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST) 
            dlg = wx.DirDialog(self, t, defDir, style=style)
        if dlg.ShowModal() == wx.ID_CANCEL: return
        inputFP = dlg.GetPath()
        dlg.Destroy() 
         
        self.initMRWidgets() 
        
        ### display input file path
        self.inputFP = inputFP
        obj = wx.FindWindowByName("inputFP_txt", self.panel["tp"])
        obj.SetValue(path.basename(inputFP))

        self.zoom_txt.SetValue("1.0")
        self.offset_txt.SetValue("0, 0")

        # delete original graph images saved as temporary files
        if self.pgd != None: self.pgd.removeGraph()

        self.pgd = ProcGraphData(self) # init instance class
                                       # for processing graph data 
        flagInitOnDataLoading = False
        _args = (self.q2m,)
        try:
            if self.eCase == "L2020": 
                self.vRW = VideoRW(self) # for reading/writing video file
                self.vRW.initReader(inputFP) # init video reader
                ### resize graph panel
                f = self.vRW.currFrame
                r = calcI2DRatio(f, self.pi["mp"]["sz"])
                sz = (int(f.shape[1]*r), int(f.shape[0]*r))
                self.panel["mp"].SetSize(sz)
                self.pi["mp"]["sz"] = sz 
                self.pi["bm"]["sz"] = (sz[0], self.pi["bm"]["sz"][1])
                self.panel["bm"].SetSize(self.pi["bm"]["sz"])
                self.gbs["bm"].Layout()
                ### update resolution 
                obj = wx.FindWindowByName("imgSavResW_txt",
                                          self.panel["bm"])
                obj.SetValue(str(f.shape[1]))
                obj = wx.FindWindowByName("imgSavResH_txt",
                                          self.panel["bm"])
                obj.SetValue(str(f.shape[0]))
                ### update frames in the widgets
                obj = wx.FindWindowByName("startFrame_txt", 
                                          self.panel["ml"])
                obj.SetValue('0')
                obj = wx.FindWindowByName("endFrame_txt", self.panel["ml"])
                obj.SetValue(str(self.vRW.nFrames-1))
                obj = wx.FindWindowByName("frameIntv_txt", self.panel["ml"])
                obj.SetValue(str(self.vRW.nFrames))
                ### update ROI
                self.roi = [675, 0, 300, 1024] 
                obj = wx.FindWindowByName("roi_txt", self.panel["ml"])
                obj.SetValue(str(self.roi).strip("[]"))
                # init some variables in ProcGraphData
                self.pgd.initOnDataLoading()

            elif self.eCase == "anVid":
                if not 'ProcAnVidRslt' in sys.modules:
                    from procAnVid import ProcAnVidRslt
                self.pgd.subClass = ProcAnVidRslt(self, self.pgd) 
                flagInitOnDataLoading = True 
 
            elif self.eCase == "aos":
                if not 'ProcAntOS' in sys.modules:
                    from procAntOS import ProcAntOS
                self.pgd.subClass = ProcAntOS(self, self.pgd) 
                flagInitOnDataLoading = True  
            
            elif self.eCase == "aosSec":
                if not 'ProcAntOSSec' in sys.modules:
                    from procAntOSSec import ProcAntOSSec
                self.pgd.subClass = ProcAntOSSec(self, self.pgd)  
            
            elif self.eCase == "aosI":
                startTaskThread(self, "drawGraph", self.pgd.graphAOSI, _args) 

            elif self.eCase == "V2020":
                self.pgd.initOnDataLoading()
                self.pgd.graphV2020()
                self.panel["mp"].Refresh() # display graph 
            
            elif self.eCase == "L2020CSV1":
                csvFP1 = inputFP.replace("_0.csv", "_1.csv") # CSV file with
                  # data of centroid, rect, and so forth.. 
                if not path.isfile(csvFP1):
                    msg = "CSV file with data of centroid, rect, etc"
                    msg += " is not found\n\n"
                    msg += "The below file should be present.\n"
                    msg += "%s\n"%(csvFP1)
                    wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                    return
                tF = self.pgd.initOnDataLoading
                startTaskThread(self, "initOnDataLoading", tF, _args)

            elif self.eCase == "L2020CSV2":
                fn = path.basename(inputFP)
                videoFN = fn.split("_raw_")[0] + ".mp4"
                videoFP = inputFP.replace(fn, videoFN) 
                if not path.isfile(videoFP):
                    msg = "Video file not found\n\n"
                    msg += "The below file should be present in the same folder"
                    msg += " where the CSV file is.\n"
                    msg += "%s\n"%(videoFP)
                    wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
                    return
                self.vRW = VideoRW(self)
                self.vRW.initReader(videoFP)
                tF = self.pgd.graphL2020CSV2
                startTaskThread(self, "drawGraph", tF, _args)

            if flagInitOnDataLoading: 
                tF = self.pgd.subClass.initOnDataLoading
                startTaskThread(self, "initOnDataLoading", tF, (self.q2m,))
        
        except Exception as e: # failed
            self.inputFP = ""
            self.pgd.csvFP = ""
            msg = "Failed to load CSV data\n"
            msg += str(traceback.format_exc())
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
 
        self.panel["mp"].Refresh() # draw graph
         
    #---------------------------------------------------------------------------
    
    def onPaintMP(self, event):
        """ painting graph

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.inputFP == "": return

        event.Skip()
        
        dc = wx.PaintDC(self.panel["mp"])
        dc.Clear()
        if self.pgd.graphImgIdx == -1: return

        ### draw underlying grid
        nGrid = 10
        mpSz = self.pi["mp"]["sz"]
        gridSz = [round(mpSz[0]/nGrid), round(mpSz[1]/nGrid)]
        lines = []
        for i in range(2): # horizontal & vertical
            if i == 0:
                x1 = 0; x2 = mpSz[0]
                y1 = y2 = 0 
            elif i == 1:
                x1 = x2 = 0 
                y1 = 0; y2 = mpSz[1]
            for gri in range(1, nGrid):
                if i == 0: y1 = y2 = y1 + gridSz[1]
                elif i == 1: x1 = x2 = x1 + gridSz[0]
                lines.append((x1, y1, x2, y2))
        dc.DrawLineList(lines, wx.Pen((50,50,50), 1))
        
        ### draw the generated graph image
        gImg = self.pgd.graphImg[self.pgd.graphImgIdx]
        bmp = gImg["img"].ConvertToBitmap()
        dc.DrawBitmap(bmp, gImg["offset"][0], gImg["offset"][1])
        
        ### draw the additional image, if it exists
        if "additionalImg" in gImg.keys():
            bmp = gImg["additionalImg"].ConvertToBitmap()
            dc.DrawBitmap(bmp, 0, 0)
       
        ### draw interactive parts
        if self.pgd.interactiveDrawing != {}:
            iDraw = self.pgd.interactiveDrawing
            for k in iDraw.keys():
                p = iDraw[k]
                if k == "drawText":
                    dc.SetTextForeground((0,0,0))
                    dc.DrawTextList(p["textList"], p["coords"], 
                                    p["foregrounds"])

                elif k == "drawLine":
                    dc.DrawLineList(p["lines"], p["pens"])

                elif k == "drawCircle":
                    for i in range(len(p)):
                        dc.SetPen(p[i]["pen"]) 
                        dc.SetBrush(p[i]["brush"]) 
                        dc.DrawCircle(p[i]["x"], p[i]["y"], p[i]["r"])

                elif k == "drawRectangle":
                    for i in range(len(p)):
                        dc.SetPen(p[i]["pen"]) 
                        dc.SetBrush(p[i]["brush"]) 
                        dc.DrawRectangle(p[i]["x1"], p[i]["y1"], 
                                         p[i]["x2"]-p[i]["x1"], 
                                         p[i]["y2"]-p[i]["y1"])

                elif k == "drawBitmap":
                    for i in range(len(p)):
                        dc.DrawBitmap(p[i]["bitmap"], p[i]["x"], p[i]["y"]) 

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
        
        elif rData[0] == "aborted":
            postProcTaskThread(self, flag)
            wx.MessageBox(rData[1], "Info.", wx.OK|wx.ICON_INFORMATION)
        
        elif rData[0].startswith("finished"):
            self.callback(rData, flag)
   
    #---------------------------------------------------------------------------
    
    def onMLBDown(self, event):
        """ Processing when mouse L-buttton pressed down 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or self.inputFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition() # mouse pointer position
        mState = wx.GetMouseState()

        flagPlaySnd = False

        if mState.ControlDown(): # Ctrl key is pressed
            gImg = self.pgd.graphImg[self.pgd.graphImgIdx]
            self.prevOffset = gImg["offset"]
            self.panel[pk].mousePressedPt = mp

        else:
            if self.eCase in ["aos", "anVid"]:
                if len(self.flags["drawingROI"]) > 0: 
                    self.panel[pk].mousePressedPt = mp
                elif len(self.flags["selectPt"]) > 0:
                    ### update the value in textCtrl
                    x, y = mp 
                    gImg = self.pgd.graphImg[self.pgd.graphImgIdx]
                    x = int((x - gImg["offset"][0]) / gImg["ratDisp2OrigImg"]) 
                    y = int((y - gImg["offset"][1]) / gImg["ratDisp2OrigImg"])
                    k = list(self.flags["selectPt"].keys())[0]
                    w = wx.FindWindowByName("%sPt_txt"%(k), self.panel["ml"])
                    w.SetValue("%i, %i"%(x, y))
                    self.flags["selectPt"] = {} # remove the flag
                    ### set bitmap image to default
                    w = wx.FindWindowByName("%sPt_btn"%(k), self.panel["ml"])
                    set_img_for_btn(self.targetBtnImgPath, w)
                    ### init image
                    cho = wx.FindWindowByName("process_cho", self.panel["ml"])
                    cho.SetSelection(0)
                    startTaskThread(self,
                                    "drawGraph", 
                                    self.pgd.subClass.drawGraph,
                                    args=(self.q2m,))

        if flagPlaySnd: wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))

    #---------------------------------------------------------------------------
    
    def onMLBUp(self, event):
        """ Processing when mouse L-buttton clicked 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or self.inputFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition()
        mState = wx.GetMouseState()

        flagPlaySnd = False

        if pk == "mp":
            if self.colorPick != "" and not mState.ControlDown():
            # color pick button is pressed and ctrl key is not pressed
                ck, mLbl = self.colorPick.split("-")
                ### get color value of the clicked pixel
                gi = self.pgd.graphImgIdx
                img = self.pgd.graphImg[gi]["img"]
                offset = self.pgd.graphImg[gi]["offset"]
                x = mp[0] - offset[0]
                y = mp[1] - offset[1]
                r = img.GetRed(x, y)
                g = img.GetGreen(x, y)
                b = img.GetBlue(x, y)
                col = {}
                # convert it to HSV value
                col["H"], col["S"], col["V"] = rgb2cvHSV(r, g, b)
                ### set slider bar to the HSV value
                for k in col.keys():
                    sN = "c-%s-%s-%s_sld"%(ck, k, mLbl.capitalize())
                    sld = wx.FindWindowByName(sN, self.panel["ml"])
                    sld.SetValue(col[k])
                # call function for slider value change
                self.onSlider(None, objName=sN)
                ### init
                cursor = wx.Cursor(wx.CURSOR_ARROW)
                self.panel["mp"].SetCursor(cursor)
                bN = "pipette-%s_btn"%(self.colorPick)
                btn = wx.FindWindowByName(bN, self.panel["ml"])
                btn.SetBackgroundColour("#333333")
                self.colorPick = ""
                flagPlaySnd = True
            
            if self.eCase == "aos":
                if len(self.flags["drawingROI"]) > 0:
                    ### update the rect value in textCtrl
                    x, y = self.panel[pk].mousePressedPt
                    w = abs(mp[0]-x)
                    h = abs(mp[1]-y)
                    gImg = self.pgd.graphImg[self.pgd.graphImgIdx]
                    x = int((x - gImg["offset"][0]) / gImg["ratDisp2OrigImg"]) 
                    y = int((y - gImg["offset"][1]) / gImg["ratDisp2OrigImg"])
                    w = int(w / gImg["ratDisp2OrigImg"])
                    h = int(h / gImg["ratDisp2OrigImg"])
                    k = list(self.flags["drawingROI"].keys())[0]
                    txt = wx.FindWindowByName("%sROI_txt"%(k), self.panel["ml"])
                    txt.SetValue("%i, %i, %i, %i"%(x, y, w, h))
                    # remove the rect drawing
                    del(self.pgd.interactiveDrawing["drawRectangle"])
                    self.flags["drawingROI"] = {} # remove the flag
                    self.pgd.interactiveDrawing = {} # clear interactive drawing
                    self.panel["mp"].Refresh()
                    ### set bitmap image to default
                    w = wx.FindWindowByName("%sROI_btn"%(k), self.panel["ml"])
                    set_img_for_btn(self.targetBtnImgPath, w)
                    ### init image
                    cho = wx.FindWindowByName("process_cho", self.panel["ml"])
                    cho.SetSelection(0)
                    startTaskThread(self, 
                                    "drawGraph", 
                                    self.pgd.subClass.drawGraph,
                                    (self.q2m,))

                    flagPlaySnd = True
                    
        if flagPlaySnd: wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav"))

        self.panel[pk].mousePressedPt = [-1, -1] # init mouse pressed point
        
    #---------------------------------------------------------------------------
    
    def onMMove(self, event):
        """ Mouse pointer moving 
        Show some info

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or self.inputFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition()
        mState = wx.GetMouseState()

        if pk == "mp": # mouse moved in "mp" panel
            if mState.ControlDown(): # Ctrl key is pressed
                if self.panel["mp"].mousePressedPt != [-1, -1]:
                    ### set graph drawing offset
                    offset = list(self.prevOffset)
                    pPt = self.panel["mp"].mousePressedPt
                    offset[0] += (mp[0] - pPt[0])
                    offset[1] += (mp[1] - pPt[1])
                    self.setGraphOffset(offset) 
            else:
                self.pgd.procUI("mouseMove", mp)
                self.panel["mp"].Refresh()
     
    #---------------------------------------------------------------------------
    
    def onMRBUp(self, event):
        """ Mouse right click

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.flags["blockUI"] or self.inputFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mp = event.GetPosition()

        flagPlaySnd = False

        if pk == "mp": # mouse right button clicked in "ml" panel
            if self.eCase == "L2020CSV2":
                self.vRW.getFrame(self.pgd.fIdx, useCAPPROP=True)
                self.pgd.procUI("rightClick", mp)
                self.panel["mp"].Refresh()
                flagPlaySnd = True

            elif self.eCase == "aos":
                if len(self.flags["selectPt"]) > 0:
                # if it was setting a point-of-interest,
                # set it back to [-1, -1]
                    k = list(self.flags["selectPt"].keys())[0]
                    w = wx.FindWindowByName("%sPt_txt"%(k), self.panel["ml"])
                    # delete the point (back to [-1, -1])
                    w.SetValue("-1, -1")
                    self.flags["selectPt"] = {} # remove the flag
                    ### set bitmap image to default
                    w = wx.FindWindowByName("%sPt_btn"%(k), self.panel["ml"])
                    set_img_for_btn(self.targetBtnImgPath, w)
                    ### init image
                    cho = wx.FindWindowByName("process_cho", self.panel["ml"])
                    cho.SetSelection(0)
                    startTaskThread(self,
                                    "drawGraph", 
                                    self.pgd.subClass.drawGraph,
                                    args=(self.q2m,))

        if flagPlaySnd: wxSndPlay(path.join(P_DIR, "sound", "snd_rClick.wav"))

    #---------------------------------------------------------------------------
    
    def onMWheel(self, event):
        """ Mouse wheel event

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        
        if self.flags["blockUI"] or self.inputFP == "": return

        pk = event.GetEventObject().panelKey # panel key
        mWhRot = event.GetWheelRotation()
        mState = wx.GetMouseState()
        if mState.ControlDown():
        # Ctrl + mouse-wheel event
            if pk == "mp": # event in "mp" panel, where graph is shown
                ratChange = 0.05 * (mWhRot/abs(mWhRot))
                gi = self.pgd.graphImgIdx
                self.pgd.graphImg[gi]["ratDisp2OrigImg"] += ratChange
                rat = self.pgd.graphImg[gi]["ratDisp2OrigImg"]
                self.zoom_txt.SetValue(str(rat))
                self.pgd.zoomNStore() # zoom the graph image and store it
                self.panel["mp"].Refresh() # re-draw the graph

        elif mState.AltDown():
        # Alt + mouse-wheel event
            if mWhRot/abs(mWhRot) < 0: direction = "to_left_10"
            else: direction = "to_right_10"
            self.setGraphOffset(direction) 

    #---------------------------------------------------------------------------

    def onKeyPress(self, event):
        """ Process key-press event
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        event.Skip()
        
        kc = event.GetKeyCode()
        mState = wx.GetMouseState()

        if mState.ControlDown():
        # CTRL modifier key is pressed  
            if kc == ord("S"): # save a graph 
                self.save()

            elif kc == ord("R"): # save a raw data 
                self.save(savType="raw")
            
            elif kc == ord("D"): # delete a graph 
                if self.pgd != None: self.pgd.removeGraph(self.pgd.graphImgIdx)

            '''
            elif kc == ord("Q"): # close the app
                self.onClose(None)
            
            elif kc == ord("O"): # open a file/directory
                self.openInputF(None)
            '''

        elif mState.ShiftDown():
        # SHIFT modifier key is pressed
            if kc == ord("S"): # save all graphs 
                self.save(isSavingAll=True) 

            elif kc == ord("R"): # save all raw data 
                self.save(savType="raw", isSavingAll=True)

            elif kc == ord("D"): # delete all graphs
                if self.pgd != None: self.pgd.removeGraph()

        elif mState.AltDown():
        # ALT modifier key is pressed
            
            if kc in [wx.WXK_LEFT, wx.WXK_RIGHT, ord("J"), ord("L")]:
                if kc in [wx.WXK_LEFT, ord("J")]: 
                    self.setGraphOffset("to_left") 
                elif kc in [wx.WXK_RIGHT, ord("L")]: 
                    self.setGraphOffset("to_right") 

    #---------------------------------------------------------------------------
    
    def highlightThumbnail(self, thIdx=-1):
        """ Highlight the thumbnail image with the given index

        Args:
            thIdx (int): Index of thumbnail image

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if thIdx != -1: # index to highlight is given
            ### restore (de-highlight) the previously selected thumbnail image
            pIdx = self.pgd.graphImgIdx
            if pIdx != -1:
                bmp = self.pgd.graphImg[pIdx]["thumbnail"].ConvertToBitmap()
                obj = wx.FindWindowByName("thumbnail_%i"%(pIdx), 
                                          self.panel["mr"])
                obj.SetBitmap(bmp)
        else: # index is not given, highlight with the current graph-image-index
            thIdx = self.pgd.graphImgIdx

        graphImg = self.pgd.graphImg[thIdx]

        ### highlight (drawing yellow border) the given thumbnail image 
        thumbnail = wx.FindWindowByName("thumbnail_%i"%(thIdx), 
                                        self.panel["mr"])
        bmp = graphImg["thumbnail"].ConvertToBitmap()
        dc = wx.MemoryDC(bmp)
        w, h = dc.GetSize()
        dc.SetPen(wx.Pen((255,255,0), 5)) 
        dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
        dc.DrawRectangle(0, 0, w, h)
        del dc
        thumbnail.SetBitmap(bmp)

        ### update zoom ratio
        _txt = graphImg["ratDisp2OrigImg"]
        self.zoom_txt.SetValue(str(_txt))
        _txt = graphImg["offset"]
        self.offset_txt.SetValue(str(_txt))
        
        ### update image size
        w = wx.FindWindowByName("imgSavResW_txt", self.panel["bm"])
        w.SetValue(str(graphImg["imgSz"][0]))
        w = wx.FindWindowByName("imgSavResH_txt", self.panel["bm"])
        w.SetValue(str(graphImg["imgSz"][1]))

    #---------------------------------------------------------------------------
    
    def onClickThumbnail(self, event):
        """ process mouse-click on thumbnail image (wx.StaticBitmap)

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.flags["blockUI"] or self.inputFP == "": return

        wxSndPlay(path.join(P_DIR, "sound", "snd_click.wav")) 

        ### highlight the currently selected thumbnail
        obj = event.GetEventObject()
        self.highlightThumbnail(obj.index) 

        self.pgd.graphImgIdx = obj.index # store the selected thumbnail index
        self.pgd.interactiveDrawing = {} # clear interactive drawing
        self.panel["mp"].Refresh() # re-draw the graph

    #---------------------------------------------------------------------------
    
    def updateMRWid(self):
        """ update widget (for thumbnail images) in middle-right panel
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        self.initMRWidgets()
        for idx, gi in enumerate(self.pgd.graphImg.keys()):
            bmp = self.pgd.graphImg[gi]["thumbnail"].ConvertToBitmap()
            name = "thumbnail_%i"%(gi)
            sBmp = wx.StaticBitmap(self.panel["mr"], -1, bmp, name=name)
            sBmp.Bind(wx.EVT_LEFT_UP, self.onClickThumbnail)
            sBmp.index = gi 
            add2gbs(self.gbs["mr"], sBmp, (int(idx/2),idx%2), (1,1))
            self.mrWid.append(sBmp)
        self.gbs["mr"].Layout()
        self.panel["mr"].SetupScrolling()
        if self.pgd.graphImgIdx != -1: self.highlightThumbnail() 

    #---------------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread
        
        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        addData = rData[-1] # additional data

        try: msg = addData["retMsg"]
        except: msg = "" 
        if msg != "":
            if msg.startswith("ERROR"):
                title = "Error"
                icon = wx.ICON_ERROR
            else:
                title = "Info."
                icon = wx.ICON_INFORMATION
            wx.MessageBox(addData["retMsg"], title, wx.OK|icon)

        if flag == "drawGraph": 

            try: additionalImg = addData["additionalImg"] 
            except: additionalImg = None

            # store graph image
            self.pgd.storeGraphImg(rData[1], self.pi["mp"]["sz"], additionalImg)
            gi = self.pgd.graphImgIdx 

            if len(rData) > 2 and rData[2] != None:
                # store raw data
                self.pgd.storeGraphData(rData[2])
            
            if self.eCase == "L2020":
                ### store start and end frame
                self.pgd.graphImg[gi]["startFrame"] = rData[3] 
                self.pgd.graphImg[gi]["endFrame"] = rData[4] 

                if self.vRW.fi < self.endFI:
                # heatmap generation not finished yet
                    ### start another heatmap generation
                    startFrame = self.vRW.fi+1
                    nextEndFrame = min(startFrame+self.frameIntv-1, self.endFI)
                    args = (startFrame, nextEndFrame, self.q2m,)
                    wx.CallLater(5, startTaskThread, self, "drawGraph", 
                                 self.pgd.graphL2020, args)

            elif self.eCase == "L2020CSV2":
                self.pgd.graphImg[gi]["valueData"] = rData[3]

            elif self.eCase in ["aos", "anVid"]:
                if self.eCase == "aos":
                    ### store cam index
                    cho = wx.FindWindowByName("camIdx_cho", self.panel["ml"])
                    try: camIdx = int(cho.GetString(cho.GetSelection()))
                    except: camIdx = 99
                    self.pgd.graphImg[gi]["camIdx"] = camIdx 

                ### store process name 
                w = wx.FindWindowByName("process_cho", self.panel["ml"])
                proc = w.GetString(w.GetSelection())
                self.pgd.graphImg[gi]["proc"] = proc

                if self.eCase == "aos":
                    ### if this process was 'initImg', delete other 'initImg'
                    if proc == "initImg": 
                        for _gi in list(self.pgd.graphImg.keys()):
                            if _gi == gi: continue
                            if self.pgd.graphImg[_gi]["proc"] == "initImg": 
                                self.pgd.removeGraph(_gi)
                self.pgd.graphImgIdx = gi

                ### store additional graph info
                gInfo = rData[3]
                for key in gInfo.keys():
                    self.pgd.graphImg[gi][key] = gInfo[key] 

                v = self.pgd.subClass.v
                if gInfo["gEIdx"] == -1 and "fnKIdx" in v.keys():
                # reached the end of data; there's a 'filename key index'
                    v["fnKIdx"] += 1 # increase filename key index
                    if v["fnKIdx"] < len(v["data"]):
                    # there're more data files to draw
                        # draw another graph for the next file 
                        wx.CallLater(5, startTaskThread, self, "drawGraph", 
                                     self.pgd.subClass.drawGraph,
                                     args=(self.q2m,))
                    else:
                    # reached the end of data files
                        self.pgd.subClass.v["fnKIdx"] = 0 # init. index

                if gInfo["gEIdx"] != -1:
                # didn't reach the end of data
                    next_gSIdx = gInfo["gEIdx"] + 1
                    # start another graph generation
                    wx.CallLater(5, startTaskThread, self, "drawGraph", 
                                 self.pgd.subClass.drawGraph,
                                 args=(self.q2m, next_gSIdx,))
                
            self.updateMRWid() 
 
        elif flag == "drawGraph4sav":
                rData[1].SelectObject(wx.NullBitmap)
                img = self.bmp4sav.ConvertToImage()
                img.SaveFile(self.imgFP4sav, wx.BITMAP_TYPE_PNG)
                msg = 'Saved\n'
                msg += path.basename(self.imgFP4sav) + "\n"
                msg += " in output folder."
                showStatusBarMsg(self, msg, 3000)
                del self.bmp4sav
                del self.imgFP4sav
        
        elif flag == "initOnDataLoading":
            if self.eCase == "L2020CSV1":
                self.pgd.hmArr = rData[1]
                info = rData[2]
                self.pgd.info = info
                for i, perc in enumerate(self.percLst):
                    for mLbl in ["min", "max"]:
                        obj = wx.FindWindowByName("hmRng-%i-%s_txt"%(i, mLbl), 
                                                  self.panel["ml"])
                        obj.SetValue(str(info["percVal"][mLbl][i]))
                self.pgd.graphL2020CSV1() 

            elif self.eCase in ["aos", "anVid"]:
                rD = rData[1]
                ### store returned data from init process 
                for k in rD.keys():
                    self.pgd.subClass.v[k] = rD[k]

                w = wx.FindWindowByName("process_cho", self.panel["ml"])
                w.SetSelection(0) # initImg
                startTaskThread(self, "drawGraph", self.pgd.subClass.drawGraph,
                                args=(self.q2m,))

                if self.eCase == "anVid":
                    for fnK in rD["fImg"].keys():
                        fH, fW = rD["fImg"][fnK].shape[:2]
                        for i in range(2): # for 2 ROIs
                            roiK = f'roi{i:02d}'
                            (x1, y1), (x2, y2) = rD["tunnel"][fnK][roiK]
                            x = x1 + int((x2-x1)/2)
                            y = y1 + int((y2-y1)/2)
                            if "before" in fnK: wn = f'tunnelBefore{i}Pt_txt'
                            else: wn = f'tunnelAfter{i}Pt_txt'
                            w = wx.FindWindowByName(wn, self.panel["ml"])
                            w.SetValue(f'{x}, {y}')

            '''
            elif self.eCase == "aos":
                ### store returned data from init process 
                self.pgd.subClass.camIdx = rData[1]["camIdx"]
                self.pgd.subClass.camIdx4i = rData[1]["camIdx4i"]
                self.pgd.subClass.imgFiles = rData[1]["imgFiles"]
                self.pgd.subClass.data = rData[1]["data"]
                self.pgd.subClass.ci = rData[1]["ci"]
                self.pgd.subClass.keys = rData[1]["keys"]
                self.pgd.subClass.temperature = rData[1]["temperature"]

                cho = wx.FindWindowByName("process_cho", self.panel["ml"])
                cho.SetSelection(0)
                startTaskThread(self,
                                "drawGraph", 
                                self.pgd.subClass.drawGraph,
                                args=(self.q2m,))
            '''
            
            showStatusBarMsg(self, "", -1)
        
        elif flag == "readFrames":
            if self.eCase == "L2020":
                ### reached the frame in video module, start heatmap generation
                nextEndFrame = min(self.startFI+self.frameIntv-1, self.endFI)
                args = (self.startFI, nextEndFrame, self.q2m,)
                startTaskThread(self, "drawGraph", self.pgd.graphL2020, 
                               args=args)
        
        postProcTaskThread(self, flag)
        self.panel["mp"].Refresh() # display graph 
        self.panel["mr"].Refresh() # display thumbnail images
        self.panel["mr"].SetFocus()
    
    #---------------------------------------------------------------------------
    
    def save(self, savType="graph", isSavingAll=False):
        """ Save data 

        Args:
            savType (str): graph; saving graph, raw: saving raw data
            isSavingAll (bool): saving the current graph or all graph

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        if self.inputFP == "" or \
          (savType == "raw" and self.eCase in self.eCase_noRawData):
            return

        def saveImg(imgFP, fp4sav): # get image to save 
            img = cv2.imread(imgFP)
            ### resize if required 
            obj = wx.FindWindowByName("imgSavResW_txt", self.panel["bm"])
            w = int(obj.GetValue())
            obj = wx.FindWindowByName("imgSavResH_txt", self.panel["bm"])
            h = int(obj.GetValue())
            if w != img.shape[1] or h != img.shape[0]:
                img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)
            # save
            cv2.imwrite(fp4sav, img)
       
        pgd = self.pgd
        msg = ""
        if isSavingAll: idxRng = pgd.graphImgIdxLst 
        else: idxRng = [pgd.graphImgIdx]
        msg = "Saved\n"
        for idx in idxRng:
            ### determine file path to write
            # get path & input file (folder) name
            oPath, fn = path.split(self.inputFP) 
            ext = "." + fn.split(".")[-1]
            if savType == "graph": sExt = ".png"
            elif savType == "raw":
                if self.eCase == "L2020": sExt = ".csv"
                else: sExt = ".npy"
            data = pgd.graphImg[idx] 
            
            ### determine file path to save
            if self.eCase == "L2020":
                newTxt = "_%s_%s"%(savType, data["startFrame"])
                newTxt += "_%s%s"%(data["endFrame"], sExt) 
                newFN = fn.replace(ext, newTxt)
                fp4sav = self.inputFP.replace(fn, newFN)
            
            elif self.eCase == "aos":
                w = wx.FindWindowByName("fnPrefix_txt", self.panel["ml"])
                prefixTxt = w.GetValue().strip()
                if prefixTxt != "":
                    fn = prefixTxt+"_"+fn # put prefix to the filename

                newTxt = "_%s"%(data["proc"])
                if data["timeLbl"] != "":
                    newTxt += "_%s"%(data["timeLbl"]).replace(" ","")
                w = wx.FindWindowByName("savCamIdx_chk", self.panel["ml"])
                if w.GetValue():
                    newTxt += "_cam%i"%(data["camIdx"])
                if data["dPtIntvSec"] is not None: 
                    newTxt += f'_intv{data["dPtIntvSec"]}' # add bundle-interval
                w = wx.FindWindowByName("fnSuffix_txt", self.panel["ml"])
                suffixTxt = w.GetValue().strip()
                if suffixTxt != "":
                    newTxt += "_" + suffixTxt # put suffix to the filename
                newTxt += sExt # put save file extension
                if path.isdir(self.inputFP): newFN = fn + newTxt
                else: newFN = fn.replace(ext, newTxt)
                fp4sav = path.join(oPath, newFN)
            
            elif self.eCase == "anVid":
                fp4sav = pgd.subClass.getSavFilePath(idx, sExt)
            
            else:
                if path.isdir(self.inputFP): 
                    fp4sav = self.inputFP + f"{idx:02d}{sExt}"
                else:
                    ext = "." + fn.split(".")[-1]
                    fp4sav = self.inputFP.replace(ext, f"{idx:02d}{sExt}")

            if savType == "graph":
            # save graph image
                imgFP = "tmp_origImg%i.png"%(idx)
                saveImg(imgFP, fp4sav)
                msg += fp4sav + "\n\n"
            elif savType == "raw":
            # save raw data
                for fp in glob("tmp_data*%s"%(sExt)):
                    tmp = fp.replace(sExt, "").split("_")
                    if tmp[2] != str(idx): # graph index does not match
                        continue # ignore this file
                    newTxt = ""
                    for i in range(3, len(tmp)): newTxt += f'_{tmp[i]}'
                    newFP = fp4sav.replace(sExt, f"{newTxt}{sExt}")
                    ### if ths sub-class has save-file-path function, 
                    ###   call it to verify
                    getFP = getattr(pgd.subClass, "getSavFilePath", None)
                    if callable(getFP):
                        newFP = getFP(idx, sExt, newFP)
                    if newFP is not None:
                        copyfile(fp, newFP)
                        msg += newFP + "\n\n"
       
        if msg != "": wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
   
    #---------------------------------------------------------------------------

    def setGraphOffset(self, v):
        """ Set offset of the graph 
        
        Args:
            v (list/str): offset values for x & y coordinates
        
        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        gi = self.pgd.graphImgIdx
        
        if type(v) == str:

            prevV = self.pgd.graphImg[gi]["offset"]
            prevVStr = str(prevV).strip("[]")

            if v.startswith("to_left"):
                vItems = v.split("_")
                if len(vItems) == 3:
                    newX = prevV[0] + int(vItems[-1]) 
                else:
                    newX = prevV[0] + self.pi["mp"]["sz"][0]
                v = [newX, prevV[1]]

            elif v.startswith("to_right"):
                vItems = v.split("_")
                if len(vItems) == 3:
                    newX = prevV[0] - int(vItems[-1]) 
                else:
                    newX = prevV[0] - self.pi["mp"]["sz"][0]
                v = [newX, prevV[1]]

            else:
            # this value is string from 'self.offset_txt'
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
        self.pgd.graphImg[gi]["offset"] = v 
        self.panel["mp"].Refresh()

    #---------------------------------------------------------------------------
    
    def saveVideo(self):
        """ Save video (with revised head direction line)
        
        Args: None
        
        Returns: None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        w = int(np.ceil(self.vRW.currFrame.shape[1]*self.ratFImgDispImg))
        h = int(np.ceil(self.vRW.currFrame.shape[0]*self.ratFImgDispImg))
        video_fSz = (w, h) # output video frame size
        timestamp = get_time_stamp().replace("_","")[:14]
        if self.vRW.vRecVideoCodec in ['avc1', 'h264']: ext = ".mp4"
        elif self.vRW.vRecVideoCodec == 'xvid': ext = ".avi"
        inputExt = "." + self.inputFP.split(".")[-1]
        self.savVidFP = self.inputFP.replace(inputExt, 
                                             "_rev_%s%s"%(timestamp, ext))
        self.vRW.initWriter(self.savVidFP, 
                            video_fSz, 
                            self.callback, 
                            self.makeDispImg,
                            self.bp_sTxt)
        self.flags["blockUI"] = True 
    
    #---------------------------------------------------------------------------

    def config(self, flag):
        """ saving/loading configuration of the current experiment-case 

        Args:
            flag (str): save, load, 
                        save_general, save_eCase, load_general, load_eCase

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        ### general config
        configFP = path.join(FPATH, f'config')
        if flag in ["save", "save_general"]:
            config = dict(eCase_cho = self.eCase)
            fh = open(configFP, "wb")
            pickle.dump(config, fh)
            fh.close()
        elif flag in ["load", "load_general"]:
            if path.isfile(configFP):
                fh = open(configFP, "rb")
                config = pickle.load(fh)
                fh.close()
                w = wx.FindWindowByName("eCase_cho", self.panel["tp"])
                val = None
                for i, ecchStr in enumerate(self.eCaseChoices):
                    if ecchStr.startswith(config["eCase_cho"]):
                        val = ecchStr
                        break
                if val is not None: w.SetSelection(w.FindString(val))
                self.eCase = config["eCase_cho"]
            self.initMLWidgets() # set up the left panel
       
        ##### [begin] eCase config -----
        configFP = path.join(FPATH, f'config_{self.eCase}')

        wids = {}
        '''
        wids["tp"] = []
        for wi, wn in enumerate(wids["tp"]):
            wids["tp"][wi] = wx.FindWindowByName(wn, self.panel["tp"])
        '''
        wids["ml"] = self.mlWid
        w2ignore = ["process_cho"]

        if flag in ["save", "save_eCase"]:
            config = {}
            for pk in wids.keys():
                config[pk] = {}
                for w in wids[pk]: # through widgets to save
                    wn = w.GetName()
                    if wn in w2ignore: continue
                    val = widgetValue(w) # get the widget value
                    if val != "": config[pk][wn] = val
            fh = open(configFP, "wb")
            pickle.dump(config, fh)
            fh.close()

        elif flag in ["load", "load_eCase"]:
            if path.isfile(configFP):
            # config file exists
                fh = open(configFP, "rb")
                config = pickle.load(fh)
                fh.close()
                for pk in config.keys():
                    for wn in config[pk].keys():
                        if wn in w2ignore: continue
                        w = wx.FindWindowByName(wn, self.panel[pk])
                        # set the widget value
                        widgetValue(w, config[pk][wn], "set") 
        ##### [end] eCase config -----

    #---------------------------------------------------------------------------
    
    def runScript(self, q2m, q2t, scriptTxt, csvTxt):
        """ [OBSOLETE] obsolete for now... use it later?
        Run script in UI to manipulate CSV values (as a thread).

        Args:
            q2m (queue.Queue): Queue to main thread
            q2t (queue.Queue): Queue to this thread
            scriptTxt (str): Python script text to edit CSV text 
            csvTxt (str): Original CSV text 
        
        Returns: None
        """ 
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        msg = "Running script..."
        q2m.put((msg,), True, None)
        
        lines = csvTxt.split("\n")
        try:
            exec(scriptTxt) # run script (which will change 'lines' of CSV)
        except:
            msg = "Finished"
            msg2 = "Script execution failed."
            q2m.put((msg,msg2,), True, None)
            return

        msg = "Update CSV ..."
        q2m.put((msg,), True, None)

        ### update CSV text
        csvTxt = ""
        for line in lines:
            csvTxt += line + "\n"
        csvTxt = csvTxt.rstrip("\n")

        msg = "Finished script running"
        msg2 = 'Script was executed.'
        q2m.put((msg,msg2,csvTxt,), True, None)

    #---------------------------------------------------------------------------

    def displayKeyMapping(self, event):
        """ Display help string 

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if FLAGS["debug"]: MyLogger.info(str(locals()))

        title = "Key mapping"
        msg = ""
        img = load_img("keyMapping.png")
        iSz = img.GetSize()
        sz = (self.wSz[0], self.wSz[1])
        rat = calcI2DRatio(iSz, sz)
        iSz = (int(iSz[0]*rat), int(iSz[1]*rat))
        img = img.Rescale(iSz[0], iSz[1], wx.IMAGE_QUALITY_HIGH)
        addW = [[{"type":"sBmp", "nCol":2, "bmp":wx.Bitmap(img), 
                  "border":10, "size":iSz}]]
        sz = (iSz[0]+20, sz[1])
        dlg = PopupDialog(self, -1, title, msg, size=sz, 
                          font=self.fonts[2], flagDefOK=True, addW=addW)
        dlg.ShowModal()
        dlg.Destroy()

    #---------------------------------------------------------------------------

#===============================================================================

class STC(wx.stc.StyledTextCtrl):
    def __init__(self, parent, pos, 
                 size, fgCol="#000000", bgCol="#cccccc", caretFGCol="#ffffff"):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        lineNumFGCol = bgCol 
        lineNumBGCol = fgCol

        wx.stc.StyledTextCtrl.__init__(self, 
                                       parent, 
                                       -1, 
                                       pos=pos, 
                                       size=size, 
                                       style=wx.SIMPLE_BORDER)
        self.StyleClearAll()
        self.SetViewWhiteSpace(wx.stc.STC_WS_VISIBLEALWAYS)
        self.SetIndentationGuides(1)
        self.SetViewEOL(0)
        self.SetIndent(4) 
        self.SetMarginType(0, wx.stc.STC_MARGIN_NUMBER)
        self.SetMarginWidth(0, self.TextWidth(0,'9999'))
        fontStr = "face:Monaco,fore:%s,back:%s,size:12"%(fgCol, bgCol)
        self.StyleSetSpec(wx.stc.STC_STYLE_DEFAULT, fontStr) 
        self.StyleSetSpec(wx.stc.STC_P_DEFAULT, fontStr) 
        fontStr = "face:Monaco,fore:%s,"%(lineNumFGCol)
        fontStr += "back:%s,size:12"%(lineNumBGCol)
        self.StyleSetSpec(wx.stc.STC_STYLE_LINENUMBER, fontStr) 
        self.SetCaretStyle(2)
        self.SetCaretForeground(caretFGCol)
        self.StyleSetBackground(wx.stc.STC_STYLE_DEFAULT, bgCol) 
        self.SetSTCCursor(0) # To make cursor as a pointing arrow. 
          # Can't find reference for STC_CURSOR, 
          # but '1' resulted in the text-input cursor.

#===============================================================================

class DataVisualizerApp(wx.App):
    def OnInit(self):
        if FLAGS["debug"]: MyLogger.info(str(locals()))
        self.frame = DataVisualizerFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#===============================================================================

HELPSTR = {}
HELPSTR["L2020"] = """
Draw heatmap and calculate relevant data for structured nest experiment (Linda Sartoris, 2020)

1) Heatmap
The program goes through each frame of the video, looking for ant colors. Each pixel where ant color is found in each frame is counted. The resultant data is an array which has the same width and height of the input video. Each number in the array is a positive integer, indicating how many frames have the ant color in the corresponding pixel. Then, this array is converted to a heatmap image, using colors associated with ranges of data. Both the array and the generated image can be saved in CSV and PNG format respectively.

HSV ant color ranges are ant's body color and two ranges for detecting color markers on ant's gaster.

Heatmap can be generated with certain frame interval to see changes of gathering through timeline.


2) Generating data for statistical analysis
For statistical analysis, the program calculates four data, namely bRects, rect,
rectL and centroid.

‘blob’ means a connected chunk in the binary image, generated with the
HSV color detection. The connected chunk is detected with an OpenCV’s function, connectedComponentsWithStats.
The 'bRects' is a list of bounding rects (rect = [x, y, width, height]), each of which surrounds a connected ant blob.
The 'rect' is a rectangle (x, y, width, height) that surrounds all the found ant
blobs.
The 'rectL' is a rectangle (x, y, width, height) that surrounds only the large size (aggregation of ants) ant blobs.
The 'centroid' is x- and y-coordinate of center of mass. The mass here is all the white (255) pixels in the binary image from the aforementioned color detection.

In [debug] mode, a user can check whether ant blobs, found with HSV color
detection, bRects, rect, rectL and centroid seem to be correct in frame image.

- 2020.10.
"""
HELPSTR["L2020CSV1"] = """
Drawing the heatmap from the first CSV raw data (heatmap data) file from L2020.
It draws some additional information and user can change data range for heatmap coloring.

- 2020.12.
"""
HELPSTR["L2020CSV2"] = """
Drawing a graph with the second CSV raw data (mean nearest neighbor distance & aggregation dispersal) from L2020.

- 2021.02.
"""
HELPSTR["anVid"] = """
Graph generated from CSV data of AnVid.
Currently it processes data from AnVid, which analyzed video recordings of Michaela's experiment (2024.April) on ants' responses to fungal volatiles.

- 2024.02.
"""
HELPSTR["aos"] = """
Graph using data from AntOS. 

- 2020.11.
"""
HELPSTR["V2020"] = """
Graph of virus presences in multiple ant species.
Figure for Viljakainen et al (2020).

- 2020.04.
"""


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-w': GNU_notice(1)
        elif sys.argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        app = DataVisualizerApp(redirect = False)
        app.MainLoop()

