# coding: UTF-8

"""
An open-source software written in Python to produce a graph.
  Also, it could be used as an interactive graph for a talk.

This program was coded and tested in Ubuntu 18.04.

Jinook Oh, Cremer group, Institute of Science and Technology Austria 
Aug. 2020.

Dependency:
    Python (3.7)
    wxPython (4.0)
    OpenCV (3.4)
    NumPy (1.18)

------------------------------------------------------------------------
Copyright (C) 2020 Jinook Oh & Sylvia Cremer 
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
"""

import sys, queue
from os import path, mkdir, remove
from shutil import copyfile
from glob import glob
from copy import copy

import wx, wx.adv, wx.stc
#from wx.lib.wordwrap import wordwrap
import wx.lib.scrolledpanel as SPanel 
import numpy as np

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is
sys.path.append(FPATH) # add FPATH to path
sys.path.append(path.split(FPATH)[0]) # add parent folder to path; for modFFC

from modFFC import *
from modProcGraph import ProcGraphData 
from modVideoRW import VideoRW

DEBUG = False 
__version__ = "0.1.20200925"

#===============================================================================

class GraphDrawerFrame(wx.Frame):
    """ Frame for drawing a graph
    for saving it as a image file or interactive graph on screen.

    Args:
        None
     
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self):
        if DEBUG: print("GraphDrawerFrame.__init__()")
        
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
              "pyDrawGraph v.%s"%(__version__), 
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
        self.flagBlockUI = False # block user input 
        pi = self.setPanelInfo() # set panel info
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.timer["sb"] = None # timer for status bar message display
        self.mlWid = [] # wx widgets in middle left panel
        self.mrWid = [] # wx widgets in middle right panel
        self.inputFP = ""
        self.origCSVTxt = "" # origianl CSV text from CSV file 
        self.colTitles = [] # column titles of CSV data
        self.numData = None # numpy array, containing numeric data
        self.strData = None # numpy array, containing string data
        self.frameImgFP = [] # list of file paths of 
          # frame images (graph which will become a frame of a video file)
        self.pgd = None # instance of ProcGraphData class
        self.debugging = False
        self.graphTypeChoices = [
                "L2020: structured nest [Sartoris et al]",
                "J2020: pilot work with founding queens [Jinook]",
                ]
        self.graphType = self.graphTypeChoices[0].split(":")[0].strip()
        ##### [end] setting up attributes -----
         
        ### create panels and its widgets
        btnSz = (35, 35)
        vlSz = (-1, 20) # size of vertical line separator
        for pk in pi.keys():
            self.panel[pk] = SPanel.ScrolledPanel(self, 
                                                  pos=pi[pk]["pos"],
                                                  size=pi[pk]["sz"],
                                                  style=pi[pk]["style"])
            self.panel[pk].SetBackgroundColour(pi[pk]["bgCol"])
            self.gbs[pk] = wx.GridBagSizer(0,0)
            w = [] # each itme represents a row in the left panel
            if pk == "tp":
                w.append([
                    {"type":"sTxt", "label":"Graph type:", "nCol":1,
                     "fgColor":"#cccccc"},
                    {"type":"cho", "nCol":1, "name":"graphType",
                     "choices":self.graphTypeChoices, 
                     "val":self.graphTypeChoices[0]},
                    {"type":"btn", "nCol":1, "name":"open", "size":btnSz,
                     "img":path.join(FPATH,"btn_imgs","open.png")},
                    {"type":"txt", "nCol":1, "name":"inputFP", 
                     "val":"[Opened input file]", "style":wx.TE_READONLY,
                     "size":(300,-1), "fgColor":"#cccccc"},
                    {"type":"sLn", "nCol":1, "size":vlSz, 
                     "style":wx.LI_VERTICAL},
                    {"type":"chk", "nCol":1, "name":"debug", "label":"debug", 
                     "val":self.debugging, "style":wx.CHK_2STATE, 
                     "fgColor":"#cccccc"},
                    {"type":"sTxt", "nCol":1, "fgColor":"#cccccc",
                     "label":"frames to jump for debugging:"},
                    {"type":"txt", "nCol":1, "name":"debugFrameIntv", 
                     "val":"1", "numOnly":True, "size":(50,-1)}
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
                     "img":path.join(FPATH,"btn_imgs","save.png"),
                     "tooltip":"Save graph"},
                    {"type":"sLn", "nCol":1, "size":vlSz, 
                     "style":wx.LI_VERTICAL},
                    {"type":"btn", "nCol":1, "name":"saveRawData", "size":btnSz,
                     "img":path.join(FPATH,"btn_imgs","saveR.png"),
                     "tooltip":"Save raw data as CSV"}
                    ])
            elif pk == "br":
                w.append([
                    {"type":"btn", "nCol":1, "name":"saveAll", "size":btnSz,
                     "img":path.join(FPATH,"btn_imgs","save.png"),
                     "tooltip":"Save all graphs"},
                    {"type":"sLn", "nCol":1, "size":vlSz, 
                     "style":wx.LI_VERTICAL},
                    {"type":"btn", "nCol":1, "name":"saveAllRawData", 
                     "size":btnSz,
                     "img":path.join(FPATH,"btn_imgs","saveR.png"),
                     "tooltip":"Save all raw data as CSV"}
                    ])
            if w != []: addWxWidgets(w, self, pk) 
            self.panel[pk].SetSizer(self.gbs[pk])
            self.gbs[pk].Layout()
            self.panel[pk].SetupScrolling()
            
        ### Bind events to the middle panel (graph panel)
        self.panel["mp"].Bind(wx.EVT_PAINT, self.onPaintMP)
        self.panel["mp"].Bind(wx.EVT_LEFT_UP, self.onClickGraph)
        self.panel["mp"].Bind(wx.EVT_RIGHT_UP, self.onMouseRClick)
        #self.panel["mp"].Bind(wx.EVT_MOTION, self.onMouseMove)

        self.initMLWidgets() # set up middle left panel
      
        ### keyboard binding
        exitId = wx.NewIdRef(count=1)
        self.Bind(wx.EVT_MENU, self.onClose, id = exitId)
        accel_tbl = wx.AcceleratorTable([
                            (wx.ACCEL_CTRL,  ord('Q'), exitId),
                                        ])
        self.SetAcceleratorTable(accel_tbl)
         
        ### make this frame modal
        if __name__ != "__main__":
            _dirs = path.normpath(FPATH).split("/")
            if len(_dirs) > 1 and "cremergroupapp" in _dirs[-2].lower():
                self.makeModal(True)
        
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
        if DEBUG: print("GraphDrawerFrame.makeModal()")
        
        if modal and not hasattr(self, '_disabler'):
            self._disabler = wx.WindowDisabler(self)
        if not modal and hasattr(self, '_disabler'):
            del self._disabler

    #---------------------------------------------------------------------------
    
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: print("GraphDrawerFrame.setPanelInfo()")

        wSz = self.wSz 
        pi = {} # information of panels
        if sys.platform.startswith("win"):
            style = (wx.TAB_TRAVERSAL|wx.SIMPLE_BORDER)
            bgCol = "#777777"
        else:
            style = (wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
            bgCol = "#333333"

        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), sz=(wSz[0], 50), bgCol=bgCol, style=style)
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
                        bgCol=bgCol, style=style)
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
        
        Args:
            None
        
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: print("GraphDrawerFrame.initMLWidgets()")
        
        pk = "ml"
        pSz = self.pi[pk]["sz"]
        hlSz = (int(pSz[0]*0.95), -1)
        
        for i, w in enumerate(self.mlWid): # through widgets in the panel
            try:
                self.gbs[pk].Detach(w) # detach 
                w.Destroy() # destroy
            except:
                pass
        
        ##### [begin] set up middle left panel -----
        w = [] # each itme represents a row in the left panel 
        if self.graphType == "L2020":
            ##### [begin] set widgets for HSV colors to track -----
            self.defHSVVal = {}
            self.defHSVVal["col0"] = {} # ant body color
            self.defHSVVal["col0"]["min"] = (0, 0, 0)
            self.defHSVVal["col0"]["max"] = (180, 150, 100)
            self.defHSVVal["col1"] = {} # color markers
            self.defHSVVal["col1"]["min"] = (50, 80, 80)
            self.defHSVVal["col1"]["max"] = (180, 255, 200)
            self.defHSVVal["col2"] = {} # yellow color marker
            self.defHSVVal["col2"]["min"] = (10, 200, 150)
            self.defHSVVal["col2"]["max"] = (30, 255, 250)
            for ci in range(3): # three color ranges
                for mLbl in ["min", "max"]: # HSV min & max values
                    ck = "col%i"%(ci)
                    col = dict(H=0, S=0, V=0)
                    col["H"], col["S"], col["V"] = self.defHSVVal[ck][mLbl]
                    rgbVal = cvHSV2RGB(col["H"], col["S"], col["V"])
                    w.append([
                      {"type":"sTxt", "label":"Color%i [HSV-%s.]"%(ci, mLbl), 
                       "nCol":2},
                      {"type":"panel", "nCol":1, "bgColor":rgbVal,
                       "size":(20,20), "name":"col%i%s"%(ci, mLbl.capitalize())}
                      ])
                    tmp = []
                    for k in ["H", "S", "V"]:
                        if k == "H": maxVal = 180
                        else: maxVal = 255
                        tmp.append(
                            {"type":"sld", "nCol":1, "val":col[k],
                             "name":"col%i%s%s"%(ci, k, mLbl.capitalize()), 
                             "size":(int(pSz[0]*0.3), -1), "border":1,
                             "minValue":0, "maxValue":maxVal, 
                             "style":wx.SL_VALUE_LABEL}
                            )
                    w.append(tmp) 
            ##### [end] set widgets for HSV colors to track -----
            
            ### widgets for button to set HSV colors back to default 
            w.append([{"type":"btn", "size":(int(pSz[0]*0.85),-1), "nCol":3,
                       "name":"resetHSV", 
                       "label":"Reset all HSV values to default"}])
            w.append([{"type":"sLn", "size":(int(pSz[0]*0.85),-1), "nCol":3,
                       "style":wx.LI_HORIZONTAL}])
            
            ### widgets for changing color scheme
            w.append([{"type":"sTxt", "label":"Background color", "nCol":2},
                      {"type":"cPk", "nCol":1, "color":(0,0,0), 
                       "name":"bgCol", "size":(60,-1)}])
            #w.append([{"type":"sTxt", "label":"Low heat color", "nCol":1}])
            
            ### widgets for frame range to calculate heatmap
            w.append([{"type":"sTxt", "label":"start-frame", "nCol":2},
                      {"type":"txt", "name":"startFrame", "val":"0",
                       "nCol":1, "numOnly":True, "size":(60,-1)}])
            w.append([{"type":"sTxt", "label":"end-frame", "nCol":2},
                      {"type":"txt", "name":"endFrame", "val":"1",
                       "nCol":1, "numOnly":True, "size":(60,-1)}])
            ### widgets for frame intervals for heatmap generation
            w.append([{"type":"sTxt", "label":"frame-interval", "nCol":2},
                      {"type":"txt", "name":"frameIntv", "val":"1",
                       "nCol":1, "numOnly":True, "size":(60,-1)}])
            ### widgets for minimum area of an ant
            w.append([{"type":"sTxt", "label":"ant min. area", "nCol":2},
                      {"type":"txt", "name":"aMinArea", "val":"100",
                       "nCol":1, "numOnly":True, "size":(60,-1)}])
            ### button widget to draw the graph
            w.append([{"type":"btn", "name":"draw", "label":"Draw heatmap",
                       "nCol":3}])
        self.gbs[pk] = wx.GridBagSizer(0,0) 
        self.mlWid, pSz = addWxWidgets(w, self, pk)
        if pSz[0] > self.pi[pk]["sz"][0]:
            self.panel[pk].SetSize(pSz[0], self.pi[pk]["sz"][1])
        ### 
        self.panel[pk].SetSizer(self.gbs[pk])
        self.gbs[pk].Layout()
        self.panel[pk].SetupScrolling()
        ##### [end] set up middle left panel -----
    
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
        if DEBUG: print("GraphDrawerFrame.onButtonPressDown()")

        ret = preProcUIEvt(self, event, objName, "btn")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret
        if flag_term: return

        if self.flagBlockUI or not obj.IsEnabled(): return
        self.playSnd("leftClick")

        if objName == "open_btn": self.openInputFile()

        elif objName == "save_btn": self.save()

        elif objName == "saveAll_btn": self.save(isSavingAll=True) 

        elif objName == "saveRawData_btn": self.save(savType="raw")
        
        elif objName == "saveAllRawData_btn": self.save("raw", True)

        elif objName == "resetHSV_btn":
            for ci in range(3): # three color ranges
                for mLbl in ["min", "max"]: # HSV min & max values
                    ck = "col%i"%(ci)
                    col = dict(H=0, S=0, V=0)
                    col["H"], col["S"], col["V"] = self.defHSVVal[ck][mLbl]
                    for k in ["H", "S", "V"]:
                        sldN = "col%i%s%s_sld"%(ci, k, mLbl.capitalize())
                        sld = wx.FindWindowByName(sldN, self.panel["ml"])
                        sld.SetValue(col[k])
                    pN = "%s%s_panel"%(ck, mLbl.capitalize()) 
                    rgbVal = cvHSV2RGB(col["H"], col["S"], col["V"])
                    obj = wx.FindWindowByName(pN, self.panel["ml"])
                    obj.SetBackgroundColour(rgbVal)
                    obj.Refresh()
       
        elif objName == "draw_btn":
            if self.graphType == "L2020":
            # Heatmap for Linda (2020)
                self.pgd.graphImg = []
                self.pgd.graphImgIdx = 0
                if self.debugging:
                    self.pgd.graphL2020(self.mpDC, -1, -1, self.q2m)
                    self.panel["mp"].Refresh() # display graph 
                else:
                    for i, w in enumerate(self.mrWid): # widgets in the panel
                        try:
                            self.gbs["mr"].Detach(w) # detach 
                            w.Destroy() # destroy
                        except:
                            pass
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
                        startHeavyTask(self, "drawGraph", self.pgd.graphL2020, 
                                       args=args)
                    else:
                        self.vRW.getFrame(startFrame, self.callback)
     
    #---------------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.CheckBox event 
                with the given name. 
        
        Returns: None
        """
        if DEBUG: print("GraphDrawerFrame.onCheckBox()")

        ret = preProcUIEvt(self, event, objName, "chk")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "debug_chk":
            self.debugging = objVal
            if self.debugging:
                if hasattr(self, "vRW") and self.inputFP != "":
                    self.vRW.initReader(self.inputFP) # init video reader
    
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
        if DEBUG: print("GraphDrawerFrame.onChoice()")
        
        ret = preProcUIEvt(self, event, objName, "cho")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName == "graphType_cho":
            ## store graph type
            self.graphType = objVal.split(":")[0].strip()
            self.initMLWidgets() # set up middle left panel

    #---------------------------------------------------------------------------
    
    def onSlider(self, event, objName=""):
        """ wx.Slider was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.Choice event 
                with the given name. 
        
        Returns:
            None
        """
        if DEBUG: print("GraphDrawerFrame.onSlider()")
        
        ret = preProcUIEvt(self, event, objName, "sld")
        flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal = ret 
        if flag_term: return

        if objName.startswith("col"):
            ### update chosen HSV color on the corresponding panel
            tmp = objName.split("_")[0]
            colN = tmp[:4]
            minOrMax = tmp[-3:]
            val = []
            for lbl in ["H", "S", "V"]:
                objName = "%s%s%s_sld"%(colN, lbl, minOrMax)
                obj = wx.FindWindowByName(objName, self.panel["ml"])
                val.append(obj.GetValue())
            rgbVal = cvHSV2RGB(val[0], val[1], val[2])
            objName = "%s%s_panel"%(colN, minOrMax) 
            obj = wx.FindWindowByName(objName, self.panel["ml"])
            obj.SetBackgroundColour(rgbVal)
            obj.Refresh()

    #---------------------------------------------------------------------------
    
    def onTextCtrlChar(self, event, objName="", isNumOnly=False):
        """ Character entered in wx.TextCtrl.
        Currently using to allow entering numbers only.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.Choice event 
                with the given name.
            isNumOnly (bool): allow number entering only.
        
        Returns:
            None
        """
        if DEBUG: print("GraphDrawerFrame.onTextCtrlChar()")
        
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
    
    def onClose(self, event):
        """ Close this frame. 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onClose()")
        
        self.makeModal(False)
        stopAllTimers(self.timer)
        for fp in glob("tmp_*.*"):
            if path.isfile(fp): remove(fp)
        self.Destroy()
    
    #---------------------------------------------------------------------------

    def openInputFile(self):
        """ Open input file. 
        
        Args:
            None
        
        Returns:
            None 
        """
        if DEBUG: print("GraphDrawerFrame.openInputFile()")

        if self.graphType == "L2020": fType = "file"; ext = "mp4"
        elif self.graphType == "J2020": fType = "dir"; ext = "csv" 
        else: fType = "file"; ext = "csv"

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
         
        ### display input file path
        self.inputFP = inputFP
        obj = wx.FindWindowByName("inputFP_txt", self.panel["tp"])
        obj.SetValue(path.basename(inputFP))

        if fType == "file" and ext == "csv":
            ### set CSV text
            f = open(inputFP, 'r')
            csvTxt = f.read()
            f.close()
            self.stcCSV.SetEditable(True)
            self.stcCSV.SetText(csvTxt)
            self.stcCSV.SetEditable(False) # CSV text is read-only
            self.origCSVTxt = copy(csvTxt) # keep original CSV text 

        self.pgd = ProcGraphData(self) # init instance class
                                       # for processing graph data 
        try:
            if self.graphType == "L2020":
                self.pgd.graphImg = []
                self.pgd.graphImgIdx = 0
                self.vRW = VideoRW(self) # for reading/writing video file
                self.vRW.initReader(inputFP) # init video reader
                ### resize graph panel
                f = self.vRW.currFrame
                r = calcI2DIRatio(f, self.pi["mp"]["sz"])
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
                self.pgd.initOnDataLoading(inputFP)
            elif self.graphType == "J2020":
                self.pgd.initOnDataLoading(inputFP)
                startHeavyTask(self, 
                               "drawGraph", 
                               self.pgd.graphJ2020, 
                               args=(0, self.q2m,))
        except Exception as e: # failed
            self.inputFP = ""
            self.pgd.csvFP = ""
            """
            self.stcCSV.SetEditable(True)
            self.stcCSV.SetText("")
            self.stcCSV.SetEditable(False) # CSV text is read-only
            """
            msg = "Failed to load CSV data\n"
            msg += str(e)
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
        if DEBUG: print("GraphDrawerFrame.onPaintMP()")

        if self.inputFP == "": return

        event.Skip()
        
        dc = wx.PaintDC(self.panel["mp"])
        self.mpDC = dc
        gImg = self.pgd.graphImg
        if len(gImg) == 0: return
        idx = self.pgd.graphImgIdx
        dc.DrawBitmap(gImg[idx]["img"].ConvertToBitmap(), 0, 0)

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
            return
        
        elif rData[0].startswith("finished"):
            self.callback(rData, flag)
    
    #---------------------------------------------------------------------------
    
    def onClickGraph(self, event):
        """ Processing when user clicked graph

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onClickGraph()")
        
        if self.flagBlockUI or self.inputFP == "": return

        self.playSnd("leftClick") 
        
        mp = event.GetPosition()
        
        if self.graphType.startswith("V2020"):
            ### check whether classification label (in legend) is clicked 
            for cl in self.pgd.clR.keys():
                r = self.pgd.clR[cl]
                if r[0] <= mp[0] <= r[2] and r[1] <= mp[1] <= r[3]:
                # clicked
                    self.pgd.initUITask() # delete current uiTask
                    ### process clicked class
                    self.pgd.uiTask["showThisClassOnly"] = cl
                    self.bp_sTxt.SetLabel(cl)
                    self.panel["mp"].Refresh() # re-draw graph
                    return
            ### check whether virus label (in legend) is clicked 
            for vl in self.pgd.vlR.keys():
                r = self.pgd.vlR[vl]
                if r[0] <= mp[0] <= r[2] and r[1] <= mp[1] <= r[3]:
                # clicked
                    self.pgd.initUITask() # delete current uiTask
                    ### process clicked virus 
                    self.pgd.uiTask["showThisVirusOnly"] = vl
                    self.bp_sTxt.SetLabel(vl)
                    self.panel["mp"].Refresh() # re-draw graph
                    return
            ### check whether virus presence circle is clicked 
            vpPt = self.pgd.vpPt
            rad = self.pgd.vCR
            sD = self.pgd.strData
            for vl in vpPt.keys(): # virus label
                lst = []
                for si in range(self.pgd.numSpecies): lst += vpPt[vl][si]
                for x, y in lst:
                    if x-rad <= mp[0] <= x+rad and y-rad <= mp[1] <= y+rad:
                    # clicked
                        row = np.where(sD==vl)[0][0]
                        cl = sD[row,1] # classification
                        vlStr = "%s [%s]"%(vl, cl)
                        self.pgd.uiTask["showVirusLabel"] = [x, y, vlStr]
                        self.bp_sTxt.SetLabel(vlStr)
                        self.panel["mp"].Refresh() # re-draw graph
                        return
            ### nothing specific is clicked, delete label of bp_sTxt 
            self.bp_sTxt.SetLabel("")
            self.pgd.initUITask() # delete current uiTask
            self.panel["mp"].Refresh() # re-draw graph
    
    #---------------------------------------------------------------------------
    
    def onClickThumbnail(self, event):
        """ process mouse-click on thumbnail image

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onClickThumbnail()")

        if self.flagBlockUI or self.inputFP == "": return

        self.playSnd("leftClick") 
        self.pgd.graphImgIdx = event.GetEventObject().index
        self.panel["mp"].Refresh()

    #---------------------------------------------------------------------------
    
    def onMouseMove(self, event):
        """ Mouse pointer moving on graph area
        Show some info

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onMouseMove()")

        if self.flagBlockUI or self.inputFP == "": return

        mp = event.GetPosition()
     
    #---------------------------------------------------------------------------
    
    def onMouseRClick(self, event):
        """ Mouse right click on graph area.
        [Currently no functionality implmented]

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onMouseRClick()")
        
        if self.flagBlockUI or self.inputFP == "": return

        #self.panel["mp"].Refresh() # re-draw graph

    #---------------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread
        
        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if DEBUG: print("GraphDrawerFrame.callbackFunc()")

        if flag == "drawGraph":
            showStatusBarMsg(self, "Heatmap generated.", 3000)
            idx = len(self.pgd.graphImg)-1
            self.pgd.graphImgIdx = idx 
            ### display thumbnail image of the generated graph 
            bmp = self.pgd.graphImg[-1]["thumbnail"].ConvertToBitmap()
            sBmp = wx.StaticBitmap(self.panel["mr"], -1, bmp)
            sBmp.Bind(wx.EVT_LEFT_UP, self.onClickThumbnail)
            sBmp.index = idx
            add2gbs(self.gbs["mr"], sBmp, (int(idx/2),idx%2), (1,1))
            self.mrWid.append(sBmp)
            self.gbs["mr"].Layout()
            self.panel["mr"].SetupScrolling()
            if self.graphType == "L2020":
                if self.vRW.fi < self.endFI:
                # heatmap generation not finished yet
                    ### start another heatmap generation
                    startFrame = self.vRW.fi+1
                    nextEndFrame = min(startFrame+self.frameIntv-1, self.endFI)
                    args = (startFrame, nextEndFrame, self.q2m,)
                    wx.CallLater(5, startHeavyTask, self, "drawGraph", 
                                 self.pgd.graphL2020, args)

            elif self.graphType == "J2020":
                showStatusBarMsg(self, "Graph generated.", 3000)
                ai = len(self.pgd.graphImg)
                if ai < 4: # there are more ants to process
                    wx.CallLater(5, startHeavyTask, self, "drawGraph", 
                                 self.pgd.graphJ2020, (ai, self.q2m,))

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
        
        elif flag == "readFrames":
            if self.graphType == "L2020":
                ### reached the frame in video module, start heatmap generation
                nextEndFrame = min(self.startFI+self.frameIntv-1, self.endFI)
                args = (self.startFI, nextEndFrame, self.q2m,)
                startHeavyTask(self, "drawGraph", self.pgd.graphL2020, 
                               args=args)
        
        self.flagBlockUI = False
        if self.th != None:
            try: self.th.join()
            except: pass
        if flag in self.timer.keys():
            try: self.timer[flag].Stop()
            except: pass
        self.panel["mp"].Refresh() # display graph 
        self.panel["mr"].Refresh() # display thumbnail images 
    
    #---------------------------------------------------------------------------
    
    def save(self, savType="graph", isSavingAll=False):
        """ Save data 

        Args:
            savType (str): graph; saving graph, raw: saving raw data
            isSavingAll (bool): saving the current graph or all graph

        Returns:
            None
        """
        if DEBUG: print("GraphDrawerFrame.save()")

        if self.inputFP == "": return

        ### save graph
        if self.graphType == "L2020":
            if isSavingAll: idxRng = range(len(self.pgd.graphImg))
            else: idxRng = [self.pgd.graphImgIdx]
            msg = "Saved\n"
            for idx in idxRng:
                ### determine file path to write
                #timestamp = get_time_stamp().replace("_","")[:14]
                fn = path.basename(self.inputFP) # current input file name 
                ext = "." + fn.split(".")[-1]
                if savType == "graph": sExt = ".png"
                elif savType == "raw": sExt = ".csv"
                data = self.pgd.graphImg[idx]
                newTxt = "_%s_%s"%(savType, data["startFrame"])
                newTxt += "_%s%s"%(data["endFrame"], sExt) 
                newFN = fn.replace(ext, newTxt)
                fp4sav = self.inputFP.replace(fn, newFN)
                if savType == "graph":
                    # get image to save 
                    img = cv2.imread("tmp_origImg%i.png"%(idx))
                    ### resize if required 
                    obj = wx.FindWindowByName("imgSavResW_txt", 
                                              self.panel["bm"])
                    w = int(obj.GetValue())
                    obj = wx.FindWindowByName("imgSavResH_txt", 
                                              self.panel["bm"])
                    h = int(obj.GetValue())
                    if w != img.shape[1] or h != img.shape[0]:
                        img = cv2.resize(img, 
                                         (w,h), 
                                         interpolation=cv2.INTER_CUBIC)
                    # save
                    cv2.imwrite(fp4sav, img)
                    msg += fp4sav + "\n\n"
                elif savType == "raw":
                    for fp in glob("tmp_data*%s"%(sExt)):
                        tmp = fp.replace(sExt, "").split("_")
                        if tmp[2] == str(idx): # matches graph index
                            newFP = fp4sav.replace(sExt, 
                                                   "_%s%s"%(tmp[-1], sExt))
                            copyfile(fp, newFP)
                            msg += newFP + "\n\n"
        
        """
        ### save CSV
        msg = "Saving CSV ..."
        self.q2m.put((msg,), True, None)
        csvFN = fn.replace(".csv", "_graph_%s.csv"%(timestamp))
        csvFP = path.join(self.outputPath, csvFN)
        fh = open(csvFP, 'w')
        fh.write(self.stcCSV.GetText())
        fh.close()
        msg2 += csvFN + "\n"
        """
        wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
    
    #---------------------------------------------------------------------------
    
    def saveVideo(self):
        """ Save video (with revised head direction line)
        
        Args: None
        
        Returns: None
        """
        if DEBUG: print("GraphDrawerFrame.saveVideo()")

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
        self.flagBlockUI = True 
    
    #---------------------------------------------------------------------------
    
    def playSnd(self, flag=""):
        """ Play sound 

        Args:
            flag (str): Which sound to play.

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.playSnd()")

        if flag == "leftClick":
            ### play click sound
            snd_click = wx.adv.Sound(path.join(FPATH, "snd_click.wav"))
            snd_click.Play(wx.adv.SOUND_ASYNC)

    #---------------------------------------------------------------------------
    
    def runScript(self, q2m, q2t, scriptTxt, csvTxt):
        """ Run script in UI to manipulate CSV values (as a thread).

        Args:
            q2m (queue.Queue): Queue to main thread
            q2t (queue.Queue): Queue to this thread
            scriptTxt (str): Python script text to edit CSV text 
            csvTxt (str): Original CSV text 
        
        Returns: None
        """ 
        if DEBUG: print("GraphDrawerFrame.runScript()") 

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

#===============================================================================

class STC(wx.stc.StyledTextCtrl):
    def __init__(self, parent, pos, 
                 size, fgCol="#000000", bgCol="#cccccc", caretFGCol="#ffffff"):
        if DEBUG: print("STC.__init__")
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

class GraphDrawerApp(wx.App):
    def OnInit(self):
        if DEBUG: print("GraphDrawerApp.OnInit()")
        self.frame = GraphDrawerFrame()
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
        app = GraphDrawerApp(redirect = False)
        app.MainLoop()

