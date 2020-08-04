# coding: UTF-8

"""
An open-source software written in Python
  to import CSV data, then produce a graph with it,
  (especially for when the graph is complicated).
  Also, it could be used as an interactive graph for a talk.
Originally developed for a graph in a paper of 
  Sylvia Cremer and Lumi Viljakainen 
  about viruses in different species and populations.

This program was coded and tested in macOS 10.13.

Jinook Oh, Cremer group, Institute of Science and Technology Austria 
Jan. 2020.

Dependency:
    wxPython (4.0)

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
"""

import sys, queue
from threading import Thread 
from os import path, mkdir
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

from modFFC import GNU_notice, get_time_stamp, getWXFonts
from modFFC import updateFrameSize, add2gbs, receiveDataFromQueue
from modFFC import stopAllTimers, set_img_for_btn
from modProcGraph import ProcGraphData 

DEBUG = False 
__version__ = "0.1.1"

#=======================================================================

class GraphDrawerFrame(wx.Frame):
    """ For drawing a graph based on a CSV file,
    to save it as a image file or interactive graph on screen.

    Args:
        None
     
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """ 

    def __init__(self):
        if DEBUG: print("GraphDrawerFrame.__init__()")

        ### init 
        wPos = (0, 20)
        wg = wx.Display(0).GetGeometry()
        wSz = (wg[2], int(wg[3]*0.85))
        wx.Frame.__init__(
              self,
              None,
              -1,
              "pyDrawGraph v.%s"%(__version__), 
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX),
              )
        self.SetBackgroundColour('#AAAABB')
        iconPath = path.join(FPATH, "icon.png")
        if __name__ == '__main__' and path.isfile(iconPath):
            self.SetIcon(wx.Icon(iconPath)) # set app icon
        self.Bind(wx.EVT_CLOSE, self.onClose)

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
        self.csvFP = "" # CSV file path
        self.outputPath = path.join(FPATH, "output") # output file path
        if not path.isdir(self.outputPath): mkdir(self.outputPath)
        self.origCSVTxt = "" # origianl CSV text from CSV file 
        self.colTitles = [] # column titles of CSV data
        self.numData = None # numpy array, containing numeric data
        self.strData = None # numpy array, containing string data
        self.frameImgFP = [] # list of file paths of 
          # frame images (graph which will become a frame of a video file)
        self.fImgThumbnailSz = [pi["rp"]["sz"][0]-10, 1] # thumbnail image
          # size of a saved frame image (graph)
        self.fImgThumbnailSz[1] = int(self.fImgThumbnailSz[0] * 0.75)
        ### set various resolutions for image saving
        self.imgSavRes = []
        iRes = (800, 600)
        for i in range(20):
            w = int(iRes[0] + i * 200)
            h = int(w * 0.75)
            self.imgSavRes.append((w, h))
        self.pgd = None # instance of ProcGraphData class
        self.graphTypeChoices = [
            "CV2020: Ant-Virus paper [Cremer & Viljakainen]"
            ]
        ##### [end] setting up attributes -----
        
        updateFrameSize(self, wSz)
        
        ### create panels
        for k in pi.keys():
            self.panel[k] = SPanel.ScrolledPanel(self, 
                                                 pos=pi[k]["pos"],
                                                 size=pi[k]["sz"],
                                                 style=pi[k]["style"])
            self.panel[k].SetBackgroundColour(pi[k]["bgCol"])

        ##### [begin] set up top panel interface -----
        vlSz = (-1, 20) # size of vertical line separator
        self.gbs["tp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        sTxt = wx.StaticText(self.panel['tp'], 
                             -1,
                             label="Graph type:")
        sTxt.SetForegroundColour("#cccccc")
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,1))
        col += 1
        cho = wx.Choice(self.panel['tp'], 
                        -1, 
                        choices=self.graphTypeChoices,
                        name="graphType_cho",
                        size=(150,-1))
        cho.SetSelection(0)
        add2gbs(self.gbs["tp"], cho, (row,col), (1,1))
        col += 1 
        btn = wx.Button(self.panel["tp"],
                        -1,
                        size=(30,30),
                        name="openCSV_btn")
        set_img_for_btn(path.join(FPATH, "img_open.png"), btn) 
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        txt = wx.TextCtrl(self.panel['tp'], 
                          -1,
                          name="csvFP_txt",
                          value="[Opened CSV file]",
                          size=(150,-1),
                          style=wx.TE_READONLY)
        txt.SetBackgroundColour("#cccccc")
        add2gbs(self.gbs["tp"], txt, (row,col), (1,1))
        col += 1
        add2gbs(self.gbs["tp"],
                wx.StaticLine(self.panel["tp"],
                              -1,
                              size=vlSz,
                              style=wx.LI_VERTICAL),
                (row,col),
                (1,1)) # vertical line separator 
        col += 1
        btn = wx.Button(self.panel['tp'], 
                                  -1, 
                                  #label='Quit',
                                  size=(30,30),
                                  name="quit_btn")
        set_img_for_btn(path.join(FPATH, "img_quit.png"), btn) 
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        self.panel["tp"].SetSizer(self.gbs["tp"])
        self.gbs["tp"].Layout()
        self.panel["tp"].SetupScrolling()
        ##### [end] set up top panel interface -----

        ##### [begin] set up left panel interface -----
        self.gbs["lp"] = wx.GridBagSizer(0,0)
        lpSz = pi["lp"]["sz"]
        row = 0; col = 0
        sTxt = wx.StaticText(self.panel['lp'], 
                             -1,
                             label="Current CSV")
        sTxt.SetForegroundColour('#ffffff')
        add2gbs(self.gbs["lp"], sTxt, (row,col), (1,1))
        row += 1
        self.stcCSV = STC(self.panel["lp"], 
                          (0,0), 
                          (lpSz[0],lpSz[1]-30),
                          fgCol="#cccccc",
                          bgCol="#555555")
        self.stcCSV.SetEditable(False) # CSV text is read-only 
        add2gbs(self.gbs["lp"], self.stcCSV, (row,col), (1,1), bw=0) 
        self.panel["lp"].SetSizer(self.gbs["lp"])
        self.gbs["lp"].Layout()
        #self.panel["lp"].SetupScrolling()
        ##### [end] set up left panel interface -----

        ##### [begin] set up script panel interface -----
        self.gbs["sp"] = wx.GridBagSizer(0,0)
        spSz = pi["sp"]["sz"]
        row = 0; col = 0
        sTxt = wx.StaticText(self.panel['sp'], 
                             -1,
                             label="Python script to modify CSV lines")
        sTxt.SetForegroundColour('#ffffff')
        add2gbs(self.gbs["sp"], sTxt, (row,col), (1,1))
        row += 1
        self.stcScript = STC(self.panel["sp"], (0,0), (spSz[0],spSz[1]-60)) 
        add2gbs(self.gbs["sp"], self.stcScript, (row,col), (1,2), bw=0) 
        scriptStr = "### Go through each line in lines (Python List).\n"
        scriptStr += "### If you change 'lines' and click 'Run script',\n"
        scriptStr += "###  the above CSV lines and graph will modified\n"
        scriptStr += "# ----------------------------------------------\n"
        scriptStr += "for li, line in enumerate(lines): # through each line\n"
        scriptStr += "  cols = [x.strip() for x in line.split(',')]"
        scriptStr += " # list of each column data of the current line\n"
        scriptStr += "  line = ''\n"
        scriptStr += "  for ci in range(len(cols)):\n"
        scriptStr += "    line += cols[ci] + ', '\n"
        scriptStr += "  line = line.rstrip(', ')\n"
        scriptStr += "  lines[li] = line\n"
        self.stcScript.SetText(scriptStr)
        row += 1; col = 1 
        btn = wx.Button(self.panel['sp'], 
                        -1, 
                        name="runScript_btn",
                        size=(30,30))
        set_img_for_btn(path.join(FPATH, "img_run.png"), btn) 
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["sp"], btn, (row,col), (1,1), int(spSz[0]*0.2), wx.LEFT)
        self.panel["sp"].SetSizer(self.gbs["sp"])
        self.gbs["sp"].Layout()
        #self.panel["sp"].SetupScrolling()
        ##### [end] set up script panel interface -----

        ### set up graph-panel
        self.panel["gp"].Bind(wx.EVT_PAINT, self.onPaint)
        self.panel["gp"].Bind(wx.EVT_LEFT_UP, self.onClickGraph)
        self.panel["gp"].Bind(wx.EVT_RIGHT_UP, self.onMouseRClick)
        #self.panel["gp"].Bind(wx.EVT_MOTION, self.onMouseMove)

        ##### [begin] set up graph (image) saving interface -----
        gspSz = pi["gsp"]["sz"]
        cho = wx.Choice(self.panel['gsp'], 
                        -1, 
                        choices=[str(x) for x in self.imgSavRes], 
                        name="imgSavRes_cho",
                        pos=(int(gspSz[0]*0.75), 2),
                        size=(int(gspSz[0]*0.19),-1))
        cho.SetSelection(6)
        btn = wx.Button(self.panel["gsp"],
                        -1,
                        name="saveGraph_btn",
                        pos=(int(gspSz[0]*0.95), 1),
                        size=(20,20)
                        )
        set_img_for_btn(path.join(FPATH, "img_save.png"), btn) 
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        ##### [end] set up graph (image) saving interface -----
        
        ##### [begin] set up graph (video) saving interface -----
        self.gbs["vsp"] = wx.GridBagSizer(0,0)
        vspSz = pi["vsp"]["sz"]
        row = 0; col = 0
        btn = wx.Button(self.panel["vsp"],
                        -1,
                        size=(30,30),
                        name="openCSVs_btn")
        set_img_for_btn(path.join(FPATH, "img_open.png"), btn) 
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["vsp"], btn, (row,col), (1,1), 5, wx.LEFT)
        col += 1
        chk = wx.CheckBox(self.panel['vsp'], 
                          -1, 
                          name="interpolate_chk",
                          label="",
                          style=wx.CHK_2STATE)
        chk.Bind(wx.EVT_CHECKBOX, self.onCheckBox)
        chk.Disable()
        add2gbs(self.gbs["vsp"], chk, (row,col), (1,1), 5, wx.LEFT|wx.TOP)
        col += 1
        sTxt = wx.StaticText(self.panel['vsp'], 
                             -1,
                             label="interpolate")
        sTxt.SetForegroundColour('#cccccc')
        add2gbs(self.gbs["vsp"], sTxt, (row,col), (1,1), 5, wx.TOP)
        col += 1
        btn = wx.Button(self.panel["vsp"],
                        -1,
                        name="saveVid_btn",
                        size=(30,30))
        set_img_for_btn(path.join(FPATH, "img_video.png"), btn) 
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        btn.Disable()
        _bw = int(vspSz[0]*0.1)
        add2gbs(self.gbs["vsp"], btn, (row,col), (1,1), _bw, wx.LEFT)
        self.panel["vsp"].SetSizer(self.gbs["vsp"])
        self.gbs["vsp"].Layout()
        ##### [end] set up graph (video) saving interface -----

        ##### [begin] set up bottom panel interface -----
        self.gbs["bp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        self.bp_sTxt = wx.StaticText(self.panel['bp'], -1, label="")
        self.bp_sTxt.SetBackgroundColour('#dddddd')
        self.bp_sTxt.SetForegroundColour('#000000')
        add2gbs(self.gbs["bp"], self.bp_sTxt, (row,col), (1,1))
        self.panel["bp"].SetSizer(self.gbs["bp"])
        self.gbs["bp"].Layout()
        #self.panel["bp"].SetupScrolling()
        ##### [end] set up bottom panel interface -----

        ### keyboard binding
        exit_btnId = wx.NewIdRef(count=1)
        #space_btnId = wx.NewIdRef(count=1) # for continuous playing 
        self.Bind(wx.EVT_MENU, self.onClose, id = exit_btnId)
        #self.Bind(wx.EVT_MENU, self.onSpace, id = space_btnId)
        accel_tbl = wx.AcceleratorTable([
                            (wx.ACCEL_CMD,  ord('Q'), exit_btnId ),
                            #(wx.ACCEL_NORMAL, wx.WXK_SPACE, space_btnId),
                                        ])
        self.SetAcceleratorTable(accel_tbl)
       
        ### make this frame modal
        if __name__ != "__main__":
            _dirs = path.normpath(FPATH).split("/")
            if len(_dirs) > 1 and "cremergroupapp" in _dirs[-2].lower():
                self.makeModal(True)
        
        #self.openCSVFile()
    
    #-------------------------------------------------------------------
  
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

    #-------------------------------------------------------------------
    
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
        # top panel for major buttons
        pi["tp"] = dict(pos=(0, 0), 
                        sz=(wSz[0], 50), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        tpSz = pi["tp"]["sz"]
        # bottom panel for short info
        pi["bp"] = dict(pos=(0, wSz[1]-30), 
                        sz=(wSz[0], 30), 
                        bgCol="#dddddd", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        bpSz = pi["bp"]["sz"]
        # left panel for showing CSV lines 
        pi["lp"] = dict(pos=(0, tpSz[1]), 
                        sz=(int(wSz[0]*0.25), 
                            int((wSz[1]-tpSz[1]-bpSz[1])*0.5)),
                        bgCol="#333333",
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        lpSz = pi["lp"]["sz"]
        # panel for writing script
        pi["sp"] = dict(pos=(0, tpSz[1]+lpSz[1]),
                        sz=lpSz,
                        bgCol="#333333",
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        gph = wSz[1]-tpSz[1]-bpSz[1]-30
        gpw = gph * 1.3333
        # panel for showing graph
        pi["gp"] = dict(pos=(lpSz[0], tpSz[1]),
                        sz=(gpw, gph), 
                        bgCol="#888888",
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        gpSz = pi["gp"]["sz"]
        # panel for showing graph image saving interface 
        pi["gsp"] = dict(pos=(lpSz[0], wSz[1]-bpSz[1]-30),
                        sz=(gpSz[0], 30),
                        bgCol="#333333",
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER) 
        rw = wSz[0]-lpSz[0]-gpSz[0] # right panel width
        # panel for showing graph video saving interface 
        pi["vsp"] = dict(pos=(wSz[0]-rw, wSz[1]-bpSz[1]-30),
                        sz=(rw, 30),
                        bgCol="#333333",
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER) 
        vspSz = pi["vsp"]["sz"]
        # right panel for showing frame images
        pi["rp"] = dict(pos=(wSz[0]-rw, tpSz[1]),
                        sz=(rw, wSz[1]-tpSz[1]-bpSz[1]-vspSz[1]),
                        bgCol="#888888",
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        rpSz = pi["rp"]["sz"] 
        return pi

    #-------------------------------------------------------------------

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

        if objName == '':
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
            obj = wx.FindWindowByName(objName, self.panel["tp"])

        if self.flagBlockUI: return
        if not obj.IsEnabled(): return
        
        self.playSnd("leftClick")

        if objName in ["runScript_btn", "saveGraph_btn"]:
            if self.csvFP == "": return # there's no opened CSV file
            self.flagBlockUI = True 
            ### set timer for updating progress 
            self.timer["prog"] = wx.Timer(self)
            self.Bind(wx.EVT_TIMER,
                      lambda event: self.onTimer(event, "prog"),
                      self.timer["prog"])
            self.timer["prog"].Start(10) 

        if objName == "quit_btn": self.onClose(None)
        elif objName == "openCSV_btn": self.openCSVFile()
        elif objName == "runScript_btn":
            scriptTxt = self.stcScript.GetText() # python script text
            ### don't allow importing other libararies
            if "import" in scriptTxt:
                ### remove all lines, which contain "import"
                while "import" in scriptTxt:
                    idx1 = scriptTxt.index("import")
                    idx2 = scriptTxt.index("\n", idx1) + 1
                    scriptTxt = scriptTxt[:idx1] + scriptTxt[idx2:]
                self.stcScript.SetText(scriptTxt)
            ### start thread to run script
            args = (self.q2m, self.q2t, scriptTxt, copy(self.origCSVTxt), ) 
            self.th = Thread(target=self.runScript, args=args)
            self.th.start()
        elif objName == "saveGraph_btn":
            ### start thread to save 
            self.th = Thread(target=self.saveGraph)
            self.th.start()
     
    #-------------------------------------------------------------------

    def onCheckBox(self, event, objName=""):
        """ wx.CheckBox was changed.
        
        Args:
            event (wx.Event)
            objName (str, optional): objName to emulate wx.CheckBox event 
                with the given name. 
        
        Returns: None
        """
        if DEBUG: print("AnimalBehaviourCoderFrame.onCheckBox()")

        if self.flagBlockUI: return 
        
        if objName == "":
            obj = event.GetEventObject()
            objName = obj.GetName()
            isChkBoxEvent = True 
        else:
        # funcion was called by some other function without wx.Event
            obj = wx.FindWindowByName(objName, self.panel["tp"])
            isChkBoxEvent = False 
        
        objVal = obj.GetValue() # True/False 

        if objName == "interpolate_chk":
            pass

    #-------------------------------------------------------------------
    
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
        self.Destroy()
    
    #-------------------------------------------------------------------

    def openCSVFile(self):
        """ Open data CSV file. 
        
        Args:
            None
        
        Returns:
            None 
        """
        if DEBUG: print("GraphDrawerFrame.openCSVFile()")

        ### choose result CSV file 
        wc = 'CSV files (*.csv)|*.csv' 
        dlg = wx.FileDialog(self, 
                            "Open CSV file",
                            defaultDir=FPATH,
                            wildcard=wc, 
                            style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_CANCEL: return
        dlg.Destroy()
        csvFP = dlg.GetPath()
        
        ### check file existence
        if not path.isfile(csvFP):
        # file doesn't exist
            msg = "File doesn't exist."%(csvFP)
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return

        ### update CSV file path
        self.csvFP = csvFP
        obj = wx.FindWindowByName("csvFP_txt", self.panel["tp"])
        obj.SetValue(path.basename(csvFP))
        ### set CSV text
        f = open(csvFP, 'r')
        csvTxt = f.read()
        f.close()
        self.stcCSV.SetEditable(True)
        self.stcCSV.SetText(csvTxt)
        self.stcCSV.SetEditable(False) # CSV text is read-only
        self.origCSVTxt = copy(csvTxt) # keep original CSV text
        ### get graph type
        obj = wx.FindWindowByName("graphType_cho", self.panel["tp"])
        self.graphType = obj.GetString(obj.GetSelection())
        self.graphType = self.graphType.split(":")[0].strip()
         
        self.pgd = ProcGraphData(self, csvFP) # init instance class
          # for processing graph data 
        self.loadData() # load CSV data
     
    #-------------------------------------------------------------------
    
    def loadData(self):
        """ load CSV data 

        Args: None
        
        Returns: None
        """
        if DEBUG: print("GraphDrawerFrame.loadData()")

        csvTxt = self.stcCSV.GetText()
        try:
            self.pgd.loadData(csvTxt) # load CSV data
        except Exception as e: # failed to load data
            self.csvFP = ""
            self.pgd.csvFP = ""
            self.stcCSV.SetEditable(True)
            self.stcCSV.SetText("")
            self.stcCSV.SetEditable(False) # CSV text is read-only
            msg = "Failed to load CSV data\n"
            msg += str(e)
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
        self.panel["gp"].Refresh() # draw graph
    
    #-------------------------------------------------------------------
    
    def onPaint(self, event):
        """ painting graph

        Args:
            event (wx.Event)

        Returns:
            None
        """
        if DEBUG: print("GraphDrawerFrame.onPaint()")

        if self.csvFP == "": return

        event.Skip()
        
        dc = wx.PaintDC(self.panel["gp"])        
        self.pgd.drawGraph(dc)
            
    #-------------------------------------------------------------------

    def onTimer(self, event, flag):
        """ Processing on wx.EVT_TIMER event
        
        Args:
            event (wx.Event)
            flag (str): Key (name) of timer
        
        Returns:
            None
        """
        #if DEBUG: print("VideoRW.onTimer()") 

        ### receive (last) data from queue
        rData = None
        while True: 
            ret = receiveDataFromQueue(self.q2m)
            if ret == None: break
            rData = ret # store received data
        if rData == None: return
        
        if flag == "prog":
            if rData[0].startswith("Finished"):
                self.th.join()
                self.flagBlockUI = False 
                self.bp_sTxt.SetLabel("")
                wx.MessageBox(rData[1], "Info", wx.OK|wx.ICON_INFORMATION)
                if rData[0] == "Finished script running":
                # thread ran Python script, editting CSV text
                    self.stcCSV.SetEditable(True)
                    self.stcCSV.SetText(rData[2]) # update CSV
                    self.stcCSV.SetEditable(False) # CSV text is read-only
                    self.loadData() # re-load CSV data
            else:
                self.bp_sTxt.SetLabel(rData[0])
                self.panel["bp"].Refresh()
    
    #-------------------------------------------------------------------
    
    def onClickGraph(self, event):
        """ Processing when user clicked graph

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onClickGraph()")
        
        if self.flagBlockUI or self.csvFP == "": return

        self.playSnd("leftClick") 
        
        mp = event.GetPosition()
        
        if self.graphType.startswith("CV2020"):
            
            ### check whether classification label (in legend) is clicked 
            for cl in self.pgd.clR.keys():
                r = self.pgd.clR[cl]
                if r[0] <= mp[0] <= r[2] and r[1] <= mp[1] <= r[3]:
                # clicked
                    self.pgd.initUITask() # delete current uiTask
                    ### process clicked class
                    self.pgd.uiTask["showThisClassOnly"] = cl
                    self.bp_sTxt.SetLabel(cl)
                    self.panel["gp"].Refresh() # re-draw graph
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
                    self.panel["gp"].Refresh() # re-draw graph
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
                        self.panel["gp"].Refresh() # re-draw graph
                        return
            ### nothing specific is clicked, delete label of bp_sTxt 
            self.bp_sTxt.SetLabel("")
            self.pgd.initUITask() # delete current uiTask
            self.panel["gp"].Refresh() # re-draw graph
     
    #-------------------------------------------------------------------
    
    def onMouseMove(self, event):
        """ Mouse pointer moving on graph area
        Show some info

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onMouseMove()")

        if self.flagBlockUI or self.csvFP == "": return

        mp = event.GetPosition()
     
    #-------------------------------------------------------------------
    
    def onMouseRClick(self, event):
        """ Mouse right click on graph area.
        [Currently no functionality implmented]

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onMouseRClick()")
        
        if self.flagBlockUI or self.csvFP == "": return

        #self.panel["gp"].Refresh() # re-draw graph

    #-------------------------------------------------------------------
    
    def callback(self, rData, flag=""):
        """ call back function after running thread
        
        Args:
            rData (tuple): Received data from queue at the end of thread running
            flag (str): Indicator of origianl operation of this callback
        
        Returns:
            None
        """
        if DEBUG: print("GraphDrawerFrame.callbackFunc()")
        
        if flag == "finalizeSavingVideo":
            msg = 'Saved.\n'
            msg += self.savVidFP 
            wx.MessageBox(msg, "Info", wx.OK|wx.ICON_INFORMATION)
            self.jumpToFrame(0) 
        # show current frame
        self.displayFrameImage(self.vRW.currFrame, flagMakeDispImg=True) 
        self.flagBlockUI = False
        self.panel["gp"].Refresh() # re-draw graph
    
    #-------------------------------------------------------------------
    
    def onSpace(self, event):
        """ start/stop continuous play 

        Args:
            event (wx.Event)

        Returns:
            None
        """ 
        if DEBUG: print("GraphDrawerFrame.onSpace()")

        if self.flagBlockUI or self.csvFP == "": return

        if self.isRunning == False:
            self.isRunning = True
            self.timer["run"] = wx.CallLater(5, self.play)
        else:
            try: # stop timer
                self.timer["run"].Stop() 
                self.timer["run"] = None
            except: pass
            self.isRunning = False
            
    #-------------------------------------------------------------------
    
    def play(self):
        """ load the next frame and move forward if it's playing.

        Args: None

        Returns: None
        """ 
        if DEBUG: print("GraphDrawerFrame.play()")

        self.jumpToFrame(-1) # load next frame
        
        if self.isRunning:
            if self.vRW.fi >= self.endDataIdx: # reached end of available data
                self.onSpace(None) # stop 
            else:
                self.timer["run"] = wx.CallLater(5, self.play) # to next frame

    #-------------------------------------------------------------------
    
    def saveGraph(self):
        """ Save the current graph and CSV

        Args: None

        Returns: None
        """
        if DEBUG: print("GraphDrawerFrame.save()") 
        
        ### file names to write
        timestamp = get_time_stamp().replace("_","")[:14]
        fn = path.basename(self.csvFP) # current CSV file name 
        graphFN = fn.replace(".csv", "_graph_%s.png"%(timestamp)) 
        graphFP = path.join(self.outputPath, graphFN)
        csvFN = fn.replace(".csv", "_graph_%s.csv"%(timestamp))
        csvFP = path.join(self.outputPath, csvFN)
        
        msg = "Saving graph ..."
        self.q2m.put((msg,), True, None)

        ### save graph
        obj = wx.FindWindowByName("imgSavRes_cho", self.panel["gsp"])
        res = obj.GetString(obj.GetSelection()).strip().strip("()")
        res = [int(x) for x in res.split(",")] # resolution of graph image
        bmp = wx.Bitmap(res[0], res[1], depth = -1) 
        memDC = wx.MemoryDC()
        memDC.SelectObject(bmp)
        self.pgd.drawGraph(memDC, flag="save") # draw graph
        memDC.SelectObject(wx.NullBitmap)
        img = bmp.ConvertToImage()
        img.SaveFile(graphFP, wx.BITMAP_TYPE_PNG)

        msg = "Saving CSV ..."
        self.q2m.put((msg,), True, None)

        ### save CSV
        fh = open(csvFP, 'w')
        fh.write(self.stcCSV.GetText())
        fh.close()
        
        msg = "Finished saving graph"
        msg2 = 'Saved\n'
        msg2 += graphFN + "\n"
        msg2 += csvFN + "\n"
        msg2 += " in output folder."
        self.q2m.put((msg,msg2), True, None)
    
    #-------------------------------------------------------------------
    
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
        self.savVidFP = self.csvFP.replace(".csv", "_rev_%s%s"%(timestamp, ext))
        self.vRW.initWriter(self.savVidFP, 
                            video_fSz, 
                            self.callback, 
                            self.makeDispImg,
                            self.bp_sTxt)
        self.flagBlockUI = True 
    
    #-------------------------------------------------------------------
    
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

    #-------------------------------------------------------------------
    
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

    #-------------------------------------------------------------------

#=======================================================================

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

#=======================================================================

class GraphDrawerApp(wx.App):
    def OnInit(self):
        if DEBUG: print("GraphDrawerApp.OnInit()")
        self.frame = GraphDrawerFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#=======================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-w': GNU_notice(1)
        elif sys.argv[1] == '-c': GNU_notice(2)
    else:
        GNU_notice(0)
        app = GraphDrawerApp(redirect = False)
        app.MainLoop()

