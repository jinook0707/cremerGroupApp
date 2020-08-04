# coding: UTF-8

"""
A convenience app to launch various apps, used in Cremer group 
    for studying social immunity of ants.

This program was coded and tested in Ubuntu Linux 18.04. 

Jinook Oh, Cognitive Biology department, University of Vienna
2020.Jun.

Dependency:
    wxPython (4.0)

------------------------------------------------------------------------
Copyright (C) 2020 Jinook Oh, Sylvia Cremer 
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

v.0.1 [20200617]: initial development 
"""

import sys
from os import getcwd, path
from glob import glob

import wx
import wx.lib.scrolledpanel as SPanel 

from modFFC import GNU_notice, getWXFonts, updateFrameSize, stopAllTimers
from modFFC import load_img, add2gbs, setupStaticText, PopupDialog

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is

DEBUG = False
__version__ = "0.1"

#===============================================================================

class CremerGroupAppFrame(wx.Frame):
    """ Frame for 
        
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self):
        if DEBUG: print("CremerGroupAppFrame.__init__()")

        ### init frame
        wPos = [0, 25]
        wg = wx.Display(0).GetGeometry()
        wSz = (int(wg[2]*0.333), int(wg[3]*0.5))
        wx.Frame.__init__(
              self,
              None,
              -1,
              "CremerGroupApp v.%s"%(__version__),
              pos = tuple(wPos),
              size = tuple(wSz),
              style=wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER|wx.MAXIMIZE_BOX),
              )
        self.SetBackgroundColour('#333333')
        iconPath = path.join(FPATH, "icon.png")
        if path.isfile(iconPath):
            self.SetIcon(wx.Icon(iconPath)) # set app icon
        
        ##### [begin] setting up attributes -----
        self.wPos = wPos # window position
        self.wSz = wSz # window size
        self.fonts = getWXFonts(initFontSz=8, numFonts=3)
        pi = self.setPanelInfo()
        self.pi = pi # pnael information
        self.gbs = {} # for GridBagSizer
        self.panel = {} # panels
        self.timer = {} # timers
        self.chosenAppIdx = 0
        self.appNames = [] # names of apps
        self.appDesc = [] # description string for each app
        for p in sorted(glob(path.join(FPATH, "*"))):
            if path.isdir(p) and path.isfile(path.join(p, "__init__.py")):
                dirName = path.basename(p)
                self.appNames.append(dirName) # store app name 
                  # it should be as same as its containing folder name
                ### get description of the file,
                ###   which should be located at top of the file.
                appFilePath = path.join(p, "%s.py"%(dirName))
                fh = open(appFilePath, "r")
                lines = fh.readlines()
                fh.close()
                isInitCommentStarted = False
                desc = ""
                for line in lines:
                    if line[:3] in ["\"\"\"", "'''"]: # multiline comment
                        if not isInitCommentStarted: # comment starting
                            isInitCommentStarted = True
                            line = line[3:]
                        else: # comment ended
                            break
                    if isInitCommentStarted:
                        desc += line
                # store description
                self.appDesc.append(desc)
        ##### [end] setting up attributes -----
        
        ### create panels
        for pk in pi.keys():
            self.panel[pk] = SPanel.ScrolledPanel(
                                                  self,
                                                  name="%s_panel"%(pk),
                                                  pos=pi[pk]["pos"],
                                                  size=pi[pk]["sz"],
                                                  style=pi[pk]["style"],
                                                 )
            self.panel[pk].SetBackgroundColour(pi[pk]["bgCol"])
        
        ##### [begin] set up top panel interface -----
        tpSz = pi["tp"]["sz"]
        self.gbs["tp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        sTxt = setupStaticText(self.panel["tp"], 
                               label="App: ", 
                               font=self.fonts[1], 
                               fgColor="#cccccc", 
                               bgColor="#333333")
        add2gbs(self.gbs["tp"], sTxt, (row,col), (1,1))
        col += 1
        cho = wx.Choice(self.panel["tp"],
                        -1,
                        name="app_cho",
                        size=(200,-1),
                        choices=self.appNames)
        cho.Bind(wx.EVT_CHOICE, self.onChoice)
        cho.SetSelection(0)
        self.chosenAppIdx = 0
        add2gbs(self.gbs["tp"], cho, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label= "What's this app?",
                        size=(150, -1),
                        name="desc_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        col += 1
        btn = wx.Button(self.panel["tp"],
                        -1,
                        label= "Run App",
                        size=(100, -1),
                        name="runApp_btn")
        btn.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
        add2gbs(self.gbs["tp"], btn, (row,col), (1,1))
        self.panel["tp"].SetSizer(self.gbs["tp"])
        self.gbs["tp"].Layout()
        self.panel["tp"].SetupScrolling()
        ##### [end] set up top panel interface -----

        ##### [begin] set up bottom panel interface -----
        bpSz = pi["bp"]["sz"]
        self.gbs["bp"] = wx.GridBagSizer(0,0)
        row = 0; col = 0
        self.desc_txt = wx.TextCtrl(self.panel["bp"],
                                    value="",
                                    size=(bpSz[0]-4, bpSz[1]-tpSz[1]),
                                    style=wx.TE_READONLY|wx.TE_MULTILINE)
        self.desc_txt.SetForegroundColour("#cccccc")
        self.desc_txt.SetFont(self.fonts[2])
        add2gbs(self.gbs["bp"], self.desc_txt, (row,col), (1,1))
        self.panel["bp"].SetSizer(self.gbs["bp"])
        self.gbs["bp"].Layout()
        self.panel["bp"].SetupScrolling()
        ##### [end] set up bottom panel interface -----
        
        ### set up menu
        menuBar = wx.MenuBar()
        pyABCMenu = wx.Menu()
        quitItem = pyABCMenu.Append(wx.Window.NewControlId(), 
                                    item="Quit\tCTRL+Q")
        menuBar.Append(pyABCMenu, "&CremerGroupApp")
        self.Bind(wx.EVT_MENU, self.onClose, quitItem)
        self.SetMenuBar(menuBar) 

        ### keyboard binding
        exitId = wx.NewIdRef(count=1)
        self.Bind(wx.EVT_MENU, self.onClose, id=exitId)
        accel_tbl = wx.AcceleratorTable([
                        (wx.ACCEL_CTRL,  ord('Q'), exitId ), 
                                        ]) 
        self.SetAcceleratorTable(accel_tbl) 

        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        self.timer["sb"] = None
        
        updateFrameSize(self, wSz)
        self.Bind(wx.EVT_CLOSE, self.onClose)

    #---------------------------------------------------------------------------
   
    def setPanelInfo(self):
        """ Set up panel information.
        
        Args:
            None
         
        Returns:
            pi (dict): Panel information.
        """
        if DEBUG: print("CremerGroupApp.setPanelInfo()")
        
        wSz = self.wSz # window size
        pi = {} # panel information to return
        # top panel 
        pi["tp"] = dict(pos=(0, 0), 
                        sz=(wSz[0], 50), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        tpSz = pi["tp"]["sz"]
        # bottom panel
        pi["bp"] = dict(pos=(0, tpSz[1]), 
                        sz=(wSz[0], wSz[1]-tpSz[1]), 
                        bgCol="#333333", 
                        style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        return pi
    
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
        if DEBUG: print("CremerGroupApp.onButtonPressDown()")

        if objName == '':
            obj = event.GetEventObject()
            objName = obj.GetName()
        else:
            obj = wx.FindWindowByName(objName, self.panel["tp"])

        if not obj.IsEnabled(): return

        if objName == "desc_btn":
            if self.chosenAppIdx == -1: return
            self.desc_txt.SetValue(self.appDesc[self.chosenAppIdx])
        
        elif objName == "runApp_btn":
            if self.chosenAppIdx == -1: return
            aName = self.appNames[self.chosenAppIdx].upper()
            runningFrame = None
            if aName == "DRAWGRAPH":
                from drawGraph import drawGraph
                runningFrame = drawGraph.GraphDrawerFrame()
            elif aName == "LONGTERMREC":
                from longTermRec import longTermRec
                runningFrame = longTermRec.LongTermRecFrame()
            elif aName == "ABC_DATANAV":
                from abc_dataNav import abc_dataNav
                runningFrame = abc_dataNav.ABC_DataNavFrame()
            if runningFrame != None: runningFrame.Show()
            APP.SetTopWindow(runningFrame)
    
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
        if DEBUG: print("CremerGroupAppFrame.onChoice()")

        if objName == "":
            obj = event.GetEventObject()
            objName = obj.GetName()
            isChoiceEvent = True 
        else:
        # funcion was called by some other function without wx.Event
            obj = wx.FindWindowByName(objName, self.panel["tp"])
            isChoiceEvent = False 
        
        if not obj.IsEnabled(): return
        objVal = obj.GetSelection()
        objStr = obj.GetString(objVal)

        if objName == "app_cho":
            self.chosenAppIdx = objVal
            self.desc_txt.SetValue("")

    #---------------------------------------------------------------------------

    def onClose(self, event):
        """ Close this frame.
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: print("CremerGroupAppFrame.onClose()")

        stopAllTimers(self.timer)
        wx.CallLater(100, self.Destroy)
    
    #---------------------------------------------------------------------------

#===============================================================================

class CremerGroupApp(wx.App):
    def OnInit(self):
        if DEBUG: print("CremerGroupApp.OnInit()")
        self.frame = CremerGroupAppFrame()
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
        APP = CremerGroupApp(redirect = False)
        APP.MainLoop()


