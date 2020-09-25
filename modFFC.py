# coding: UTF-8
"""
Frequenty used functions and classes

Dependency:
    wxPython (4.0), 
    Numpy (1.17), 

last editted: 2020.09.15.
"""

import sys, errno, colorsys
from threading import Thread 
from os import path, strerror
from datetime import datetime

import wx
import wx.lib.scrolledpanel as sPanel
import numpy as np
import cv2

DEBUG = False

#-------------------------------------------------------------------------------

def GNU_notice(idx=0):
    """ Function for printing GNU copyright statements

    Args:
        idx (int): Index to determine which statement to print out.

    Returns:
        None

    Examples:
        >>> GNU_notice(0)
        Copyright (c) ...
        ...
        run this program with option '-c' for details.
    """
    if DEBUG: print("modFFC.GNU_notice()")

    if idx == 0:
        year = datetime.now().year
        msg = "Copyright (c) %i Jinook Oh, Sylvia Cremer.\n"%(year)
        msg += "This program comes with ABSOLUTELY NO WARRANTY;"
        msg += " for details run this program with the option `-w'."
        msg += "This is free software, and you are welcome to redistribute"
        msg += " it under certain conditions;"
        msg += " run this program with the option `-c' for details."
    elif idx == 1:
        msg = "THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED"
        msg += " BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING"
        msg += " THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE"
        msg += " PROGRAM 'AS IS' WITHOUT WARRANTY OF ANY KIND, EITHER"
        msg += " EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE"
        msg += " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A"
        msg += " PARTICULAR PURPOSE."
        msg += " THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE"
        msg += " PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU"
        msg += " ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR"
        msg += " CORRECTION."
    elif idx == 2:
        msg = "You can redistribute this program and/or modify it under" 
        msg += " the terms of the GNU General Public License as published"
        msg += " by the Free Software Foundation, either version 3 of the"
        msg += " License, or (at your option) any later version."
    print(msg)

#-------------------------------------------------------------------------------

def chkFPath(fp):
    """ Check whether file/folder exists
    If not found, raise FileNotFoundError

    Args:
        fp: file or folder path to check

    Returns:
        None
    
    Examples:
        >>> chkFPath('./test/test1.txt')

    Raises:
       FileNotFoundError: When 'fp' is not a valid file-path. 
    """
    if DEBUG: print("modFFC.chkFPath()")
    
    rslt = False 
    if path.isdir(fp): rslt = True
    elif path.isfile(fp): rslt = True
    if rslt == False:
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), fp)

#-------------------------------------------------------------------------------

def get_time_stamp(flag_ms=False):
    """ Function to return string which contains timestamp.

    Args:
        flag_ms (bool, optional): Whether to return microsecond or not

    Returns:
        ts_ret (str): Timestamp string to return

    Examples:
        >>> print(get_time_stamp())
        2019_09_10_16_21_56
    """
    if DEBUG: print("modFFC.get_time_stamp()")
    
    ts = datetime.now()
    ts_ret = ('%.4i_%.2i_%.2i_%.2i_%.2i_%.2i')%(ts.year, 
                                                ts.month, 
                                                ts.day, 
                                                ts.hour, 
                                                ts.minute, 
                                                ts.second)
    if flag_ms == True: ts_ret += '_%.6i'%(ts.microsecond)
    return ts_ret

#-------------------------------------------------------------------------------

def writeFile(file_path, txt='', mode='a'):
    """ Function to write a text or numpy file.

    Args:
        file_path (str): File path for output file.
        txt (str): Text to print in the file.
        mode (str, optional): File opening mode.

    Returns:
        None

    Examples:
        >>> writeFile('logFile.txt', 'A log is written.', 'a')
    """
    if DEBUG: print("writeFile()")
    
    f = open(file_path, mode)
    f.write(txt)
    f.close()

#-------------------------------------------------------------------------------

def str2num(s, c=''):
    """ Function to convert string to an integer or a float number.
    
    Args: 
        s (str): String to process 
        c (str): Intented conversion

    Returns:
        oNum (None/ int/ float):
          Converted number or None (when it failed to convert).

    Examples:
        >>> print(str2num('test'))
        None
        >>> print(str2num('3'))
        3 
        >>> print(str2num('3.0'))
        3.0
        >>> print(str2num('3.0', 'int'))
        3
    """
    if DEBUG: print("modFFC.str2num()")
    
    oNum = None 
    if c != '': # conversion method is given
        try: oNum = eval('%s(%s)'%(c, s)) # try the intended conversion
        except: pass
    else: # no conversion is specified
        try:
            oNum = int(s) # try to convert to integer first
        except:
            try: oNum = float(s) # then, float
            except: pass
    return oNum 

#-------------------------------------------------------------------------------

def lst2rng(inputLst):
    """ Convert a list to a list of ranges; change consecutive numbers into
    a range ([beginning-number, end-number]).
    
    Args: 
        inputLst (list): list of integers 

    Returns:
        outputLst (list): list of ranges

    Examples:
        >>> lst2rng([1, 2, 3, 5, 6, 10, 11, 12, 100, 102, 103]) 
        [[1, 3], [5, 6], [10, 12], [100, 100], [102, 103]]
    """
    if DEBUG: print("modFFC.lst2rng()")

    outputLst = [[inputLst[0]]]
    consecIdx = outputLst[-1][0]
    for i in range(1, len(inputLst)):
        currIdx = inputLst[i]
        consecIdx += 1
        if consecIdx == currIdx:
            continue
        else:
            outputLst[-1].append(consecIdx-1)
            outputLst.append([currIdx]) 
            consecIdx = outputLst[-1][0]
    if len(outputLst[-1]) == 1:
        outputLst[-1].append(consecIdx)
    return outputLst

#-------------------------------------------------------------------------------

def load_img(fp, size=(-1,-1), flag='wx'):
    """ Load an image using wxPython or OpenCV functions.

    Args:
        fp (str): File path of an image to load. 
        size (tuple): Output image size.
        flag (str): 'wx' or 'cv'

    Returns:
        img (wx.Image)

    Examples:
        >>> img1 = load_img("test.png")
        >>> img2 = load_img("test.png", size=(300,300))
    """
    if DEBUG: print("modFFC.load_img()")
    
    chkFPath(fp) # chkeck whether file exists
    
    if flag == 'wx':
        tmp_null_log = wx.LogNull() # for not displaying 
          # the tif library warning
        img = wx.Image(fp, wx.BITMAP_TYPE_ANY)
        del tmp_null_log
        if size != (-1,-1) and type(size[0]) == int and \
          type(size[1]) == int: # appropriate size is given
            if img.GetSize() != size:
                img = img.Rescale(size[0], size[1])
    
    elif flag == 'cv':
        img = cv2.imread(fp)
        if size != (-1,-1) and type(size[0]) == int and type(size[1]) == int:
            img = cv2.resize(img, size)

    return img

#-------------------------------------------------------------------------------

def getColorInfo(img, pt, m=1):
    """ Get color information around the given position (pt)

    Args:
        img (numpy.ndarray): Image array
        pt (tuple): x, y coordinate
        m (int): Margin to get area around the 'pt'

    Returns:
        colInfo (dict): Information about color
    """ 
    if DEBUG: print("modFFC.getColorInfo()")

    r = [pt[0]-m, pt[1]-m, pt[0]+m+1, pt[1]+m+1] # rect
    roi = img[r[1]:r[3],r[0]:r[2]] # region of interest
    col = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    col = col.reshape((col.shape[0]*col.shape[1], col.shape[2]))
    colInfoKey1 = ['hue', 'sat', 'val']
    colInfoKey2 = ['med', 'std']
    colInfo = {}
    for k1i in range(len(colInfoKey1)):
        k1 = colInfoKey1[k1i]
        for k2 in colInfoKey2:
            k = k1 + "_" + k2 
            if k2 == 'med': colInfo[k] = int(np.median(col[:,k1i]))
            elif k2 == 'std': colInfo[k] = int(np.std(col[:,k1i]))
    return colInfo 

#-------------------------------------------------------------------------------

def getCamIdx(nCam=3):
    """ Returns indices of attached webcams, after confirming
      each webcam returns its image.

    Args:
        nCam (int): Number of cams to check 

    Returns:
        idx (list): Indices of webcams

    Examples:
        >>> getCamIdx()
        [0]
    """
    if DEBUG: print("modFFC.getCamIdx()")

    idx = []
    for i in range(nCam):
        cap = cv2.VideoCapture(i)
        ret, f = cap.read()
        del(cap)
        if ret == True: idx.append(i)
    return idx

#-------------------------------------------------------------------------------
    
def drawBMPonMemDC(bmpSz, dFunc):
    """ Make memoryDC and draw bitmap with given drawing function 
    
    Args:
        bmpSz (tuple): Width and height of bitmap to draw
        dFunc (function): To draw something on bitmap 
    
    Returns:
        bmp (wx.Bitmap): Drawn bitmap image 
    """
    if DEBUG: print("modFFC.drawBMPonMemDC()") 

    bmp = wx.Bitmap(bmpSz[0], bmpSz[1], depth=32)
    memDC = wx.MemoryDC()
    memDC.SelectObject(bmp)
    memDC.SetBackground(wx.Brush((0,0,0,255)))
    memDC.Clear()
    dFunc(memDC) # draw 
    memDC.SelectObject(wx.NullBitmap)
    return bmp 

#-------------------------------------------------------------------------------

def set_img_for_btn(imgPath, btn, imgPCurr=None, imgPDis=None, 
                    imgPFocus=None, imgPPressed=None):
    """ Set image(s) for a wx.Button

    Args:
        imgPath (str): Path of default image file. 
        btn (wx.Button): Button to put image(s).
        imgPCurr (str): Path of image for when mouse is over.
        imgPDis (str): Path of image for when button is disabled.
        imgPFocus (str): Path of image for when button has the keyboard focus.
        imgPPressed (str): Path of image for when button was pressed.

    Returns:
        btn (wx.Button): Button after processing.

    Examples:
        >>> btn = set_img_for_btn('btn1img.png', wx.Button(self, -1, 'testButton'))
    """
    if DEBUG: print("modFFC.set_img_for_btn()")
    
    imgPaths = dict(all=imgPath, current=imgPCurr, disabled=imgPDis,
                    focus=imgPFocus, pressed=imgPPressed)
    for key in imgPaths.keys():
        fp = imgPaths[key]
        if fp == None: continue
        img = load_img(fp)
        bmp = wx.Bitmap(img)
        if key == 'all': btn.SetBitmap(bmp)
        elif key == 'current': btn.SetBitmapCurrent(bmp)
        elif key == 'disabled': btn.SetBitmapDisabled(bmp)
        elif key == 'focus': btn.SetBitmapFocus(bmp)
        elif key == 'pressed': btn.SetBitmapPressed(bmp)
    return btn

#-------------------------------------------------------------------------------

def getWXFonts(initFontSz=8, numFonts=5, fSzInc=2, 
               fontFaceName="Arial", weight=wx.FONTWEIGHT_NORMAL, 
               style=wx.FONTSTYLE_NORMAL, underline=False):
    """ For setting up several fonts (wx.Font) with increasing size.

    Args:
        initFontSz (int): Initial (the smallest) font size.
        numFonts (int): Number of fonts to return.
        fSzInc (int): Increment of font size.
        fontFaceName (str, optional): Font face name.
        weight (int): Font weight.
        style (int): Font style.
        underline (bool): To underline text or not.

    Returns:
        fonts (list): List of several fonts (wx.Font)

    Examples:
        >>> fonts = getWXFonts(8, 3)
        >>> fonts = getWXFonts(8, 3, 5, 'Arial')
    """
    if DEBUG: print("modFFC.getWXFonts()")

    if fontFaceName == "":
        if 'darwin' in sys.platform: fontFaceName = "Monaco"
        else: fontFaceName = "Courier"
    fontSz = initFontSz 
    fonts = []  # larger fonts as index gets larger 
    for i in range(numFonts):
        fonts.append(
                        wx.Font(
                                fontSz, 
                                wx.FONTFAMILY_SWISS, 
                                style, 
                                weight,
                                underline, 
                                faceName=fontFaceName,
                               )
                    )
        fontSz += fSzInc 
    return fonts

#-------------------------------------------------------------------------------

def addWxWidgets(w, self, pk):
    """ Make wxPython widgets

    Args:
        w (list): List of widget data
        self (wx.Frame): Frame to work on
        pk (str): Panel key string

    Return:
        widLst (list): List of made widgets 
        pSz (list): Resultant panel size after adding widgets
    """
    panel = self.panel[pk]
    gbs = self.gbs[pk]
    pSz = [0, 0] # panel size
    widLst = []
    row = 0
    for ri in range(len(w)):
        _width = 0
        col = 0
        for ci in range(len(w[ri])):
            wd = w[ri][ci]
            if "size" in wd.keys(): size = wd["size"]
            else: size = (-1, -1)
            if "style" in wd.keys(): style = wd["style"]
            else: style = 0
            if wd["type"] == "sTxt": # wx.StaticText
                _w = wx.StaticText(panel, -1, label=wd["label"], size=size,
                                   style=style) 

            if wd["type"] == "sLn": # wx.StaticLine
                _w = wx.StaticText(panel, -1, size=size, style=style) 
            
            elif wd["type"] == "txt": # wx.TextCtrl
                _w = wx.TextCtrl(panel, -1, value=wd["val"], size=size, 
                                 style=style)
                if "numOnly" in wd.keys() and wd["numOnly"]:
                    _w.Bind(
                        wx.EVT_CHAR, 
                        lambda event: self.onTextCtrlChar(event,
                                                          isNumOnly=True)
                        )
            elif wd["type"] == "btn": # wx.Button
                lbl = ""
                if "label" in wd.keys(): lbl = wd["label"]
                _w = wx.Button(panel, -1, label=lbl, size=size,
                               style=style)
                if "img" in wd.keys(): set_img_for_btn(wd["img"], _w) 
                if hasattr(self, "onButtonPressDown") and \
                  callable(getattr(self, "onButtonPressDown")): 
                    _w.Bind(wx.EVT_LEFT_DOWN, self.onButtonPressDown)
            elif wd["type"] == "chk": # wx.CheckBox
                _w = wx.CheckBox(panel, id=-1, label=wd["label"], size=size,
                                 style=style)
                val = False
                if "val" in wd.keys(): val = wd["val"]
                _w.SetValue(val)
                if hasattr(self, "onCheckBox") and \
                  callable(getattr(self, "onCheckBox")): 
                    _w.Bind(wx.EVT_CHECKBOX, self.onCheckBox)
            elif wd["type"] == "cho": # wx.Choice
                _w = wx.Choice(panel, -1, choices=wd["choices"], size=size,
                               style=style)
                if hasattr(self, "onChoice") and \
                  callable(getattr(self, "onChoice")): 
                    _w.Bind(wx.EVT_CHOICE, self.onChoice)
                _w.SetSelection(_w.FindString(wd["val"]))
            elif wd["type"] == "radB": # wx.RadioBox
                _w = wx.RadioBox(panel, -1, label=wd["label"], size=size,
                                 choices=wd["choices"], style=style,
                                 majorDimension=wd["majorDimension"])
                if hasattr(self, "onRadioBox") and \
                  callable(getattr(self, "onRadioBox")): 
                    _w.Bind(wx.EVT_RADIOBOX, self.onRadioBox)
                _w.SetSelection(_w.FindString(wd["val"]))
            elif wd["type"] == "sld": # wx.Slider
                _w = wx.Slider(panel, -1, size=size, value=wd["val"],
                               minValue=wd["minValue"], maxValue=wd["maxValue"],
                               style=style)
                if hasattr(self, "onSlider") and \
                  callable(getattr(self, "onSlider")): 
                    _w.Bind(wx.EVT_SCROLL, self.onSlider)
            elif wd["type"] == "cPk": # wx.ColourPickerCtrl
                _w = wx.ColourPickerCtrl(panel, -1, size=size, 
                                         colour=wd["color"], style=style)
                if hasattr(self, "onColourPicker") and \
                  callable(getattr(self, "onColourPicker")): 
                    _w.Bind(wx.EVT_COLOURPICKER_CHANGED, self.onColourPicker)
            elif wd["type"] == "panel": # wx.Panel
                _w = wx.Panel(panel, -1, size=size, style=style)
            if "name" in wd.keys(): _w.SetName("%s_%s"%(wd["name"], wd["type"]))
            if "wrapWidth" in wd.keys(): _w.Wrap(wd["wrapWidth"])
            if "font" in wd.keys(): _w.SetFont(wd["font"])
            if "fgColor" in wd.keys(): _w.SetForegroundColour(wd["fgColor"]) 
            if "bgColor" in wd.keys(): _w.SetBackgroundColour(wd["bgColor"])
            if "tooltip" in wd.keys(): _w.SetToolTip(wd["tooltip"])
            widLst.append(_w)
            if "border" in wd.keys(): bw = wd["border"]
            else: bw = 5 
            if "flag" in wd.keys(): flag = wd["flag"]
            else: flag = (wx.ALIGN_CENTER_VERTICAL|wx.ALL)
            add2gbs(gbs, _w, (row,col), (1, wd["nCol"]), bw=bw, flag=flag)
            _width += _w.GetSize()[0]
            col += wd["nCol"] 
        row += 1
        if _width > pSz[0]: pSz[0] = _width
        pSz[1] += _w.GetSize()[1] + 10
    return widLst, pSz

#-------------------------------------------------------------------------------

def setupStaticText(panel, label, name=None, size=None, 
                    wrapWidth=None, font=None, fgColor=None, bgColor=None):
    """ Initialize wx.StatcText widget with more options
    
    Args:
        panel (wx.Panel): Panel to display wx.StaticText.
        label (str): String to show in wx.StaticText.
        name (str, optional): Name of the widget.
        size (tuple, optional): Size of the widget.
        wrapWidth (int, optional): Width for text wrapping.
        font (wx.Font, optional): Font for wx.StaticText.
        fgColor (wx.Colour, optional): Foreground color 
        bgColor (wx.Colour, optional): Background color 

    Returns:
        wx.StaticText: Created wx.StaticText object.

    Examples :
        (where self.panel is a wx.Panel, and self.fonts[2] is a wx.Font object)
        >>> sTxt1 = setupStaticText(self.panel, 'test', font=self.fonts[2])
        >>> sTxt2 = setupStaticText(self.panel, 
                                    'Long text................................',
                                    font=self.fonts[2], 
                                    wrapWidth=100)
    """ 
    if DEBUG: print("modFFC.setupStaticText()")

    sTxt = wx.StaticText(panel, -1, label)
    if name != None: sTxt.SetName(name)
    if size != None: sTxt.SetSize(size)
    if wrapWidth != None: sTxt.Wrap(wrapWidth)
    if font != None: sTxt.SetFont(font)
    if fgColor != None: sTxt.SetForegroundColour(fgColor) 
    if bgColor != None: sTxt.SetBackgroundColour(bgColor)
    return sTxt

#-------------------------------------------------------------------------------

def updateFrameSize(wxFrame, w_sz):
    """ Set window size exactly to a user-defined window size (w_sz)
    , excluding counting menubar/border/etc.

    Args:
        wxFrame (wx.Frame): Frame to resize.
        w_sz (tuple): Client size. 

    Returns:
        None

    Examples:
        >>> updateFrameSize(self, (800,600))
    """
    if DEBUG: print("updateFrameSize()")

    ### set window size to w_sz, excluding counting menubar/border/etc.
    _diff = (wxFrame.GetSize()[0]-wxFrame.GetClientSize()[0], 
             wxFrame.GetSize()[1]-wxFrame.GetClientSize()[1])
    _sz = (w_sz[0]+_diff[0], w_sz[1]+_diff[1])
    wxFrame.SetSize(_sz) 
    wxFrame.Refresh()

#-------------------------------------------------------------------------------

def add2gbs(gbs, 
            widget, 
            pos, 
            span=(1,1), 
            bw=5, 
            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL):
    """ Add 'widget' to given 'gbs'.
    
    Args:
        gbs (wx.GridBagSizer).
        widget (wxPython's widget such as wx.StaticText, wx.Choice, ...).
        pos (tuple): x and y cell indices for positioning 'widget' in 'gbs'.
        span (tuple): width and height in terms of cells in 'gbs'.
        bw (int): Border width.
        flag (int): Flags for styles.
    
    Returns:
        None
    
    Examples:
        >>> add2gbs(self.gbs["ui"], sTxt, (0,0), (1,1))
    """
    if DEBUG: print("modFFC.add2gbs()")
    
    gbs.Add(widget, pos=pos, span=span, border=bw, flag=flag)

#-------------------------------------------------------------------------------

def preProcUIEvt(frame, event, objName, objType):
    """ Conduct some common processes when there was UI event in
    an wxPython object.
    
    Args:
        frame (wx.Frame): Frame that calls this function 
        event (wx.Event)
        objName (str): Name of the widget, supposed to cause the event.
        objType (str): Type of the widget.
    
    Returns:
        flag_term (bool): flag to quit function
        obj (object): widget that caused the event.
        objName (str): Name of the object.
        wasFuncCalledViaWxEvent (bool): whehter caller function was 
            called via wx.Event or directly called from other function.
        objVal (str): Value of the object, if available.
    """
    if DEBUG: print("modFFC.preProcUIEvt()")

    if objName == "":
        obj = event.GetEventObject()
        objName = obj.GetName()
        wasFuncCalledViaWxEvent = True
    else:
    # funcion was called by some other function without wx.Event
        obj = wx.FindWindowByName(objName, frame)
        wasFuncCalledViaWxEvent = False
   
    if frame.flagBlockUI or obj.IsEnabled() == False: flag_term = True 
    else: flag_term = False
   
    objVal = ""
    if objType in ["txt", "spin", "chk", "sld"]:
        objVal = obj.GetValue()
    elif objType in ["cho", "radB"]:
        objVal = obj.GetString(obj.GetSelection()) # text of chosen option
    
    return flag_term, obj, objName, wasFuncCalledViaWxEvent, objVal 

#-------------------------------------------------------------------------------
    
def startHeavyTask(frame, taskName, targetFunc, args=None): 
    """ start potentially heavy task, using Thread

    Args:
        frame (wx.Frame): Frame that calls this function 
        taskName (str): Name of the task.
        targetFunc (func): Function to run for the task
        args (tuple): Arguments to pass to the target function

    Returns:
        None
    """
    if DEBUG: print("modFFC.startHeavyTask()")

    ### set timer for loading data
    frame.timer[taskName] = wx.Timer(frame)
    frame.Bind(wx.EVT_TIMER,
              lambda event: frame.onTimer(event, taskName),
              frame.timer[taskName])
    frame.timer[taskName].Start(10)
   
    frame.flagBlockUI = True

    ### start thread to write
    if args == None: frame.th = Thread(target=targetFunc)
    else: frame.th= Thread(target=targetFunc, args=args)
    wx.CallLater(10, frame.th.start)

#-------------------------------------------------------------------------------

def stopAllTimers(timer):
    """ Stop all running wxPython timers
    
    Args:
        timer (dict): container of all timers to stop.
    
    Returns:
        timer (dict) 
    """
    if DEBUG: print("modFFC.stopAllTimers()")

    for k in timer.keys():
        if timer[k] != None:
            try: timer[k].Stop()
            except: pass
            timer[k] = None
    return timer

#-------------------------------------------------------------------------------

def showStatusBarMsg(wxFrame, txt, delTime=5000):
    """ Show message on status bar

    Args:
        wxFrame (wx.Frame)
        txt (str): Text to show on status bar
        delTime (int): Duration (in milliseconds) to show the text

    Returns:
        None
    """
    if DEBUG: print("modFFC.showStatusBarMsg()")

    if wxFrame.timer["sb"] != None:
        ### stop status-bar timer
        wxFrame.timer["sb"].Stop()
        wxFrame.timer["sb"] = None
    
    # show text on status bar 
    wxFrame.statusbar.SetStatusText(txt)
    
    ### change status bar color
    if txt == '': bgCol = wxFrame.sbBgCol 
    else: bgCol = '#33aa33'
    wxFrame.statusbar.SetBackgroundColour(bgCol)
    wxFrame.statusbar.Refresh()

    if txt != '' and delTime != -1:
    # showing message and deletion time was given.
        # schedule to delete the shown message
        wxFrame.timer["sb"] = wx.CallLater(delTime, showStatusBarMsg,
                                           wxFrame, '') 


#-------------------------------------------------------------------------------

def cvHSV2RGB(h, s, v): 
    """ convert openCV's HSV color values to RGB values

    Args:
        h (int): Hue (0-180)
        s (int): Saturation (0-255)
        v (int): Value (0-255)

    Returns:
        (tuple): Tuple of RGB values 
    """ 
    if DEBUG: print("modFFC.cvHSV2RGB()")

    h = h / 180.0
    s = s / 255.0
    v = v / 255.0
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

#-------------------------------------------------------------------------------

def getConspicuousCol(col): 
    """ get conspicuous color of a rgb color

    Args:
        col (tuple): RGB color tuple

    Returns:
        (tuple): Tuple of RGB values 
    """ 
    if DEBUG: print("modFFC.getConspicuousCol()")
    
    if col[:3] in [(0, 0, 0), (255, 255, 255)]:
        return (255-col[0], 255-col[1], 255-col[2])
    else:
        col = [col[x]/255.0 for x in range(3)]
        hsv = colorsys.rgb_to_hsv(col[0], col[1], col[2])
        h = (hsv[0] + 0.5) % 1
        s = 1.0 - hsv[1]
        v = 1.0 - hsv[2]
        tmp = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    return tmp

#-------------------------------------------------------------------------------

def calcI2DIRatio(img, dispSz): 
    """ Calculate ratio for resizing frame image to 
        display image (in StaticBitmap, paintDC, etc)

    Args:
        img (numpy.ndarray): Input image.
        dispSz (tuple): Width and height of StaticBitmap to display image.

    Returns:
        ratImg2DispImg (float): Float number for resizing image later.
    """ 
    if DEBUG: print("modFFC.calcFI2DIRatio()")

    if img.shape[1] > dispSz[0] or img.shape[0] > dispSz[1]:
        ratImg2DispImg = float(dispSz[0]) / img.shape[1]
        w = img.shape[1]*ratImg2DispImg
        h = img.shape[0]*ratImg2DispImg
        if h > dispSz[1]:
            ratImg2DispImg = float(dispSz[1]) / img.shape[0]
    else:
        ratImg2DispImg = 1.0
    return ratImg2DispImg 

#-------------------------------------------------------------------------------

def convt_idx_to_ordinal(number):
    """ Convert zero-based index number to ordinal number string
    0->1st, 1->2nd, ...

    Args:
        number (int): An unsigned integer number.

    Returns:
        number (str): Converted string

    Examples:
        >>> convt_idx_to_ordinal(0)
        '1st'
    """
    if DEBUG: print("modFFC.convt_idx_to_ordinal()")
    
    if number == 0: return "1st"
    elif number == 1: return "2nd"
    elif number == 2: return "3rd"
    else: return "%ith"%(number+1)

#-------------------------------------------------------------------------------

def convt_180_to_360(ang):
    """ convert -180~180 degree to 0~360 degree

    Args:
        angle (int): Output angle, 0 indicates right, 90 indicates up, 
            180 indicates left, -90 indicates down
        
    Returns:
       angle (int): Input angle. 0 indicates right, 90 indicates up,
            180 indicates left, 270 indicates down. 
    
    Examples:
        >>> convt_180_to_360(45)
        45
        >>> convt_180_to_360(-45)
        315
    """ 
    if ang < 0: ang = 360 + ang
    return ang

#-------------------------------------------------------------------------------

def convt_360_to_180(angle):
    """ Convert 360 degree system to 180 degree system.

    Args:
        angle (int): Input angle. 0 indicates right, 90 indicates up,
            180 indicates left, 270 indicates down.

    Returns:
        angle (int): Output angle, 0 indicates right, 90 indicates up, 
            180 indicates left, -90 indicates down

    Examples:
        >>> convt_360_to_180(90)
        90
        >>> convt_360_to_180(270)
        -90
    """
    if angle <= 180: return angle
    else: return -(360 % angle)

#-------------------------------------------------------------------------------

def calc_pt_w_angle_n_dist(angle, dist, bPosX=0, bPosY=0, flagScreen=False):
    """ Calculates a point when a angle and a distance is given.

    Args:
        angle (int): 0 indicates right, 90 indicates up, 
            180 or -180 indicates left, -90 indicates down.
        dist (int): Distance in pixel.
        bPosX (int): x-coordinate of base-position.
        bPosY (int): y-coordinate of base-position.
        flagScreen (bool): whether it's for displaying it on screen.
          (y coordinate will be reversed)

    Returns:
        (int, int): x,y coordinate

    Examples:
        >>> calc_pt_w_angle_n_dist(90, 20)
        (0, 20)
        >>> calc_pt_w_angle_n_dist(180, 20)
        (-20, 0)
        >>> calc_pt_w_angle_n_dist(-135, 20)
        (-14, -14)
        >>> calc_pt_w_angle_n_dist(-135, 20, 100, 100)
        (85, 85)
        >>> calc_pt_w_angle_n_dist(-135, 20, 100, 100, True)
        (85, 114)
    """
    if DEBUG: print("modFFC.calc_pt_w_angle_n_dist()")

    s = np.sin(np.deg2rad(angle))
    c = np.cos(np.deg2rad(angle))
    x = int(bPosX + c*dist)
    if flagScreen: y = int(bPosY - s*dist)
    else: y = int(bPosY + s*dist)
    return (x, y)

#-------------------------------------------------------------------------------

def calc_line_angle(pt1, pt2):
    """ Calculates angle of a line, defined with two points (pt1 and pt2)

    Args:
        pt1 (tuple): x, y coordinate of the first point
        pt2 (tuple): x, y coordinate of the second point

    Returns:
        (int): angle of the line; 0=right, 90=upward, -90=downward, 180=left

    Examples:
        >>> calc_line_angle((0,0), (1,0))
        0
        >>> calc_line_angle((0,0), (1,1))
        -45
        >>> calc_line_angle((0,0), (-1,-1))
        135
    """
    return int(np.degrees(np.arctan2(-(pt2[1]-pt1[1]),pt2[0]-pt1[0])))

#-------------------------------------------------------------------------------

def calc_angle_diff(ang1, ang2):
    """ Calculates angle difference between two angles

    Args:
        ang1 (int): Angle between -180 and 180
        ang2 (int): Angle between -180 and 180

    Returns:
        (int): Angle difference (smallest)

    Examples:
        >>> calc_angle_diff(0, 90)
        90
        >>> calc_angle_diff(180, 45)
        135
        >>> calc_angle_diff(180, -90)
        90
    """
    angle_diff = 0
    if (ang1 >= 0 and ang2 >= 0) or (ang1 < 0 and ang2 < 0):
        angle_diff = abs(ang1-ang2)
    elif (ang1 >=0 and ang2 < 0) or (ang1 < 0 and ang2 >= 0):
        ad1 = abs(ang1) + abs(ang2)
        ad2 = 180-abs(ang1) + 180-abs(ang2)
        angle_diff = min(ad1, ad2)
    return angle_diff

#-------------------------------------------------------------------------------

def calc_pt_line_dist(pt, line, flag_line_ends=True):
    """ Calculates distance from a point to a line

    Args:
        pt (tuple): a point
        line (tuple): a line, defined with two points; ((x1,y1), (x2,y2))
        flag_line_ends : whether line ends at (x1,y1) & (x2,y2) 
            or indefinitely extends

    Returns:
        (float): the distance between point and line

    Examples:
        >>> calc_pt_line_dist((0,0), ((1,0), (0,1)))
        0.7071067811865476
        >>> calc_pt_line_dist((0,0), ((1,0), (0.5,0.25)), True)
        0.5590169943749475
        >>> calc_pt_line_dist((0,0), ((1,0), (0.5,0.25)), False)
        0.4472135954999579
    """
    lpt1 = line[0]; lpt2 = line[1]
    ldx = lpt2[0]-lpt1[0]
    ldy = lpt2[1]-lpt1[1]
    sq_llen = ldx**2 + ldy**2 # square length of line
    if sq_llen == 0: # line is a point
        return np.sqrt( (pt[0]-lpt1[0])**2 + (pt[1]-lpt1[1])**2 )
    u = ( (pt[0]-lpt1[0])*ldx + (pt[1]-lpt1[1])*ldy ) / float(sq_llen)
    x = lpt1[0] + u * ldx
    y = lpt1[1] + u * ldy
    if flag_line_ends:
        if u < 0.0: x, y = lpt1[0], lpt1[1] # beyond lpt1-end of segment
        elif u > 1.0: x, y = lpt2[0], lpt2[1] # beyond lpt2-end of segment
    dx = pt[0] - x
    dy = pt[1] - y
    return np.sqrt(dx**2 + dy**2)

#-------------------------------------------------------------------------------

def rot_pt(pt, ct, deg):
    """ Rotate (counter-clockwise) point;pt, around center point;ct
    * y-coordinate follows computer screen coordinate system,
    where 0 is the top row and the row index increases as it comes down

    Args:
        pt (tuple): Point to rotate
        ct (tuple): Center point
        deg (float): Angel to rotate

    Returns:
        (tuple): Rotated point. 
    
    Examples:
        >>> rot_pt((2,2), (1,1), 45)
        (2, 1)
        >>> rot_pt((2,2), (1,1), 180)
        (0, 0)
        >>> rot_pt((2,2), (1,1), -90)
        (0, 2)
    """ 
    if DEBUG: print("modFFC.rot_pt()")

    r = np.deg2rad(deg)
    tx = pt[0]-ct[0]
    ty = pt[1]-ct[1]
    x = (tx * np.cos(r) + ty * np.sin(r)) + ct[0]
    y = (-tx * np.sin(r) + ty * np.cos(r)) + ct[1]
    return (int(np.round(x)), int(np.round(y)))

#-------------------------------------------------------------------------------

def receiveDataFromQueue(q, logFile=''):
    """ Receive data from a queue.

    Args:
        q (Queue): Queue to receive data.
        logFile (str): File path of log file.

    Returns:
        rData (): Data received from the given queue. 

    Examples:
        >>> receiveDataFromQueue(Queue(), 'log.txt')
    """
    if DEBUG: print("modFFC.receiveDataFromQueue()")

    rData = None
    try:
        if q.empty() == False: rData = q.get(False)
    except Exception as e:
        em = "%s, [ERROR], %s\n"%(get_time_stamp(), str(e))
        if path.isfile(logFile) == True: writeFile(logFile, em)
        print(em)
    return rData    

#===============================================================================

class PopupDialog(wx.Dialog):
    """ Class for showing a message to a user.
    Most simple messages can be dealt using wx.MessageBox.
    This class was made to use it as a base class for a dialog box
      with more widgets such as a dialog box to enter
      subject's information (id, gender, age, prior experiences, etc)
      before running an experiment.
    
    Args:
        parent (wx.Frame): Parent object (probably, wx.Frame or wx.Panel).
        id (int): ID of this dialog.
        title (str): Title of the dialog.
        msg (str): Message to show.
        iconFP (str): File path of an icon image.
        font (wx.Font): Font of message string.
        pos (None/ tuple): Position to make the dialog window.
        size (tuple): Size of dialog window.
        flagOkayBtn (bool): Whether to show Ok button.
        flagCancelBtn (bool): Whether to show Cancel button.
        flagDefOK (bool): Whether Ok button has focus by default (so that 
          user can just press enter to dismiss the dialog window).
    """
    def __init__(self, 
                 parent=None, 
                 id=-1, 
                 title="Message", 
                 msg="", 
                 iconFP="", 
                 font=None, 
                 pos=None, 
                 size=(300, 200), 
                 flagOkayBtn=True, 
                 flagCancelBtn=False, 
                 flagDefOK=False):
        if DEBUG: print("PopupDialog.__init__()")

        ### init Dialog
        wx.Dialog.__init__(self, parent, id, title)
        self.SetSize(size)
        if pos == None: self.Center()
        else: self.SetPosition(pos)
        self.Center()
        # init panel
        panel = sPanel.ScrolledPanel(self, -1, pos=(0,0), size=size)

        ### font setup 
        if font == None:
            font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.NORMAL, 
                           wx.FONTWEIGHT_NORMAL, False, "Arial", 
                           wx.FONTENCODING_SYSTEM)

        ##### [begin] set up widgets -----
        gbs = wx.GridBagSizer(0,0)
        row = 0; col = 0
        ### icon image
        if iconFP != "" and path.isfile(iconFP) == True:
            bmp = wx.Bitmap(wxLoadImg(iconFP))
            icon_sBmp = wx.StaticBitmap(panel, -1, bmp)
            iconBMPsz = icon_sBmp.GetBitmap().GetSize()
            add2gbs(gbs, icon_sBmp, (row,col), (1,1))
            col += 1 
        else:
            iconFP = ""
            iconBMPsz = (0, 0)
        ### message to show
        sTxt = wx.StaticText(panel, -1, label=msg)
        sTxt.SetSize((size[0]-max(iconBMPsz[0],100)-50, -1))
        sTxt.SetFont(font)
        if iconFP == "": sTxt.Wrap(size[0]-30)
        else: sTxt.Wrap(size[0]-iconBMPsz[0]-30)
        if iconFP == "": _span = (1,2)
        else: _span = _span = (1,1)
        add2gbs(gbs, sTxt, (row,col), _span)
        ### okay button
        row += 1; col = 0
        btn = wx.Button(panel, wx.ID_OK, "OK", size=(100,-1))
        add2gbs(gbs, btn, (row,col), (1,1))
        if flagOkayBtn: # okay button is shown
            if flagCancelBtn == False or flagDefOK == True:
            # cancel button won't be made or default-okay is set True 
                panel.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
                btn.SetDefault()
        else:
            btn.Hide()
        ### cancel button
        col += 1
        if flagCancelBtn:
            btn = wx.Button(panel, wx.ID_CANCEL, "Cancel", size=(100,-1))
            add2gbs(gbs, btn, (row,col), (1,1))
        else:
            sTxt = wx.StaticText(panel, -1, label=" ")
            add2gbs(gbs, sTxt, (row,col), (1,1))
        ### lay out
        panel.SetSizer(gbs)
        gbs.Layout()
        panel.SetupScrolling()
        ##### [end] set up widgets -----
    
    #---------------------------------------------------------------------------

    def onKeyPress(self, event):
        """ Process key-press event
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: print("PopupDialog.onKeyPress()")

        if event.GetKeyCode() == wx.WXK_RETURN: 
            self.EndModal(wx.ID_OK)
    
#===============================================================================

if __name__ == '__main__':
    pass
