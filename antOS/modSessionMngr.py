# coding: UTF-8
"""
This is for managing an experimental session procedure of CATOS-AntOS program.

last edited on 2023-07-16
"""

import queue
import datetime as dt
from os import remove
from shutil import copyfile
from time import sleep, time

import wx

from modFFC import *
from modCV import *

MyLogger = setMyLogger("AntOS_modSessionMngr")
DEBUG = False 

#===============================================================================

class ESessionManager:
    """ Experimental session manager
    """
    def __init__(self, parent):
        if DEBUG: MyLogger.info(str(locals()))
        
        self.parent = parent
        self.state = ""
        
        if parent.chosenCamIdx == []:
            msg = "Empty camera index list."
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return

        ##### [begin] setting up attributes -----
        self.classTag = "sessionMngr"
        self.sessionStartTime = -1 
        self.timer = {}
        self.state = "pause" # current state of the session/ 
                             # currently, 'inSession' or 'pause'
        cdt = dt.datetime.now()
        txt = wx.FindWindowByName("initSessionDelay_txt", parent.panel["ml"])
        try: initDelay = int(txt.GetValue())
        except: initDelay = 0
        txt = wx.FindWindowByName("sessionDur_txt", parent.panel["ml"])
        try: endDelay = int(txt.GetValue())
        except: endDelay = 24
        # session-start date-time
        self.startDT = cdt + dt.timedelta(seconds=initDelay) 
        # session-end date-time
        self.endDT = self.startDT + dt.timedelta(hours=endDelay)
        self.intv = {} # intervals to conduct operations
        self.intv["reg"] = 100 # interval (in milliseconds) for routine 
        self.intv["procFrame"] = 900 # interval (in milliseconds)
                                     #   for processing frame image
        w = wx.FindWindowByName("frameSav_txt", parent.panel["ml"])
        # interval (in seconds) for saving frame
        self.intv["frameSav"] = int(w.GetValue()) 
        self.lT = {} # last time operations were done.
        self.lT["frameSav"] = {}
        for ci in parent.chosenCamIdx:
            self.lT["frameSav"][ci] = -1 # time when frame image was saved
        if parent.flags["raspberryPi"]:
            ledDelay = max(0, initDelay-30) # Turn on light 30 sec earlier 
            # set IR LED light-on time
            self.irTurnOnTime = cdt + dt.timedelta(seconds=ledDelay)

            w = wx.FindWindowByName("tempRead_txt", parent.panel["ml"])
            # interval (in seconds) to read temperature sensor
            self.intv["tempRead"] = int(w.GetValue()) 
            self.lT["tempRead"] = -1 # last reading time 
        self.timerKeys = ["sMReg"]
        self.flagIR1 = False # to turn on/off the second IR LED
        ##### [end] setting up attributes -----
        
        ### set regular timer for checking and processing data  
        parent.timer["sMReg"] = wx.Timer(parent)
        parent.Bind(wx.EVT_TIMER,
                    lambda event: self.onTimer(event, "sMReg"),
                    parent.timer["sMReg"])
        parent.timer["sMReg"].Start(self.intv["reg"])

        ### set timer for processing frame image 
        parent.timer["procFrame"] = wx.Timer(parent)
        parent.Bind(wx.EVT_TIMER,
                    lambda event: self.onTimer(event, "procFrame"),
                    parent.timer["procFrame"])
        parent.timer["procFrame"].Start(self.intv["procFrame"])

        parent.log("Mod init.", self.classTag) # leave a log

    #---------------------------------------------------------------------------
    
    def startSession(self):
        """ Start session.
        
        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent
        if p.flags["raspberryPi"]:
            p.startMods(mod='raspGPIO') # start raspberry-pi GPIO mod
        btn = wx.FindWindowByName("chkCamView_btn", p.panel["tp"])
        btn.Disable() # disable checking camera view button
        self.state = "inSession"

        ### set result file path for this session
        rsltFN = "rslt_%s.csv"%(get_time_stamp())
        header = "Cam-idx, Timestamp, Key, Value\n"
        self.rsltFP = []
        for roiIdx in range(p.numROIs):
            self.rsltFP.append(path.join(p.roiOutputFP[roiIdx], rsltFN))
            writeFile(self.rsltFP[roiIdx], header)

        ### init some variables
        self.prev_grey = {} # greyscale image of the previous frame
        for ci in p.chosenCamIdx:
            self.prev_grey[ci] = []
            for roiIdx in range(p.numROIs):
                self.prev_grey[ci].append(None)

        # start camera 
        p.startMods('videoIn', dict(videoInFPSLimit=3))

        self.params = {} # minor parameters 

        self.sessionStartTime = time() # store session start time
        
        p.log("Beginning of session.", self.classTag) # leave a log

    #---------------------------------------------------------------------------
    
    def stopSession(self):
        """ Stop session.
        
        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent
        p.stopMods(mod='videoIn') # stop camera 
        if p.flags["raspberryPi"] and p.mods["raspGPIO"] != None:
            p.mods["raspGPIO"].sendSignal("irLed0", False) # turn off IR LED 
            if self.flagIR1:
                p.mods["raspGPIO"].sendSignal("irLed1", False)
            p.stopMods(mod='raspGPIO') # stop raspberry-pi GPIO mod
        p.sTime_txt.SetLabel('0:00:00')
        btn = wx.FindWindowByName("chkCamView_btn", p.panel["tp"])
        btn.Enable() # enable checking camera view button

        ### copy log files to each ROI folder
        logFiles = list(glob(path.join(p.outputFP, "log_*.txt"))) 
        for fp in logFiles:
            for folderPath in p.roiOutputFP:
                fn = path.basename(fp)
                newFP = path.join(folderPath, fn)
                copyfile(fp, newFP)
        #for fp in logFiles: remove(fp) # delete log files in output folder

        self.state = "pause"
        
        p.log("End of session.", self.classTag) # leave a log

        self.sessionStartTime = -1

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
        
        p = self.parent
        if flag == "sMReg": # regular timer of session-manager
            cdt = dt.datetime.now() 

            if p.flags["raspberryPi"] and p.mods["raspGPIO"] != None:
                if self.irTurnOnTime != -1 and cdt >= self.irTurnOnTime:
                    # turn on IR LED light
                    p.mods["raspGPIO"].sendSignal("irLed0", True)
                    if self.flagIR1:
                        p.mods["raspGPIO"].sendSignal("irLed1", True)
                    self.irTurnOnTime = -1 

            if self.state == "pause":
                if cdt >= self.startDT:
                    self.startSession() # start session
                    lbl = "Session start"
                else:
                    secStr = str((self.startDT - cdt).seconds)
                    lbl = "Session starts after %s"%(secStr)
                p.sessionStartSTxt.SetLabel(lbl)
            elif self.state == "inSession":
                if cdt >= self.endDT:
                    self.stopSession() # finish the session
                    p.stopMods(mod="sessionMngr")

                if p.flags["raspberryPi"]:
                    ### temperature loggin
                    if self.lT["tempRead"] == -1 or \
                      time()-self.lT["tempRead"] >= self.intv["tempRead"]:
                        p.mods["raspGPIO"].logTemp()
                        self.lT["tempRead"] = time()

        elif flag == "procFrame":
            if self.state == "inSession":
                self.procFrameImg() # process frame image
    
    #---------------------------------------------------------------------------
    
    def procFrameImg(self):
        """ process frame image 

        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        p = self.parent

        ### set minor parameters
        if self.params == {}:
            for ci in p.chosenCamIdx:
                if not ci in p.mods["videoIn"].keys(): self.params={}; return 
                viMod = p.mods["videoIn"][ci]
                if not hasattr(viMod, "frame"): self.params={}; return 
                fSh = viMod.frame.shape
                self.params[ci] = {}
                # threshold for size (w+h) of an motion contour threshold 
                self.params[ci]["motionThr"] = [fSh[0]*0.01, fSh[0]*0.25]
                # radius for drawing a motion point
                self.params[ci]["motionPtRad"] = max(1, int(fSh[0]*0.002))
                ### params for drawing the centroid of color blob
                self.params[ci]["colCentRad"] = max(1, int(fSh[0]*0.007))
                self.params[ci]["colCentThck"] = max(1, int(fSh[0]*0.003))
                # thresholds for size (w+h) of an ant color contour
                self.params[ci]["antCntThr"] = (max(1, int(fSh[0]*0.005)),
                                                int(fSh[0]*0.25))

        rois = [] # region of interest (for masking image)
        for roiIdx in range(p.numROIs):
            obj = wx.FindWindowByName("roi%i_txt"%(roiIdx), p.panel["ml"])
            try: rois.append([int(x) for x in obj.GetValue().split(",")])
            except: rois.append([])
        ts = get_time_stamp(flag_ms=True) # time-stamp
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        
        for i, ci in enumerate(p.chosenCamIdx):
        # go through camera indices
        # !! Currently (2023.Jan), only one camera is being used. If multiple
        #    cameras are required, ROIs setting in GUI and processing here must 
        #    be modified for each camera !!
            if not ci in p.mods["videoIn"].keys(): continue
            viMod = p.mods["videoIn"][ci]
            if not hasattr(viMod, "frame"): continue
            params = self.params[ci]
            origFrame = viMod.frame.copy() # get frame image
            fSh = origFrame.shape
            if len(rois[i]) != 4: rois[i] = (0, 0, fSh[1], fSh[0])

            frameImgSaved = False
            for roiIdx in range(p.numROIs):
            # go through ROIs
                rslt = {}
                
                ### cut off image with ROI
                r = rois[roiIdx]
                rx1 = max(0, r[0])
                rx2 = min(fSh[1], r[0]+r[2])
                ry1 = max(0, r[1])
                ry2 = min(fSh[0], r[1]+r[3])
                frame = origFrame[ry1:ry2, rx1:rx2]
                #frame = maskImg(frame, [rois[roiIdx]], col=(255,255,255))

                ### save frame image
                if self.lT["frameSav"][ci] == -1 or \
                  time()-self.lT["frameSav"][ci] >= self.intv["frameSav"]:
                    fp = path.join(p.roiOutputFP[roiIdx], 
                                   "%s_cam-%02i.jpg"%(ts, ci))
                    cv2.imwrite(fp, frame)
                    frameImgSaved = True
              
                # greyscale image 
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

                ##### [begin] motion detection -----
                m_pts = [] # motion points
                m_thr = 30 #15 threshold for cv2.threshold
                maxNM = 100 # max number of motion contours in a frame
                if type(self.prev_grey[ci][roiIdx]) == np.ndarray and \
                  self.prev_grey[ci][roiIdx].shape == grey.shape:
                    diff = cv2.absdiff(grey, self.prev_grey[ci][roiIdx])
                    ret, diff = cv2.threshold(diff, m_thr, 255, 
                                              cv2.THRESH_BINARY) 
                    #diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
                    cnts, hierarchy = cv2.findContours(diff,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        # ignore too small or large motion segment
                        mThr = params["motionThr"]
                        if bw+bh < mThr[0] or bw+bh > mThr[1]:
                            continue
                        _x = bx + int(bw/2)
                        _y = by + int(bh/2)
                        m_pts.append((_x, _y))
                        if len(m_pts) > maxNM:
                        # too many motions are recorded in a frame
                            m_pts = []
                            break
                self.prev_grey[ci][roiIdx] = grey.copy()
                if len(m_pts) > 0:
                    # store the motion points 
                    rslt["motionPts"] = m_pts
                ##### [end] motion detection -----

                if len(m_pts) > 0: # there were some motion point(s)
                    ##### [begin] color detection -----
                    for cTag in p.cTags:
                        hsv = {} 
                        for mm in ["Min", "Max"]:
                            hsv[mm] = []
                            for ck in ["H", "S", "V"]:
                                hsv[mm].append(p.hsv[cTag][ck][mm])
                        fcRslt = findColor(frame, 
                                           tuple(hsv["Min"]), 
                                           tuple(hsv["Max"]))
                        fcRsltSum = np.sum(fcRslt)
                        if fcRsltSum > 0:
                            ### get contours of the found color
                            cnts, hierarchy = cv2.findContours(
                                                       fcRslt,
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE
                                                       )
                            if len(cnts) > 0:
                                # thresholds for a contour size
                                thrMin, thrMax = params["antCntThr"] 
                                if cTag == "ants":
                                    rslt["antCnts"] = []
                                elif cTag == "focalAntMarker":
                                    cntInfo = []
                                for cnt in cnts:
                                # go through each blob's contour
                                    bx, by, bw, bh = cv2.boundingRect(cnt)
                                    # ignore too small or too large blob 
                                    if bw+bh < thrMin or bw+bh > thrMax: 
                                        continue
                                    if cTag == "ants":
                                        # store
                                        rslt["antCnts"].append((bx, by, bw, bh))
                                    elif cTag == "focalAntMarker":
                                        cntInfo.append([bw+bh, bx, by, bw, bh])
                                if cTag == "focalAntMarker" and len(cntInfo) > 0:
                                    ### store the largest blob's center point
                                    bsz, bx, by, bw, bh = sorted(cntInfo, 
                                                             reverse=True)[0]
                                    rslt["colCent"] = (bx+int(bw/2), 
                                                       by+int(bh/2)) 
                    ##### [end] color detection -----
              
                if p.flags["showFrame"]:
                    #### [begin] display some results on frame image -----
                    pt1 = (rx1, ry1)
                    pt2 = (rx2, ry2)
                    cv2.rectangle(origFrame, pt1, pt2, (0,200,0), 2)
                    if len(m_pts) > 0: # there were some motion point(s)
                        '''
                        ### [for DEBUGGING]
                        if p.flags["showCol"] and fcRsltSum > 0:
                            fcIdx = np.where(fcRslt==255)
                            frame[fcIdx] = (50, 127, 255) # mark detected colors
                        '''
                        if "colCent" in rslt.keys():
                            # display centroid of ant's color
                            cv2.circle(frame, rslt["colCent"], 
                                       params["colCentRad"], (255,50,50), 
                                       params["colCentThck"])
                        if "antCnts" in rslt.keys():
                            ### draw each contour bounding rect
                            for cnt in rslt["antCnts"]:
                                x, y, w, h = cnt
                                cv2.rectangle(frame, (x,y), (x+w, y+h), 
                                              (0,255,255), 2)
                    ### draw motion points
                    for mpt in m_pts:
                        x = mpt[0] + rois[roiIdx][0]
                        y = mpt[1] + rois[roiIdx][1]
                        cv2.circle(origFrame, (x,y), params["motionPtRad"], 
                                   (50,255,50), -1)
                    #### [end] display some results on frame image ----- 

                ### write results
                oStr = ""
                for key, value in rslt.items():
                    v = ""
                    if key == "colCent":
                        if value != (-1, -1):
                            v = str(value).replace(",", "/")
                    elif key in ["motionPts", "antCnts"]:
                        for pt in value: 
                            v += str(pt).replace(",", "/")
                    else:
                        v = str(value)
                    if v != "": oStr += "%i, %s, %s, %s\n"%(ci, ts, key, v)
                if oStr != "": writeFile(self.rsltFP[roiIdx], oStr)

            # display the frame image on UI
            p.displayCamFrame(ci, origFrame)

            if frameImgSaved:
                self.lT["frameSav"][ci] = time()

    #---------------------------------------------------------------------------
    
    def close(self):
        """ closesession 

        Args: None
        
        Returns: None
        """
        if DEBUG: MyLogger.info(str(locals()))

        if self.state == "inSession": self.stopSession()

        ### stop timers, started by this class
        for k in self.timerKeys:
            if self.parent.timer[k] != None:
                try: self.parent.timer[k].Stop()
                except: pass
                self.parent.timer[k] = None
        
        self.parent.log("Mod close.", self.classTag) # leave a log
    
    #---------------------------------------------------------------------------
    
#===============================================================================




