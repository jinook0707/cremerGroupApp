# coding: UTF-8
"""
For color-triggered recording in 'recCams' program.

last edited on 2022-03-27
"""
import sys
from os import path, mkdir, rename
from time import time
from datetime import datetime, timedelta
import logging
logging.basicConfig(
    format="%(asctime)-15s [%(levelname)s] %(funcName)s: %(message)s",
    level=logging.INFO
    )

_path = path.realpath(__file__)
FPATH = path.split(_path)[0] # path of where this Python file is
sys.path.append(FPATH) # add FPATH to path
P_DIR = path.split(FPATH)[0] # parent directory of the FPATH
sys.path.append(P_DIR) # add parent directory 

from modFFC import *
from modCV import *
import cv2
CV_Ver = [int(x) for x in cv2.__version__.split(".")]

DEBUG = False

#===============================================================================

class ColorTriggeredRecording:
    """ Class for color-triggered recording in 'recCams' program.

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    def __init__(self, mainFrame):
        if DEBUG: logging.info(str(locals()))

        ##### [begin] setting up attributes -----
        self.main = mainFrame 
        self.flags = {}
        self.flags["state"] = "watching" # 'watching' or 'storing'
        self.fBuffer = [] # for storing past frame images (with time info)
        self.sLen = 5 # in seconds; if new color-spot is found, 
                       #   frames from -sLen to +sLen will be saved as files 
        self.sSTime = -1 # time when storing started
        ##### [end] setting up attributes -----
   
    #---------------------------------------------------------------------------
    
    def initVars(self):
        """ Init. some variables 

        Args: None

        Returns: None
        """
        if DEBUG: logging.info(str(locals()))
        
        # list of rects bounding previously found color blobs (x, y, w, h)
        self.cRects = [] 

    #---------------------------------------------------------------------------
    
    def procFrame(self, frame, avgFPS):
        """ process frame image

        Args:
            frame (numpy.ndarray): Frame image
            avgFPS (int): Average FPS sent from VideoIn

        Returns:
            frame (numpy.ndarray): Frame image after processing
        """
        if DEBUG: logging.info(str(locals()))

        if hasattr(self, "prevT"):
            print("[procFrame;%s] %.3f"%(self.flags["state"], 
                                         time()-self.prevT))
        self.prevT = time()

        currTime = datetime.now()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        fbCol = [50, 255, 50] # coloring color (on image displayed 
                              #   on 'ip' panel) for found blob
        w = wx.FindWindowByName("bMinRad_txt", self.main.panel["lp"])
        bMinRad = int(w.GetValue()) # min. radius of a new color blob to find
        w = wx.FindWindowByName("bMaxRad_txt", self.main.panel["lp"])
        bMaxRad = int(w.GetValue()) # max. radius of a new color blob to find
        
        # store the current time and frame image in the buffer list
        self.fBuffer.append([currTime, frame.copy()])

        ### limit the buffer list length
        blLen = int(self.sLen*2 * avgFPS)
        if len(self.fBuffer) >= blLen:
            self.fBuffer = self.fBuffer[-blLen:]
        
        # blur frame
        frame = cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT)
        ### find color in the frame
        hsvP = {}
        ci = 0 # currently there's only one color to detect
        for mLbl in ["Min", "Max"]:
            val = []
            for hsvLbl in ["H", "S", "V"]:
                wn = "c-%i-%s-%s_sld"%(ci, hsvLbl, mLbl) # widget name
                w = wx.FindWindowByName(wn, self.main.panel["lp"])
                val.append(w.GetValue())
            hsvP[mLbl] = tuple(val)
        # detect color
        fcRslt = findColor(frame, hsvP["Min"], hsvP["Max"])
        # decrease noise 
        fcRslt = cv2.morphologyEx(fcRslt, cv2.MORPH_OPEN, kernel, iterations=2)
        
        fcIdx = np.where(fcRslt==255)
        frame[fcIdx] = fbCol

        # find blobs using connectedComponentsWithStats
        ccOutput = cv2.connectedComponentsWithStats(fcRslt, connectivity=8) 
       
        ### draw circles with the user defined radiuses to show its size 
        cv2.circle(frame, (bMaxRad, bMaxRad), bMinRad, (0,0,255), -1)
        cv2.circle(frame, (bMaxRad, bMaxRad), bMaxRad, (0,0,255), 1)
        ### draw already registered blob rects
        for cr in self.cRects:
            x1, y1, _w, _h = cr
            x2 = x1 + _w
            y2 = y1 + _h
            cv2.rectangle(frame, (x1, y1), (x2, y2), fbCol, 1)

        if self.flags["state"] == "watching":
        
            ##### [begin] check out new acceptable color blob -----
            isNewBlobFound = False 
            nLabels = ccOutput[0] # number of labels
            labeledImg = ccOutput[1]
            # stat matrix; (left, top, width, height, area)
            stats = list(ccOutput[2])
            newRects = []
            for li in range(1, nLabels):
            # go through each blob label
                l, t, w, h, a = stats[li]
                cpt = [l+int(w/2), t+int(h/2)]

                if min(w, h) >= bMinRad*2:
                # min. length is acceptable 
                    if max(w, h) <= bMaxRad*2:
                    # max. length is acceptable
                        if len(self.cRects) == 0: isNewRect = True
                        else:
                            isNewRect = True 
                            for cr in self.cRects:
                            # go through previously found color blob rects
                                _l, _t, _w, _h = cr
                                _cpt = [_l+int(_w/2), _t+int(_h/2)]
                                dist = np.sqrt((cpt[0]-_cpt[0])**2 + \
                                               (cpt[1]-_cpt[1])**2)
                                if dist < bMinRad*2:
                                # distance to the previously found blob 
                                #   is too close
                                    isNewRect = False
                                #print(li, l, _l, t, _t, l+w, _l+_w, t+h, _t+_h)
                                if l <= _l and t <= _t and \
                                  l+w >= _l+_w and t+h >= _t+_h:
                                # the current blob's rect contains 
                                #   the previously found blob's rect 
                                    isNewRect = False 
                                    break
                        if isNewRect:
                            isNewBlobFound = True
                            # store the rect of the new blob
                            newRects.append([l-bMinRad, t-bMinRad, 
                                             w+bMinRad*2, h+bMinRad*2])
                #print(li, (l,t,w,h), newRects)
            
            if newRects != []: self.cRects += newRects
            ##### [end] check out new acceptable color blob -----

            if isNewBlobFound: 
                # store frame-save beginning time
                self.saveBTime = currTime - timedelta(seconds=self.sLen)
                # store frame-save end time
                self.saveETime = currTime + timedelta(seconds=self.sLen)
                # store the storing start time 
                self.sSTime = time()
                # change the state; will store frames in buffer
                #   for the next 'shLen' seconds
                self.flags["state"] = "storing"

        elif self.flags["state"] == "storing":
            ### display the recording mark while storing
            w = int(frame.shape[1]*0.8)
            h = int(frame.shape[0]*0.8)
            x1 = int(frame.shape[1]/2 - w/2)
            y1 = int(frame.shape[0]/2 - h/2)
            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), fbCol, 1)
            cv2.circle(frame, (x1+15, y1+15), 10, (0,0,255), -1) 

            if time() - self.sSTime >= self.sLen:
            # time for storing has passed

                '''
                ### save frame images in a new folder
                newFolderName = get_time_stamp()
                folderPath = path.join(self.main.outputDir, newFolderName)
                mkdir(folderPath) # make a new folder 
                flagSave = False
                imgCnt = 0
                bT = None # beginning time
                eT = None # end time
                for bTime, bFrame in self.fBuffer:
                    if flagSave:
                        if bTime > self.saveETime: # end of saving time
                            break
                        imgCnt += 1
                        fp = path.join(folderPath, "%06i.jpg"%(imgCnt))
                        cv2.imwrite(fp, bFrame)
                        eT = bTime
                    else:
                        if bTime >= self.saveBTime: # beginning of saving time
                            flagSave = True
                            bT = bTime
                
                ### add FPS value at the end of folder name
                fps = int(imgCnt / (eT-bT).seconds)
                rename(folderPath, folderPath+"_fps%i"%(fps))
                '''

                ### init variables
                self.saveBTime = -1
                self.saveETime = -1
                self.sSTime = -1
                self.flags["state"] = "watching" # back to watching state
        return frame

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == "__main__":
    pass
 
