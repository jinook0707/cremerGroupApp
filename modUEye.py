# coding: UTF-8
"""
Class for using UEye camera from IDS

Dependency:
    PyUEye (4.95.0)

last edited on 2022-03-25
"""
import sys
from os import path, remove, mkdir
from time import time, sleep

import cv2
try:
    from pyueye import ueye
    FLAG_PYUEYE = True
except Exception as e:
    FLAG_PYUEYE = False

from modFFC import *

DEBUG = False

#-------------------------------------------------------------------------------

def getUEyeCamIdx(nCam=4):
    """ Returns indices of attached uEye cameras

    Args:
        nCam (int): Number of cameras to check 

    Returns:
        idx (list): Indices of webcams

    Examples:
        >>> getUEyeCamIdx()
        [0]
    """
    if DEBUG: print("modUEye.getUEyeCamIdx()")

    idx = []
    if not FLAG_PYUEYE: return idx 
    for i in range(1, nCam+1):
    # Index 0: first available camera
    # Index 1-254: The camera with the specified camera ID
        cam = ueye.HIDS(i)
        ret = ueye.is_InitCamera(cam, None)
        if ret == ueye.IS_SUCCESS: idx.append(i)
        ueye.is_ExitCamera(cam)
    return idx

#===============================================================================

class VideoInUEye:
    """ Class for reading frames from a UEye camera 

    Args:
        parent: parent object

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent, cIdx, desiredRes=(-1,-1), fpsLimit=15, 
                 outputFormat="image", ssIntv=1.0, params={}):
        if DEBUG: print("VideoInUEye.__init__()")
        
        ##### [begin] setting up attributes on init. -----
        self.classTag = "videoInUEye-%i"%(cIdx)
        self.parent = parent # parent
        self.cIdx = cIdx # index of cam
        self.outputFormat = outputFormat # image or video
        self.fpsLimit = fpsLimit # limit of frames per second
        self.ssIntv = ssIntv # interval in seconds for saving image from Cam
        self.imgExt = "jpg" # file type when saving frames to images
        self.desiredRes = desiredRes
        self.params = params # other parameters
        self.color = dict(U=100, V=100) # color space U & V
        ##### [end] setting up attributes on init. -----

        self.initCam()
                        
        parent.log("Mod init.", self.classTag) 

    #---------------------------------------------------------------------------

    def initCam(self):
        """ Initialize camera with the given camera index
        
        Args:
            None

        Returns:
            None
        """
        if DEBUG: print("VideoInUEye.initCam()")
        
        self.cam = ueye.HIDS(self.cIdx) # Index 0: first available camera;
                               # 1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImgMem = ueye.c_mem_p()
        self.memId = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        if self.desiredRes != (-1, -1):
            self.recAOI.s32Width = self.desiredRes[0]
            self.recAOI.s32Height = self.desiredRes[1]
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24) # 24: bits per pixel for color mode; 
                                          # take 8 bits per pixel for monochrome
        self.channels = 3 # 3: self.channels for color mode(RGB); 
                          # take 1 channel for monochrome
        self.m_nColMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
        self.bytesPerPixel = int(self.nBitsPerPixel/8)

        ### Starts the driver and establishes the connection to the camera
        nRet = ueye.is_InitCamera(self.cam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        ### Reads out the data hard-coded in the non-volatile camera memory 
        ###   and writes it to the data structure that self.cInfo points to
        nRet = ueye.is_GetCameraInfo(self.cam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        ### You can query additional information about the sensor type 
        ###   used in the camera
        nRet = ueye.is_GetSensorInfo(self.cam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault(self.cam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

        # Set display mode to DIB; IS_SET_DM_DIB/ IS_SET_DM_DIRECT3D
        nRet = ueye.is_SetDisplayMode(self.cam, ueye.IS_SET_DM_DIB)

        ### Set the right color mode
        cModeVal = int.from_bytes(self.sInfo.nColorMode.value, byteorder='big')
        if cModeVal == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.cam, self.nBitsPerPixel, self.m_nColMode)
            self.bytesPerPixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColMode: ", self.m_nColMode)
            print("\tnBitsPerPixel: ", self.nBitsPerPixel)
            print("\tbytesPerPixel: ", self.bytesPerPixel)
            print()
        elif cModeVal == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytesPerPixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColMode: ", self.m_nColMode)
            print("\tnBitsPerPixel: ", self.nBitsPerPixel)
            print("\tbytesPerPixel: ", self.bytesPerPixel)
            print()
        elif cModeVal == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytesPerPixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColMode: ", self.m_nColMode)
            print("\tnBitsPerPixel: ", self.nBitsPerPixel)
            print("\tbytesPerPixel: ", self.bytesPerPixel)
            print()
        else:
            # for monochrome camera models use Y8 mode
            self.m_nColMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytesPerPixel = int(self.nBitsPerPixel / 8)
            print("else")

        ### enable some automatic parameter setting
        enable = ueye.DOUBLE(1)
        disable = ueye.DOUBLE(0)
        zero = ueye.DOUBLE(0)
        _pLst = [
                    #ueye.IS_SET_ENABLE_AUTO_GAIN,
                    #ueye.IS_SET_ENABLE_AUTO_SHUTTER,
                    #ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE,
                    #ueye.IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER,
                 ]
        for _p in _pLst:
            ueye.is_SetAutoParameter(self.cam, _p, enable, zero) 
        '''
        _pLst = [
                    #ueye.IS_SET_ENABLE_AUTO_GAIN,
                    #ueye.IS_SET_ENABLE_AUTO_SHUTTER,
                    #ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE,
                    #ueye.IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER,
                 ]
        for _p in _pLst:
            ueye.is_SetAutoParameter(self.cam, _p, disable, zero)
        '''

        ### set pixel-clock
        pixelClock = ueye.uint(self.params["pixelClock"])
        nRet = ueye.is_PixelClock(self.cam, 
                                  ueye.IS_PIXELCLOCK_CMD_SET, 
                                  pixelClock, 
                                  ueye.sizeof(pixelClock))
        if nRet != ueye.IS_SUCCESS:
            print("Pixel-clock ERROR")
        
        ### set exposure time
        expTime = ueye.double(self.params["exposureTime"])
        nRet = ueye.is_Exposure(self.cam, 
                                ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, 
                                expTime, 
                                ueye.sizeof(expTime))  
        if nRet != ueye.IS_SUCCESS:
            print("Exposure-time ERROR")

        ### set gamma 
        gamma = ueye.INT(self.params["gamma"])
        nRet = ueye.is_Gamma(self.cam, 
                                ueye.IS_GAMMA_CMD_SET, 
                                gamma, 
                                ueye.sizeof(gamma))  
        if nRet != ueye.IS_SUCCESS:
            print("Gamma ERROR")

        # Can be used to set the size and position of 
        #   an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(self.cam, 
                           ueye.IS_AOI_IMAGE_GET_AOI, 
                           self.rectAOI, 
                           ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        ### Prints out some information about the camera and the sensor
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", self.width)
        print("Maximum image height:\t", self.height)
        print()

        # Allocates an image memory for an image having its dimensions 
        #   defined by width and height and its color depth 
        #   defined by self.nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.cam, 
                                     self.width, 
                                     self.height, 
                                     self.nBitsPerPixel, 
                                     self.pcImgMem, 
                                     self.memId)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.cam, self.pcImgMem, self.memId)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.cam, self.m_nColMode)

        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.cam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR") 

        # Enables the queue mode for existing image memory sequences
        self.inqImgRet = ueye.is_InquireImageMem(self.cam,
                                                 self.pcImgMem, 
                                                 self.memId, 
                                                 self.width, 
                                                 self.height, 
                                                 self.nBitsPerPixel, 
                                                 self.pitch)
        
        if self.inqImgRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

    #---------------------------------------------------------------------------

    def run(self, q2m, q2t, recFolder="", flagSendFrame=True):
        """ Function for thread to retrieve image
        and store it as video or image
        
        Args:
            q2m (queue.Queue): Queue to main thread to return message.
            q2t (queue.Queue): Queue from main thread.
            recFolder (str): Folder to save recorded videos/images.
            flagSendFrame (bool): Whether to send frame image to main thread.
        
        Returns:
            None
        """
        if DEBUG: print("VideoInUEye.run()")

        #-----------------------------------------------------------------------
        def handleErr(self, q2m, msg=""): 
            print(msg)
            q2m.put(["ERROR", msg], True, None)
            self.parent.log("Thread stops.", self.classTag)
        #-----------------------------------------------------------------------
       
        def startRecording(cIdx, oFormat, recFolder, fps, fSz, fpsLimit, 
                           ssIntv):
            if DEBUG: print("VideoInUEye.run.startRecording()")
            
            log = "recording starts"
            log += " [%s]"%(oFormat)
            if oFormat == 'video':
                # Define the codec and create VideoWriter object
                #fourcc = cv2.VideoWriter_fourcc(*'X264')
                fourcc = cv2.VideoWriter_fourcc(*'avc1') # for saving mp4 video
                #fourcc = cv2.VideoWriter_fourcc(*'xvid') # for saving avi video
                ofn = "output_cam%.2i_%s.mp4"%(cIdx, get_time_stamp())
                ofp = path.join(recFolder, ofn)
                if fpsLimit >= 1:
                    # get average of the past 10 fps records
                    ofps = int(np.average(fps[-11:-1]))
                else:
                    ofps = fpsLimit 
                # set 'out' as a video writer
                out = cv2.VideoWriter(ofp, fourcc, ofps, fSz, True)
                log += " [%s] [FPS: %i] [FPS-limit: %i]"%(ofn, ofps, fpsLimit)
            elif oFormat == 'image':
                ofn = "output_cam%.2i_%s"%(cIdx, get_time_stamp())
                ofn = path.join(recFolder, ofn)
                # 'out' is used as an index of a image file
                out = 1
                log += " [%s] [Snapshot-interval: %s]"%(ofn, str(ssIntv))
                if not path.isdir(ofn): mkdir(ofn)
            self.parent.log(log, self.classTag)
            return out, ofn

        #-----------------------------------------------------------------------
        
        def stopRecording(out, cIdx):
            if DEBUG: print("VideoInUEye.run.stopRecording()")
            if isinstance(out, cv2.VideoWriter): out.release()
            out = None
            self.parent.log("recording stops.", self.classTag) # log
            return out

        #-----------------------------------------------------------------------

        q2tMsg = '' # queued message sent from main thread
        ofn = '' # output file or folder name
        out = None # videoWriter or number '1' for image file
        fpIntv = 1.0/self.fpsLimit # interval between each frame
        lastFrameProcTime = time()-fpIntv # last frame processing time
        imgSaveTime = time()-self.ssIntv # last time image was saved
        fpsTimeSec = time()
        fpsLastLoggingTime = time()
        fpsLst = [0]
        flagFPSLogging = False
        frameSz = (self.width.value, self.height.value)

        while(self.inqImgRet == ueye.IS_SUCCESS):
            ### limit frame processing 
            if self.fpsLimit != -1:
                if time()-lastFrameProcTime < fpIntv:
                    sleep(0.001)
                    continue
                lastFrameProcTime = time()

            ### fps
            if self.fpsLimit >= 1:
                if time()-fpsTimeSec > 1:
                    if flagFPSLogging:
                        if time()- fpsLastLoggingTime > 60: # one minute passed
                            log = "FPS during the past minute "
                            log += str(fpsLst[:60])
                            self.parent.log(log, self.classTag)
                            fpsLastLoggingTime = time()
                    else:
                        print("[%s] FPS: %i"%(self.classTag, fpsLst[-1]))
                    fpsLst.append(0)
                    fpsTimeSec = time()
                    # keep fps records of the past one minute
                    if len(fpsLst) > 61: fpsLst.pop(0)
                else:
                    fpsLst[-1] += 1

            try:
                ### process queue message (q2t)
                if q2t.empty() == False:
                    try: q2tMsg = q2t.get(False)
                    except: pass
                if q2tMsg != "":
                    if q2tMsg == "quit":
                        break
                    elif q2tMsg == "rec_init":
                        if out == None:
                            out, ofn = startRecording(self.cIdx, 
                                                      self.outputFormat, 
                                                      recFolder,
                                                      fpsLst, 
                                                      frameSz, 
                                                      self.fpsLimit, 
                                                      self.ssIntv)
                    elif q2tMsg == "rec_stop":
                        if out != None:
                            out = stopRecording(out, self.cIdx)

                    elif q2tMsg.startswith("color"):
                        cs, val = q2tMsg.split("-")
                        cs = cs[-1] # U or V
                        self.color[cs] = int(val)
                        # set saturation 
                        nRet = ueye.is_SetSaturation(self.cam,
                                                     self.color["U"], 
                                                     self.color["V"])

                    elif q2tMsg.startswith("expTime"):
                        expTime = ueye.double(float(q2tMsg.split("-")[1]))
                        # set exposure
                        nRet = ueye.is_Exposure(self.cam, 
                                            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, 
                                            expTime, 
                                            ueye.sizeof(expTime))  
                    
                    elif q2tMsg.startswith("gamma"):
                        gamma = ueye.INT(int(q2tMsg.split("-")[1]))
                        nRet = ueye.is_Gamma(self.cam, 
                                             ueye.IS_GAMMA_CMD_SET, 
                                             gamma, 
                                             ueye.sizeof(gamma))  

                    q2tMsg = ""

                # get frame image
                frame = ueye.get_data(self.pcImgMem, 
                                      self.width, 
                                      self.height, 
                                      self.nBitsPerPixel, 
                                      self.pitch, 
                                      copy=True)

                frame = np.reshape(frame, (self.height.value, 
                                           self.width.value, 
                                           self.bytesPerPixel))
                frame = frame[:,:,:3]

                if self.outputFormat == 'video':
                # video recording
                    if out != None:
                        out.write(frame) # write a frame to video
                
                elif self.outputFormat == 'image':
                # image recording
                    if time()-imgSaveTime >= self.ssIntv:
                    # interval time has passed
                        if out != None:
                            imgSaveTime = time()
                            ### save frame image
                            ts = get_time_stamp(flag_ms=True)
                            fp = path.join(ofn, "f_%s.%s"%(ts, self.imgExt))
                            q = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                            cv2.imwrite(fp, frame, q)
                        
                if flagSendFrame and q2m.empty():
                    ### send frame via queue to main thread with some info
                    if self.fpsLimit >= 1:
                        if len(fpsLst) > 10:
                            avgFPS = int(np.average(fpsLst[-11:-1]))
                        else:
                            avgFPS = fpsLst[-1]
                    else:
                        avgFPS = -1
                    _d = ["frameImg", self.cIdx, frame, avgFPS]
                    q2m.put(_d, True, None)

            except Exception as e:
                msg = "ERROR:: %s"%(e)
                handleErr(self, q2m, msg)
                return

            sleep(0.001)

        self.parent.log("Thread stops.", self.classTag)

    #---------------------------------------------------------------------------

    def close(self):
        """
        close the mod 

        Args:
            None

        Returns:
            None 
        """
        if DEBUG: print("VideoInUEye.close()")

        # releases an image memory that was allocated using is_AllocImageMem() 
        #   and removes it from the driver management
        ueye.is_FreeImageMem(self.cam, self.pcImgMem, self.memId)

        # disables the camera handle and releases the data structures 
        #   and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.cam)
        
        self.parent.log("Mod stops.", self.classTag)

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass



