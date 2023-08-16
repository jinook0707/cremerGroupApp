# coding: UTF-8
"""
Classes for using FLIR camera (from Point Grey)

Dependency:
    PySpin (2.4.0.143)

last editted: 2023-02-18
"""
import sys, queue
from time import time, sleep

import cv2, PySpin

from modFFC import *

DEBUG = False
#===============================================================================

class Spinnaker:
    """ Class for using PySpin; Spinnaker, to use camera from FLIR 

    Args:
        parent: parent object

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent):
        if DEBUG: print("Spinnaker.__init__()")
        
        ##### [begin] setting up attributes on init. -----
        self.parent = parent
        # Retrieve singleton reference to system object
        self.snSys = PySpin.System.GetInstance()
        v = self.snSys.GetLibraryVersion() # get current library version
        self.version = "%d.%d.%d.%d"%(v.major, v.minor, v.type, v.build)
        # retrieve list of cameras from the system
        self.camLst = self.snSys.GetCameras()
        self.nCams = self.camLst.GetSize()
        if self.nCams == 0:
            self.close()
            print("ERROR:: No camera detected.")
        
    #---------------------------------------------------------------------------

    def getCam(self, cIdx):
        """
        get camera instance

        Args:
            cIdx (int): Camera index to retrieve. 

        Returns:
            None 
        """
        if DEBUG: print("Spinnaker.getCam()")

        cam = self.camLst[cIdx]
        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            cam.Init() # initialize camera
            nodemap = cam.GetNodeMap() # retrieve GenICam nodemap

        except PySpin.SpinnakerException as ex:
            self.close()
            print('Error: %s' % ex)

        return (cam, nodemap, nodemap_tldevice)

    #---------------------------------------------------------------------------

    def deinitCam(self, cam):
        """
        deinitialize the camera instance 

        Args:
            cam: PySpin's camera instance 

        Returns:
            None 
        """
        if DEBUG: print("Spinnaker.deinitCam()")

        cam.DeInit()
        del cam

    #---------------------------------------------------------------------------

    def close(self):
        """
        close the instance of Spinnaker 

        Args:
            None

        Returns:
            None 
        """
        if DEBUG: print("Spinnaker.close()")

        self.camLst.Clear() # clear camera list before releasing system
        self.snSys.ReleaseInstance() # release system instance

    #---------------------------------------------------------------------------

#===============================================================================

class VideoInSpinnaker:
    """ Class for reading frames from a FLIR camera

    Args:
        parent: parent object

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent, cIdx):
        if DEBUG: print("Spinnaker.__init__()")
        
        ##### [begin] setting up attributes on init. -----
        self.classTag = "videoInSpinnaker-%i"%(cIdx)
        self.parent = parent # parent
        self.cIdx = cIdx # index of cam
        c = parent.spinnaker.getCam(cIdx)
        self.cam, self.nodemap, self.nodemap_tldevice = c
        ##### [end] setting up attributes on init. -----
                        
        parent.log("Mod init.", self.classTag) 

    #---------------------------------------------------------------------------

    def run(self, q2m, q2t):
        """
        continuously get frame images from a FLIR camera 

        Args:
            q2m (queue.Queue): Queue to main thread to return message.
            q2t (queue.Queue): Queue from main thread.

        Returns:
            None
        """
        if DEBUG: print("Spinnaker.getImg()")

        #-----------------------------------------------------------------------
        def handleErr(self, q2m, msg=""): 
            print(msg)
            q2m.put(["ERROR", msg], True, None)
            self.parent.log("Thread stops.", self.classTag)
        #-----------------------------------------------------------------------
        
        q2tMsg = ""

        ### set pixel format
        #pfStr = "BayerRG8" 
        pfStr = "Mono8"

        if pfStr != "Mono8":
            nodePF = PySpin.CEnumerationPtr(self.nodemap.GetNode("PixelFormat"))
            if not PySpin.IsAvailable(nodePF) or not PySpin.IsWritable(nodePF):
                msg = "Unable to set Pixel Format (enum retrieval). Aborting..."
                handleErr(self, q2m, msg)
                return
            nodePFbyName = nodePF.GetEntryByName(pfStr)
            if not PySpin.IsAvailable(nodePFbyName) or \
               not PySpin.IsReadable(nodePFbyName):
                msg = "Unable to set Pixel Format to"
                msg += " %s (entry retrieval)."%(pfStr)
                msg += " Aborting..."
                handleErr(self, q2m, msg)
                return
            pixelFormat = nodePFbyName.GetValue()
            nodePF.SetIntValue(pixelFormat)

        sNodemap = self.cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(
                                sNodemap.GetNode('StreamBufferHandlingMode')
                                )
        if not PySpin.IsAvailable(node_bufferhandling_mode) or \
           not PySpin.IsWritable(node_bufferhandling_mode):
            msg = "Unable to set stream buffer handling mode.. Aborting..."
            handleErr(self, q2m, msg)
            return

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or \
           not PySpin.IsReadable(node_newestonly):
            msg = "Unable to set stream buffer handling mode.. Aborting..."
            handleErr(self, q2m, msg)
            return

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        try:
            nodeAcquiMode = PySpin.CEnumerationPtr(
                                        self.nodemap.GetNode('AcquisitionMode')
                                        )
            if not PySpin.IsAvailable(nodeAcquiMode) or \
               not PySpin.IsWritable(nodeAcquiMode):
                msg = "Unable to set acquisition mode to continuous"
                msg += " (enum retrieval). Aborting..."
                handleErr(self, q2m, msg)
                return

            # Retrieve entry node from enumeration node
            nodeAcquiModeCont = nodeAcquiMode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(nodeAcquiModeCont) or \
               not PySpin.IsReadable(nodeAcquiModeCont):
                msg = "Unable to set acquisition mode to continuous"
                msg += " (entry retrieval). Aborting..."
                handleErr(self, q2m, msg)
                return
            # Retrieve integer value from entry node
            acquisition_mode_continuous = nodeAcquiModeCont.GetValue()
            # Set integer value from entry node as new value of enumeration node
            nodeAcquiMode.SetIntValue(acquisition_mode_continuous)

            #  Begin acquiring images
            #
            #  *** notes ***
            #  What happens when the camera begins acquiring images depends on
            #  the acquisition mode. Single frame captures only a single image,
            #  multi frame catures a set number of images, 
            #  and continuous captures acontinuous stream of images.
            #
            #  *** later ***
            #  Image acquisition must be ended when no more images are needed.
            self.cam.BeginAcquisition()

            """
            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras
            #  from overwriting one another.
            #  Grabbing image IDs could also accomplis hthis.
            devSerialNum = ''
            node_devSerialNum = PySpin.CStringPtr(
                            self.nodemap_tldevice.GetNode('DeviceSerialNumber')
                            )
            if PySpin.IsAvailable(node_devSerialNum) and \
               PySpin.IsReadable(node_devSerialNum):
                devSerialNum = node_devSerialNum.GetValue()
                print("Device serial number retrieved as %s..."%(devSerialNum))
            """

            fps = 0
            fpsLst = []
            fpsTime = time()
            while(True):
                ### FPS measure
                if time()-fpsTime < 1:
                    fps += 1
                else:
                    print("[%s] FPS: %i"%(self.classTag, fps))
                    fpsLst.append(fps)
                    if len(fpsLst) > 60: fpsLst.pop(0)
                    fps = 0
                    fpsTime = time()

                try:
                    ### process queue message (q2t)
                    if q2t.empty() == False:
                        try: q2tMsg = q2t.get(False)
                        except: pass
                    if q2tMsg != "":
                        if q2tMsg == "quit":
                            break
                        q2tMsg = ""

                    #  Retrieve next received image
                    #
                    #  *** notes ***
                    #  Capturing an image houses images on the camera buffer.
                    #  Trying to capture an image that does not exist will hang 
                    #  the camera.
                    #
                    #  *** later ***
                    #  Once an image from the buffer is saved and/or no longer
                    #  needed, the image must be released in order to keep the
                    #  buffer from filling up.
                    imgRslt = self.cam.GetNextImage(1000) 
                    
                    if imgRslt.IsIncomplete(): # ensure image completion
                        msg = "Image incomplete with image status"
                        msg += " %d ..."%(imgRslt.GetImageStatus())
                        print(msg)
                    else:
                        #imgRslt.Save("output/%s.Raw"%(get_time_stamp(True)))
                        if q2m.empty():
                            '''
                            fp = "output/%s.Raw"%(get_time_stamp(True))
                            imgRslt.Save(fp)
                            '''
                            ### convert the image result to BGR frame image 
                            #    frame=imgRslt.Convert(PySpin.PixelFormat_BGR8)
                            p = PySpin.ImageProcessor()
                            if pfStr.startswith("BayerRG"):
                                _pf = PySpin.PixelFormat_BGR8
                            elif pfStr == "Mono8":
                                _pf = PySpin.PixelFormat_Mono8
                            frame = p.Convert(imgRslt, 
                                              PySpin.PixelFormat_Mono8)
                            frame = frame.GetNDArray() # convert to array
                            if len(fpsLst) > 10:
                                _fps = int(np.mean(fpsLst[-11:-1]))
                            else:
                                _fps = fps
                            _d = ["frameImg", self.cIdx, frame, _fps]
                            # send frame and other data via queue to main
                            q2m.put(_d, True, None)

                    imgRslt.Release() # release image
                    # *** Images retrieved directly from the camera (i.e. 
                    # non-converted images) need to be released in order to 
                    # keep from filling the buffer.

                except PySpin.SpinnakerException as ex:
                    msg = "ERROR:: %s"%(ex)
                    handleErr(self, q2m, msg)
                    return

                sleep(0.001)

            self.cam.EndAcquisition() # *** ending acquisition appropriately 
              # helps ensure that devices clean up properly and do not need 
              # to be power-cycled to maintain integrity.

        except PySpin.SpinnakerException as ex:
            msg = "ERROR:: %s"%(ex)
            handleErr(self, q2m, msg)
            return
        
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
        if DEBUG: print("VideoInSpinnaker.close()")
        self.parent.spinnaker.deinitCam(self.cam)
        self.parent.log("Mod stops.", self.classTag)

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass



