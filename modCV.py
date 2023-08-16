# coding: UTF-8
"""
Functions & classes for computer vision, 
  using OpenCV, NumPy, SciPy, etc.

Dependency:
    Numpy (1.17),
    Scipy (1.5),
    OpenCV (4.1),
    Scikit-Video (1.1)

last editted: 2023-07-12
"""

import sys, queue, subprocess
from threading import Thread 
from os import path, remove, mkdir
from glob import glob
from time import time, sleep
from PIL import ImageFont, ImageDraw, Image

import wx, cv2
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

try:
    from pyueye import ueye
    FLAG_PYUEYE = True
except Exception as e:
    FLAG_PYUEYE = False

try:
    if sys.platform.startswith("win"):
        import skvideo
        ### set path where ffmpeg.exe and ffprobe.exe are located
        '''
        ### OBSOLUTE; too slow
        ffmpegPaths = glob("C:/Program Files/**/ffmpeg.exe", recursive=True)
        ffprobePaths = glob("C:/Program Files/**/ffprobe.exe", recursive=True)
        for fp in ffmpegPaths:
            if fp in ffprobePaths:
                break
        fp = path.split(fp)[0]
        '''
        fp = path.join(path.split(path.realpath(__file__))[0], "bin")
        skvideo.setFFmpegPath(fp)
    import skvideo.io
except:
    pass

from modFFC import *
MyLogger = setMyLogger("modFFC")

DEBUG = False

#-------------------------------------------------------------------------------

def getOpenCVVersion():
    """ Return OpenCV version as a float number.

    Args:
        None

    Returns:
        cvVer (float): OpenCV's version 
    """
    if DEBUG: print("modCV.getOpenCVVersion()")

    cvVer = cv2.__version__.split(".")
    return float("%s.%s"%(cvVer[0], cvVer[1]))

#-------------------------------------------------------------------------------

def convt_cvImg2wxImg(imgArr, toBMP=False): 
    """ convert openCV image to wxPython wx.Image or wx.Bitmap

    Args:
        arr (numpy.ndarray): input image array 
        toBMP (bool): if True, coonvert it to wx.Bitmap 

    Returns:
        (tuple): Tuple of RGB values 
    """ 
    if DEBUG: print("modCV.convt_cvImg2wxImg()")

    iSz = (imgArr.shape[1], imgArr.shape[0])
    imgArr = cv2.cvtColor(imgArr, cv2.COLOR_BGR2RGB)
    wxImg = wx.Image(iSz[0], iSz[1])
    wxImg.SetData(imgArr.tostring())
    if toBMP: return wxImg.ConvertToBitmap()
    else: return wxImg

#-------------------------------------------------------------------------------

def convt_wxImg2cvImg(img, fromBMP=False):
    """ convert wxPython image to openCV image 

    Args:
        img (wx.Image of wx.Bitamp): input image 
        fromBMP (bool): True, if 'img' is wx.Bitmap

    Returns:
        arr (numpy.ndarray): numpy array (BGR image)
    """ 
    if DEBUG: print("modCV.convt_wxImg2cvImg()")

    if fromBMP: img = img.ConvertToImage()
    arr = np.frombuffer(img.GetDataBuffer(), dtype=np.uint8)
    sz = img.GetSize()
    nc = int(len(arr) / sz[0] / sz[1])
    arr = arr.reshape((sz[1], sz[0], nc))
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

#-------------------------------------------------------------------------------

def convt_mplFig2npArr(fig):
    """ convert matplotlib figure to numpy array.

    Args:
        fig (matplotlib.pyplot.figure)

    Returns:
        imgArr (numpy.ndarray)
    """
    if DEBUG: logging.info(str(locals()))

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    nCol, nRow = fig.canvas.get_width_height()
    imgArr = np.fromstring(buf, dtype=np.uint8).reshape(nRow, nCol, 3)
    imgArr = imgArr[:,:,::-1]
    imgArr = np.ascontiguousarray(imgArr, dtype=np.uint8)
    return imgArr

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
    if DEBUG: print("modCV.cvHSV2RGB()")

    h = h / 180.0
    s = s / 255.0
    v = v / 255.0
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

#-------------------------------------------------------------------------------

def rgb2cvHSV(r, g, b): 
    """ convert RGB values to openCV's HSV color values

    Args:
        r (int): Red (0-255)
        g (int): Green (0-255)
        b (int): Blue (0-255)

    Returns:
        (tuple): Tuple of HSV values 
    """ 
    if DEBUG: print("modCV.rgb2cvHSV()")

    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = int(h * 180)
    s = int(s * 255)
    v = int(v * 255)
    return (h, s, v) 

#-------------------------------------------------------------------------------

def highContrast(img): 
    """ enhance contrast 

    Args:
        img (numpy.ndarray): Image array (BGR)

    Returns:
        img (numpy.ndarray): Image array with enhanced contrast 
    """ 
    if DEBUG: print("modCV.highContrast()")

    # converting image to LAB Color model
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    # applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # converting image from LAB Color model to RGB model
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return img

#-------------------------------------------------------------------------------

def drawSkeleton(img, thr=[30,50]): 
    """ draw skeleton (center line) of white blobs in binary image 

    Args:
        img (numpy.ndarray): Binary image (0 or 255) which has blob(s).
        thr (list): Two parameter for cv2.Canny edge detection.

    Returns:
        ret (numpy.ndarray): Binary image which has center lines of the blobs. 
    """ 
    if DEBUG: print("modCV.drawSkeleton()")

    img = cv2.Canny(img, thr[0], thr[1])
    img, cnts, hierarchy = cv2.findContours(img, cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)
    ret = np.zeros(img.shape, dtype=np.uint8)
    for ci in range(len(cnts)):
        img[:,:] = 0
        cv2.drawContours(img, cnts, ci, 255, thickness=cv2.FILLED)
        img = cv2.ximgproc.thinning(img)
        ret = cv2.add(ret, img)
    return ret 

#-------------------------------------------------------------------------------

def drawCvxHullDefects(img, lineCol=200, lineThck=3, dotCol=100, dotRad=5):
    """ draw defects of convex hull with binary blobs 

    Args:
        img (numpy.ndarray): Binary image (0 or 255) which has blob(s).
        lineCol (int): Color of line.
        lineThck (int): Thickness of line.
        dotCol (int): Color of defect dot.
        dotRad (int): Radius of defect dot.

    Returns:
        img (numpy.ndarray): Binary image with defects drawn. 
    """ 
    if DEBUG: print("modCV.drawCvxHullDefects()")

    img, cnts, hierarchy = cv2.findContours(img, 
                                            cv2.RETR_LIST, 
                                            cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        if defect == None: continue
        for di in range(defects.shape[0]):
            s,e,f,d = defects[di, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            if lineThck > 0:
                cv2.line(img, start, end, lineCol, lineThck)
            if dotRad > 0:
                cv2.circle(img, far, dotRad, dotCol, -1)
    return img

#-------------------------------------------------------------------------------

def detectLinesF(img, length_threshold=10, distance_threshold=1.414213562,
                 canny_th1=50, canny_th2=50, canny_aperture_size=3,
                 do_merge=False):
    """ convenience function for detecting lines using the fast line detector  

    Args:
        img (numpy.ndarray): Binary image (0 or 255) which has blob(s).
        * Other arguments are same as createFastLineDetector
    
    Returns:
        pts (numpy.ndarray): Three points of lines; both ends + middle point
    """ 
    if DEBUG: print("modCV.detectLines()")

    fld = cv2.ximgproc.createFastLineDetector(
                            length_threshold, distance_threshold, 
                            canny_th1, canny_th2, canny_aperture_size, do_merge)
    lines = fld.detect(img)
    linePts = []
    for line in lines:
        x1 = int(line[0][0])
        y1 = int(line[0][1])
        x2 = int(line[0][2])
        y2 = int(line[0][3])
        minX = min(x1, x2)
        minY = min(y1, y2)
        xm = min(x1,x2) + int(abs(x1-x2)/2)
        ym = min(y1,y2) + int(abs(y1-y2)/2)
        linePts.append([(x1,y1), (xm, ym), (x2, y2)])
    return linePts

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
    if DEBUG: print("modCV.getColorInfo()")

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

def getCamIdx(maxIdx=4):
    """ Returns indices of attached webcams, after confirming
      each webcam returns its image.

    Args:
        maxIdx (int): Max cam index to check 

    Returns:
        idx (list): Indices of webcams

    Examples:
        >>> getCamIdx()
        [0]
    """
    if DEBUG: print("modCV.getCamIdx()")

    idx = []
    for i in range(maxIdx):
        cap = cv2.VideoCapture(i)
        ret, f = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if ret==False or f is None or fps == 0: continue
        idx.append(i)
        cap.release()
    print("Camera indices: %s"%(str(idx)))
    return idx

#-------------------------------------------------------------------------------

def getPossibleResolutions():
    """ Get possible resolutions of all cameras attached to the computer.

    Args:
        None

    Returns:
        pRes (dict): Possible resolutions of each camera.
    """
    if DEBUG: print("modCV.getPossibleResolutions()")

    # known resolutions
    resolutions = [(16, 16), (42, 11), (32, 32), (40, 30), (42, 32), (48, 32), 
       (60, 40), (84, 48), (64, 64), (72, 64), (128, 36), (75, 64), (150, 40), 
       (96, 64), (96, 64), (128, 48), (96, 65), (102, 64), (101, 80), (96, 96), 
       (240, 64), (160, 102), (128, 128), (160, 120), (160, 144), (144, 168), 
       (160, 152), (160, 160), (140, 192), (160, 200), (224, 144), (208, 176), 
       (240, 160), (220, 176), (160, 256), (208, 208), (256, 192), (280, 192), 
       (256, 212), (432, 128), (256, 224), (240, 240), (256, 240), (320, 192), 
       (320, 200), (256, 256), (256, 256), (320, 208), (320, 224), (320, 240), 
       (320, 256), (384, 224), (368, 240), (376, 240), (272, 340), (400, 240), 
       (512, 192), (320, 320), (432, 240), (560, 192), (400, 270), (512, 212), 
       (384, 288), (480, 234), (400, 300), (480, 250), (312, 390), (512, 240), 
       (320, 400), (640, 200), (480, 272), (512, 256), (512, 256), (416, 352), 
       (480, 320), (640, 240), (640, 240), (640, 256), (512, 342), (368, 480), 
       (496, 384), (800, 240), (512, 384), (640, 320), (640, 350), (640, 360), 
       (480, 500), (512, 480), (720, 348), (720, 350), (640, 400), (720, 364), 
       (800, 352), (600, 480), (640, 480), (640, 512), (768, 480), (800, 480), 
       (848, 480), (854, 480), (800, 600), (960, 540), (832, 624), (960, 544), 
       (1024, 576), (960, 640), (1024, 600), (1024, 640), (960, 720), 
       (1136, 640), (1138, 640), (1024, 768), (1024, 800), (1152, 720), 
       (1152, 768), (1280, 720), (1120, 832), (1280, 768), (1152, 864), 
       (1334, 750), (1280, 800), (1152, 900), (1024, 1024), (1366, 768), 
       (1280, 854), (1280, 960), (1600, 768), (1080, 1200), (1440, 900), 
       (1440, 900), (1280, 1024), (1440, 960), (1600, 900), (1400, 1050), 
       (1440, 1024), (1440, 1080), (1600, 1024), (1680, 1050), (1776, 1000), 
       (1600, 1200), (1600, 1280), (1920, 1080), (1440, 1440), (2048, 1080), 
       (1920, 1200), (2048, 1152), (1792, 1344), (1920, 1280), (2280, 1080), 
       (2340, 1080), (1856, 1392), (2400, 1080), (1800, 1440), (2880, 900), 
       (2160, 1200), (2048, 1280), (1920, 1400), (2520, 1080), (2436, 1125), 
       (2538, 1080), (1920, 1440), (2560, 1080), (2160, 1440), (2048, 1536), 
       (2304, 1440), (2256, 1504), (2560, 1440), (2304, 1728), (2560, 1600), 
       (2880, 1440), (2960, 1440), (2560, 1700), (2560, 1800), (2880, 1620), 
       (2560, 1920), (3440, 1440), (2736, 1824), (2880, 1800), (2560, 2048), 
       (2732, 2048), (3200, 1800), (2800, 2100), (3072, 1920), (3000, 2000), 
       (3840, 1600), (3200, 2048), (3240, 2160), (3200, 2400), (3840, 2160), 
       (4096, 2160), (3840, 2400), (4096, 2304), (5120, 2160), (4480, 2520), 
       (4096, 3072), (4500, 3000), (5120, 2880), (5120, 3200), (5120, 4096), 
       (6016, 3384), (6400, 4096), (6400, 4800), (6480, 3240), (7680, 4320), 
       (7680, 4800), (8192, 4320), (8192, 4608), (10240, 4320), (8192, 8192), 
       (15360, 8640)] 

    camIdx = getCamIdx()

    pRes = {}
    for ci in camIdx:
        cap = cv2.VideoCapture(ci)
        pRes[ci] = {}
        for i, res in enumerate(resolutions): 
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            pRes[ci]["%i x %i"%(width, height)] = "OK"
    return pRes 

#-------------------------------------------------------------------------------

def getFontScale(cvFont, thresholdPixels=20, thick=1):
    """ get OpenCV font scale given the font and the target pixel size 

    Args:
        cvFont (cv2.font): Font to use
        thresholdPixels (int): Threshold to determine the font scale 
        thick (int): Thickness

    Returns:
        fs (float): Font scale
        w (int): Width of character
        h (int): Height of character
        bl (int): Baseline
    """
    if DEBUG: print("modCV.getFontScale()")

    fs = 0.1
    while True:
        (w, h), bl = cv2.getTextSize("X", cvFont, fs, thickness=thick)
        if bl + h > thresholdPixels: break 
        fs += 0.1
    return (fs, w, h, bl)

#-------------------------------------------------------------------------------

def clustering(pt_list, threshold, criterion='distance'):
    """ Cluster given points

    Args:
        pt_list (list): List of points.
        threshold (int): Threshold to cluster.
        criterion (str): Criterion for fclusterdata function.

    Returns:
        nGroups (int): Number of groups.
        groups (list): List of groups.
    """
    if DEBUG: print("modCV.clustering()")

    cRslt = []
    try: cRslt = list(fclusterdata(np.asarray(pt_list), 
                                   threshold, 
                                   criterion=criterion,
                                   metric='euclidean'))
    except: pass
    nGroups = 0
    groups = []
    if cRslt != []:
        groups = []
        nGroups = max(cRslt)
        for i in range(nGroups): groups.append([])
        for i in range(len(cRslt)):
             groups[cRslt[i]-1].append(pt_list[i])
    return nGroups, groups

#-------------------------------------------------------------------------------

def preProcImg(img, dilation=1, erosion=1, flagGrey=False):
    """ pre-processing image.

    Args:
        img (numpy.ndarray): Image (BGR image) array to process. 
        dilation (int): Number of iteration for cv2.dilate. 
        erosion (int): Number of iteration for cv2.erode. 
        flagGrey (bool): Turn to grey image.

    Returns:
        img (numpy.ndarray): image after pre-processing. 
    """
    if DEBUG: print("modCV.preProcImg()")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    if dilation > 0:
        img = cv2.dilate(img, kernel, iterations=dilation)
    if erosion > 0:
        img = cv2.erode(img, kernel, iterations=erosion)
    if flagGrey:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#-------------------------------------------------------------------------------

def findColor(img, hsvMin, hsvMax, r=None):
    """ Find a color(range: hsvMin - hsvMax) in an area of an image.

    Args:
        img (numpy.ndarray): Input image (BGR).
        hsvMin (tuple): Minimum values to find the color.
        hsvMax (tuple): Maximum values to find the color.
        r (None/tuple): Rect (x1,y1,x2,y2). If None, use entire image.

    Returns:
        retImg (numpy.ndarray): Binary image indicating the found color.
    """
    if DEBUG: print("modCV.findColor()")
    
    mask = np.zeros((img.shape[0], img.shape[1]) , dtype=np.uint8)
    if r == None:
        # Upper Left, Lower Left, Lower Right, Upper Right
        pts = [(0,0), (0,img.shape[0]), 
               (img.shape[1],img.shape[0]), (img.shape[1],0)]
    else:
        pts = [(r[0], r[1]), (r[0], r[3]), (r[2], r[3]), (r[2], r[1])] 
    cv2.fillConvexPoly(mask, np.asarray(pts), 255)
    tmpColImg = cv2.bitwise_and(img, img, mask=mask)
    tmpColImg = cv2.cvtColor(tmpColImg, cv2.COLOR_BGR2HSV)
    retImg = cv2.inRange(tmpColImg, hsvMin, hsvMax)
    __, retImg = cv2.threshold(retImg, 50, 255, cv2.THRESH_BINARY)
    if hsvMin[0] == 0: # finding red color
        hsvMin = (180-hsvMax[0], hsvMin[1], hsvMin[2])
        hsvMax = (180, hsvMax[1], hsvMax[2])
        retImg_ = cv2.inRange(tmpColImg, hsvMin, hsvMax)
        __, retImg_ = cv2.threshold(retImg_, 50, 255, cv2.THRESH_BINARY)
        retImg = cv2.add(retImg, retImg_)
    return retImg

#-------------------------------------------------------------------------------

def getCentroid(img):
    """ Get the centroid of binary image. 

    Args:
        img (numpy.ndarray): Input binary image.

    Returns:
        (tuple): Coordinate of the centroid.
    """
    if DEBUG: print("modCV.getCentroid()")

    m = cv2.moments(img)
    if m["m00"] != 0:
        mx = int(m["m10"]/m["m00"])
        my = int(m["m01"]/m["m00"])
    else:
        mx = -1
        my = -1
    return (mx, my)

#-------------------------------------------------------------------------------

def maskImg(img, rois, col):
    """ mask image with the regions of interests 

    Args:
        img (numpy.ndarray): Input image.
        rois (list): list of region of interests (x, y, w, h) or (x, y, r)
        col (tuple): fill color for masked area 

    Returns:
        img (numpy.ndarray): Output image. 
    """
    if DEBUG: print("modCV.maskImg()")

    sh = img.shape
    bgImg = np.zeros(sh, dtype=np.uint8)
    cv2.rectangle(bgImg, (0,0), (sh[1], sh[0]), col, -1)
    mask = np.zeros((img.shape[0], img.shape[1]) , dtype=np.uint8)
    if len(sh) == 2: mBGCol = 0
    else: mBGCol = (0,0,0)
    for roi in rois:
        if len(roi) == 4:
            pt1 = (roi[0], roi[1])
            pt2 = (roi[0]+roi[2], roi[1]+roi[3])
            cv2.rectangle(mask, pt1, pt2, 255, -1)
            cv2.rectangle(bgImg, pt1, pt2, mBGCol, -1)
        elif len(roi) == 3:
            x, y, r = roi
            cv2.circle(mask, (x,y), r, 255, -1)
            cv2.circle(bgImg, (x,y), r, mBGCol, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.add(bgImg, img)
    return img

#-------------------------------------------------------------------------------

def drawEllipse(img, center, axes, angle, startAngle, endAngle, color,
                thickness=1, lineType=cv2.LINE_AA, shift=10):
    """ wrapper for OpenCV's ellipse function
    """
    if DEBUG: print("modCV.drawEllipse()")

    ### uses the shift to accurately get sub-pixel resolution for arc
    ### taken from https://stackoverflow.com/a/44892317/5087436
    '''
    The shift parameter indicates the number of "fractional bits" 
    in the center and axes coordinate values, 
    so that's why the coordinates multiplied by powers of 2 
    (multiplying by 2 is the same as shifting the bits in their 
    integer binary representations to the left by one). 
    This shift trick is handy with many other opencv functions as well, 
    but its usage is not documented very well (especially in python).
    '''
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    return cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color,
                       thickness, lineType, shift)


#-------------------------------------------------------------------------------

def drawTxtWithPIL(img, txt, pos, fontName="LiberationMono-Regular.ttf", 
                   fontSz=12, col=(0,0,0)):
    """ Draw text using PIL library on the given image. 

    Args:
        img (numpy.ndarray): input image array.
        txt (str): text to write.
        pos (tuple): x & y coordinates.
        fontName (str): font name.
        fontSz (int): font size.
        col (tuple): color (RGB).

    Returns:
        (numpy.ndarray): output image array 
    """ 
    if DEBUG: print("modCV.drawTxtWithPIL()")

    text_to_show = "The quick brown fox jumps over the lazy dog"  
 
    # convert to RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # to PIL image
    pilImg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilImg) 
    # font to use
    font = ImageFont.truetype(fontName, fontSz)  
    # draw the text  
    draw.text(pos, txt, font=font, fill=col, stroke_fill=col)
    
    # convert back to BGR and return
    return cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR) 

#-------------------------------------------------------------------------------

def convt_raw2png(fp, pFormat="BayerRG", res=(3000,4000), compression=6):
    """ Convert RAW format image file (from FLIR camera), to PNG file.
      * In 'SpinView' software, raw file format can be checked (or changed)
          in 'Image Format Control' of 'feature' tree.

    Args:
        fp (str): File path of the RAW file.
        pFormat (str): Pixel format of FLIR camera recording.
        res (tuple): Resolution of image for image array (height x widht).
        compression (int): PNG compression; 0-9.

    Returns:
        None
    """
    if DEBUG: print("modCV.convt_FLIRRaw2jpg()")

    img = np.fromfile(fp, dtype=np.uint8).reshape(res)
    if pFormat == "BayerRG":
        cImg = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
    elif pFormat == "BayerGR":
        cImg = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    elif pFormat == "BayerGB":
        cImg = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
    elif pFormat == "BayerBG":
        cImg = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    newFP = fp.replace(".Raw", ".png")
    cv2.imwrite(newFP, cImg, [cv2.IMWRITE_PNG_COMPRESSION, compression])

#===============================================================================

class VideoIn:
    """ class for retrieving images from video camera 
    
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent, cIdx, desiredRes=(-1,-1), fpsLimit=30, 
                 outputFormat="image", ssIntv=1.0, params={}):
        if DEBUG: print("VideoIn.__init__()")
        
        ##### [begin] class attributes -----
        self.classTag = "videoIn-%i"%(cIdx)
        self.parent = parent # parent
        self.cIdx = cIdx # index of cam
        self.outputFormat = outputFormat # image or video
        self.fpsLimit = fpsLimit # limit of frames per second
        self.ssIntv = ssIntv # interval in seconds for saving image from Cam
        self.params = params # other parameters
        ### set video capture
        cvVer = getOpenCVVersion() 
        if sys.platform.startswith("win") and cvVer > 3.4: 
            self.cap = cv2.VideoCapture(cIdx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(cIdx)
        sleep(0.5) # some delay for cam's initialization
        #self.cap.set(cv2.CAP_PROP_FOURCC, 0x47504A4D) # MJPG
        ### set resolution
        if desiredRes != (-1,-1): # desired resolution available
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desiredRes[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desiredRes[1])
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ### read initial frames
        print("[%s] Reading inital frames..."%(self.classTag))
        for i in range(100):
            ret, frame = self.cap.read()
            if ret: break
            sleep(0.01)
        if not ret:
            print("[%s] failed to init."%(self.classTag))
            parent.log("Failed to init.", self.classTag)
        else:
            self.fSz = (frame.shape[1], frame.shape[0]) # store frame size
            print("Cam index-%i resolution: %s"%(cIdx, str(self.fSz)))
        self.initFrame = frame # store initial frame
        ##### [end] class attributes -----
                        
        parent.log("Mod init.", self.classTag) 
    
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
        if DEBUG: print("VideoIn.run()")

        #-----------------------------------------------------------------------
        def startRecording(cIdx, oFormat, recFolder, fps, fSz, 
                           fpsLimit, ssIntv):
            if DEBUG: print("VideoIn.run.startRecording()")
            # Define the codec and create VideoWriter object
            #fourcc = cv2.VideoWriter_fourcc(*'X264')
            fourcc = cv2.VideoWriter_fourcc(*'avc1') # for saving mp4 video
            #fourcc = cv2.VideoWriter_fourcc(*'xvid') # for saving avi video
            
            log = "recording starts"
            log += " [%s]"%(oFormat)
            if oFormat == 'video':
                ofn = "output_cam%.2i_%s.mp4"%(cIdx, get_time_stamp())
                ofp = path.join(recFolder, ofn)
                # get average of the past 10 fps records
                ofps = int(np.average(fps[-11:-1]))
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
            if DEBUG: print("VideoIn.run.stopRecording()")
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
        ssIntv = copy(self.ssIntv)
        imgSaveTime = time()-ssIntv # last time image was saved
        fpsTimeSec = time()
        fpsLastLoggingTime = time()
        fps = [0]
        flagFPSLogging = False

        ##### [begin] infinite loop of thread -----
        while self.cap.isOpened():
            ### limit frame processing 
            if self.fpsLimit != -1:
                if time()-lastFrameProcTime < fpIntv:
                    sleep(0.001)
                    continue
                lastFrameProcTime = time()
           
            ### fps
            if time()-fpsTimeSec > 1:
                if flagFPSLogging:
                    if time()- fpsLastLoggingTime > 60: # one minute passed
                        log = "FPS during the past minute %s"%(str(fps[:60]))
                        self.parent.log(log, self.classTag)
                        fpsLastLoggingTime = time()
                else:
                    print("\r[%s] FPS: %i"%(self.classTag, fps[-1]), end="")
                fps.append(0)
                fpsTimeSec = time()
                # keep fps records of the past one minute
                if len(fps) > 61: fps.pop(0)
            else:
                fps[-1] += 1
            
            ### process queue message (q2t)
            if q2t.empty() == False:
                try: q2tMsg = q2t.get(False)
                except: pass
            if q2tMsg != "":
                if q2tMsg == "quit":
                    break
                elif q2tMsg == 'rec_init':
                    if out == None:
                        out, ofn = startRecording(
                                        self.cIdx, self.outputFormat, recFolder,
                                        fps, self.fSz, self.fpsLimit, ssIntv
                                        )
                elif q2tMsg == 'rec_stop':
                    if out != None:
                        out = stopRecording(out, self.cIdx)
                q2tMsg = ""
            
            ret, frame = self.cap.read() # retrieve a frame image

            if not ret: # frame image not retrieved
                sleep(0.001)
                continue
            
            if self.outputFormat == 'video':
            # video recording
                if out != None:
                    out.write(frame) # write a frame to video
            
            elif self.outputFormat == 'image':
            # image recording
                if time()-imgSaveTime >= ssIntv:
                # interval time has passed
                    if out != None:
                        imgSaveTime = time()
                        ### save frame image
                        ts = get_time_stamp(flag_ms=True)
                        if "imgExt" in self.params.keys():
                            ext = self.params["imgExt"]
                        else:
                            ext = "png"
                        fp = path.join(ofn, f'f_{ts}.{ext}')
                        if ext == "jpg":
                            qVal = 95
                            if "jpegQuality" in self.params.keys():
                                qVal = self.params["jpegQuality"]
                            q = [int(cv2.IMWRITE_JPEG_QUALITY), qVal]
                            cv2.imwrite(fp, frame, q)
                        else:
                            cv2.imwrite(fp, frame)
                    
            if flagSendFrame and q2m.empty():
                ### send frame via queue to main thread
                if len(fps) > 10: avgFPS = int(np.average(fps[-11:-1]))
                else: avgFPS = fps[-1]
                q2m.put(["frameImg", self.cIdx, frame, avgFPS], True, None)
        ##### [end] infinite loop of thread -----
        
        if out != None and type(out) != int:
            out.release()
        self.parent.log("Thread stops.", self.classTag)
    
    #---------------------------------------------------------------------------

    def close(self):
        """ Release VideoCapture of this Cam
        
        Args: None
        
        Returns: None
        """
        if DEBUG: print("VideoIn.close()")

        self.cap.release()
        self.parent.log("Mod close.", self.classTag) 
    
    #---------------------------------------------------------------------------
    
#===============================================================================

class VideoRW:
    """ Class for reading/writing frame from/to video file 

    Args:
        parent: parent object

    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent, codec="XVID", 
                 useSKVideo=False, skIn={}, skOut={}):
        if DEBUG: print("VideoRW.__init__()")
        
        ##### [begin] setting up attributes on init. -----
        self.parent = parent
        self.fPath = "" # file path of video
        self.vCap = None # VideoCapture object of OpenCV
        self.vCapFSz = (-1, -1) # frame size (w, h) of frame image 
          # of current video
        self.currFrame = None # current frame image (ndarray)
        self.nFrames = 0 # total number of frames
        self.fi = -1 # current frame index
        self.th = None # thread
        self.q2m = queue.Queue() # queue from thread to main
        self.q2t = queue.Queue() # queue from main to thread 
        self.timer = {} # timers for this class
        self.vRecVideoCodec = codec # codec when using OpenCV's VideoWriter
                        # Currently, (H264/AVC1 (.mp4) or XVID/MJPG (.avi))
        self.flagSKVideo = useSKVideo # Using scikit-video package.
        self.skIn = skIn # input parameters for scikit-video FFmpegWriter.
        self.skOut = skOut # output parameters for scikit-video FFmpegWriter.
        ##### [end] setting up attributes on init. -----

    #---------------------------------------------------------------------------
    
    def initReader(self, fPath):
        """ init. video file reading 

        Args:
            fPath (str): Path of video file to read. 

        Returns:
            None
        """
        if DEBUG: print("VideoRW.initReader()") 

        self.fPath = fPath
        if path.isfile(self.fPath):
            # init video capture
            self.vCap = cv2.VideoCapture(fPath)
            # get total number of frames
            self.nFrames = int(self.vCap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.nFrames = len(glob(path.join(self.fPath, "*.jpg")))
        self.fi = -1
        self.getFrame(-1) # read the 1st frame
        # store frame size
        self.vCapFSz = (self.currFrame.shape[1], self.currFrame.shape[0]) 

    #---------------------------------------------------------------------------
    
    def initWriter(self, fPath, video_fSz=None, vRecFPS=None, 
                   callbackFunc=None, procFunc=None):
        """ init. video file writing 

        Args:
            fPath (str): Path of video file to record.
            video_fSz (None/tuple): Video frame size to record.
            vRecFPS (None/int): output filw FPS.
            callbackFunc (function): Callback function to call after writing.
            procFunc (function): Function to call to process 
              before saving each frame image. 

        Returns:
            None
        """
        if DEBUG: print("VideoRW.initWriter()") 

        # if file already exists, delete it
        if path.isfile(fPath): remove(fPath)
        
        if vRecFPS is None: vRecFPS = 30
        self.writingVideoFrameSz = video_fSz 
        self.callbackFunc = callbackFunc
        if self.flagSKVideo:
            if self.skIn == {}: skIn = {"-r":str(vRecFPS)}
            else: skIn = self.skIn
            if not "-r" in self.skOut.keys():
                self.skOut["-r"] = str(vRecFPS)
            self.video_rec = skvideo.io.FFmpegWriter(fPath,
                                                     inputdict=skIn,
                                                     outputdict=self.skOut)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'%s'%(self.vRecVideoCodec))
            # init video writer
            self.video_rec = cv2.VideoWriter(fPath, 
                                             fourcc=fourcc, 
                                             fps=vRecFPS, 
                                             frameSize=video_fSz, 
                                             isColor=True)
        if callbackFunc != None:
            ### set timer for updating current frame index 
            self.timer["writeFrames"] = wx.Timer(self.parent)
            self.parent.Bind(wx.EVT_TIMER,
                        lambda event: self.onTimer(event, "writeFrames"),
                        self.timer["writeFrames"])
            self.timer["writeFrames"].Start(10) 
            ### start thread to write 
            self.callbackFunc = callbackFunc # store callback function
            self.th = Thread(target=self.writeFrames, 
                             args=(self.video_rec, self.q2m, procFunc,))
            wx.CallLater(20, self.th.start)
                
    #---------------------------------------------------------------------------
    
    def getFrame(self, targetFI=-1, useCAPPROP=True, callbackFunc=None):
        """ Retrieve a frame image with a given index or 
        just the next frame when index is not given. 

        Args:
            targetFI (int): Target frame index to retrieve.
            useCAPPROP (bool): Whether to use cv2.CAP_PROP_POS_FRAMES to seek.
            callbackFunc (None/function): Callback function when targetFI
              is not -1, meaning it'd be thread running.

        Returns:
            ret (bool): Notifying whether the image retrieval was successful.
            frame (numpy.ndarray): Frame image.
        """
        if DEBUG: print("VideoRW.getFrame()")

        if self.fi >= self.nFrames: return

        ##### [begin] image file reading -----
        if path.isdir(self.fPath): # folder with images
            if targetFI == -1:
                self.fi = min(self.fi+1, self.nFrames) 
                targetFI = self.fi
            else:
                self.fi = targetFI
            try: 
                fLst = glob(path.join(self.fPath, "*.jpg"))
                self.currFrame = cv2.imread(fLst[targetFI]) 
                ret = True
            except Exception as e:
                em = "%s, [ERROR], %s\n"%(get_time_stamp(), str(e))
                ret = False
            return ret
        ##### [end] image file reading -----
        
        ##### [begin] video file reading ----- 
        def readFrame(fi, nFrames, vCap):
        # funtioinc to read frame(s) until it's read successfully
            numReadFrames = 0 
            for i in range(fi, nFrames):
                ret, frame = vCap.read() # read next frame
                numReadFrames += 1
                if ret: break # stop reading, if it was successful
            return ret, numReadFrames, frame
        if targetFI == 0:
            self.vCap.release()
            self.vCap = cv2.VideoCapture(self.fPath)
            self.fi = -1
            targetFI = -1
        if targetFI == -1 or targetFI == self.fi+1:
        # target index is not given 
        #   or the given index is the next of the current frame
            ret, nRF, frame = readFrame(self.fi, self.nFrames, self.vCap) 
            if ret: # if a frame was successfully retrieved
                self.fi += nRF 
                self.currFrame = frame
            return ret
        else:
        # otherwise
            if useCAPPROP: # use set capture property to get the frame
                self.fi = targetFI-1
                self.vCap.set(cv2.CAP_PROP_POS_FRAMES, self.fi)
                ret, nRF, frame = readFrame(self.fi, self.nFrames, self.vCap) 
                if ret:
                    self.fi += nRF 
                    self.currFrame = frame
                    return True
                else: # failed to jump to the target frame-index
                    self.vCap.release()
                    self.vCap = cv2.VideoCapture(self.fPath)
                    self.fi = -1
                    ret = self.getFrame(-1)
                    return ret 
            else:
            # read frame sequentially;
            # useful when dealing with partially broken video file 
                if targetFI > self.fi:
                    nRead = targetFI - self.fi 
                elif targetFI < self.fi:
                    self.vCap.release()
                    self.vCap = cv2.VideoCapture(self.fPath)
                    self.fi = -1 
                    nRead = targetFI + 1
                else:
                    return True
                ### start thread to read
                self.th = Thread(target=self.readFrames, 
                                 args=(self.fi, nRead, self.q2m,))
                self.th.start()
                self.callbackFunc = callbackFunc # store callback function
                self.targetFI = targetFI
                ### set timer for updating current frame index 
                self.timer["readFrames"] = wx.Timer(self.parent)
                self.parent.Bind(wx.EVT_TIMER,
                        lambda event: self.onTimer(event, "readFrames"),
                        self.timer["readFrames"])
                self.timer["readFrames"].Start(10)
                return True
        ##### [end] video file reading -----
    
    #---------------------------------------------------------------------------

    def readFrames(self, fi, n, q2m):
        """ read frames from video
        
        Args:
            fi (int): Current frame index
            n (int): Number of frames to read
            q2m (queue.Queue): Queue to send data to main
        
        Returns:
            None
        """
        if DEBUG: print("VideoRW.readFrames()") 
        
        for i in range(n):
            ret, frame = self.vCap.read()
            #if not ret: break
            q2m.put((fi,), True, None)
            fi += 1
        q2m.put((fi, frame), True, None)

    #---------------------------------------------------------------------------
    
    def writeFrames(self, video_rec, q2m, procFunc):
        """ Write frames to VideoRecorder to save.

        Args:
            video_rec (cv2.VideoWriter)
            q2m (queue.Queue): Queue to send data to main thread.
            procFunc (function): Function to call before saving.

        Returns:
            None
        """
        if DEBUG: print("VideoRW.writeFrames()")
        
        ### write video frames
        for fi in range(self.nFrames):
            if fi > 0: self.getFrame(-1)
            frame = self.currFrame.copy()
            if procFunc != None: frame = procFunc(frame)
            # write a frame
            if self.flagSKVideo: self.video_rec.writeFrame(frame[:,:,::-1])
            else: video_rec.write(frame) 
            msg = "Writing video.. frame-idx: %i/%i"%(fi, self.nFrames-1)
            q2m.put((msg,), True, None)
        q2m.put((msg, frame), True, None)
    
    #---------------------------------------------------------------------------
    
    def writeFrame(self, frame):
        """ Write a single frame to VideoRecorder to save.

        Args:
            frame (np.array): frame image

        Returns:
            None
        """
        if DEBUG: print("VideoRW.writeFrame()")

        try:
            ### resize frame
            if hasattr(self, "writingVideoFrameSz"):
                wVFSz = self.writingVideoFrameSz
                if wVFSz != None and wVFSz != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, wVFSz)
            ### write the frame
            if self.flagSKVideo: self.video_rec.writeFrame(frame)
            else: self.video_rec.write(frame)
        except Exception as e:
            print(e)
     
    #---------------------------------------------------------------------------
    
    def extractImgsFromVideo(self, folderPath, q2m, callbackFunc=None):
        """ extract images (read & save as images) from video 
        
        Args: 
            folderPath (str): Folder path to save images.
            q2m (queue.Queue): Queue to send data to main.
            callbackFunc (None/function): Callback function to call 
              after processing.
        
        Returns:
            None
        """
        if DEBUG: print("VideoRW.extractImgsFromVideo()")
         
        ### init video
        self.vCap.release()
        self.vCap = cv2.VideoCapture(self.fPath)
        ### read & save frame images
        fi = -1
        for i in range(self.nFrames):
            ret, frame = self.vCap.read()
            if not ret: break
            fi += 1
            fp = path.join(folderPath, "f%07i.jpg"%fi)
            cv2.imwrite(fp, frame)
            msg = "extracting frames .. %i/ %i"%(fi, self.nFrames)
            q2m.put(("displayMsg", msg,), True, None)
        
        ### re-init video
        self.vCap.release()
        self.vCap = cv2.VideoCapture(self.fPath)
        self.fi = -1 
        self.getFrame(-1) # read the 1st frame
        
        q2m.put(("extractionFinished", ), True, None) 
        if callbackFunc != None:
            callbackFunc("extractionFinished")
    
    #---------------------------------------------------------------------------

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

        if flag == "readFrames":
        # navigating (reading frames) to a specific frame
            if len(rData) == 1:
                showStatusBarMsg(self.parent, "frame - %i"%(rData[0]), -1)
            elif len(rData) == 2:
            # reached target frame index
                self.fi, self.currFrame = rData
                self.timer["readFrames"].Stop()
                self.timer["readFrames"] = None
                self.targetFI = -1
                showStatusBarMsg(self.parent, "", -1)
                self.th.join()
                self.th = None
                self.callbackFunc(rData, flag)
        
        elif flag == "writeFrames":
            if len(rData) == 1:
                if callable(getattr(self.parent, "showStatusBarMsg", None)):
                    self.parent.showStatusBarMsg(rData[0], -1)
            elif len(rData) == 2:
                self.timer["writeFrames"].Stop()
                self.timer["writeFrames"] = None
                if callable(getattr(self.parent, "showStatusBarMsg", None)):
                    self.parent.showStatusBarMsg("", -1)
                self.closeWriter() 
                self.callbackFunc(rData, "finalizeSavingVideo") 
    
    #---------------------------------------------------------------------------

    def closeReader(self):
        """ close videoCapture 

        Args: None

        Returns: None
        """
        if DEBUG: print("VideoRW.closeReader()") 
        
        if path.isfile(self.fPath):
            self.vCap.release() # close video capture instance
        self.vCap = None
        self.fPath = ""
        self.fi = -1
        self.nFrames = 0

    #---------------------------------------------------------------------------

    def closeWriter(self):
        """ close videoWriter

        Args: None

        Returns: None
        """
        if DEBUG: print("VideoRW.closeWriter()")

        ### finish recorder
        try:
            if self.flagSKVideo: self.video_rec.close()
            else: self.video_rec.release()
        except:
            pass
        self.video_rec = None
        
    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__':
    pass



