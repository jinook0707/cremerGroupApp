# coding: UTF-8
"""
This module is for pyDrawGraph.
It has specific processing algorithms for different type of data, graph, etc. 

Dependency:
    Numpy (1.17), 
"""

from sys import platform
from os import path
from copy import copy

import wx
import numpy as np

from modFFC import str2num, getWXFonts, rot_pt, calc_line_angle
from modFFC import calc_pt_w_angle_n_dist

DEBUG = False

#=======================================================================

class ProcGraphData():
    """ Class for processing data.
    
    Args:
        parent (wx.Frame): Parent frame
        csvFP (str): File path of data CSV file
    
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    
    def __init__(self, parent, csvFP):
        if DEBUG: print("ProcGraphData.__init__()")

        ##### [begin] setting up attributes -----
        self.parent = parent
        self.csvFP = parent.csvFP # file path of data CSV
        self.colTitles = [] # data column titles
        self.numData = None # Numpy array of numeric data
        self.strData = None # Numpy array of character data
        self.fonts = parent.fonts
        self.uiTask = {} # task from UI (such as to show one virus, class, etc)
        self.uiTask["showVirusLabel"] = None
        self.uiTask["showThisVirusOnly"] = None 
        self.uiTask["showThisClassOnly"] = None 
        ##### [end] setting up attributes -----
    
    #-------------------------------------------------------------------
  
    def loadData(self, csv=""):
        """ Load data from CSV file

        Args:
            csv (str): String of CSV. If this is not given,
              self.csvFP will be used to open file and use its string.

        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.loadData()")

        gType = self.parent.graphType
        if gType == "CV2020":
        # for data of Syliva and Lumi (2020)
            self.delimiter = ","
            # column indices of numeric data
            self.numericDataIdx = [2, 3, 4, 5, 6, 7, 8, 9, 10] 
            # column indices of string data
            self.stringDataIdx = [0, 1] 
            self.npNumericDataType = np.int8
            self.numSpecies = 3
            self.numPopulations = 3
            self.vCR = -1 # radius of virus presence circle
            self.vpPt = {} # dictionary of virus presence points 
              # key will be virus such as 'LHUV-1'
              # value will be list of x,y coordinates where it appears in graph
            self.vlR = {} # dictionary of virus labels' rects (in legend)
            self.clR = {} # virus classifications' rects (in legend)
              # both rects are in form of (x1, y1, x2, y2)
            ##### [begin] set colors
            # set colors for each species
            self.sColor = [
                            wx.Colour(0,200,0),
                            wx.Colour(0,0,255),
                            wx.Colour(150,150,0),
                          ]
            # set colors for each population 
            self.pColor = [
                            [[50,200,50],
                             [100,200,100],
                             [150,200,150]],
                            [[0,0,175],
                             [50,50,175],
                             [100,100,175]],
                            [[175,175,0],
                             [175,175,90],
                             [175,175,150]],
                          ]
            ### set colors for each classification
            self.cColor = {}
            cl = "Bunyavirales" 
            self.cColor[cl] = wx.Colour(255,50,255)
            # -----
            cl = "Mononegavirales;Partitiviridae"
            self.cColor[cl] = wx.Colour(255,100,25)
            cl = "Mononegavirales;Rhabdoviridae"
            self.cColor[cl] = wx.Colour(255,150,75)
            # -----
            cl = "Narnaviridae"
            self.cColor[cl] = wx.Colour(200,100,200)
            # -----
            cl = "Nodaviridae"
            self.cColor[cl] = wx.Colour(100,50,100)
            # -----
            cl = "Permutotetraviridae"
            self.cColor[cl] = wx.Colour(200,50,100)
            # -----
            cl = "Picornavirales"
            self.cColor[cl] = wx.Colour(255,0,0)
            cl = "Picornavirales;Dicistroviridae;Aparavirus"
            self.cColor[cl] = wx.Colour(255,50,50)
            cl = "Picornavirales;Polycipiviridae"
            self.cColor[cl] = wx.Colour(255,100,100)
            cl = "Picornavirales;Polycipiviridae;Sopolycivirus"
            self.cColor[cl] = wx.Colour(255,150,150)
            # -----
            cl = "Totiviridae"
            self.cColor[cl] = wx.Colour(150,0,150)
            # -----
            cl = "Unclassified"
            self.cColor[cl] = wx.Colour(100,100,100)
            ##### [end] set colors for each classification
        else:
            msg = "Unknown graph type: %s"%(gType)
            wx.MessageBox(msg, "Error", wx.OK|wx.ICON_ERROR)
            return

        colTitles = []  # data column titles
        numData = [] # numeric data
        strData = [] # string data
        ### load CSV data 
        if csv == "": # CSV is not given
            f = open(self.csvFP, 'r')
            csv = f.read()
            f.close()
        lines = csv.split("\n")
        for line in lines:
            items = [x.strip() for x in line.split(self.delimiter)]
 
            if gType == "CV2020":
            # for data of Syliva and Lumi (2020)
                while "" in items: items.remove("") # remove empty data
                if len(items) <= 1: continue # ignore emtpy line.
                  # line iwth one item is just a comment line. Also ignore.
                
                ### store column titles
                # The first proper (len(items)>1) line is assumed to be
                # the title line
                if colTitles == []:
                    for ii in range(len(items)):
                        colTitles.append(items[ii])
                    continue 

                ### store data
                rsd = [] # row of string data
                rnd = [] # row of numeric data
                for ci in range(len(colTitles)): # through columns
                    val = items[ci].strip()
                    if ci in self.numericDataIdx:
                        val = str2num(val)
                        if val == None: val = -1 
                        rnd.append(val) # add numeric data
                    else:
                        val = val.replace(" ","") # remove blanks
                        rsd.append(val) # add string data
                ### add a data line
                numData.append(rnd)
                strData.append(rsd)
         
        self.colTitles = colTitles
        self.numData = np.asarray(numData, self.npNumericDataType)
        self.strData = np.char.asarray(strData)
         
        if gType == "CV2020":
            self.numViruses = self.numData.shape[0] 
            # number of virus presences in inner circle
            self.numVPresence = np.sum(self.numData)
            # degree between two virus presence dots 
            self.vpDeg = -360.0 / self.numVPresence # minus value = clockwise
    
    #-------------------------------------------------------------------
  
    def drawGraph(self, dc, flag=""):
        """ Draw graph with loaded data 
        
        Args:
            dc (wx.PaintDC): PaintDC to draw graph on.
            flag (str): Flag to indicate certain task such as 'save'
         
        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.drawGraph()")

        gType = self.parent.graphType 
        if gType == "CV2020": self.graphCV2020(dc, flag)
    
    #-------------------------------------------------------------------
  
    def graphCV2020(self, dc, flag):
        """ Draw graph for ant-virus study by Cremer and Viljakainen (2020)  
        
        Args:
            dc (wx.PaintDC): PaintDC to draw graph on.
            flag (str): Flag to indicate certain task such as 'save'
        
        Returns:
            None
        """ 
        if DEBUG: print("ProcGraphData.graphCV2020()")
        
        dcSz = dc.GetSize()
        arcType = 1 # 0: type of connecting arc with straight line 
              # 1: simple arc line type
        maxRad = min(dcSz) / 2 # maximum radius of circle in this DC 
        icRad = int(maxRad * 0.6) # radius of inner circle of circular graph,
          # where virus presence line will be drawn.
        popArcRad = int(dcSz[1]*0.03) # radius of population arc
        spArcRad = int(dcSz[1]*0.12) # radius of species arc
        baseLLen = int(dcSz[1] * 0.007) # base length of straight line 
          # (of virus presence), line length increases stepwise 
          # with this base length
        ct = (int(dcSz[0]*0.38), int(dcSz[1]/2)) # center point of panel
        cntVP = 0 # for counting overall virus presence circles
        fontSz = int(dcSz[1] * 0.01) # font size, depending on DC size
        fSzInc = int(dcSz[1] * 0.005)
        fonts = getWXFonts(fontSz, numFonts=3, fSzInc=fSzInc)
        emphFonts = getWXFonts(fontSz, 
                               numFonts=3,
                               fSzInc=fSzInc,
                               weight=wx.FONTWEIGHT_BOLD,
                               style=wx.FONTSTYLE_ITALIC, 
                               underline=False)
        vCR = int(dcSz[1]*0.01) # radius of virus presnece circle 
        vpPt1 = {} # This dictionary will have dictionaries with keys of 
          # virus labels (such as 'LHUV-1'),
          # and it'll have two nested lists of points (where center of virus 
          # presence circle is) in each ant species.
          # e.g.) "vpPt1['LHUV-1'][0][1]" has point of the second 
          # virus (LHUV-1 virus) presence circle of the first ant species.
        vpDeg = {} # Similar structure with vpPt1,
          # but it's degree of rotation of vpPt1 points
        for ri in range(self.strData.shape[0]):
            vl = self.strData[ri,0] # virus label 
            vpPt1[vl] = []
            vpDeg[vl] = []
            ### append list for each ant species
            for si in range(self.numSpecies): 
                vpPt1[vl].append([])
                vpDeg[vl].append([])
        
        ##### [begin] drawing graph -----
        dc.SetBackground(wx.Brush(wx.Colour(255,255,255)))
        dc.Clear()
        ### calculate positions of virus presence circles
        ###   & draw some base parts of graph
        for si in range(self.numSpecies):
            sS = (-1, -1) # species arc start point
            sE = (-1, -1) # species arc end point
            for pi in range(self.numPopulations):
                pS = (-1, -1) # population arc start point
                pE = (-1, -1) # population arc end point
                for ri in range(self.numData.shape[0]): # row (= virus)
                    vl = self.strData[ri,0] # virus label 
                    cl = self.strData[ri,1] # classification label  
                    ci = si*self.numPopulations + pi # column index
                      # column = a population of an ant species
                    if self.numData[ri,ci] == 0: # virus is not present
                        continue # to the next virus
                    
                    deg = cntVP * self.vpDeg # degree to rotate for 
                      # drawing virus presence circle
                    deg += 180 # starting from right, instead of left side
                    cntVP += 1 # counting overall virus presences 
                    
                    ### points for virus presence
                    pt1 = rot_pt((ct[0]+icRad, ct[1]), 
                                 ct, 
                                 deg) # center of circle
                    vpPt1[vl][si].append(pt1) # store coordinate
                    vpDeg[vl][si].append(deg)
                    
                    ### store population & species arc points
                    paPt = rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                  ct, 
                                  deg)
                    saPt = rot_pt((ct[0]+icRad+spArcRad, ct[1]), 
                                  ct, 
                                  deg)
                    if pS == (-1, -1):
                        pS = rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                     ct, 
                                     deg-self.vpDeg/2)
                        pSDeg = copy(deg)
                    pE = rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                 ct, 
                                 deg+self.vpDeg/2)
                    pEDeg = copy(deg)
                    if sS == (-1, -1):
                        sS = copy(saPt)
                        sSDeg = copy(deg)
                    sE = copy(saPt)
                    sEDeg = copy(deg)
                        
                if pS != (-1, -1) and pE != (-1, -1):
                    if pS == pE:
                        pEDeg = pSDeg + 1
                        pE= rot_pt((ct[0]+icRad+popArcRad, ct[1]), 
                                    ct, 
                                    pEDeg)
                    ### draw population arc
                    pCol = self.pColor[si][pi]
                    lighterPCol = tuple(min(x+75, 255) for x in pCol)
                    dc.SetPen(wx.Pen(lighterPCol, 0, wx.TRANSPARENT))
                    dc.SetBrush(wx.Brush(lighterPCol))
                    if self.vpDeg > 0:
                        dc.DrawArc(pS[0], pS[1], pE[0], pE[1], ct[0], ct[1])
                    else:
                        dc.DrawArc(pE[0], pE[1], pS[0], pS[1], ct[0], ct[1])
                    ### draw text of population label
                    dc.SetFont(fonts[2])
                    dc.SetTextForeground(tuple(pCol))
                    popLabel = self.colTitles[self.numericDataIdx[ci]]
                    popLabel = popLabel.split("[")[1].replace("]","")
                    _x, _y = rot_pt((ct[0]+icRad+int(dcSz[1]*0.04), ct[1]), 
                                     ct, 
                                     pSDeg+(pEDeg-pSDeg)/2)
                    w, h = dc.GetTextExtent(popLabel) 
                    if _x <= ct[0]: _x -= w
                    if _y <= ct[1]: _y -= h
                    dc.DrawText(popLabel, _x, _y)
            if sS != (-1, -1) and sE != (-1, -1):
                ### draw species arc
                dc.SetPen(wx.Pen(self.sColor[si], int(dcSz[1]*0.01)))
                dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                if self.vpDeg > 0:
                    dc.DrawArc(sS[0], sS[1], sE[0], sE[1], ct[0], ct[1]) 
                else:
                    dc.DrawArc(sE[0], sE[1], sS[0], sS[1], ct[0], ct[1]) 
                ### draw text of species label
                dc.SetFont(emphFonts[2])
                dc.SetTextForeground(self.sColor[si])
                spLabel = self.colTitles[self.numericDataIdx[ci]]
                spLabel = spLabel.split("[")[0]
                _x, _y = rot_pt(
                        (ct[0]+icRad+spArcRad+int(dcSz[1]*0.005), ct[1]), 
                        ct, 
                        sSDeg+(sEDeg-sSDeg)/2
                        )
                w, h = dc.GetTextExtent(spLabel) 
                if _x <= ct[0]: _x -= w
                if _y <= ct[1]: _y -= h
                dc.DrawText(spLabel, _x, _y)

        ### [begin] draw connecting lines & virus presence circles, 
        ###   except viruses occurred in multiple ant speceis
        lLen = []
        for si in range(self.numSpecies):
            lLen.append(baseLLen) # Length of straight part 
              # of connecting line. This will increase as more connecting
              # lines are drawn in each species
        ### dicts for viruses which occurred multiple ant species
        vMSpPt1 = {} # pt1 
        vMSpDeg = {} # degree 
        vMSpCl = {} # classification
        connLinTh = int(dcSz[1]*0.002) # connecting line thickness
        connLinTh2 = int(dcSz[1]*0.004) # connecting line thickness
          # for viruses present in multiple ant species
        for ri in range(self.strData.shape[0]):
        # go through each virus
            vl = self.strData[ri,0] # virus label
            cl = self.strData[ri,1] # classification label 
            
            ### there's a specfic class to display 
            #   and this virus doesn't belong to it,
            #   continue without drawing
            if self.uiTask["showThisClassOnly"] != None:
                if not self.uiTask["showThisClassOnly"] == cl:
                    continue
            
            ### there's a specfic virus to display 
            #   and this virus is not,
            #   continue without drawing
            if self.uiTask["showThisVirusOnly"] != None:
                if not self.uiTask["showThisVirusOnly"] == vl:
                    continue
                        
            cntLst = [] 
            cntAll = 0
            cntSp = 0
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                cntAll += cnt
                if cnt > 0: cntSp += 1
            
            if cntSp > 1: # if virus occurred in more than one species
                vMSpPt1[vl] = vpPt1[vl]
                vMSpDeg[vl] = vpDeg[vl]
                vMSpCl[vl] = cl
                continue # don't draw it in this section
             
            
            arcPt1 = None
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vpPt1[vl][si][i]
                    pt2 = rot_pt((ct[0]+icRad-vCR-lLen[si], ct[1]), 
                                  ct, 
                                  vpDeg[vl][si][i]) # where connecting line ends
                    if arcType == 0:
                    # graph with arcs + straight lines
                        if cntAll > 1:
                        # there're more than one presences of this virus 
                            ### draw straight line part of connecting line
                            dc.SetPen(wx.Pen(self.sColor[si], connLinTh))
                            dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                            dc.DrawLine(pt1[0], pt1[1], pt2[0], pt2[1])
                            # store pt2 
                            if arcPt1 == None: arcPt1 = copy(pt2)
                    else:
                    # graph with simple arcs 
                        if i > 0:
                            ### draw arc between virus presence in this species
                            dc.SetPen(wx.Pen(self.sColor[si], connLinTh))
                            dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                            _pt = vpPt1[vl][si][i-1]
                            deg1 = vpDeg[vl][si][i]
                            deg2 = vpDeg[vl][si][i-1]
                            deg = deg1 + (deg2-deg1)/2
                            aCt = rot_pt((ct[0]+icRad, ct[1]),
                                          ct, 
                                          deg)
                            if self.vpDeg > 0: 
                                dc.DrawArc(pt1[0], pt1[1],
                                           _pt[0], _pt[1], 
                                           aCt[0], aCt[1])
                            else:
                                dc.DrawArc(_pt[0], _pt[1],
                                           pt1[0], pt1[1], 
                                           aCt[0], aCt[1])
                if arcType == 0:
                    if cntAll > 1 and cnt > 0:
                        lLen[si] += baseLLen
                        ### draw arc part of the connecting line
                        dc.SetPen(wx.Pen(self.sColor[si], connLinTh))
                        dc.SetBrush(wx.Brush('#000000', wx.TRANSPARENT))
                        dc.DrawArc(arcPt1[0], arcPt1[1],
                                   pt2[0], pt2[1], 
                                   ct[0], ct[1])
       
            for si in range(self.numSpecies):
                cnt = len(vpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vpPt1[vl][si][i]
                    ### draw virus presence circle
                    dc.SetPen(wx.Pen('#000000', 1, wx.TRANSPARENT))
                    dc.SetBrush(wx.Brush(self.cColor[cl]))
                    dc.DrawCircle(pt1[0], pt1[1], vCR)
        ### [end] draw connecting lines & virus presence circles
        
        ### [begin] draw connecting line & virus presence circles,
        ###   which present in multiple ant species
        #lLen = max(lLen) + baseLLen*4 # starting line length for these viruses
        len4ArcCt= int(dcSz[1]*0.02) # length to calculate center point 
          # for DrawArc
        for vl in vMSpPt1.keys():
            arcPt1 = None
            pts4isa = [] # list of points to draw arc line across ant species 
            for si in range(self.numSpecies):
                cnt = len(vMSpPt1[vl][si])
                for i in range(cnt): # virus presences in species
                    pt1 = vMSpPt1[vl][si][i]
                    if i > 0:
                        ### draw arc line between 
                        ###   virus presences in this species
                        deg1 = vMSpDeg[vl][si][i-1]
                        deg2 = vMSpDeg[vl][si][i]
                        deg = deg1 + (deg2-deg1)/2
                        aCt = rot_pt((ct[0]+icRad-len4ArcCt, ct[1]),
                                      ct, 
                                      deg)
                        dc.SetPen(wx.Pen("#000000", connLinTh2))
                        dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
                        if self.vpDeg > 0:
                            dc.DrawArc(pt1[0], pt1[1], 
                                       pPt1[0], pPt1[1], 
                                       aCt[0], aCt[1])
                        else:
                            dc.DrawArc(pPt1[0], pPt1[1], 
                                       pt1[0], pt1[1], 
                                       aCt[0], aCt[1])
                        ### store middle point of arc line
                        ###   to draw arc line between ant species
                        dist = np.sqrt((pt1[0]-pPt1[0])**2+(pt1[1]-pPt1[1])**2)
                        deg = calc_line_angle(aCt, ct)
                        pt4isa = rot_pt((aCt[0]+dist/2, aCt[1]),
                                        aCt,
                                        deg)
                        pts4isa.append(pt4isa)
                    # store pt1 for the next loop
                    pPt1 = copy(pt1)
                    '''
                    ### draw straight line part of connecting line
                    dc.SetPen(wx.Pen("#000000", connLinTh2))
                    dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
                    dc.DrawLine(pt1[0], pt1[1], pt2[0], pt2[1])
                    if arcPt1 == None: arcPt1 = copy(pt2)
                    '''
                if cnt == 1: pts4isa.append(pt1)
            ### draw arc line between virus presences across ant species
            dc.SetPen(wx.Pen("#000000", connLinTh2))
            dc.SetBrush(wx.Brush("#000000", wx.TRANSPARENT))
            for i in range(1, len(pts4isa)):
                pt1 = pts4isa[i-1]
                pt2 = pts4isa[i]
                ptsDist = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
                minX = min(pt1[0], pt2[0]); maxX = max(pt1[0], pt2[0])
                minY = min(pt1[1], pt2[1]); maxY = max(pt1[1], pt2[1])
                ptsCt = (int(minX+(maxX-minX)/2), int(minY+(maxY-minY)/2))
                ptsDeg = calc_line_angle(pt1, pt2)
                ptsDeg += 90
                aCt = calc_pt_w_angle_n_dist(ptsDeg,
                                             int(ptsDist/2), 
                                             ptsCt[0],
                                             ptsCt[1], 
                                             flagScreen=True)
                dc.DrawArc(pt1[0], pt1[1], pt2[0], pt2[1], aCt[0], aCt[1])
            # draw arc part of the connecting line
            #dc.DrawArc(arcPt1[0], arcPt1[1], pt2[0], pt2[1], ct[0], ct[1])
            #lLen += baseLLen
            len4ArcCt += baseLLen 
         
        for vl in vMSpPt1.keys():
            for si in range(self.numSpecies):
                for i in range(len(vMSpPt1[vl][si])):
                # virus presences in species
                    pt1 = vMSpPt1[vl][si][i]
                    ### draw virus presence circle
                    dc.SetPen(wx.Pen('#000000', 1, wx.TRANSPARENT))
                    dc.SetBrush(wx.Brush(self.cColor[vMSpCl[vl]]))
                    dc.DrawCircle(pt1[0], pt1[1], vCR)
        ### [end] draw connecting line & virus presence circles
        ##### [end] drawing graph ----- 
         
        ##### [begin] drawing legend ----- 
        _x1 = int(dcSz[0]*0.76)
        _x2 = int(dcSz[0]*0.92)
        _y = int(dcSz[1]*0.05)
        _yInc = int(fonts[1].GetPixelSize()[1] + dcSz[1]*0.001)
        vlR = {}
        clR = {}
        
        # temporary strData, sorted with virus label
        sD = self.strData[self.strData[:,0].argsort(axis=0)]
        for cl in list(np.unique(sD[:,1])):
        # go through each classification
            onlyCL = self.uiTask["showThisClassOnly"]
            ### write classification label
            if (onlyCL != None and onlyCL == cl):
            # if it's showing only this class
                dc.SetFont(emphFonts[0]) # emphasize this label
            else:
                dc.SetFont(fonts[0])
            dc.SetTextForeground("#999999")
            dc.DrawText(cl, _x1, _y)
            if len(self.clR) == 0: # first time to draw graph
                w, h = dc.GetTextExtent(cl)
                # store rect of text (classification in legend)
                clR[cl] = (_x1, _y, _x1+w, _y+h)
            ### write virus label
            dc.SetFont(fonts[1])
            # set text color for classification of the virus
            dc.SetTextForeground(self.cColor[cl])
            for ri in range(sD.shape[0]): # row (= virus)
                if cl == sD[ri,1]:
                    vl = sD[ri,0]
                    onlyVL = self.uiTask["showThisVirusOnly"]
                    if (onlyCL != None and onlyCL == cl) or \
                      (onlyVL != None and onlyVL == vl):
                    # if it's showing only this classification or viris
                        dc.SetFont(emphFonts[1]) # emphasize this label
                        # write virus label
                        dc.DrawText(vl, _x2, _y)
                        dc.SetFont(fonts[1])
                    else:
                        # write virus label
                        dc.DrawText(vl, _x2, _y)
                    if len(self.vlR) == 0: # first time to draw graph
                        w, h = dc.GetTextExtent(vl)
                        # store rect of text (virus label in legend)
                        vlR[vl] = (_x2, _y, _x2+w, _y+h)
                    _y += _yInc # increase y-coordinate
        ##### [end] drawing legend -----
        
        ##### [begin] drawing virus label user clicked -----
        dc.SetFont(fonts[2])
        if self.uiTask["showVirusLabel"] != None:
            x, y, virus = self.uiTask["showVirusLabel"]
            gpSz = self.parent.pi["gp"]["sz"]
            rat = (dcSz[0]/gpSz[0], dcSz[1]/gpSz[1])
            x = int(x * rat[0])
            y = int(y * rat[1])
            dc.SetTextForeground("#666666")
            dc.DrawText(virus, x, y)
        ##### [end] drawing virus label user clicked -----
        
        if len(self.vpPt) == 0: # first time to draw graph
            self.vCR = vCR # store radius of virus presence circle
            self.vpPt = vpPt1 # store virus presence points
            self.vlR = vlR # store rects of virus labels in legend
            self.clR = clR # store rects of classification labels in legend

    #-------------------------------------------------------------------
  
    def initUITask(self):
        """ Delete all current UI tasks.
        
        Args: None
        
        Returns: None
        """ 
        if DEBUG: print("ProcGraphData.initUITask()")

        for key in self.uiTask.keys():
            self.uiTask[key] = None

    #-------------------------------------------------------------------
    
#=======================================================================
