# coding: UTF-8
""" Currently for rules of a single ant to be used in AntSim program.

last edited: 2023-10-08
"""
import numpy as np
import cv2

from initVars import *
from modFFC import *

DEBUG = False

#===============================================================================

class AAggregate:
    """ Class to represent a hypothetical ant aggregate from the same colony, 
    with its attributes & behavioural rules.
    Currently to be used in AntSim program only.
    
    Args:
        None
     
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """

    def __init__(self, parent, params={}):
        if DEBUG: MyLogger.info(str(locals()))

        ##### [begin] setting up attributes on init. -----
        self.parent = parent
        # simulation type
        self.simTyp = parent.simTyp
        ### number of workers in this aggregate
        if "nw" in params.keys(): self.nw = params["nw"]
        else: self.nw = parent.mPa["nWorkerInG"] 
        ### degree of interaction with each other
        ###   (currently boolean; 0 or 1, but will be a float in the future)
        if "interaction" in params.keys():
            self.interaction = params["interaction"]
        else:
            self.interaction = parent.mPa["interaction"]
        ### threshold ranges of activity & inactivity boosting
        if "thR" in params.keys(): self.thR = params["thR"]
        else: self.thR = parent.mPa["thR"]
        ### set the thresholds 
        if "th" in params.keys():
            self.th = params["th"]
        else:
            self.th = {}
            for k in self.thR.keys():
                self.th[k] = np.random.randint(self.thR[k][0], self.thR[k][1]) 
        # log file path
        self.logFP = parent.logFP
        ##### [end] setting up attributes on init. -----

        # set probabilities
        self.adjProb()
   
    #---------------------------------------------------------------------------
    
    def initSim(self, params={}):
        """ initialization before simulation

        Args:
            params (dict): parameters

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        prnt = self.parent
        
        self.rALst = [] # list of activity durations of the aggregate 
        self.rILst = [] # list of inactivity durations of the aggregate 
        
        # info for the current activity bout
        self.boutInfo = dict(
                                bIdx = [0] * self.nw, # beginning index
                                eIdx = [0] * self.nw, # end index
                                bType = ["inact"] * self.nw, # inact/activity
                                ) 
        
        self.flagActive = False # aggregate's activity 
                                #   (nActPhase > 0 means it's active)
        self.nActPhase = 0 # number of active phases of individual workers
        self.idxAorIBegun = 0 # index of overall active or inactive phase begun

    #---------------------------------------------------------------------------

    def adjProb(self, flagVerb=True):
        """ adjust probabilities of behavior of ants in this aggregate
        Args:
            flagVerb (bool): print/log messages

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        prnt = self.parent

        minThr = 3 # minimum threshold seconds 
                   #   (currently 3 due to 2-second-data-bundling)
        tarF = 0.05 # target fraction to increase the sum

        self.prob = {} # probabilities
        devFrac = 1/20
        noAdj = False # [*] for debugging [*]
        msgW = "\n"
        for thK in self.thR.keys():
            bThR = self.thR[thK] # boosting threshold range
            if bThR is None: continue

            probK = thK[:2]

            if thK == "ADB": lbl = "activity"
            elif thK == "IDB": lbl = "inactivity"
            elif thK == "WDB": lbl = "walking-distance"
            
            if flagVerb:
                msg = f'\n* {lbl} threshold range: {str(bThR)}'
                print(msg)
                writeFile(self.logFP, msg+"\n")
       
            self.prob[probK] = [] 
            # fraction of the threshold range to give to individual workers
            #   deviation from the colony threshold 
            dev = int((bThR[1]-bThR[0]) * devFrac)
            
            msgW += f'\n{lbl}\n'
            ### boost motion activity probabilities of longer durations
            dTh = self.th[thK] # threshold for an output (= a colony)
            for wi in range(self.nw):
                msgW += f'[{wi+1}/{self.nw}]'
                if noAdj:
                    _dTh = 0
                else:
                    # deviate an worker's threshold from the colony threshold
                    _dTh = dTh + np.random.randint(-dev, dev)
                    _dTh = min(max(bThR[0], _dTh), bThR[1])
                _dTh = max(minThr, _dTh) # minimum threshold 
                op = prnt.prob[probK].copy() # original probability
                msgW += f', {_dTh}-{len(op)}'
                boostF = 1.0 # initial boost factor
                _summed = np.sum(op[_dTh:])
                while _summed == 0:
                # sum should be higher than zero
                    _dTh -= 1 
                    _summed = np.sum(op[_dTh:])
                _initSum = copy(_summed)
                while _summed < tarF:
                    boostF = boostF*1.1
                    _summed = np.sum(op[_dTh:]*boostF)
                    if flagVerb:
                        msg = f'\r summed {lbl}: {_summed}, boost F.: {boostF}'
                        print(msg, end="")
                msgW += f', boost-factor: {boostF}\n'
                op[_dTh:] *= boostF # boost longer duration range
                _sum = 1.0 - np.sum(op[_dTh:])
                # suppress short duration range
                op[:_dTh] *= (_sum / np.sum(op[:_dTh])) 
                # store this worker's probability
                self.prob[probK].append(op)
        
        # set aggregate's walking probability
        wProb = np.random.uniform(prnt.walkProbRng[0], prnt.walkProbRng[1])
        self.prob["wProb"] = []
        ### adjust each worker's walking probability with +/- 2%, and store it
        for wi in range(self.nw):
            self.prob["wProb"].append(wProb + np.random.uniform(-0.02, 0.02)) 

        if not self.simTyp.startswith("sweep"): # non-sweep operation
            if flagVerb:
                print(msgW)
                writeFile(self.logFP, msgW+"\n") 

    #---------------------------------------------------------------------------
    
    def simulate(self, di, cntLIDB, data, aPosData):
        """ simulate ant activity data

        Args:
            di (int): Current data index.
            cntLIDB (list): List of # of active nestmates for each worker.
            data (list): Output activity (n of motions) data.
            aPosData (list): Output ant's position data.

        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        prnt = self.parent
        aArr = prnt.arenaArr.copy()

        if aPosData[0] == []:
        # no position data, position ants
            ### position ants in random positions
            for wi in range(self.nw):
                # coordinates where ant can be positioned
                ys, xs = np.where(aArr==0)
                x = np.random.choice(xs) # choose x-coordinate
                y = np.random.choice(ys) # choose y-coordinate
                aPosData[wi].append((x, y)) # store the current position
                aArr[y, x] = 1 # mark the occupied position in the array

        for wi in range(self.nw):

            if self.boutInfo["eIdx"][wi] == di:
            # reached end of an act/inact
               
                if self.boutInfo["bType"][wi] == "act":
                    self.boutInfo["bType"][wi] = "inact"
                    _p = self.prob["ID"][wi].copy()
                    if self.nw > 1 and self.interaction:
                        ### more active workers lead to higher chance that 
                        ###   this ant will have longer inactivity 
                        ###   in the coming phase
                        # count active workers with chance
                        #countedA = int(np.round(self.nActPhase * uniform(0,1)))
                        countedA = self.nActPhase
                        if countedA > 0: 
                        # at least, one active ant is counted 
                            # store this ant's counted active workers 
                            cntLIDB[wi] += countedA
                            # set threshold to adjust the probability 
                            _thDur = countedA * 4
                            _p[:_thDur] = 0 # suppress shorter durations 
                            _r = 1.0 - np.sum(_p[:_thDur])
                            # boost longer duration range accordingly
                            _p[_thDur:] *= (_r / np.sum(_p[_thDur:]))
                            ''' !!! [TEMP;debugging]
                            fp = path.join(self.rsltDir,
                                           "probInd_%i.png"%(_th))
                            if not path.isfile(fp):
                                fig, ax = plt.subplots()
                                xLst = list(range(300))
                                plt.bar(xLst, height=_p[:len(xLst)])
                                _t = "Inactivity probability th. %i"%(_th)
                                plt.title(_t)
                                plt.savefig(fp)
                            '''

                    _dur = np.random.choice(prnt.pXLst["ID"], p=_p)
                    self.nActPhase -= 1 # decrease n of active phase

                elif self.boutInfo["bType"][wi] == "inact":
                    self.boutInfo["bType"][wi] = "act"
                    _dur = np.random.choice(prnt.pXLst["AD"], 
                                            p=self.prob["AD"][wi])
                    self.nActPhase += 1 # increase n of active phase


                self.boutInfo["bIdx"][wi] = di
                self.boutInfo["eIdx"][wi] = di + _dur 
            
            cx, cy = aPosData[wi][-1] # current ant position
            
            if self.boutInfo["bType"][wi] == "act":
            # this ant is currently active
                # determine number of motions for this data-index 
                nM = np.random.choice(prnt.mpLst, p=prnt.mpProb)
                # store the numder of motion data
                data[wi].append(nM)
               
                _prob = np.random.rand()
                if _prob < self.prob["wProb"][wi]:
                # walking occurs with this ant's walking probability
                    collisionChk = True 
                    ah, aw = aArr.shape # arena width & height
                    while collisionChk:
                        # get walking distance
                        _dist = np.random.choice(prnt.pXLst["WD"], 
                                                 p=self.prob["WD"][wi])
                        # WD probability has % data so that divided by 100
                        # then multiple the ant body size (in mm). 
                        # the arena size is also in millimeters.
                        _dist = int(_dist / 100 * prnt.mPa["antSz"])
                        # direction is random
                        _dir = np.random.randint(0, 361)
                        # calculate moved coordinate with _dir & _dist
                        mx, my = calc_pt_w_angle_n_dist(_dir, _dist, cx, cy, 
                                                        flagScreen=True)
                        if 0 <= mx < aw and 0 <= my < ah and \
                          aArr[my, mx] == 0:
                        # this is a movable spot
                            collisionChk = False # passed the collision check
                    ### mark the moved ants
                    aArr[cy, cx] = 0
                    aArr[my, mx] = 1
                    # store the moved position
                    aPosData[wi].append((mx, my))
                else:
                # no walking
                    aPosData[wi].append((cx, cy)) # keep the current position 

            elif self.boutInfo["bType"][wi] == "inact":
            # this ant is currently inactive
                data[wi].append(0) # N of motion = 0
                aPosData[wi].append((cx, cy)) # keep the current position 

            if self.flagActive:
            # the aggregate has been active
                if self.nActPhase == 0 or di == prnt.mPa["nDP"]-1:
                # nobody is active or reached end of data
                    self.flagActive = False
                    # store activity duration
                    self.rALst.append(di-self.idxAorIBegun) 
                    # store the beginning index
                    self.idxAorIBegun = copy(di) 
            else:
            # the aggregate has been inactive
                if self.nActPhase > 0 or di == prnt.mPa["nDP"]-1:
                # someone is active or reacehd end of data
                    self.flagActive = True
                    # store inactiity duration
                    self.rILst.append(di-self.idxAorIBegun) 
                    # store the beginning index
                    self.idxAorIBegun = copy(di) 

        return cntLIDB, data, aPosData

    #---------------------------------------------------------------------------

#===============================================================================

if __name__ == '__main__': pass

