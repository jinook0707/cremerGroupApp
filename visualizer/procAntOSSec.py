# coding: UTF-8
"""
This module is for the secondary data processing with the data from 'aos' in Visualizer.

Dependency:
    Numpy (1.17)
    SciPy (1.4)
    OpenCV (3.4)

last edited: 2023-08-05
"""
from os import path
from glob import glob
from copy import copy

from scipy.signal import savgol_filter, find_peaks
from scipy.stats import iqr 

from modFFC import *
from modCV import *
from modGraph import *

DEBUG = False

#===============================================================================

class ProcAntOSSec:
    """ Class for processing data from 'aos' and generate graphs.
    
    Args:
        parent (wx.Frame): Parent frame
    
    Attributes:
        Each attribute is commented in 'setting up attributes' section.
    """
    
    def __init__(self, mainFrame, parent):
        if DEBUG: MyLogger.info(str(locals()))

        self.mainFrame = mainFrame # main object with wx.Frame
        self.parent = parent

    #---------------------------------------------------------------------------

    def drawGraph(self):
        """ Draw graph with data from 'aos' 
        
        Args:
            None
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        main = self.mainFrame # wx.Frame in visualizer.py

        w = wx.FindWindowByName("aType_cho", main.panel["ml"])
        aType = widgetValue(w)

        if aType.startswith("n1610"):
            aType = aType.lstrip("n1610;")
            if aType in ["intensity", "dist2b", "proxMCluster"]:
                self.drawGraph_n1610() 
            elif aType == "actInact":
                self.drawGraph_n1610_actInact()
        elif aType == "n1;saMVec":
                self.drawGraph_n1_mVector()
        elif aType == "n1;saDist2c": # for n1 data
                self.drawGraph_dist2c("c")
        elif aType == "dist2ca": # for n6 & n n10 data
                self.drawGraph_dist2c("ca")
        showStatusBarMsg(main, "", -1)

    #---------------------------------------------------------------------------

    def drawGraph_n1610(self):
        """ Draw graph with data from 'aos' for n1610 experiment in 2022 
        
        Args:
            None
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        _debug = False
        flagSavImgs = False # save result graph images during this process
        main = self.mainFrame # wx.Frame in visualizer.py
        outputFolder, _ = path.split(main.inputFP) # output folder
        pa = self.getParamFromML()
        pa["aType"] = pa["aType"].lstrip("n1610;")

        ### set max. value for certain graph type
        maxY = {}
        if pa["aType"] == "intensity":
            maxY["psd"] = 2500
            nSLst = ["n01", "n06", "n10"]
        elif pa["aType"] == "dist2b":
            maxY["psd"] = 600 
            nSLst = ["n01", "n06", "n10"]
        elif pa["aType"] == "proxMCluster":
            maxY["psd"] = -1 
            nSLst = ["n06", "n10"]

        fLst = sorted(glob(path.join(main.inputFP, "*.npy")))

        # get items from filename matching with the given parameters
        dateStr, fpLst = self.n1610_getFNItems(fLst, pa) 
        
        fpLst = np.asarray(fpLst)
        
        ### prepare dSrc dict
        dSrc = {} # data source 
        for di, dS in enumerate(sorted(dateStr)):
            a = fpLst[fpLst[:,2] == dS] # matching date-string
            if sorted(list(a[:,1])) == nSLst: # n-string should match
                dSrc[dS] = {}
                for nS in nSLst:
                    dSrc[dS][nS] = a[a[:,1]==nS][0][0]
            else:
                dateStr[di] = None
                continue
        while None in dateStr: dateStr.remove(None)

        if len(dSrc) == 0:
            msg = "No files, matching with the given parameters, were found"
            msg += f" in the folder ({main.inputFP})."
            wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
            main.flags["blockUI"] = False
            return

        if _debug:
            for ds in dSrc.keys():
                print(ds)
                for ns in dSrc[ds].keys():
                    print(ns, path.basename(dSrc[ds][ns]))

        pa["nRow"] = int(np.ceil(len(dateStr) / pa["nCol"]))

        ##### [begin] get data -----
        data = {} # data 
        dataDev = {} # data for deviation from mean 
        
        if pa["h2proc"] > 0:
            # length of data list to get
            dL2get = int(pa["h2proc"] * 60 * 60 / pa["dPtIntvSec"])
        if pa["h2procDev"] > 0:
            # length of deviation data list to get
            dL2getDev = int(pa["h2procDev"] * 60 * 60 / pa["dPtIntvSec"])

        for ds in dSrc.keys():
            data[ds] = {}
            dataDev[ds] = {}
            for ns in dSrc[ds].keys():

                fn = dSrc[ds][ns]

                data[ds][ns] = []
                dataDev[ds][ns] = []
                d = np.load(fn)
                inLeadingZeros = True 
                iDT = None # to store the first datetime 
                for ri in range(d.shape[0]):

                    _data = float(d[ri][pa["aType"]])

                    ### trim the leading zeros
                    if inLeadingZeros:
                        if _data == 0: continue
                        else: inLeadingZeros = False

                    dt = d["datetime"][ri]
                    if iDT is None:
                        iDT = get_datetime(dt)
                        continue
                    cDT = get_datetime(dt)

                    elapsedHour = ((cDT-iDT).total_seconds())/60/60 

                    flagBreak = dict(d=False, dDev=False)
                    
                    if elapsedHour >= pa["h2ignore"]: # hours to ignore passed
                        if pa["h2proc"] <= 0: # no hour limit
                            data[ds][ns].append(_data) # store data
                        else: # there's a hour limit for data 
                            if len(data[ds][ns]) >= dL2get:
                            # data length reached the limit 
                                flagBreak["d"] = True
                            else:
                                # store data
                                data[ds][ns].append(_data)

                    ### data for deviation graph
                    if elapsedHour >= pa["h2ignoreDev"]:
                    # hours to ignore passed
                        if pa["h2procDev"] <= 0: # no hour limit
                            dataDev[ds][ns].append(_data) # store data
                        else: # there's a hour limit for data 
                            if len(dataDev[ds][ns]) >= dL2getDev:
                            # data length reached the limit 
                                flagBreak["dDev"] = True
                            else:
                                # store data
                                dataDev[ds][ns].append(_data) 

                    if flagBreak["d"] and flagBreak["dDev"]: break 
                
                # trim tailing zero padding
                data[ds][ns] = np.trim_zeros(
                                            np.asarray(data[ds][ns]), trim='b'
                                            ).tolist()
                dataDev[ds][ns] = np.trim_zeros(
                                        np.asarray(dataDev[ds][ns]), trim='b'
                                        ).tolist()

                if _debug:
                    msg = f"{ds} ({ns}) [{str(iDT)} ~ {str(cDT)}]\n"
                    msg += f"  - number of data-points: {len(data[ds][ns])}\n"
                    msg += f"  - sum of number of motions: "
                    msg += f"{int(np.sum(data[ds][ns]))}"
                    print(msg)
        ##### [end] get data -----
        
        if flagSavImgs: rsltMsg = f"Graphs -----\n"

        ##### [begin] Power Spectral Density plot -----
        setPLTParams(nRow=pa["nRow"], nCol=pa["nCol"], spSz=(2.0,1.2))
        if pa["aType"] == "proxMCluster": _shY = False
        else: _shY = True
        for ns in nSLst:
            fig, ax = plt.subplots(pa["nRow"], pa["nCol"], 
                                   sharey=_shY, sharex=True, tight_layout=True)
            ri = 0
            ci = 0
            psdMVLst = []
            for ds in sorted(data.keys()):
                if ns not in data[ds].keys(): continue

                params = dict(dPtIntvSec=pa["dPtIntvSec"], maxY=maxY["psd"])
                ret = drawPSD(data[ds][ns], ax[ri, ci], "", params)
                psdMVLst.append(ret["psdMV"])
                ci += 1
                if ci >= pa["nCol"]:
                    ci = 0 
                    ri += 1 
            plt.gcf().text(0.01, 0.93, "power (RMS, x10^3)", fontsize=8)

            if _shY:
                ### set y-ticks
                if maxY["psd"] == -1: _mY = max(psdMVLst)
                else: _mY = maxY["psd"]
                plt.ylim(0, _mY)
                setYFormat(_mY, None, "{:,.1f}", False)
                #y, ytL = genYTickLbls(psdMV, nTicks=3)
                #plt.yticks(y, ytL, fontsize=6)

            fig.suptitle("[PSD] Data from ants; %s"%(ns), fontsize=12)
            graph = convt_mplFig2npArr(fig)
            if flagSavImgs: 
                fn = f"vRslt_psd_{pa['aType']}_{ns}.png"
                rsltMsg += f"{fn}\n"
                fp = path.join(outputFolder, fn)
                cv2.imwrite(fp, graph)
            # display the result graph
            main.callback(("finished", graph), flag="drawGraph")
        ##### [end] Power Spectral Density plot -----

        ##### [begin] comparison of aggreagates -----
        _lbls = []
        _data = []
        for ds in sorted(data.keys()):
            for ns in sorted(data[ds].keys()):
                _lbls.append(f"{ds}_{ns}")
                _data.append(data[ds][ns])
        retGraphs = proc_mComp(None, _data, _lbls, pa["dPtIntvSec"])
        for k in retGraphs.keys():
            if flagSavImgs:
                fn = f"vRslt_mComp_{pa['aType']}_{k}.png"
                rsltMsg += f"{fn}\n"
                fp = path.join(outputFolder, fn)
                cv2.imwrite(fp, retGraphs[k])
            # display the result graph
            main.callback(("finished", retGraphs[k]), flag="drawGraph")
        ##### [end] comparison of aggreagates -----

        ##### [begin] draw deviation graph -----
        flagDevNorm = True # normalization (division by max value)
        for ns in nSLst:
            fig, ax = plt.subplots(pa["nRow"], pa["nCol"], 
                                   sharey=True, sharex=True, tight_layout=True) 
            ri = 0; ci = 0
            print(f"processing {ns} for deviation graph..")

            for ds in sorted(dataDev.keys()):
                if ns not in data[ds].keys(): continue
                if _debug: print(f"processing {ds}_{ns} ..")
                
                d = dataDev[ds][ns]
                meV = np.mean(d)
                d = d - meV
               
                wlf = 1 #1/1.1 #1/6
                # determine window length in terms of number of bundled data 
                # points (default length is one hour)
                windowLen = int(3600*wlf/pa["dPtIntvSec"]) 
                hWL = int(windowLen/2)
                gd = []
                for i in range(hWL, len(d)-hWL, windowLen):
                    gd.append(np.sum(d[i-hWL:i+hWL]))

                if flagDevNorm:
                    ### normalize
                    gd = np.asarray(gd) 
                    gd = gd / np.max(np.abs(gd))
               
                # draw bar graph
                ax[ri,ci].bar(list(range(len(gd))), height=gd) 

                ### draw smooth line
                wLen = 12 * int(wlf**-1) # window length 
                gsd = savgol_filter(gd, window_length=wLen, polyorder=3)
                ax[ri,ci].plot(gsd, c=(1.0,0,0), linewidth=0.75)

                ### write the first time when it goes below average (0)
                for i, gsdV in enumerate(gsd):
                    if gsdV < 0: break
                if wlf == 1.0: txt = "%i h"%(i+1)
                else: txt = "%.3f h"%((i+1) * wlf)
                ax[ri,ci].text(i, max(gd), txt, fontsize=12,
                           horizontalalignment="left", verticalalignment="top")

                if flagDevNorm:
                    ax[ri,ci].set_ylim(-1.0, 1.0)
                else:
                    # set y-tick format for this subplot
                    setYFormat(max(gd), ax[ri,ci])
               
                ci += 1
                if ci >= pa["nCol"]:
                    ci = 0 
                    ri += 1 
         
            supTitle = f"Diff. from mean [{pa['aType']}]; {ns}"
            fig.suptitle(supTitle)
            graph = convt_mplFig2npArr(fig)
            if flagSavImgs: 
                fn = f"vRslt_diffFMean_{pa['aType']}_{ns}.png"
                rsltMsg += f"{fn}\n"
                fp = path.join(outputFolder, fn)
                cv2.imwrite(fp, graph)
            # display the result graph
            main.callback(("finished", graph), flag="drawGraph")
        ##### [end] draw deviation graph -----

        plt.close("all")
        if flagSavImgs:
            rsltMsg += f"----------\n\nsaved in\n{main.inputFP}."
            wx.MessageBox(rsltMsg, "Info.", wx.OK|wx.ICON_INFORMATION)

    #---------------------------------------------------------------------------

    def drawGraph_n1610_actInact(self):
        """ Draw graph with data from 'aos' for n1610 experiment in 2022;
        regarding activity/inactivity 
        
        Args:
            None
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        _debug = False 
        flagSavImgs = False # save result graph images during this process
        main = self.mainFrame # wx.Frame in visualizer.py
        outputFolder, _ = path.split(main.inputFP) # output folder
        pa = self.getParamFromML()
        pa["aType"] = pa["aType"].lstrip("n1610;")
        pa["dPtIntvSec"] = 2 # 2-second-bundled-data only
        
        nSLst = ["n01", "n06", "n10"]
        mKeysInput = ["mboutsec", "inactivitysec"]
        mKeysOutput = ["activitysec", "inactivitysec"]

        # set duration (in sec.) range to count
        durRng = dict(inactivitysec=[0, 2000], activitysec=[0, 5000])

        fLst = sorted(glob(path.join(main.inputFP, "*.npy")))

        # get items from filename matching with the given parameters
        dateStr, fpLst = self.n1610_getFNItems(fLst, pa) 
        if _debug:
            print("fpLst -----")
            for _fpL in fpLst: print(_fpL)
        if fpLst == []:
            msg = "No files, matching with the given parameters, were found"
            msg += f" in the folder ({main.inputFP})."
            wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
            main.flags["blockUI"] = False
            return
        fpLst = np.asarray(fpLst)
        
        ##### [begin] get data -----
        data = {}
        # data for activity/inactivity probability distributions of n01 
        data4prob = {} 
        for ki, k in enumerate(mKeysInput):
            inputDLbl = "intensity%s"%(k.capitalize())
            mk = mKeysOutput[ki]
            data[mk] = {}
            data4prob[mk] = {}
            for ns in nSLst:
                data[mk][ns] = []
                data4prob[mk][ns] = []

                fa = fpLst[fpLst[:,1]==ns] # file-paths, matching with n-string
                for fp, _ns, _ds in fa:
                    if inputDLbl not in fp: continue
                    
                    msg = "Processing %s .."%(path.basename(fp))
                    arr = np.load(fp)

                    _data = arr[inputDLbl].tolist()
                    _dt = arr["datetime"]
                    data[mk][ns].append([])
                    data4prob[mk][ns].append([])

                    ### remove data from the first date(s)
                    iDT = get_datetime(_dt[0])
                    for ri in range(1, _dt.shape[0]):
                    # go through each data index
                        cDT = get_datetime(_dt[ri])
                        elapsedHour = ((cDT-iDT).total_seconds())/60/60 
                        if elapsedHour >= pa["h2ignore"]:
                        # hours to ignore passed
                            sIdx = copy(ri) # get the starting index
                            break

                    _data = np.asarray(_data[sIdx:])
                    
                    msg += ", Max val.: %i"%(np.max(_data))

                    data[mk][ns][-1] = _data.tolist()
                    for sec in range(durRng[mk][0], durRng[mk][1]+1):
                        # count how many of this duration/interval occurred
                        durCnt = np.count_nonzero(_data==sec)
                        data4prob[mk][ns][-1].append(durCnt)

                    print(msg)
        ##### [end] get data -----

        ##### [begin] saving activity/inactivity prob. dist. -----
        kwa = {}
        for mk in mKeysOutput:
            dArr = np.asarray(data4prob[mk]["n01"])
            summed = np.sum(dArr, 1) # sum each row (of each data file)
            prob = dArr / summed[:,None] # divide each row with summed values 
                                         #   to get occurrence probability
            fp = path.join(outputFolder, f"prob_{mk}.npy")
            np.save(fp, np.asarray(prob))
            ap = np.mean(prob, 0) # average probability
            if mk == "activitysec":
                kwa["acdp"] = np.mean(prob, 0) # average probability
                kwa["acdpXLim"] = 300 
            elif mk == "inactivitysec":
                kwa["iadp"] = np.mean(prob, 0) # average probability
                kwa["iadpXLim"] = 300
        rGraph = drawProbDistributions(kwa)
        for k in rGraph.keys():
            if flagSavImgs:
                fn = f"vRslt_{k}.png"
                fp = path.join(outputFolder, fn)
                cv2.imwrite(fp, rGraph[k])
            # display the result graph
            main.callback(("finished", rGraph[k]), flag="drawGraph")
        ##### [end] saving activity/inactivity prob. dist. -----

        ##### [begin] draw activity/inactivity graphs -----
        for ns in nSLst:
            nw = int(ns.lstrip("n"))
            rGraph = drawDurGraphs(
                            nw, 
                            data["activitysec"][ns], 
                            data["inactivitysec"][ns],
                            False)
            for k in rGraph.keys():
                if flagSavImgs:
                    fn = f"vRslt_{pa['aType']}_{k}_{ns}.png"
                    fp = path.join(outputFolder, fn)
                    cv2.imwrite(fp, rGraph[k])
                # display the result graph
                main.callback(("finished", rGraph[k]), flag="drawGraph")
        ##### [end] draw activity/inactivity graphs -----
        
        plt.close("all")

    #---------------------------------------------------------------------------

    def drawGraph_n1_mVector(self):
        """ Draw graph with data from 'aos' for n1610 experiment in 2022;
        regarding direction & distance (relative to body-length) of 
        movements of a single ant 
        
        Args:
            None
         
        Returns:
            None
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        _debug = True 
        flagSavImgs = False # save result graph images during this process
        main = self.mainFrame # wx.Frame in visualizer.py
        outputFolder, _ = path.split(main.inputFP) # output folder
        pa = self.getParamFromML()
        pa["aType"] = pa["aType"].lstrip("n1;")
        pa["dPtIntvSec"] = 1 # 1-second-bundled-data only
        pa["maxVal"] = dict(direction=360, dist=0)
        
        fLst = sorted(glob(path.join(main.inputFP, "*.npy")))

        # get items from filename matching with the given parameters
        dateStr, fpLst = self.n1610_getFNItems(fLst, pa) 
        if _debug:
            print("fpLst -----")
            for _fpL in fpLst: print(_fpL)
        if fpLst == []:
            msg = "No files, matching with the given parameters, were found"
            msg += f" in the folder ({main.inputFP})."
            wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
            main.flags["blockUI"] = False
            return
        fpLst = np.asarray(fpLst)
        
        ##### [begin] get data -----
        if pa["h2proc"] > 0:
            # length of data list to get
            dL2get = int(pa["h2proc"] * 60 * 60 / pa["dPtIntvSec"])

        data = dict(direction=[], dist=[]) # read data
        data4prob = {} # data for probability distributions 
        for dk in data.keys(): data4prob[dk] = []
            
        for fp, ns, ds in fpLst:
            msg = f'Processing {path.basename(fp)} ..'
            arr = np.load(fp)

            _d = {}
            for dk in data.keys():
                data[dk].append([])
                data4prob[dk].append([])
                _d[dk] = arr[dk]
            
            _dt = arr["datetime"]

            iDT = None
            for ri in range(1, _dt.shape[0]):
            # go through each data index

                if _d["direction"][ri] == -1: continue # no movement occurred

                if iDT is None:
                    iDT = get_datetime(_dt[ri])
                    continue

                cDT = get_datetime(_dt[ri])
                elapsedHour = ((cDT-iDT).total_seconds())/60/60 
                
                if elapsedHour >= pa["h2ignore"]: # hours to ignore passed
                    if pa["h2proc"] <= 0: # no hour limit
                        for dk in data.keys():
                            data[dk][-1].append(_d[dk][ri]) # store data
                    else: # there's a hour limit for data 
                        if len(data[dk][-1]) >= dL2get:
                        # data length reached the limit 
                            break 
                        else:
                            for dk in data.keys():
                                data[dk][-1].append(_d[dk][ri]) # store data
           
            # update max. value of 'dist'
            _max = max(data["dist"][-1])
            if _max > pa["maxVal"]["dist"]: pa["maxVal"]["dist"] = _max 
          
        distMF = 100 # multiply this to the distance value (make it to %)
        for dk in data.keys():
            _max = pa["maxVal"][dk]
            if dk == "dist": _max = int(_max * distMF)
            for i in range(len(fpLst)):
                ### convert to array
                _data = np.asarray(data[dk][i])
                if dk == "dist":
                    _data *= distMF
                    _data = np.around(_data)
                    _data = _data.astype(np.uint16)
                ### make data4prob
                for val in range(_max+1):
                    # count how many of this value occurred
                    cnt = np.count_nonzero(_data==val)
                    data4prob[dk][i].append(cnt)

            print(msg)
        ##### [end] get data -----

        ##### [begin] draw prob. dist. graph & save the prob. as npy -----
        for dk in data.keys():
            dArr = np.asarray(data4prob[dk])
            summed = np.sum(dArr, 1) # sum each row (of each data file)
            prob = dArr / summed[:,None] # divide each row with summed values 
                                         #   to get occurrence probability
            if dk == "direction": gKey = "drp"
            elif dk == "dist": gKey = "dsp"
            ### display probability graphs of each file
            for ri in range(prob.shape[0]):
                g = drawProbDistributions({f'{gKey}':prob[ri]})
                cvFont = cv2.FONT_HERSHEY_PLAIN
                fThck = 1 
                _thP = int(g[gKey].shape[0] * 0.025)
                fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                         thresholdPixels=_thP,
                                                         thick=fThck)
                cv2.putText(g[gKey], f'[{fpLst[ri][2]}]', (5, txtH+txtBl), 
                    cvFont, fontScale=fScale, color=(0,0,0), thickness=fThck)
                if dk == "dist":
                    _txt = '% of body-length'
                    x = g[gKey].shape[1] - txtW*(len(_txt)+1)
                    y = g[gKey].shape[0] - (txtH*2+txtBl)
                    cv2.putText(g[gKey], _txt, (x, y), cvFont,
                        fontScale=fScale, color=(0,0,0), thickness=fThck)

                if _debug:
                    if dk == "direction": 
                        _max = max(prob[ri])
                        print(fpLst[ri][2], _max, list(prob[ri]).index(_max))
                main.callback(("finished", g[gKey]), flag="drawGraph")
            ### save probability to a file
            fp = path.join(outputFolder, f'prob_{dk}.npy')
            np.save(fp, np.asarray(prob))
            ### draw average (of all recorded individuals) probability
            rGraph = drawProbDistributions({f'{gKey}':np.mean(prob, 0)})
            if flagSavImgs:
                fn = f'vRslt_{gKey}.png'
                fp = path.join(outputFolder, fn)
                cv2.imwrite(fp, rGraph[gKey])
            # display the result graph
            main.callback(("finished", rGraph[gKey]), flag="drawGraph")

            plt.close("all")
        ##### [end] draw prob. dist. graph & save the prob. as npy -----
       
    #---------------------------------------------------------------------------

    def drawGraph_dist2c(self, flag="c"):
        """ Draw graph with data from 'aos' for n1610 experiment in 2022;
        regarding distance between the motion point of an ant and 
        the center of ROI0. 
        
        Args:
            flag (str): c or ca; from dist2c or dist2ca in procAntOS.py
         
        Returns:
            None
        """
        if DEBUG: MyLogger.info(str(locals()))

        _debug = True
        main = self.mainFrame # wx.Frame in visualizer.py
        pa = self.getParamFromML()
        pa["aType"] = f'dist2{flag}'
        pa["dPtIntvSec"] = 1 # 1-second-bundled-data only
        pa["maxVal"] = 0
        
        fLst = sorted(glob(path.join(main.inputFP, "*.npy")))

        # get items from filename matching with the given parameters
        dateStr, fpLst = self.n1610_getFNItems(fLst, pa) 
        if _debug:
            print("fpLst -----")
            for _fpL in fpLst: print(_fpL)
            print("----------")
        if fpLst == []:
            msg = "No files, matching with the given parameters, were found"
            msg += f" in the folder ({main.inputFP})."
            wx.MessageBox(msg, "Info.", wx.OK|wx.ICON_INFORMATION)
            main.flags["blockUI"] = False
            return
        fpLst = np.asarray(fpLst)
        
        ##### [begin] get data -----
        if pa["h2proc"] > 0:
            # length of data list to get
            dL2get = int(pa["h2proc"] * 60 * 60 / pa["dPtIntvSec"])

        data = {pa["aType"]:[]} # read data
        data4prob = {pa["aType"]:[]} # data for probability distributions 
            
        for fp, ns, ds in fpLst:
            print(f'Processing {path.basename(fp)} ..')
            arr = np.load(fp)

            _d = {}
            for dk in data.keys():
                data[dk].append([])
                data4prob[dk].append([])
                _d[dk] = arr[dk]
            
            _dt = arr["datetime"]

            iDT = None
            for ri in range(1, _dt.shape[0]):
            # go through each data index

                if iDT is None:
                    iDT = get_datetime(_dt[ri])
                    continue

                cDT = get_datetime(_dt[ri])
                elapsedHour = ((cDT-iDT).total_seconds())/60/60
                
                if elapsedHour >= pa["h2ignore"]: # hours-to-ignore passed
                    cVal = _d[dk][ri] # current data value
                    if cVal > -1:
                        if pa["h2proc"] <= 0: # no hour limit
                            for dk in data.keys():
                                data[dk][-1].append(cVal) # store data
                        else: # there's a hour limit for data 
                            if len(data[dk][-1]) >= dL2get:
                            # data length reached the limit 
                                break 
                            else:
                                for dk in data.keys():
                                    data[dk][-1].append(cVal) # store data
           
            # update max. value 
            _max = max(data[pa["aType"]][-1])
            if _max > pa["maxVal"]: pa["maxVal"] = _max 
          
        for dk in data.keys():
            _max = pa["maxVal"]
            for i in range(len(fpLst)):
                ### convert to array
                _data = np.asarray(data[dk][i])
                ### make data4prob
                for val in range(_max+1):
                    # count how many of this value occurred
                    cnt = np.count_nonzero(_data==val)
                    data4prob[dk][i].append(cnt)
        ##### [end] get data -----

        ##### [begin] draw prob. dist. graph & save the prob. as npy -----
        for dk in data.keys():
            dArr = np.asarray(data4prob[dk])
            summed = np.sum(dArr, 1) # sum each row (of each data file)
            prob = dArr / summed[:,None] # divide each row with summed values
                                         #   to get occurrence probability
            gKey = "dist2c"
            iqrLst = [] # to store the interquartile range
            rng70Lst = [] # to store the range that contains 70% of prob.
            ### display probability graphs of each file
            for ri in range(prob.shape[0]):
                cData = prob[ri] # data of a file 
                date = f'[{fpLst[ri][2]}]'
                iqrLst.append(iqr(cData)) # store the interquartile range
                # get percentile values and its indices
                pv, pi = getPercentileValsNPos(cData)
                #vl = [(pi[2], (0,1,0))] # vertical-line info of the median
                
                ### find peaks and give it as vertical line info
                # smooth data
                sd = savgol_filter(cData, window_length=21, polyorder=3)
                prom = pv[3]-pv[1] # use IRQ as prominence
                # find peaks
                peaks, _ = find_peaks(sd, prominence=prom)
                peakInfo = []
                for peak in peaks: peakInfo.append([sd[peak], peak])
                peakInfo = sorted(peakInfo)[-1] # get only the highest peak 
                vl = [(peakInfo[1], (0,1,0))] # peak index & color

                ### find range of 50% of data around the found peak
                idx = vl[0][0]
                rng = [idx, idx+1]
                pSum = cData[idx] 
                while pSum < 0.7:
                    rng[0] = max(0, rng[0]-1)
                    rng[1] = min(rng[1]+1, len(cData)-1)
                    pSum += cData[rng[0]]
                    pSum += cData[rng[1]]
                vl.append((rng[0], (0,0.5,0)))
                vl.append((rng[1], (0,0.5,0)))
                # store the range
                rng70Lst.append(rng[1]-rng[0])

                # draw prob. distribution graph
                g = drawProbDistributions({
                                            f'{gKey}': cData,
                                            f'{gKey}YLim': 0.035,
                                            f'vline': vl,
                                            f'xtraD': (sd, (1,0.5,0)),
                                            })

                cvFont = cv2.FONT_HERSHEY_PLAIN
                fThck = 1 
                _thP = int(g[gKey].shape[0] * 0.025)
                fScale, txtW, txtH, txtBl = getFontScale(cvFont,
                                                         thresholdPixels=_thP,
                                                         thick=fThck)
                cv2.putText(g[gKey], date, (5, txtH+txtBl), 
                    cvFont, fontScale=fScale, color=(0,0,0), thickness=fThck)
                main.callback(("finished", g[gKey]), flag="drawGraph")
            ### draw average (of all recorded individuals) probability
            rGraph = drawProbDistributions({
                                            f'{gKey}': np.mean(prob, 0),
                                            f'{gKey}YLim': 0.025 
                                            })
            # display the result graph
            main.callback(("finished", rGraph[gKey]), flag="drawGraph")

            plt.close("all")

            print("IQRs:", iqrLst)
            print("Range of 70% of data:", rng70Lst)

        ##### [end] draw prob. dist. graph & save the prob. as npy -----

    #---------------------------------------------------------------------------

    def getParamFromML(self):
        """ Get parameter values from the 'ml' panel of the main 
        
        Args:
            None
         
        Returns:
            pa (dict): parameter values 
        """ 
        if DEBUG: MyLogger.info(str(locals()))

        _debug = False 

        pa = {} # parameters
        for w in self.mainFrame.mlWid:
            key = w.GetName().split("_")[0]
            val = widgetValue(w)
            nVal = str2num(val) # convert to a number, if applicable
            if nVal == None: pa[key] = val
            else: pa[key] = nVal
        return pa

    #---------------------------------------------------------------------------
     
    def n1610_getFNItems(self, fLst, pa):
        """ (For n1610)
        Get items from filename matching with the given parameters
        
        Args:
            fLst (list): file list
            pa (dist): parameters to check
         
        Returns:
            dateStr (list): list of date-string
            fpLst (list): list of the matched file-path
        """
        if DEBUG: MyLogger.info(str(locals()))

        dateStr = []
        fpLst = []
        for fp in fLst:
            _fn = path.basename(fp).rstrip(".npy").split("_")
            if len(_fn) < 5: continue
            nS, dS, anS1, intvS, anS2 = _fn[:5]
            n = int(nS.lstrip("n"))
            _intv = int(intvS.lstrip("intv"))
            
            if pa["aType"] == "actInact":
                if anS1 != "intensity": continue
                if anS2.lower() not in ["intensitymboutsec",
                                        "intensityinactivitysec"]:
                    continue
            else:
                if pa["aType"] != anS1: continue
                if pa["aType"] == "proxMCluster" and nS == "n1": continue
                if pa["aType"] == "saMVec" and nS != "n1": continue
            if pa["dPtIntvSec"] != _intv: continue
            
            fpLst.append([fp, f'n{n:02d}', dS])
            if dS not in dateStr: dateStr.append(dS)
        return dateStr, fpLst

    #---------------------------------------------------------------------------

#===============================================================================



