function replaceAll(string, search, replace) {
  return string.split(search).join(replace);
}

function arrayRemove(arr, value) { 
    return arr.filter(function(ele){ return ele != value; });
}

function rot_pt(pt, ct, deg) {
// function to rotate (clockwise) point;pt, around center point;ct
// * y-coordinate follows computer screen coordinate system,
//where 0 is the top row and the row index increases as it comes down
    let r = -deg * (Math.PI/180);
    let tx = pt[0]-ct[0];
    let ty = pt[1]-ct[1];
    let x = (tx * Math.cos(r) + ty * Math.sin(r)) + ct[0];
    let y = (-tx * Math.sin(r) + ty * Math.cos(r)) + ct[1];
    return [x, y];
}

jQuery.ajax({ // read CSV file
    url: "data/ActiveViruses.csv",
    type: 'get',
    dataType: 'text',
    success: function(data) {
        let lines = data.split('\n');
        let items = lines[0].split(',');
        let colTitle = [];
        for (let i = 0; i < items.length; i++) {
        // set column titles
            if (items[i].replace(/\s+/g,"") == "") { continue; }
            colTitle.push(items[i]);
        }
        let aSpNames = [] // ant species names
        let numPopulations = 3; // number of populations in each ant species
        for(let ci = 2; ci < colTitle.length; ci++){
        // from column-index 2, ant species columns start
            let names = colTitle[ci].split(" ");
            let aSpName = names[0] + " " + names[1];
            if (aSpNames.includes(aSpName) == false) {
                aSpNames.push(aSpName); // store ant species names
            }
        }
        let csvData = [];
        let numVP = 0; // number of virus presences
        let vMP = {}; // viruses show multiple presences in ant species
        // [begin] parse CSV data -----
        for(let li = 1; li < lines.length; li++){
            let items = lines[li].split(',');
            if (items.length < colTitle.length) { continue; }
            let lineData = {};
            let vMP_indentLevel = 0;
            let vMP_presenceCnt = 0;
            let vMP_antSpecies = [];
            let vMP_degArr = [];
            for(let ci = 0; ci < colTitle.length; ci++){
                let colVal = items[ci].replace(/\s+/g,""); // remove blanks
                if (colVal == "1") {
                    numVP += 1; // store total number of virus presences
                    vMP_presenceCnt += 1; // count presence
                    // store ant species, in which this virus occurred
                    let names = colTitle[ci].split(" ");
                    let aSpName = names[0] + " " + names[1];
                    if (vMP_antSpecies.includes(aSpName) == false) {
                        vMP_antSpecies.push(aSpName);
                    }
                }
                // store data of this column
                lineData[colTitle[ci]] = colVal;
            }
            csvData.push(lineData); // store data of this line (= virus) 
            if (vMP_presenceCnt > 1) {
                // store data of virus, showing multiple presences
                vSp = lineData[colTitle[0]]; // virus species
                vMP[vSp] = {indentLevel: vMP_indentLevel,
                            presenceCnt: vMP_presenceCnt,
                            antSpecies: vMP_antSpecies,
                            degArr: vMP_degArr,
                            vi: li-1};
            }
        }
        // [end] parse CSV data -----
        indentLevel = []; // indentation level for each ant species
        for (let si=0; si<aSpNames.length; si++) { // ant species
            aSp = aSpNames[si];
            indentLevel.push(0); // indentation level for this ant species
            for (const vSp of Object.keys(vMP)) {
                // skip virus occurred across multiple ant species
                if (vMP[vSp]["antSpecies"].length > 1) continue;
                // skip is this virus did not occur in this ant species
                if (vMP[vSp]["antSpecies"].includes(aSp) == false) continue; 
                indentLevel[si] += 1;
                // set indentatil level
                vMP[vSp]["indentLevel"] = indentLevel[si];
            }
        }
        // starting indentation level for virus across multiple ant species
        indentLevel = Math.max(...indentLevel) + 1;
        for (const vSp of Object.keys(vMP)) {
            if (vMP[vSp]["antSpecies"].length > 1) {
                indentLevel += 1;
                // set indentatil level
                vMP[vSp]["indentLevel"] = indentLevel;
            }
        }
        // [begin] set parameters for graph -----
        let svgH  = window.innerHeight * 0.9
        let svgW  = svgH * (16/9) 
        //let gSVG = d3.select("body").append("svg")
        let gSVG = d3.select("#graph_virusInAnts");
        gSVG.attr("width", svgW).attr("height", svgH);
        let spCol = ["rgb(95,186,27)", 
                     "rgb(0,80,203)",
                     "rgb(255,225,30)"]; // color for ant species
        let pCol = [
                    ["rgb(175,200,175)",
                     "rgb(120,200,120)",
                     "rgb(0,200,0)"],
                    ["rgb(175,175,200)",
                     "rgb(120,120,200)",
                     "rgb(0,0,200)"],
                    ["rgb(200,200,170)",
                     "rgb(200,200,130)",
                     "rgb(200,200,0)"],
                  ]; // color for each population of each ant species
        let cCol = {
            "Bunyavirales": "rgb(255,127,255)",
            "Mononegavirales": "rgb(0,0,0)",
            "Narnaviridae": "rgb(127,0,255)",
            "Nodaviridae": "rgb(255,127,127)",
            "Permutotetraviridae": "rgb(127,0,0)",
            "Picornavirales": "rgb(255,0,0)",
            "Totiviridae": "rgb(255,127,0)",
            "Unclassified": "rgb(127,127,127)",
        }; // color each virus classification
        let ct = [svgW*0.37, svgH/2]; // center of pie graph
        // degree for each virus presence dot
        let vpDeg = 360.0 / numVP;
        let radIC = svgH * 0.39; // radius of inner circle, where virus 
                                // presence circles are drawn
        let radOC = svgH * 0.47; // radius where ant species arcs are drawn
        // radius of virus presence dot 
        let radVP = (2*Math.PI*radIC) / numVP * 0.2; 
        let vMPInd = radVP * 2.0 // indentation for virus, showing 
                                 // multiple presences
        let legPosX = svgW * 0.7 // position-x where graph legend starts 
        // [end] set parameters for graph -----
        let selectedEntry = []; // clicked virus or its classification
        // [begin] make data for drawing graph elements -----
        let cntDrawnVP = 0;
        let vpd = []; // virus presence dot data for drawing 
        let sad = []; // species arc data for drawing 
        let pad = []; // population arc data for drawing
        let tLeg = {}; // temporary data for drawing legend
        for (let si=0; si<aSpNames.length; si++) { // ant species
            let spBeginDeg = -1;
            for (let pi=0; pi<numPopulations; pi++) { // ant population
                ci = 2 + si*numPopulations + pi // column index for
                    // virus presence in the population of the ant
                    // +2: first two columns are for virus spceis and class.
                let pBeginDeg = -1; 
                for(let vi = 0; vi< csvData.length; vi++) { // virus
                    cT = colTitle[ci]
                    if (parseInt(csvData[vi][cT]) == 0) {
                        continue; // no presence, skip this column
                    }
                    deg = cntDrawnVP * vpDeg;
                    deg = deg + 180; // starts from the left side
                    if (spBeginDeg == -1) { spBeginDeg = deg; }
                    if (pBeginDeg == -1) { pBeginDeg = deg; }
                    cntDrawnVP += 1;
                    let vSp = csvData[vi]['Virus species'];
                    let posX = ct[0] + radIC;
                    if (vSp in vMP) {
                    // this virus has multiple presences
                        // give indentation
                        posX = ct[0] + radIC - vMPInd*vMP[vSp]["indentLevel"];
                        // store the degree for drawing arc (+90 for d3.arc)
                        vMP[vSp]["degArr"].push(deg + 90);
                    }
                    let pt = rot_pt([posX, ct[1]], ct, deg);
                    let _vpr = radVP
                    if (vSp in vMP && vMP[vSp].antSpecies.length > 1) {
                    // increas radius of this virus presence dot
                    // , if it occurs across multiple ant species
                        _vpr = radVP * 1.25;
                    }
                    let vCl = csvData[vi]['Classification']; 
                    let txtPt = [svgW * 0.9, 50+(vi+1)*svgH*0.02];
                    // store data for drawing virus presence dot
                    vpd.push({
                        x: pt[0], 
                        y: pt[1], 
                        r: _vpr, 
                        aSi: si,
                        aPi: pi, 
                        vSp: vSp,
                        vCl: vCl, 
                        col: cCol[vCl],
                    });
                    // make virus classification array in legend data
                    if (!(vCl in tLeg)) { tLeg[vCl] = []; }
                    // add virus into the classification array
                    if (tLeg[vCl].includes(vSp) == false) {
                        tLeg[vCl].push(vSp);
                    }
                }
                beginDeg = pBeginDeg + 90 - (vpDeg/2);
                // +90: change degree for drawing d3.arc 
                let endDeg = deg + 90 + (vpDeg/2);
                let txtDeg = pBeginDeg + (deg - pBeginDeg) / 2;
                let txtPt = rot_pt([ct[0]+radIC*1.05, ct[1]], ct, txtDeg);
                let txtAnchor = "start";
                if (txtPt[0] < ct[0]) { txtAnchor = "end" }
                // store data for drawing arc to represent ant population arc 
                pad.push({
                        "bDeg": beginDeg, 
                        "eDeg": endDeg, 
                        "iRad": radIC*0.35,
                        "oRad": radIC,
                        "col": pCol[si][pi],
                        "text": "Pop." + (pi+1).toString(),
                        "txtPt": txtPt,
                        "txtAnchor": txtAnchor,
                        "fontWt": "bold", 
                        "fontFamily": "sans-serif", 
                        "fontCol": pCol[si][pi],
                        "fontSz": 12,
                }); 
            }
            beginDeg = spBeginDeg + 90 - (vpDeg/2);
            let endDeg = deg + 90 + (vpDeg/2); 
            let txtDeg = spBeginDeg + (deg - spBeginDeg) / 2
            let txtPt = rot_pt([ct[0]+radOC*1.02, ct[1]], ct, txtDeg);
            let txtAnchor = "start";
            if (txtPt[0] < ct[0]) { txtAnchor = "end" }
            // store data for drawing arc to represent ant species arc 
            sad.push({
                    "bDeg": beginDeg, 
                    "eDeg": endDeg, 
                    "iRad": radOC*0.99, 
                    "oRad": radOC, 
                    "col": spCol[si],
                    "text": aSpNames[si],
                    "txtPt": txtPt,
                    "txtAnchor": txtAnchor,
                    "fontWt": "bold", 
                    "fontFamily": "Arial",
                    "fontCol": "black", 
                    "fontSz": 14,
            });
        }
        // make data for drawing arc, representing multi-presence virus
        let mad = [];
        for (const vSp of Object.keys(vMP)) {
            let data = vMP[vSp]
            let vCl = csvData[data['vi']]['Classification']
            let add_thick = 0;
            if (vMP[vSp].antSpecies.length > 1) add_thick = 1;
            mad.push({
                    "bDeg": Math.min(...data['degArr']),
                    "eDeg": Math.max(...data['degArr']),
                    "iRad": radIC - vMPInd*data["indentLevel"] - add_thick,
                    "oRad": radIC - vMPInd*data["indentLevel"] + 1 + add_thick,
                    "col": cCol[vCl],
            });
        }
        // sort & make data for drawing legend
        let leg_class = []; // virus classifications in legend
        let leg_virus = []; // virus in legend
        let posY = 20;
        Object.keys(tLeg).sort().forEach(function(key) { // key: virus class.
            leg_class.push({
                text: key,
                txtPt: [legPosX, posY],
                txtAnchor: "start",
                fontWt: "normal", 
                fontFamily: "sans-serif", 
                fontCol: cCol[key],
                fontSz: 12,
            });
            posY += svgH*0.02;
            tLeg[key].sort();
            for (let vi=0; vi<tLeg[key].length; vi++) {
                leg_virus.push({
                    text: tLeg[key][vi],
                    txtPt: [legPosX+svgW*0.1, posY],
                    txtAnchor: "start",
                    fontWt: "normal", 
                    fontFamily: "sans-serif", 
                    fontCol: cCol[key],
                    fontSz: 12,
                });
                posY += svgH*0.02;
            }
        });
        // [end] make data for drawing graph elements -----
        
        // [begin] draw arcs -----
        var arc = d3.arc()
                    .innerRadius(function(d){return d.iRad;})
                    .outerRadius(function(d){return d.oRad;})
                    .startAngle(function(d){
                                        return d.bDeg*(Math.PI/180);
                                        })
                    .endAngle(function(d){return d.eDeg*(Math.PI/180);})
        let arcData = pad.concat(sad).concat(mad);
        var pbg = gSVG.selectAll("path")
            .data(arcData)
            .enter()
            .append("path")
        var pbgAttr = pbg
            .attr("d", arc)
            .attr("fill", function(d){return d.col;})
            .attr("transform", "translate("+ct[0]+","+ct[1]+")")
        // [end] draw arcs ----- 
        
        // [begin] draw virus presence dots -----
        gSVG.selectAll("circle")
            .data(vpd)
            .enter()
            .append("circle")
            .attr("id", function(d) {
                return "vDot_"+d.vSp+"_"+d.aSi.toString()+"_"+d.aPi.toString()
                        }
            )
            .attr("cx", function(d) {return d.x;})
            .attr("cy", function(d) {return d.y;})
            .attr("r", function(d) {return d.r;})
            .attr("fill", function(d) {return d.col;})
            .on("mouseover", onMouseOverVPD)
            .on("mouseout", onMouseOutVPD);
        // [begin] draw virus presence dots -----

        function drawTxts(d, idTag) { // function to draw texts in svg
            for (let i=0; i<d.length; i++) {
                gSVG.append("text")
                    .attr("id", idTag + "_" + d[i].text)
                    .text(d[i].text)
                    .attr("x", d[i].txtPt[0]) 
                    .attr("y", d[i].txtPt[1])
                    .attr("text-anchor", d[i].txtAnchor)
                    .attr("font-weight", d[i].fontWt)
                    .attr("font-family", d[i].fontFamily)
                    .attr("font-size", d[i].fontSz)
                    .attr("fill", d[i].fontCol)
                    .on("click", onMouseClick);
            }
        }
        // draw texts (on arcs of ant species and populations, 
        //   & in graph lengend for virus classification and virus name)
        drawTxts(pad.concat(sad), "arcTxt");
        drawTxts(leg_class, "legCl");
        drawTxts(leg_virus, "legV");
        
        function onMouseOverVPD(d, i) {
        // mouse-over event on virus presence dot
            let cObj = d3.select(this) // clicked object
            selectVirusEntry(cObj, d, i);
        }
        
        function onMouseOutVPD(d, i) {
        // mouse-out event on virus presence dot
            let cObj = d3.select(this) // clicked object
            selectVirusEntry(cObj, d, i); 
        }

        function onMouseClick(d, i) {
        // mouse-click event on virus presence dot
            let cObj = d3.select(this) // clicked object
            selectVirusEntry(cObj, d, i);
        }

        function selectVirusEntry(cObj, d, i) {
        // process when user clicked virus or virus classification
            let objId = cObj.attr("id")
            let idItems = objId.split('_'); 
            let eType = idItems[0]; // entry type
            let vSp = idItems[1]; // virus species
            if (eType == "legV" || eType == "vDot") eType = "virus";
            else if (eType == "legCl") eType = "vClassification";
            let eName = eType + "_" + vSp; // entry name
            
            if (selectedEntry.includes(eName) == true) {
            // clicked entry is already in selected entries array
                if (eType == "virus") {
                    emphLegTxt(false, "legV_"+vSp);
                    // de-emphasize the corresponding virus presence dots
                    emphVPDots(false, vSp);
                    // remove selected entry
                    selectedEntry = arrayRemove(selectedEntry, eName); 
                }
                else if (eType == "vClassification") {
                // classification was clicked 
                    emphLegTxt(false, "legCl_"+vSp); // de-emphasize legend text
                    // remove selected entry
                    selectedEntry = arrayRemove(selectedEntry, eName); 
                    let vSpArr = tLeg[vSp];
                    for (let vi=0; vi<vSpArr.length; vi++) {
                    // go through viruses in the classification
                        let vSp = vSpArr[vi];
                        let vObjId = "legV_" + vSp;
                        emphLegTxt(false, vObjId); // de-emphasize
                        // de-emphasize the corresponding virus presence dots
                        emphVPDots(false, vSp);
                        // remove the entry
                        selectedEntry = arrayRemove(selectedEntry, 
                                                    "virus_"+vSp);
                    }
                }            }
            else {
            // clicked entry is a new entry
                if (eType == "virus") {
                    emphLegTxt(true, "legV_"+vSp); // emphasize legend text
                    // de-emphasize the corresponding virus presence dots
                    emphVPDots(true, vSp);
                    selectedEntry.push(eName); // add selected entry
                }
                else if (eType == "vClassification") {
                // classification was clicked 
                    emphLegTxt(true , "legCl_"+vSp); // emphasize legend text
                    selectedEntry.push(eName); // add selected entry
                    let vSpArr = tLeg[vSp];
                    for (let vi=0; vi<vSpArr.length; vi++) {
                    // go though viruses in the classification
                        let vSp = vSpArr[vi];
                        let vObjId = "legV_" + vSp;
                        emphLegTxt(true, vObjId); // emphasize
                        // emphasize the corresponding virus presence dots
                        emphVPDots(true, vSp);
                        selectedEntry.push("virus_"+vSp); // add the entry
                    }
                }
            }
        }

        function emphLegTxt(flagEmph, objId) {
        // emphasize/de-emphasize text in graph legend
            let legV = d3.select("#" + objId);
            if (flagEmph) {
            // emphasize text in graph legend
                legV.attr("font-weight", "bold");
                legV.attr("text-decoration", "underline");
            }
            else {
            // de-emphasize text in graph legend
                legV.attr("font-weight", "normal");
                legV.attr("text-decoration", "none");
            }
        }

        function emphVPDots(flagEmph, vSp) {
        // emphasize/de-emphasize virus-presence-dots
        // with its radius, connecting line and virus species text
            if (flagEmph) {
            // emphasize virus-presence-dots
                // get points for drawing connecting line
                let pts = [];
                let strokeWidth = 1;
                let lineCol = "";
                for (let vi=0; vi<vpd.length; vi++) {
                    if (vpd[vi].vSp == vSp) {
                        if (pts.length > 0) {
                            // insert an intermediate point to draw step-like
                            // line inward to the center of the pie graph
                            let x1 = pts[pts.length-1].x;
                            let y1 = pts[pts.length-1].y;
                            let x2 = vpd[vi].x;
                            let y2 = vpd[vi].y;
                            if (Math.abs(x1-ct[0])<Math.abs(x2-ct[0])) _x=x1;
                            else _x = x2;
                            if (Math.abs(y1-ct[1])<Math.abs(y2-ct[1])) _y=y1;
                            else _y = y2;
                            pts.push({"x":_x, "y":_y});
                        }
                        pts.push({"x":vpd[vi].x, "y":vpd[vi].y});
                        lineCol = vpd[vi].col;
                        let vObjId = "vDot_" + vpd[vi].vSp;
                        vObjId += "_" + vpd[vi].aSi.toString();
                        vObjId += "_" + vpd[vi].aPi.toString();
                        let vpDot = d3.select("#"+vObjId); 
                        // increase radius of virus presence dot
                        vpDot.attr("r", radVP*1.5); 
                    }
                }
                // draw virus-presence-dot connecting line
                gSVG.append("path")
                    .datum(pts)
                    .attr("class", "line") 
                    .attr("d", d3.line()
                                .curve(d3.curveLinear)
                                //.curve(d3.curveCardinal)
                                .x(function(d) {return d.x;})
                                .y(function(d) {return d.y;})
                        )
                    .attr("id", "vPConnLine_"+vSp)
                    .style("fill", "none")
                    .style("stroke", lineCol)
                    .style("stroke-width", strokeWidth)
                
                // write virus species text
                let txtAnchor = "start";
                if (pts[0].x > ct[0]) { txtAnchor = "end" }
                gSVG.append("text")
                    .attr("id", "vpTxt_"+vSp)
                    .attr("x", pts[0].x) 
                    .attr("y", pts[0].y-5)
                    .attr("font-size", radVP*3)
                    .attr("font-weight", "bold")
                    .attr("fill", "rgb(50,50,50)")
                    .attr("text-anchor", txtAnchor)
                    .text(vSp);
            }
            else {
            // de-emphasize virus-presence-dots
                // determine virus presence radius
                _vpr = radVP;
                if (vSp in vMP && vMP[vSp].antSpecies.length > 1) {
                    _vpr = radVP * 1.25;
                }
                // make previously enlarged VP-dots to its normal size 
                for (let vpdi=0; vpdi<vpd.length; vpdi++) {
                    if (vpd[vpdi].vSp == vSp) {
                        let vObjId = "vDot_" + vpd[vpdi].vSp;
                        vObjId += "_" + vpd[vpdi].aSi.toString();
                        vObjId += "_" + vpd[vpdi].aPi.toString();
                        let vpDot = d3.select("#"+vObjId); 
                        vpDot.attr("r", _vpr); 
                    }
                }
                d3.select("#vpTxt_"+vSp).remove(); // remove added virus text
                d3.select("#vPConnLine_"+vSp).remove(); // remove conn. line
            }
        }

        gSVG.on("click", function() {
        // add mouse-click event on overall svg 
            coords = d3.mouse(this);
            //console.log(coords);
        })

    },
    error: function(jqXHR, textStatus, errorThrow){
        console.log(textStatus);
    }
});
