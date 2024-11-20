###########################################################################
sPurpose=   "Interpolate Missing M/Z values of a polyline in ArcMap console or Tool"
sAuthor=    "Piash Chowdhury"
sCreated=   "09.17.2020"
sCopyright= "Piash Chowdhury"
sLicense=   "not licensed; for internal use only"
# Versions:
sVersion=   "1.0 09.17.2020 created (PC)"

#########################################################################################
# Tool assumes:
# b) records in table sorted from upstream down 
# c) selected features maintain direction, i.e. upstream line END point equal downstream line START point
# d) missing z value for vertex is 0

import arcpy, traceback, os, sys
from arcpy import env
env.outputZFlag = "Enabled"
env.outputMFlag = "Enabled"

try:
    def showPyMessage():
        arcpy.AddMessage(str(time.ctime()) + " - " + message)
    mxd = arcpy.mapping.MapDocument("CURRENT")
    destLR = sys.argv[1]
    dDest=arcpy.Describe(destLR)
    destProj=dDest.spatialReference    
#  copy all lines to manageable list and merge
    g=arcpy.Geometry()
    geometryList=arcpy.CopyFeatures_management(destLR,g)
    nLines=len(geometryList)
    allPoints=geometryList[0].getPart(0)
    for i in range (1,nLines):
        part=geometryList[i].getPart(0)
        allPoints.extend(part)
    longLine=arcpy.Polyline(allPoints)

#  calc chainages
    dictFeatures,m = {},0
    for p in allPoints:
        dist=longLine.measureOnLine(p)
        dictFeatures[m]=(dist,p.Z)
        m+=1
#  find pairs
    first,nPoints,pairs=-1,len(allPoints),[]
    for i in range(nPoints):
        (d,z)=dictFeatures[i]
        if abs (z-0.0)>0.001: first=i; break
    if first==-1 or first==(nPoints-1):
        arcpy.AddMessage("Not enough z info")
        sys.exit(0)
    pairs.append((first,d,z))
    while True:
        second =-1
        for i in range(first+1,nPoints):
            (d,z)=dictFeatures[i]
            if abs (z-0.0)>0.001: second=i; break
        if second==-1:break
        first=second
        pairs.append((second,d,z))

# interpolate
    n=len(pairs)
    for j in range(1,n):
        first,d1,z1=pairs[j-1]
        second,d2,z2=pairs[j]
        dz=(z2-z1)/(d2-d1)
        for i in range(first+1,second):
            d,z=dictFeatures[i]
            z=z1 + dz*(d-d1)
            dictFeatures[i]=(d,z)
# update            
    with arcpy.da.UpdateCursor(destLR,"Shape@") as cursor:
        aKey,m=0,0
        pz=arcpy.Point()
        newL=arcpy.Array()
        for row in cursor:
            shp=geometryList[m];m+=1
            part=shp.getPart(0)
            n=len(part)
            for i in range(n):
                p=part[i]
                d,z=dictFeatures[aKey]
                pz.X,pz.Y,pz.Z=p.X,p.Y,z
                newL.add(pz)
                aKey+=1
            newLine=arcpy.Polyline(newL,destProj,True)
            newL.removeAll() 
            arcpy.AddMessage(newLine.length)
            cursor.updateRow((newLine,))
except:
    message = "\n*** PYTHON ERRORS *** "; showPyMessage()
    message = "Python Traceback Info: " + traceback.format_tb(sys.exc_info()[2])[0]; showPyMessage()
    message = "Python Error Info: " +  str(sys.exc_type)+ ": " + str(sys.exc_value) + "\n"; showPyMessage() 



