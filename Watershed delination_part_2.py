#-------------------------------------------------------------------------------
ssPurpose=   "Part 2/2 of a watershed delineation process from a DEMin ArcMap Toolbox"
sAuthor=    "Piash Chowdhury"
sCreated=   "01.02.2021"
sCopyright= "Piash Chowdhury"
sLicense=   "not licensed; for internal use only"
# Versions:
sVersion=   "2.0 01.02.2021 created (PC)"

#-------------------------------------------------------------------------------

arcpy.AddMessage(sAuthor)
arcpy.AddMessage(sCreated)
arcpy.AddMessage(sLicense)
arcpy.AddMessage(sVersion)

import sys, string, os, arcgisscripting, arcpy
from arcpy import env
from arcpy.sa import *

#arcpy.env.parallelProcessingFactor = "100%"

snapdist=arcpy.GetParameterAsText(0)
outfolder=arcpy.GetParameterAsText(1)
outPP=arcpy.GetParameterAsText(2)
outFlowAcc=arcpy.GetParameterAsText(3)
outFlowDir=arcpy.GetParameterAsText(4)
j1=arcpy.GetParameterAsText(5)

if not str(j1) != '':
    j1='AoI'

arcpy.env.workspace = outfolder
mxd2 = arcpy.mapping.MapDocument("CURRENT")
df2 = arcpy.mapping.ListDataFrames(mxd2,"*")[0]

t10=j1+'_snapPP'
outSnap=SnapPourPoint(outPP,outFlowAcc,snapdist)

outSnap.save(os.path.join(outfolder, t10))

newLayer9 = arcpy.mapping.Layer(t10) 
arcpy.mapping.AddLayer(df2, newLayer9,"TOP")
    
arcpy.AddMessage("Step 9/10: Snap Pour Point complete")

t11=j1+'_watershed'
outWatershed=Watershed(outFlowDir,outSnap)

outWatershed.save(os.path.join(outfolder, t11))

newLayer10 = arcpy.mapping.Layer(t11) 
arcpy.mapping.AddLayer(df2, newLayer10,"TOP")

arcpy.AddMessage("Step 10/10: Watershed complete")

arcpy.RefreshActiveView()






