#-------------------------------------------------------------------------------
ssPurpose=   "Part 1/2 of a watershed delineation process from a DEM"
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

dem1=arcpy.GetParameterAsText(0)
border1=arcpy.GetParameterAsText(1)
outfolder=arcpy.GetParameterAsText(2)
j1=arcpy.GetParameterAsText(3)   #Area of interest For watershed

if not str(j1) != '':
    j1='AoI'

arcpy.env.workspace = outfolder
mxd = arcpy.mapping.MapDocument("CURRENT")
df = arcpy.mapping.ListDataFrames(mxd,"*")[0]

arcpy.AddMessage("Initialization Complete")

t1=j1+'_clip'
f1 = os.path.join(outfolder, t1)    

outClip=arcpy.Clip_management(dem1,"#",f1,border1,"0","NONE","NO_MAINTAIN_EXTENT")
arcpy.AddMessage("Step 1/10: Clip complete")

t2=j1+'_fill'
outFill=Fill(outClip)

outFill.save(os.path.join(outfolder, t2))
arcpy.AddMessage("Step 2/10: Fill complete")

t3=j1+'_flowdir'
outFlowDir=FlowDirection(outFill)

outFlowDir.save(os.path.join(outfolder, t3))
arcpy.AddMessage("Step 3/10: Flow Direction complete")

t4=j1+'_flowacc'
outFlowAcc=FlowAccumulation(outFlowDir)

outFlowAcc.save(os.path.join(outfolder, t4))
arcpy.AddMessage("Step 4/10: Flow Accumulation complete")

t5=j1+'_con'
outCon=Con(outFlowDir,outFlowAcc,outFlowAcc)

outCon.save(os.path.join(outfolder, t5))
arcpy.AddMessage("Step 5/10: Con complete")

t6=j1+'_strord'
outStreamOrder=StreamOrder(outCon,outFlowDir)

outStreamOrder.save(os.path.join(outfolder, t6))
arcpy.AddMessage("Step 6/10: Stream Order complete")

t7=j1+'_streams'
f7 = os.path.join(outfolder, t7) 
outStreamFeat=StreamToFeature(outStreamOrder,outFlowDir,f7,"SIMPLIFY")

newLayer7 = arcpy.mapping.Layer(t7) 
arcpy.mapping.AddLayer(df, newLayer7,"TOP")

def unique_values(table, field):
    with arcpy.da.SearchCursor(table, [field]) as cursor:
        return sorted({row[0] for row in cursor})


vals = len(unique_values(t7,"GRID_CODE"))
vals2=vals-5
if vals2 > 0:
    s="GRID_CODE>"+str(vals2)
else:
    s=""
        

for lyr in arcpy.mapping.ListLayers(mxd):
    if lyr.name == t7:
        lyr.definitionQuery=s


t8=j1+'_streamsLite'
arcpy.FeatureClassToFeatureClass_conversion(outStreamFeat, outfolder, t8, s)
newLayer8 = arcpy.mapping.Layer(t8) 
arcpy.mapping.AddLayer(df, newLayer8,"TOP")

arcpy.AddMessage("Step 7/10: Stream to Feature complete")		

pp=j1+"_pourPoints"
arcpy.CreateFeatureclass_management(outfolder, pp, "POINT","","","","31468")

arcpy.AddMessage("Step 8/10: Pour Points Feature Class created")

infc = os.path.join(outfolder, pp)
xy = arcpy.SpatialReference(31468)
arcpy.DefineProjection_management(infc, xy)

newLayer8 = arcpy.mapping.Layer(pp) 
arcpy.mapping.AddLayer(df, newLayer8,"TOP")

arcpy.RefreshActiveView()






















