'''
    File:        cleandata.py
    Authors:     Joel Rivas        #11-10866
                 Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
    Updated:     02/12/2017
'''

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient_descent import *
import pandas as pd
import math
import numpy as np

ordinals={
    "Lot Shape":{
       "Reg":4,
       "IR1":3,
       "IR2":2,
       "IR3":1
    },
    "Utilities":{
       "AllPub":4,    
       "NoSewr":3,   
       "NoSeWa":2,   
       "ELO":1
    },
    "Land Slope":{
       "Gtl":3,  
       "Mod":2,  
       "Sev":1 
    },
    "MS Zoning":{
       "A":1,    
       "C (all)":2,   
       "FV":3,   
       "I":4,    
       "RH":5,   
       "RL":6,   
       "RP":7,    
       "RM":8   
    },
    "Street":{
       "Grvl":1,  
       "Pave":2 
    },
    "Alley":{
       "Grvl":2,
       "Pave":3,
       "NA":1
    },  
    "Land Contour":{
       "Lvl":1,
       "Bnk":2,
       "HLS":3,  
       "Low":4  
    },
    "Lot Config":{
       "Inside":1,
       "Corner":2,
       "CulDSac":3,  
       "FR2":4,
       "FR3":5
    },
    "Land Slope":{
       "Gtl":3,
       "Mod":2, 
       "Sev":1
    },
    "Neighborhood":{
       "Blmngtn":1,
       "Blueste":2,
       "BrDale":3,
       "BrkSide":4,
       "ClearCr":5,
       "CollgCr":6,
       "Crawfor":7,
       "Edwards":8,
       "Gilbert":9,
       "Greens":10, 
       "GrnHill":11,  
       "IDOTRR":12,   
       "Landmrk":13,  
       "MeadowV":14,  
       "Mitchel":15,  
       "NAmes":16,    
       "NoRidge":17,  
       "NPkVill":18,  
       "NridgHt":19,  
       "NWAmes":20,   
       "OldTown":21,
       "SWISU":22,    
       "Sawyer":23,  
       "SawyerW":24,  
       "Somerst":25,  
       "StoneBr":26,  
       "Timber":27, 
       "Veenker":28
    },
    "Condition 1":{ 
       "Artery":1 
       "Feedr":2    
       "Norm":3, 
       "RRNn":4, 
       "RRAn":5, 
       "PosN":6, 
       "PosA":7, 
       "RRNe":8, 
       "RRAe":9,
    },
    "Condition 2":{ 
       "Artery":1,   
       "Feedr":2,       
       "Norm":3,  
       "RRNn":4, 
       "RRAn":5, 
       "PosN":6,
       "PosA":7, 
       "RRNe":8, 
       "RRAe":9 
    },
    "Bldg Type":{
       "1Fam":1, 
       "2fmCon":2,
       "Duplex":3,
       "TwnhsE":4,
       "Twnhs":5   
    },
    "House Style":{
       "1Story":1,   
       "1.5Fin":2,   
       "1.5Unf":3,   
       "2Story":4,   
       "2.5Fin":5,   
       "2.5Unf":6,   
       "SFoyer":7,   
       "SLvl":8 
    },
    "Roof Style":{
       "Flat":1,
       "Gable":2,
       "Gambrel":3,
       "Hip":4,
       "Mansard":5,
       "Shed":6
    },
    "Roof Matl":{
       "ClyTile":1,  
       "CompShg":2, 
       "Membran":3, 
       "Metal":4,   
       "Roll":5, 
       "Tar&Grv":6,  
       "WdShake":7, 
       "WdShngl":8
    },
    "Exterior 1st":{
       "AsbShng":1,   
       "AsphShn":2,   
       "BrkComm":3,   
       "BrkFace":4,   
       "CBlock":5,    
       "CemntBd":6,   
       "HdBoard":7,   
       "ImStucc":8,   
       "MetalSd":9,   
       "Other":10,    
       "Plywood":11,  
       "PreCast":12,   
       "Stone":13,    
       "Stucco":14,   
       "VinylSd":15,   
       "Wd Sdng":16,   
       "WdShing":17   
    },
    "Exterior 2nd":{
       "AsbShng":1,   
       "AsphShn":2,   
       "Brk Cmn":3,   
       "BrkFace":4,   
       "CBlock":5,    
       "CmentBd":6,   
       "HdBoard":7,   
       "ImStucc":8,   
       "MetalSd":9,   
       "Other":10,    
       "Plywood":11,  
       "PreCast":12,  
       "Stone":13,    
       "Stucco":14,   
       "VinylSd":15,   
       "Wd Shng":16,   
       "WdShing":17,
       "Wd Sdng":18   
    },
    "Mas Vnr Type":{
       "BrkCmn":2,
       "BrkFace":3
       "CBlock":4
       "None":1
       "Stone":5
       "NA":6
    },
    "Exter Qual":{
       "Ex":5,
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1
    },
    "Exter Cond":{
       "Ex":5,
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1
    },
    "Foundation":{
       "BrkTil":6,
       "CBlock":5,
       "PConc":4, 
       "Slab":3,
       "Stone":2,
       "Wood":1
    },
    "Bsmt Qual":{
       "Ex":5, 
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1,
       "NA":0
    },
    "Bsmt Cond":{
       "Ex":5, 
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1,
       "NA":0
    },
    "Bsmt Exposure":{
       "Gd":4,
       "Av":3,
       "Mn":2,
       "No":1,
       "NA":0
    },
    "BsmtFin Type 1":{
       "GLQ":6,
       "ALQ":5,   
       "BLQ":4,   
       "Rec":3,
       "LwQ":2, 
       "Unf":1,  
       "NA":0   
    },
    "BsmtFin Type 2":{
       "GLQ":6,
       "ALQ":5,   
       "BLQ":4,   
       "Rec":3,
       "LwQ":2, 
       "Unf":1,  
       "NA":0   
    },
    "Heating":{
       "Floor":0,
       "GasA":1,
       "GasW":2, 
       "Grav":3, 
       "OthW":4, 
       "Wall":5
    },
    "Heating QC":{
       "Ex":4,  
       "Gd":3,   
       "TA":2,   
       "Fa":1,   
       "Po":0
    },
    "Central Air":{
       "N":0,
       "Y":1
    },
    "Electrical":{
       "SBrkr":5, 
       "FuseA":4,    
       "FuseF":3,    
       "FuseP":2,  
       "Mix":1,
       "NA":0
    },
    "Kitchen Qual":{
       "Ex":4,
       "Gd":3,
       "TA":2,
       "Fa":1,
       "Po":0
    },
    "Functional":{
       "Typ":7, 
       "Min1":6,
       "Min2":5, 
       "Mod":4, 
       "Maj1":3,  
       "Maj2":2,  
       "Sev":1,  
       "Sal":0  
    },
    "Fireplace Qu":{
       "Ex":5, 
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1,
       "NA":0    
    },
    "Garage Type":{
       "2Types":1,
       "Attchd":2,
       "Basment":3,
       "BuiltIn":4,
       "CarPort":5,
       "Detchd":6,
       "NA":0
    },
    "Garage Finish":{
       "Fin":3,
       "RFn":2, 
       "Unf":1,
       "NA":0
    },
    "Garage Qual":{
       "Ex":5, 
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1,
       "NA":0     
    },
    "Garage Cond":{
       "Ex":5, 
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1,
       "NA":0     
    },
    "Paved Drive":{
       "Y":2,
       "P":1,
       "N":0
    },
    "Pool QC":{
       "Ex":4, 
       "Gd":3,
       "TA":2,
       "Fa":1,
       "NA":0 
    },
    "Fence":{
       "GdPrv":4,
       "MnPrv":3,
       "GdWo":2, 
       "MnWw":1,
       "NA":0
    },
    "Misc Feature":{
       "Elev":1,    
       "Gar2":2, 
       "Othr":3, 
       "Shed":4,
       "TenC":5,
       "NA":0
    },
    "Sale Type":{
       "WD ":0,  
       "CWD":1, 
       "VWD":2,  
       "New":3,  
       "COD":4,  
       "Con":5,  
       "ConLw":6,    
       "ConLI":7,  
       "ConLD":8, 
       "Oth":9
    },
    "Sale Condition":{
       "Normal":0,
       "Abnorml":1,
       "AdjLand":2,
       "Alloca":3,
       "Family":4,
       "Partial":5
    }

}

def encode(col):
    nameColumn=list(col)[0];
    print nameColumn
    for dt in col.values:
        print type(dt[0])
        si=False
        if type(dt[0])==float:
            if math.isnan(dt[0]):
                dt[0]="NA" 
        if isinstance(dt[0], np.float64):
            print dt[0]
            if math.isnan(dt[0]):
                dt[0]=0
                si=True
        if not(si):
            dt[0]=ordinals[nameColumn][dt[0]]
    
    return col

if __name__ =="__main__":
    '''
        Clean Data.
    '''

    print "Cleaning..."

    filepath   = "./data/Datos_Filtrados.csv"
    data = pd.read_csv(filepath)


    for col in data.columns:
        if not(data.dtypes[col]!="object"):
            data[[col]]=encode(data[[col]])
        else:
            if col=="Pool QC":   
                data[[col]]=encode(data[[col]])

    data.to_csv('./data/Datos_Filtrados_Result.csv', index=False)
    print "done."
