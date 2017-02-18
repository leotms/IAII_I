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

ordinals={
    "Lot Shape":{
       "Reg":3,
       "IR1":2,
       "IR2":1,
       "IR3":0
    },
    "Utilities":{
       "AllPub":3,    
       "NoSewr":2,   
       "NoSeWa":1,   
       "ELO":0  
    },
    "Land Slope":{
       "Gtl":2,  
       "Mod":1,  
       "Sev":0 
    },
    "MS Zoning":{
       "A":0,    
       "C (all)":1,   
       "FV":2,   
       "I":3,    
       "RH":4,   
       "RL":5,   
       "RP":6,    
       "RM":7   
    },
    "Street":{
       "Grvl":0,  
       "Pave":1 
    },
    "Alley":{
       "Grvl":1,
       "Pave":2,
       "NA":0
    },  
    "Land Contour":{
       "Lvl":0,
       "Bnk":1,
       "HLS":2,  
       "Low":3  
    },
    "Lot Config":{
       "Inside":0,
       "Corner":1,
       "CulDSac":2,  
       "FR2":3, 
       "FR3":4
    },
    "Land Slope":{
       "Gtl":2,
       "Mod":1, 
       "Sev":0
    },
    "Neighborhood":{
       "Blmngtn":0,
       "Blueste":1,
       "BrDale":2,
       "BrkSide":3,
       "ClearCr":4,
       "CollgCr":5,
       "Crawfor":6,
       "Edwards":7,
       "Gilbert":8,
       "Greens":9, 
       "GrnHill":10,  
       "IDOTRR":11,   
       "Landmrk":12,  
       "MeadowV":13,  
       "Mitchel":14,  
       "NAmes":15,    
       "NoRidge":16,  
       "NPkVill":17,  
       "NridgHt":18,  
       "NWAmes":19,   
       "OldTown":20,
       "SWISU":21,    
       "Sawyer":22,  
       "SawyerW":23,  
       "Somerst":24,  
       "StoneBr":25,  
       "Timber":26, 
       "Veenker":27
    },
    "Condition 1":{ 
       "Artery":0, 
       "Feedr":1,    
       "Norm":2,  
       "RRNn":3, 
       "RRAn":4, 
       "PosN":5, 
       "PosA":6, 
       "RRNe":7, 
       "RRAe":8 
    },
    "Condition 2":{ 
       "Artery":0,   
       "Feedr":1,       
       "Norm":2,  
       "RRNn":3, 
       "RRAn":4, 
       "PosN":5,
       "PosA":6, 
       "RRNe":7, 
       "RRAe":8 
    },
    "Bldg Type":{
       "1Fam":0, 
       "2fmCon":1,
       "Duplex":2,
       "TwnhsE":3,
       "Twnhs":4   
    },
    "House Style":{
       "1Story":0,   
       "1.5Fin":1,   
       "1.5Unf":2,   
       "2Story":3,   
       "2.5Fin":4,   
       "2.5Unf":5,   
       "SFoyer":6,   
       "SLvl":7 
    },
    "Roof Style":{
       "Flat":0,
       "Gable":1,
       "Gambrel":2,
       "Hip":3,
       "Mansard":4,
       "Shed":5
    },
    "Roof Matl":{
       "ClyTile":0,  
       "CompShg":1, 
       "Membran":2, 
       "Metal":3,   
       "Roll":4, 
       "Tar&Grv":5,  
       "WdShake":6, 
       "WdShngl":7
    },
    "Exterior 1st":{
       "AsbShng":0,   
       "AsphShn":1,   
       "BrkComm":2,   
       "BrkFace":3,   
       "CBlock":4,    
       "CemntBd":5,   
       "HdBoard":6,   
       "ImStucc":7,   
       "MetalSd":8,   
       "Other":9,    
       "Plywood":10,  
       "PreCast":11,   
       "Stone":12,    
       "Stucco":13,   
       "VinylSd":14,   
       "Wd Sdng":15,   
       "WdShing":16   
    },
    "Exterior 2nd":{
       "AsbShng":0,   
       "AsphShn":1,   
       "Brk Cmn":2,   
       "BrkFace":3,   
       "CBlock":4,    
       "CmentBd":5,   
       "HdBoard":6,   
       "ImStucc":7,   
       "MetalSd":8,   
       "Other":9,    
       "Plywood":10,  
       "PreCast":11,  
       "Stone":12,    
       "Stucco":13,   
       "VinylSd":14,   
       "Wd Shng":15,   
       "WdShing":16,
       "Wd Sdng":17   
    },
    "Mas Vnr Type":{
       "BrkCmn":1,
       "BrkFace":2,
       "CBlock":3,
       "None":0,
       "Stone":4,
       "NA":5
    },
    "Exter Qual":{
       "Ex":4,
       "Gd":3,
       "TA":2,
       "Fa":1,
       "Po":0
    },
    "Exter Cond":{
       "Ex":4,
       "Gd":3,
       "TA":2,
       "Fa":1,
       "Po":0
    },
    "Foundation":{
       "BrkTil":5,
       "CBlock":4,
       "PConc":3, 
       "Slab":2,
       "Stone":1,
       "Wood":0
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
    for dt in col.values:
        if type(dt[0])==float:
            if math.isnan(dt[0]):
                dt[0]="NA" 
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


    data.to_csv('./data/Datos_Filtrados_Result.csv', index=False)
    print "done."
