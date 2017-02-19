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
       "Ex":6,
       "Gd":5,
       "TA":4,
       "Fa":3,
       "Po":2,
       "NA":1
    },
    "Bsmt Cond":{
       "Ex":6,
       "Gd":5,
       "TA":4,
       "Fa":3,
       "Po":2,
       "NA":1
    },
    "Bsmt Exposure":{
       "Gd":5,
       "Av":4,
       "Mn":3,
       "No":2,
       "NA":1
    },
    "BsmtFin Type 1":{
       "GLQ":7,
       "ALQ":6,
       "BLQ":5,
       "Rec":4,
       "LwQ":3,
       "Unf":2,
       "NA":1
    },
    "BsmtFin Type 2":{
       "GLQ":7,
       "ALQ":6,
       "BLQ":5,
       "Rec":4,
       "LwQ":3,
       "Unf":2,
       "NA":1
    },
    "Heating":{
       "Floor":1,
       "GasA":2,
       "GasW":3,
       "Grav":4,
       "OthW":5,
       "Wall":6
    },
    "Heating QC":{
       "Ex":5,
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1
    },
    "Central Air":{
       "N":1,
       "Y":2
    },
    "Electrical":{
       "SBrkr":6,
       "FuseA":5,
       "FuseF":4,
       "FuseP":3,
       "Mix":2,
       "NA":1
    },
    "Kitchen Qual":{
       "Ex":5,
       "Gd":4,
       "TA":3,
       "Fa":2,
       "Po":1
    },
    "Functional":{
       "Typ":8,
       "Min1":7,
       "Min2":6,
       "Mod":5,
       "Maj1":4,
       "Maj2":3,
       "Sev":2,
       "Sal":1
    },
    "Fireplace Qu":{
       "Ex":6,
       "Gd":5,
       "TA":4,
       "Fa":3,
       "Po":2,
       "NA":1
    },
    "Garage Type":{
       "2Types":2,
       "Attchd":3,
       "Basment":4,
       "BuiltIn":5,
       "CarPort":6,
       "Detchd":7,
       "NA":1
    },
    "Garage Finish":{
       "Fin":4,
       "RFn":3,
       "Unf":2,
       "NA":1
    },
    "Garage Qual":{
       "Ex":6,
       "Gd":5,
       "TA":4,
       "Fa":3,
       "Po":2,
       "NA":1
    },
    "Garage Cond":{
       "Ex":6,
       "Gd":5,
       "TA":4,
       "Fa":3,
       "Po":2,
       "NA":1
    },
    "Paved Drive":{
       "Y":3,
       "P":2,
       "N":1
    },
    "Pool QC":{
       "Ex":5,
       "Gd":4,
       "TA":3,
       "Fa":2,
       "NA":1
    },
    "Fence":{
       "GdPrv":5,
       "MnPrv":4,
       "GdWo":3,
       "MnWw":2,
       "NA":1
    },
    "Misc Feature":{
       "Elev":2,
       "Gar2":3,
       "Othr":4,
       "Shed":5,
       "TenC":6,
       "NA":1
    },
    "Sale Type":{
       "WD ":1,
       "CWD":2,
       "VWD":3,
       "New":4,
       "COD":5,
       "Con":6,
       "ConLw":7,
       "ConLI":8,
       "ConLD":9,
       "Oth":10
    },
    "Sale Condition":{
       "Normal":1,
       "Abnorml":2,
       "AdjLand":3,
       "Alloca":4,
       "Family":5,
       "Partial":6
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
