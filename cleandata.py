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
