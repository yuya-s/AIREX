# -*- coding: utf-8 -*-

# Directory path
import sys
sys.path.append("/home") 
# from python library
import argparse
import pandas as pd

# from my library
from source.exp import AAAI
from source.exp import train1test1
from source.exp import train1test19
from source.exp import train5test1_adain
from source.exp import train5test1_proposal
from source.exp import train19test1_adain
from source.exp import train19test1_proposal
from source.exp import train19test1_fnn
from source.exp import knn19

# main code
if __name__ == "__main__":

    '''
    :parameters
        0. PATH_ROOT_SRC
        1. MODE
        2. TRIAL
        3. EPOCHs
        4. BATCH_SIZE
        5. LEARNING_RATE
        6. WEIGHT_DECAY
        7. ALPHA
        8. GAMMA
        9. ETA
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH_ROOT_SRC", type=str, default="src/main.py")
    parser.add_argument("--EXP", help="experiment mode", type=str, default="AAAI")
    parser.add_argument("--TRIAL", help="the number of trial for the automatic parameters tuning", type=int, default=1)
    parser.add_argument("--EPOCHs", help="epoch size", type=int, default=200)
    parser.add_argument("--BATCH_SIZE", help="batch size", type=int, default=32)
    parser.add_argument("--LEARNING_RATE", help="learning rate", type=float, default=0.0005)
    parser.add_argument("--WEIGHT_DECAY", help="L2 normalization", type=float, default=0.0)
    parser.add_argument("--ALPHA", help="inference loss", type=float, default=0.5)
    parser.add_argument("--GAMMA", help="adversary loss", type=float, default=1.0)
    parser.add_argument("--ETA", help="normalization", type=float, default=1.0)
    args = parser.parse_args()

    '''
    4 cities
    '''
    CITIEs4 = ["BeiJing", "TianJin", "ShenZhen", "GuangZhou"]

    '''
    20 cities
    '''
    CITIEs20 = list()
    for city in list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"]):
        with open("database/station/station_"+city+".csv", "r") as infile:
            infile = infile.readlines()[1:] # 1行目を無視
            if len(infile) >= 5:
                CITIEs20.append(city)
    CITIEs20.remove("JiNan")
    CITIEs20.remove("HeYuan")
    CITIEs20.remove("JieYang")
    CITIEs20.remove("ShaoGuan")
    CITIEs20.remove("DaTong")
    CITIEs20.remove("DeZhou")
    CITIEs20.remove("BinZhou")
    CITIEs20.remove("DongYing")
    CITIEs20.remove("ChenZhou")

    '''
    EXP
    '''

    if args.EXP == "AAAI":
        TARGET = "BeiJing"
        AAAI(TARGET, args)
        exit()

 
    if args.EXP == "Train1":
        for SOURCE in CITIEs4:
            TARGETs = CITIEs20.copy()
            TARGETs.remove(SOURCE)
            train1test19(SOURCE, TARGETs, args)
        exit()

    if args.EXP == "Test1":
        for TARGET in CITIEs4:
            SOURCEs = CITIEs20.copy()
            SOURCEs.remove(TARGET)
            train1test1(SOURCEs, TARGET, args)
        exit()

    if args.EXP == "Test5_adain":
        for TARGET in CITIEs4:
            train5test1_adain(TARGET, args)
        exit()

    if args.EXP == "Test19_adain":
        for TARGET in CITIEs4:
            train19test1_adain(TARGET, args)
        exit()

    if args.EXP == "Test19_proposal":
        for TARGET in CITIEs4:
            train19test1_proposal(TARGET, args)
        exit()

    if args.EXP == "Test19_fnn":
        for TARGET in CITIEs4:
            train19test1_fnn(TARGET, args)
        exit()

    if args.EXP == "Test19_knn":
        for TARGET in CITIEs4:
            knn19(TARGET)
        exit()

