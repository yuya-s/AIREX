# -*- coding: utf-8 -*-

# Directory path
import sys
sys.path.append("/home")

# from python library
import pandas as pd
import argparse

# from my library
from source.func import makeAAAI18
from source.func import makeTrain1
from source.func import makeTest1
from source.func import makeTest5
from source.func import makeTest19
from source.func import makeTest19_cityData
from source.func import makeTest5_cityData
from source.func import makeIdxData

if __name__ == "__main__":

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH_ROOT_SRC", type=str, default="src/makedataset.py")
    parser.add_argument("--DATASET", help="please specify dataset you want", type=str, default="AAAI")
    args = parser.parse_args()

    print("dataset [ {} ] is been made ... ".format(args.DATASET))

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
            infile = infile.readlines()[1:] 
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

    if args.DATASET == "AAAI":
        CITY = "BeiJing"
        makeAAAI18(CITY)
        exit()

    if args.DATASET == "Train1":
        makeTrain1(CITIEs20, CITIEs4)
        exit()

    if args.DATASET == "Test1":
        makeTest1(CITIEs20, CITIEs4)
        exit()

    if args.DATASET == "Test5":
        makeTest5(CITIEs4)
        makeTest5_cityData(CITIEs4)
        makeIdxData(5, 5, CITIEs4)
        exit()

    if args.DATASET == "Test19":
        makeTest19(CITIEs20, CITIEs4)
        makeTest19_cityData(CITIEs4)
        makeIdxData(19, 5, CITIEs4)
        exit()

    print("OK")
