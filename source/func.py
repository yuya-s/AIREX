# -*- coding: utf-8 -*-

# Directory path
import sys
sys.path.append("/home")

# from python library
import pickle
import _pickle
import torch
import random
import math
import bz2
import time
import numpy as np
import pandas as pd
from torch import nn
from geomloss import SamplesLoss
from torch import optim
from sklearn.metrics import mean_squared_error

# from my library
from source.model import ADAIN
from source.model import AIREX
from source.model import FNN
from source.utility import MyDataset_ADAIN
from source.utility import MyDataset_AIREX
from source.utility import MyDataset_MMD
from source.utility import MyDataset_FNN
from source.utility import get_dist_angle
from source.utility import calc_correct
from source.utility import EarlyStopping
from source.utility import get_activation
from source.utility import get_optimizer


def makeAAAI18(CITY):

    dataset = "AAAI18"
    S = list(pd.read_csv("database/station/station_{}.csv".format(CITY), dtype=object)["sid"])
    trainNum = math.floor(len(S)*0.67)
    testNum = len(S) - trainNum

    print("stationNum: {}".format(str(len(S))))
    print("trainNum: {}".format(str(trainNum)))
    print("testNum: {}".format(str(testNum)))
    print("--------------")

    for loop in range(1, 4):

        print("* Shuffle Loop: {}".format(str(loop)))
        station = S.copy()
        random.shuffle(station)
        station_train = station[:trainNum]
        station_test = station[trainNum:]

        print("* train set")
        savePath = "dataset/{}/train_{}{}".format(dataset, CITY, str(loop))
        makeTrainData(savePath, station_train)

        print("* test set")
        savePath = "dataset/{}/test_{}{}".format(dataset, CITY, str(loop))
        makeTestData_sampled(savePath, station_test, station_train)

def makeTrain1(CITIEs20, CITIEs4):

    print("trainNum: 5")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        dataset = "{}Train1".format(TARGET)
        print("* Target: {}".format(TARGET))

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))
            station_target = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
            random.shuffle(station_target)
            station_train = station_target[:5]

            print("* train set")
            savePath = "dataset/{}/train_{}{}".format(dataset, TARGET, str(loop))
            makeTrainData(savePath, station_train)

            for SOURCE in CITIEs20:

                print("* Source: {}".format(SOURCE))

                if SOURCE == TARGET:
                    station_test = station_target[5:10]
                else:
                    station_test = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                    random.shuffle(station_test)
                    station_test = station_test[:5]

                print("* test set")
                savePath = "dataset/{}/test_{}{}".format(dataset, SOURCE, str(loop))
                makeTestData_sampled(savePath, station_test, station_train)

def makeTest1(CITIEs20, CITIEs4):

    print("trainNum: 5")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        SOURCEs = CITIEs20.copy()
        SOURCEs.remove(TARGET)

        dataset = "{}Test1".format(TARGET)
        print("* Target: {}".format(TARGET))

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))
            station_test = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
            random.shuffle(station_test)
            station_test = station_test[:5]

            for SOURCE in SOURCEs:

                print("* Source: {}".format(SOURCE))

                station_train = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                random.shuffle(station_train)
                station_train = station_train[:5]

                print("* train set")
                savePath = "dataset/{}/train_{}{}".format(dataset, SOURCE, str(loop))
                makeTrainData(savePath, station_train)

                print("* test set")
                savePath = "dataset/{}/test_{}{}".format(dataset, SOURCE, str(loop))
                makeTestData_sampled(savePath, station_test, station_train)

def makeTest5(CITIEs4):

    print("trainNum: 25")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

        print("*---TARGET: {}".format(TARGET))

        if TARGET == "BeiJing":
            SOURCEs = ["LangFang", "TianJin", "BaoDing", "TangShan", "ZhangJiaKou"]
        elif TARGET == "TianJin":
            SOURCEs = ["LangFang", "CangZhou", "TangShan", "BeiJing", "BaoDing"]
        elif TARGET == "ShenZhen":
            SOURCEs = ["XiangGang", "DongGuan", "HuiZhou", "JiangMen", "GuangZhou"]
        else:
            SOURCEs = ["FoShan", "DongGuan", "JiangMen", "ShenZhen", "HuiZhou"]

        dataset = "{}Test5".format(TARGET)

        station_test = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
        random.shuffle(station_test)
        station_test = station_test[:5] 

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))

            station_train = list()
            for SOURCE in SOURCEs:
                station_source = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                random.shuffle(station_source)
                station_train += station_source[:5] 
            random.shuffle(station_train)

            print("* train set")
            savePath = "dataset/{}/train{}".format(dataset, str(loop))
            makeTrainData(savePath, station_train)

            print("* test set")
            savePath = "dataset/{}/test{}".format(dataset, str(loop))
            makeTestData_sampled(savePath, station_test, station_train)

def makeTest5_cityData(CITIEs4):

    print("trainNum: 25")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

            print("*---TARGET: {}".format(TARGET))

            for loop in range(1, 4):
                print("* Shuffle Loop: {}".format(str(loop)))

                # train data
                dataPath_train = "dataset/{}Test5/train{}".format(TARGET, str(loop))
                savePath_train = "dataset/{}Test5_city/train{}".format(TARGET, str(loop))
                # test data
                dataPath_test = "dataset/{}Test5/test{}".format(TARGET, str(loop))
                savePath_test = "dataset/{}Test5_city/test{}".format(TARGET, str(loop))
                makeCityData(dataPath_train, savePath_train, dataPath_test, savePath_test)


def makeTest19(CITIEs20, CITIEs4):

    print("trainNum: 95")
    print("testNum: 5")
    print("--------------")

    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))

    for TARGET in CITIEs4:

        print("*---TARGET: {}".format(TARGET))

        SOURCEs = CITIEs20.copy()
        SOURCEs.remove(TARGET)
        dataset = "{}Test19".format(TARGET)

        while True:
            station_test = list(pd.read_csv("database/station/station_{}.csv".format(TARGET), dtype=object)["sid"])
            random.shuffle(station_test)
            station_test = station_test[:5]
            flag = 1
            for sid in station_test:
                if len(stationData[stationData["sid"] == sid]["lat"]) == 0:
                    print("station_test: Redo")
                    flag = 0
            if flag:
                break

        for loop in range(1, 4):

            print("* Shuffle Loop: {}".format(str(loop)))

            station_train = list()
            for SOURCE in SOURCEs:

                while True:
                    station_source = list(pd.read_csv("database/station/station_{}.csv".format(SOURCE), dtype=object)["sid"])
                    random.shuffle(station_source)
                    flag = 1
                    for sid in station_source[:5]:
                        if len(stationData[stationData["sid"] == sid]["lat"]) == 0:
                            print("station_train: Redo")
                            flag = 0
                    if flag:
                        station_train += station_source[:5]
                        break

            random.shuffle(station_train)

            print("* train set")
            savePath = "dataset/{}/train{}".format(dataset, str(loop))
            makeTrainData(savePath, station_train)

            print("* test set")
            savePath = "dataset/{}/test{}".format(dataset, str(loop))
            makeTestData_sampled(savePath, station_test, station_train)

def makeTest19_cityData(CITIEs4):

    print("trainNum: 95")
    print("testNum: 5")
    print("--------------")

    for TARGET in CITIEs4:

            print("*---TARGET: {}".format(TARGET))

            for loop in range(1, 4):
                print("* Shuffle Loop: {}".format(str(loop)))

                # train data
                dataPath_train = "dataset/{}Test19/train{}".format(TARGET, str(loop))
                savePath_train = "dataset/{}Test19_city/train{}".format(TARGET, str(loop))
                # test data
                dataPath_test = "dataset/{}Test19/test{}".format(TARGET, str(loop))
                savePath_test = "dataset/{}Test19_city/test{}".format(TARGET, str(loop))
                makeCityData(dataPath_train, savePath_train, dataPath_test, savePath_test)

def makeCityData(dataPath_train, savePath_train, dataPath_test, savePath_test):

    # raw data
    stationRaw = pd.read_csv("rawdata/zheng2015/station.csv", dtype=object)
    districtRaw = pd.read_csv("rawdata/zheng2015/district.csv", dtype=object)
    cityRaw = pd.read_csv("rawdata/zheng2015/city.csv", dtype=object)

    # dataset
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    # station train
    tmp = pickle.load(bz2.BZ2File("{}/train_000.pkl.bz2".format(dataPath_train), 'rb'))
    local, tmp = tmp[0][0], tmp[2][0]
    local = [k for k, v in staticData.items() if v == local][0]
    others = list()
    for static in tmp:
        static = static[:-2]
        others.append([k for k, v in staticData.items() if v == static][0])

    station_train = dict()
    for sid in [local] + others:
        did = list(stationRaw[stationRaw["station_id"] == sid]["district_id"])[0]
        cid = list(districtRaw[districtRaw["district_id"] == did]["city_id"])[0]
        if cid in station_train.keys():
            station_train[cid].append(sid)
        else:
            station_train[cid] = list()
            station_train[cid].append(sid)

    for k, v in station_train.items():

        if len(v) > 5:
            v = list(set(v))
            v = v[:5]
            station_train[k] = v

        if len(v) < 5:
            eng_name = list(cityRaw[cityRaw["city_id"] == k]["name_english"])[0]
            tmp = list(pd.read_csv("database/station/station_{}.csv".format(eng_name), dtype=object)["sid"])
            for removed in v:
                tmp.remove(removed)
            random.shuffle(tmp)
            station_train[k].append(tmp[0])

    station_train = [v for k, v in station_train.items()]

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    dataNum = 0
    source_location = list()
    for i in range(len(station_train)):

        # location of local city
        sid_local = station_train[i][0]
        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]
        lat_local = float(cityRaw[cityRaw["city_id"] == cid_local]["latitude"])
        lon_local = float(cityRaw[cityRaw["city_id"] == cid_local]["longitude"])
        source_location.append((lat_local, lon_local))

        # location of other cities
        others_city = list()
        for j in range(len(station_train)):

            if i == j:
                continue

            sid = station_train[j][0]
            did = list(stationRaw[stationRaw["station_id"] == sid]["district_id"])[0]
            cid = list(districtRaw[districtRaw["district_id"] == did]["city_id"])[0]
            lat = float(cityRaw[cityRaw["city_id"] == cid]["latitude"])
            lon = float(cityRaw[cityRaw["city_id"] == cid]["longitude"])
            result = get_dist_angle(lat_local, lon_local, lat, lon)
            others_city.append([result["distance"], result["azimuth1"]])

        others_city = np.array(others_city)
        print(others_city)
        minimum = others_city.min(axis=0, keepdims=True)
        maximum = others_city.max(axis=0, keepdims=True)
        others_city = (others_city - minimum) / (maximum - minimum)
        others_city = list(map(lambda x: list(x), others_city))

        for station_local in station_train[i]:

            # output
            out_local_static = list()
            out_local_seq = list()
            out_others_static = list()
            out_others_seq = list()
            out_others_city = list()
            out_target = list()

            '''
            calculate distance and angle of other stations from local stations
            '''
            # lat, lon of local station
            lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
            lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

            # distance and angle
            geoVect = list()
            for j in range(len(station_train)):

                if i == j:
                    continue

                for station_others_j in station_train[j]:
                    lat = float(stationData[stationData["sid"] == station_others_j]["lat"])
                    lon = float(stationData[stationData["sid"] == station_others_j]["lon"])
                    result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                    geoVect.append([result["distance"], result["azimuth1"]])

            # normalization others' location
            geoVect = np.array(geoVect)
            minimum = geoVect.min(axis=0, keepdims=True)
            maximum = geoVect.max(axis=0, keepdims=True)
            geoVect = (geoVect - minimum) / (maximum - minimum)
            geoVect = list(map(lambda x: list(x), geoVect))

            # add geoVect to static data
            others_static = list()
            idx = 0
            for j in range(len(station_train)):

                if i == j:
                    continue

                others_static_j = list()
                for station_others_j in station_train[j]:
                    others_static_j.append(staticData[station_others_j] + geoVect[idx])
                    idx += 1
                others_static.append(others_static_j)

            '''
            concut meteorological data with aqi data of seqData of others
            '''
            seqData_others = dict()
            for j in range(len(station_train)):

                if i == j:
                    continue

                for station_others_j in station_train[j]:
                    m = _pickle.loads(_pickle.dumps(meteorologyData[station_others_j], -1)) 
                    a = _pickle.loads(_pickle.dumps(aqiData[station_others_j], -1)) 
                    for k in range(len(m)):
                        for l in range(len(m[k])):
                            m[k][l] += a[k][l]
                    seqData_others[station_others_j] = m

            '''
            local data and target data
            '''
            local_static = staticData[station_local]
            local_seq = meteorologyData[station_local]
            target = targetData[station_local]
            dataNum = len(target)

            for t in range(dataNum):

                others_seq = list()

                for j in range(len(station_train)):

                    if i == j:
                        continue

                    others_seq_j = list()
                    for station_others_j in station_train[j]:
                        others_seq_j.append(seqData_others[station_others_j][t])
                    others_seq.append(others_seq_j)

                out_local_static.append(local_static)
                out_local_seq.append(local_seq[t])
                out_others_static.append(others_static)
                out_others_seq.append(others_seq)
                out_others_city.append(others_city)
                out_target.append(target[t])

            out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_others_city, out_target)

            cityCode = str(i).zfill(3)
            stationCode = str(station_train[i].index(station_local)).zfill(3)

            with bz2.BZ2File("{}/train_{}{}.pkl.bz2".format(savePath_train, cityCode, stationCode), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save train_{}{}.pkl.bz2".format(cityCode, stationCode))

            del out_local_seq, out_local_static, out_others_seq, out_others_static, out_others_city, out_target, out_set

    with open("{}/fileNum.pkl".format(savePath_train), "wb") as fp:
        pickle.dump({"station": len(station_train[0]), "city": len(station_train), "time": dataNum}, fp)

    '''
    test data
    '''

    dataNum = 0
    source_location = list()
    for i in range(len(station_train)):

        # location of local city
        sid_local = station_train[i][0]
        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]
        lat_local = float(cityRaw[cityRaw["city_id"] == cid_local]["latitude"])
        lon_local = float(cityRaw[cityRaw["city_id"] == cid_local]["longitude"])
        source_location.append((lat_local, lon_local))

    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath_test), "rb"))["test"]
    for i in range(testNum):

        # output
        out_others_static = list()
        out_others_seq = list()
        out_others_city = list()

        '''
        calculate distance and angle of other source cities from the target city
        '''

        tmp = pickle.load(bz2.BZ2File("{}/test_{}.pkl.bz2".format(dataPath_test, str(i).zfill(3)), 'rb'))
        out_local_static, out_local_seq, out_target, tmp = tmp[0], tmp[1], tmp[4], tmp[0][0]

        # location of local city
        sid_local = [k for k, v in staticData.items() if v == tmp][0]
        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]
        lat_local = float(cityRaw[cityRaw["city_id"] == cid_local]["latitude"])
        lon_local = float(cityRaw[cityRaw["city_id"] == cid_local]["longitude"])

        others_city = list()
        max_index = 0
        max_distance = 0
        for j in range(len(source_location)):
            lat, lon = source_location[j]
            result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
            others_city.append([result["distance"], result["azimuth1"]])
            if result["distance"] > max_distance:
                max_distance = result["distance"]
                max_index = j

        station_train_copy = station_train.copy()
        others_city.remove(others_city[max_index])
        station_train_copy.remove(station_train_copy[max_index])
        others_city = np.array(others_city)
        minimum = others_city.min(axis=0, keepdims=True)
        maximum = others_city.max(axis=0, keepdims=True)
        others_city = (others_city - minimum) / (maximum - minimum)
        others_city = list(map(lambda x: list(x), others_city))

        '''
        calculate distance and angle of other stations from local stations
        '''
        # lat, lon of local station
        lat_local = float(stationData[stationData["sid"] == sid_local]["lat"])
        lon_local = float(stationData[stationData["sid"] == sid_local]["lon"])

        # distance and angle
        geoVect = list()
        for j in range(len(station_train_copy)):
            for station_others_j in station_train_copy[j]:
                lat = float(stationData[stationData["sid"] == station_others_j]["lat"])
                lon = float(stationData[stationData["sid"] == station_others_j]["lon"])
                result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                geoVect.append([result["distance"], result["azimuth1"]])

        # normalization others' location
        geoVect = np.array(geoVect)
        minimum = geoVect.min(axis=0, keepdims=True)
        maximum = geoVect.max(axis=0, keepdims=True)
        geoVect = (geoVect - minimum) / (maximum - minimum)
        geoVect = list(map(lambda x: list(x), geoVect))

        # add geoVect to static data
        others_static = list()
        idx = 0
        for j in range(len(station_train_copy)):
            others_static_j = list()
            for station_others_j in station_train_copy[j]:
                others_static_j.append(staticData[station_others_j] + geoVect[idx])
                idx += 1
            others_static.append(others_static_j)

        '''
        concut meteorological data with aqi data of seqData of others
        '''
        seqData_others = dict()
        for j in range(len(station_train_copy)):
            for station_others_j in station_train_copy[j]:
                m = _pickle.loads(_pickle.dumps(meteorologyData[station_others_j], -1))
                a = _pickle.loads(_pickle.dumps(aqiData[station_others_j], -1))
                for k in range(len(m)):
                    for l in range(len(m[k])):
                        m[k][l] += a[k][l]
                seqData_others[station_others_j] = m

        '''
        output set
        '''
        dataNum = len(out_target)
        for t in range(dataNum):

            others_seq = list()
            for j in range(len(station_train_copy)):
                others_seq_j = list()
                for station_others_j in station_train_copy[j]:
                    others_seq_j.append(seqData_others[station_others_j][t])
                others_seq.append(others_seq_j)

            out_others_static.append(others_static)
            out_others_seq.append(others_seq)
            out_others_city.append(others_city)

        out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_others_city, out_target)

        with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath_test, str(i).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save test_{}.pkl.bz2".format(str(i).zfill(3)))

        del out_local_seq, out_local_static, out_others_seq, out_others_static, out_others_city, out_target, out_set

    with open("{}/fileNum.pkl".format(savePath_test), "wb") as fp:
       pickle.dump({"station": len(station_train[0]), "city": len(station_train), "time": dataNum}, fp)


def makeTrainData(savePath, station_train):

    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    trainNum = math.floor(len(station_train)*0.9)

    tdx, vdx = 0, 0
    for station_local in station_train:

        # output
        out_local_static = list()
        out_local_seq = list()
        out_others_static = list()
        out_others_seq = list()
        out_target = list()

        station_others = station_train.copy()
        station_others.remove(station_local)

        '''
        calculate distance and angle of other stations from local stations
        '''
        # lat, lon of local station
        lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
        lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

        # distance and angle
        distance = list()
        angle = list()
        for sid in station_others:
            lat = float(stationData[stationData["sid"] == sid]["lat"])
            lon = float(stationData[stationData["sid"] == sid]["lon"])
            result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
            distance.append(result["distance"])
            angle.append(result["azimuth1"])

        # normalization
        if len(distance) == 1:
            distance = [1.0]
            angle = [1.0]
        else:
            maximum = max(distance)
            minimum = min(distance)
            distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
            maximum = max(angle)
            minimum = min(angle)
            angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

        # add
        others_static = list()
        idx = 0
        for sid in station_others:
            others_static.append(staticData[sid] + [distance[idx], angle[idx]])
            idx += 1

        '''
        concut meteorological data with aqi data of seqData of others
        '''
        seqData_others = dict()
        for sid in station_others:
            m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1)) 
            a = _pickle.loads(_pickle.dumps(aqiData[sid], -1))
            for i in range(len(m)):
                for j in range(len(m[i])):
                    m[i][j] += a[i][j]
            seqData_others[sid] = m

        '''
        local data and target data
        '''
        local_static = staticData[station_local]
        local_seq = meteorologyData[station_local]
        target = targetData[station_local]

        '''
        make dataset
        '''
        for t in range(len(target)):

            others_seq = list()
            for sid in station_others:
                others_seq.append(seqData_others[sid][t])

            out_local_static.append(local_static)
            out_local_seq.append(local_seq[t])
            out_others_static.append(others_static)
            out_others_seq.append(others_seq)
            out_target.append(target[t])

        out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

        if tdx > trainNum-1:
            with bz2.BZ2File("{}/valid_{}.pkl.bz2".format(savePath, str(vdx).zfill(3)), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save valid_{}.pkl.bz2".format(str(vdx).zfill(3)))
            vdx += 1
        else:
            with bz2.BZ2File("{}/train_{}.pkl.bz2".format(savePath, str(tdx).zfill(3)), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save train_{}.pkl.bz2".format(str(tdx).zfill(3)))
            tdx += 1

        del out_local_seq, out_local_static, out_others_seq, out_others_static, out_set

    with open("{}/fileNum.pkl".format(savePath), "wb") as fp:
        pickle.dump({"train": tdx, "valid": vdx}, fp)

def makeTestData(savePath, station_test, station_train):
    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    tdx = 0
    for station_local in station_test:

        for station_removed in station_train:

            # output
            out_local_static = list()
            out_local_seq = list()
            out_others_static = list()
            out_others_seq = list()
            out_target = list()

            station_others = station_train.copy()
            station_others.remove(station_removed)
            '''
            calculate distance and angle of other stations from local stations
            '''
            # lat, lon of local station
            lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
            lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

            # distance and angle
            distance = list()
            angle = list()
            for sid in station_others:
                lat = float(stationData[stationData["sid"] == sid]["lat"])
                lon = float(stationData[stationData["sid"] == sid]["lon"])
                result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
                distance.append(result["distance"])
                angle.append(result["azimuth1"])

            # normalization
            if len(distance) == 1:
                distance = [1.0]
                angle = [1.0]
            else:
                maximum, minimum = max(distance), min(distance)
                distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
                maximum, minimum = max(angle), min(angle)
                angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

            # add
            others_static = list()
            idx = 0
            for sid in station_others:
                others_static.append(staticData[sid] + [distance[idx], angle[idx]])
                idx += 1

            '''
            concut meteorological data with aqi data of seqData of others
            '''
            seqData_others = dict()
            for sid in station_others:
                m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1))
                a = _pickle.loads(_pickle.dumps(aqiData[sid], -1)) 
                for i in range(len(m)):
                    for j in range(len(m[i])):
                        m[i][j] += a[i][j]
                seqData_others[sid] = m

            '''
            local data and target data
            '''
            local_static = staticData[station_local]
            local_seq = meteorologyData[station_local]
            target = targetData[station_local]

            '''
            make datatmp
            '''
            for t in range(len(target)):

                others_seq = list()
                for sid in station_others:
                    others_seq.append(seqData_others[sid][t])

                out_local_static.append(local_static)
                out_local_seq.append(local_seq[t])
                out_others_static.append(others_static)
                out_others_seq.append(others_seq)
                out_target.append(target[t])

            out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

            with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath, str(tdx).zfill(3)), 'wb', compresslevel=9) as fp:
                fp.write(pickle.dumps(out_set))
                print("* save test_{}.pkl.bz2".format(str(tdx).zfill(3)))
            tdx += 1

            del out_local_seq, out_local_static, out_others_seq, out_others_static, out_set

    with open("{}/fileNum.pkl".format(savePath), "wb") as fp:
        pickle.dump({"test": tdx}, fp)

def makeTestData_sampled(savePath, station_test, station_train):
    '''
    :param station_train): a list of station ids
    :return: featureData, labelData
    '''

    # raw data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))
    meteorologyData = pickle.load(open("datatmp/meteorologyData.pkl", "rb"))
    aqiData = pickle.load(open("datatmp/aqiData.pkl", "rb"))
    targetData = pickle.load(open("datatmp/labelData.pkl", "rb"))

    '''
    featureData_t = (local_static, local_seq, others_static, others_seq)_t
    labelData_t = target_t
    '''

    tdx = 0
    for station_local in station_test:

        # output
        out_local_static = list()
        out_local_seq = list()
        out_others_static = list()
        out_others_seq = list()
        out_target = list()

        station_others = station_train.copy()
        station_remove = station_others[random.randint(0, len(station_others)-1)]
        station_others.remove(station_remove)
        '''
        calculate distance and angle of other stations from local stations
        '''
        # lat, lon of local station
        lat_local = float(stationData[stationData["sid"] == station_local]["lat"])
        lon_local = float(stationData[stationData["sid"] == station_local]["lon"])

        # distance and angle
        distance = list()
        angle = list()
        for sid in station_others:
            lat = float(stationData[stationData["sid"] == sid]["lat"])
            lon = float(stationData[stationData["sid"] == sid]["lon"])
            result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
            distance.append(result["distance"])
            angle.append(result["azimuth1"])

        # normalization
        if len(distance) == 1:
            distance = [1.0]
            angle = [1.0]
        else:
            maximum, minimum = max(distance), min(distance)
            distance = list(map(lambda x: (x - minimum) / (maximum - minimum), distance))
            maximum, minimum = max(angle), min(angle)
            angle = list(map(lambda x: (x - minimum) / (maximum - minimum), angle))

        # add
        others_static = list()
        idx = 0
        for sid in station_others:
            others_static.append(staticData[sid] + [distance[idx], angle[idx]])
            idx += 1

        '''
        concut meteorological data with aqi data of seqData of others
        '''
        seqData_others = dict()
        for sid in station_others:
            m = _pickle.loads(_pickle.dumps(meteorologyData[sid], -1))
            a = _pickle.loads(_pickle.dumps(aqiData[sid], -1)) 
            for i in range(len(m)):
                for j in range(len(m[i])):
                    m[i][j] += a[i][j]
            seqData_others[sid] = m

        '''
        local data and target data
        '''
        local_static = staticData[station_local]
        local_seq = meteorologyData[station_local]
        target = targetData[station_local]

        '''
        make datatmp
        '''
        for t in range(len(target)):

            others_seq = list()
            for sid in station_others:
                others_seq.append(seqData_others[sid][t])

            out_local_static.append(local_static)
            out_local_seq.append(local_seq[t])
            out_others_static.append(others_static)
            out_others_seq.append(others_seq)
            out_target.append(target[t])

        out_set = (out_local_static, out_local_seq, out_others_static, out_others_seq, out_target)

        with bz2.BZ2File("{}/test_{}.pkl.bz2".format(savePath, str(tdx).zfill(3)), 'wb', compresslevel=9) as fp:
            fp.write(pickle.dumps(out_set))
            print("* save test_{}.pkl.bz2".format(str(tdx).zfill(3)))
        tdx += 1

        del out_local_seq, out_local_static, out_others_seq, out_others_static, out_set

    with open("{}/fileNum.pkl".format(savePath), "wb") as fp:
        pickle.dump({"test": tdx}, fp)

def makeIdxData(cityNum, stationNum, CITIEs4):

    import bz2

    # raw data
    stationRaw = pd.read_csv("rawdata/zheng2015/station.csv", dtype=object)
    districtRaw = pd.read_csv("rawdata/zheng2015/district.csv", dtype=object)
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))

    for city_path in CITIEs4:

        for loop_path in ["1", "2", "3"]:

            # train data
            for city_id in range(cityNum):
                for station_id in range(stationNum):

                    others_static_new = list()
                    others_seq_new = list()
                    others_city_new = list()

                    dataset_path = "dataset/{}Test{}_city/train{}/train_{}{}.pkl.bz2".format(city_path, cityNum, loop_path, str(city_id).zfill(3), str(station_id).zfill(3))
                    dataset = pickle.load(bz2.BZ2File(dataset_path, "rb"))
                    local_static, local_seq, others_static, others_seq, others_city, target = dataset

                    print("load train_{}{}.pkl.bz2".format(str(city_id).zfill(3), str(station_id).zfill(3)))

                    for t in range(len(local_static)):

                        cid_list = list()
                        others_static_sorted = list()
                        others_seq_sorted = list()
                        others_city_sorted = list()

                        static = local_static[t]
                        sid_local = [k for k, v in staticData.items() if v == static][0]
                        did_local = list(stationRaw[stationRaw["station_id"] == sid_local]["district_id"])[0]
                        cid_local = list(districtRaw[districtRaw["district_id"] == did_local]["city_id"])[0]

                        for i in range(len(others_static[t])):
                            static = others_static[t][i][0]
                            sid_others = [k for k, v in staticData.items() if v == static[:-2]][0]
                            did_others = list(stationRaw[stationRaw["station_id"] == sid_others]["district_id"])[0]
                            cid_others = list(districtRaw[districtRaw["district_id"] == did_others]["city_id"])[0]
                            cid_list.append(cid_others)

                            if len(cid_list) == 1:
                                others_static_sorted.append(others_static[t][i])
                                others_seq_sorted.append(others_seq[t][i])
                                others_city_sorted.append(others_city[t][i])
                            else:
                                cid_list.sort()
                                idx = cid_list.index(cid_others)
                                others_static_sorted.insert(idx, others_static[t][i])
                                others_seq_sorted.insert(idx, others_seq[t][i])
                                others_city_sorted.insert(idx, others_city[t][i])

                        others_static_new.append(others_static_sorted)
                        others_seq_new.append(others_seq_sorted)
                        others_city_new.append(others_city_sorted)

                    cid_list.append(cid_local)
                    cid_list.sort()
                    cid_local_index = cid_list.index(cid_local)
                    print(cid_local, end=":\t")
                    print(cid_local_index)
                    print(cid_list)

                    print(np.array(local_static).shape)
                    print(np.array(local_seq).shape)
                    print(np.array(others_static_new).shape)
                    print(np.array(others_seq_new).shape)
                    print(np.array(others_city_new).shape)
                    print(np.array(target).shape)

                    out_set = (local_static, local_seq, others_static_new, others_seq_new, others_city_new, target)

                    save_path = "dataset/{}Test{}_city/train{}/train_{}{}_idx.pkl".format(city_path, cityNum, loop_path, str(city_id).zfill(3), str(station_id).zfill(3))
                    with open(save_path, "wb") as fp:
                        pickle.dump(cid_local_index, fp)

                    save_path = "dataset/{}Test{}_city/train{}/train_{}{}.pkl.bz2".format(city_path, cityNum, loop_path, str(city_id).zfill(3), str(station_id).zfill(3))
                    with bz2.BZ2File(save_path, 'wb', compresslevel=1) as fp:
                        fp.write(pickle.dumps(out_set))
                        print("save train_{}{}.pkl.bz2".format(str(city_id).zfill(3), str(station_id).zfill(3)))
                    print("--------------------")

            cid_trains = cid_list.copy()
            print("Finish train data")
            print(cid_trains)
            print("--------------------")

            # test data
            for station_id in range(stationNum):

                others_static_new = list()
                others_seq_new = list()
                others_city_new = list()

                dataset_path = "dataset/{}Test{}_city/test{}/test_{}.pkl.bz2".format(city_path, cityNum, loop_path, str(station_id).zfill(3))
                dataset = pickle.load(bz2.BZ2File(dataset_path, "rb"))
                local_static, local_seq, others_static, others_seq, others_city, target = dataset

                print("load test_{}.pkl.bz2".format(str(station_id).zfill(3)))

                for t in range(len(local_static)):

                    cid_list = list()
                    others_static_sorted = list()
                    others_seq_sorted = list()
                    others_city_sorted = list()

                    for i in range(len(others_static[t])):
                        static = others_static[t][i][0]
                        sid_others = [k for k, v in staticData.items() if v == static[:-2]][0]
                        did_others = list(stationRaw[stationRaw["station_id"] == sid_others]["district_id"])[0]
                        cid_others = list(districtRaw[districtRaw["district_id"] == did_others]["city_id"])[0]
                        cid_list.append(cid_others)

                        if len(cid_list) == 1:
                            others_static_sorted.append(others_static[t][i])
                            others_seq_sorted.append(others_seq[t][i])
                            others_city_sorted.append(others_city[t][i])
                        else:
                            cid_list.sort()
                            idx = cid_list.index(cid_others)
                            others_static_sorted.insert(idx, others_static[t][i])
                            others_seq_sorted.insert(idx, others_seq[t][i])
                            others_city_sorted.insert(idx, others_city[t][i])

                    others_static_new.append(others_static_sorted)
                    others_seq_new.append(others_seq_sorted)
                    others_city_new.append(others_city_sorted)

                cid_removed = list(set(cid_trains).difference(set(cid_list)))[0]
                cid_list.append(cid_removed)
                cid_list.sort()
                cid_local_index = cid_list.index(cid_removed)
                print(cid_removed, end=":\t")
                print(cid_local_index)
                print(cid_list)

                print(np.array(local_static).shape)
                print(np.array(local_seq).shape)
                print(np.array(others_static_new).shape)
                print(np.array(others_seq_new).shape)
                print(np.array(others_city_new).shape)
                print(np.array(target).shape)

                out_set = (local_static, local_seq, others_static_new, others_seq_new, others_city_new, target)

                save_path = "dataset/{}Test{}_city/test{}/test_{}_idx.pkl".format(city_path, cityNum, loop_path, str(station_id).zfill(3))
                with open(save_path, "wb") as fp:
                    pickle.dump(cid_local_index, fp)

                save_path = "dataset/{}Test{}_city/test{}/test_{}.pkl.bz2".format(city_path, cityNum, loop_path, str(station_id).zfill(3))
                with bz2.BZ2File(save_path, 'wb', compresslevel=1) as fp:
                    fp.write(pickle.dumps(out_set))
                    print("save test_{}.pkl.bz2".format(str(station_id).zfill(3)))
                print("--------------------")



def objective_ADAIN(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    hp = pickle.load(open("tmp/hyperParameters.pkl", "rb"))
    batch_size = hp.BATCH_SIZE
    epochs = hp.EPOCHs
    lr = hp.LEARNING_RATE
    wd = hp.WEIGHT_DECAY

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))

    # model
    model = ADAIN(inputDim_local_static=inputDim["local_static"],
                  inputDim_local_seq=inputDim["local_seq"],
                  inputDim_others_static=inputDim["others_static"],
                  inputDim_others_seq=inputDim["others_seq"])

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # evaluation function
    criterion = nn.MSELoss()

    # initialize the early stopping object
    patience = epochs
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # dataset path
    dataPath = pickle.load(open("tmp/trainPath.pkl", "rb"))

    # the number which the train/validation dataset was divivded into
    trainNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["train"]
    #validNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["valid"]

    # start training
    for step in range(int(epochs)):

        # train
        for idx in range(trainNum):

            epoch_loss = list()

            selector = "/train_{}.pkl.bz2".format(str(idx).zfill(3))
            trainData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
            for batch_i in torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True):

                print("\t|- train batch loss: ", end="")

                # initialize graduation
                optimizer.zero_grad()

                # batch data
                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)
                batch_others_static = batch_others_static.to(device)
                batch_others_seq = batch_others_seq.to(device)
                batch_target = batch_target.to(device)

                # predict
                pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)

                # calculate loss, back-propagate loss, and step optimizer
                loss = criterion(pred, batch_target)
                loss.backward()
                optimizer.step()

                # print a batch loss as RMSE
                batch_loss = np.sqrt(loss.item())
                print("%.10f" % (batch_loss))

                # append batch loss to the list to calculate epoch loss
                epoch_loss.append(batch_loss)

            epoch_loss = np.average(epoch_loss)
            print("\t\t|- epoch %d loss: %.10f" % (step + 1, epoch_loss))

        # # validate
        # print("\t\t|- validation : ", end="")
        # rmse = list()
        # accuracy = list()
        # model.eval()
        # for idx in range(validNum):
        #     selector = "/valid_{}.pkl.bz2".format(str(idx).zfill(3))
        #     validData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
        #     rmse_i, accuracy_i = validate_ADAIN(model, validData)
        #     rmse.append(rmse_i)
        #     accuracy.append(accuracy_i)
        # # model.train()
        #
        # # calculate validation loss
        # rmse = np.average(rmse)
        # accuracy = np.average(accuracy)
        # log = {'epoch': step, 'validation rmse': rmse, 'validation accuracy': accuracy}
        # logs.append(log)
        # print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

            # evaluate
            model.eval()
            rmse, accuracy = midium_evaluate_ADAIN(model)
            model.train()
            log = {'epoch': step, 'train_rmse': epoch_loss, 'test_rmse': rmse}
            logs.append(log)
            print("\t\t|- rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

            # early stopping
            early_stopping(rmse, model)
            if early_stopping.early_stop:
                print("\t\tEarly stopping")
                break

    # load the last checkpoint after early stopping
    model.load_state_dict(torch.load("tmp/checkpoint.pt"))
    rmse = early_stopping.val_loss_min

    # save model
    trial_num = trial.number
    with open("tmp/{}_model.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/{}_log.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        pickle.dump(logs, pl)

    return rmse

def validate_ADAIN(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 200

    for batch_i in torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=False):

        batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

        # to GPU
        batch_local_static = batch_local_static.to(device)
        batch_local_seq = batch_local_seq.to(device)
        batch_others_static = batch_others_static.to(device)
        batch_others_seq = batch_others_seq.to(device)

        # predict
        pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
        pred = pred.to("cpu")

        # evaluate
        pred = list(map(lambda x: x[0], pred.data.numpy()))
        batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
        result += pred
        result_label += batch_target

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def midium_evaluate_ADAIN(model):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    batch_size = 200

    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        testData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
        for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)

            # predict
            pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
            pred = pred.to("cpu")

            # evaluate
            pred = list(map(lambda x: x[0], pred.data.numpy()))
            batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
            result += pred
            result_label += batch_target

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def evaluate_ADAIN(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))

    # model
    model = ADAIN(inputDim_local_static=inputDim["local_static"],
                  inputDim_local_seq=inputDim["local_seq"],
                  inputDim_others_static=inputDim["others_static"],
                  inputDim_others_seq=inputDim["others_seq"])

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = list()
    result_label = list()

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    batch_size = 200
    iteration = 0

    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        testData = MyDataset_ADAIN(pickle.load(bz2.BZ2File(dataPath + selector, 'rb')))
        for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)

            # predict
            pred = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq)
            pred = pred.to("cpu")

            # evaluate
            pred = list(map(lambda x: x[0], pred.data.numpy()))
            batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
            result += pred
            result_label += batch_target

            iteration += len(batch_target)
            print("\t|- iteration %d / %d" % (iteration, len(testData)*testNum))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy

def objective_AIREX(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    hp = pickle.load(open("tmp/hyperParameters.pkl", "rb"))
    batch_size = hp.BATCH_SIZE
    epochs = hp.EPOCHs
    lr = hp.LEARNING_RATE
    wd = hp.WEIGHT_DECAY
    alpha = hp.ALPHA
    beta = 1.0 - alpha
    gamma = hp.GAMMA
    eta = hp.ETA

    # dataset path
    trainPath = pickle.load(open("tmp/trainPath.pkl", "rb"))
    testPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["city"]
    stationNum = pickle.load(open("{}/fileNum.pkl".format(trainPath), "rb"))["station"]

    # model
    model = AIREX(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum,
                   stationNum=stationNum)

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # loss function
    criterion_mse = nn.MSELoss()
    criterion_mmd = SamplesLoss("gaussian", blur=0.5)

    # initialize the early stopping object
    patience = epochs
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # mmd data
    mmdData = list()
    testData = list()
    testData_idx = list()
    for i in range(stationNum):
        tmp = pickle.load(bz2.BZ2File("{}/test_{}.pkl.bz2".format(testPath, str(i).zfill(3)), 'rb'))
        mmdData.append(MyDataset_MMD(tmp[:2]))
        testData.append(MyDataset_AIREX(tmp))
        testData_idx.append(pickle.load(open("{}/test_{}_idx.pkl".format(testPath, str(i).zfill(3)), 'rb')))
    print("mmd data was loaded")

    # start training
    for step in range(int(epochs)):

        start = time.time()
        epoch_loss = list()

        stationSelector = [random.randrange(0, 5) for i in range(cityNum)]

        for idx in range(len(stationSelector)):

            repeat_loss = list()
            selectPath = "{}/train_{}{}.pkl.bz2".format(trainPath, str(idx).zfill(3), str(stationSelector[idx]).zfill(3))
            trainData = MyDataset_AIREX(pickle.load(bz2.BZ2File(selectPath, "rb")))
            trainData = list(torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=False))

            selectPath = "{}/train_{}{}_idx.pkl".format(trainPath, str(idx).zfill(3), str(stationSelector[idx]).zfill(3))
            local_index = pickle.load(open(selectPath, "rb"))

            for batch_i in range(len(trainData)):

                print("\t|- mid-loss: ", end="")

                optimizer.zero_grad()

                # batch data
                batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = trainData[batch_i]

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)
                batch_others_static = batch_others_static.to(device)
                batch_others_seq = batch_others_seq.to(device)
                batch_others_city = batch_others_city.to(device)
                batch_target = batch_target.to(device)

                # predict
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, local_index)

                # loss append
                batch_loss_moe = criterion_mse(y_moe, batch_target)
                tmp = np.sqrt(float(batch_loss_moe.item()))
                repeat_loss.append(tmp)

                batch_loss_mtl = 0
                for y_mtl_i in y_mtl:
                    batch_loss_mtl += (1/(len(y_mtl))) * criterion_mse(y_mtl_i, batch_target)

                # mmd target
                batch_local_static = list()
                batch_local_seq = list()
                for i in range(stationNum):
                    mmdData_i = list(torch.utils.data.DataLoader(mmdData[i], batch_size=batch_size, shuffle=False))
                    mmdData_i = mmdData_i[batch_i]
                    batch_local_static.append(mmdData_i[0])
                    batch_local_seq.append(mmdData_i[1])

                # stack
                batch_local_static = torch.cat(batch_local_static, dim=0)
                batch_local_seq = torch.cat(batch_local_seq, dim=0)

                # to GPU
                batch_local_static = batch_local_static.to(device)
                batch_local_seq = batch_local_seq.to(device)

                # calculate mmd target
                mmd_target = model.encode(batch_local_static, batch_local_seq)

                # mmd source
                batch_loss_mmd = criterion_mmd(mmd_target, y_mmd) ** 2

                # loss (multi-task learning)
                batch_loss = (alpha * batch_loss_moe) + (beta * batch_loss_mtl) + (gamma * batch_loss_mmd) + (eta * etp)
                batch_loss.backward()
                optimizer.step()

                print("{}, total: {}".format(str(tmp), str(float(batch_loss.item()))))

            repeat_loss = np.mean(repeat_loss)
            epoch_loss.append(repeat_loss)
            print("\t\t|- repeat loss: {}".format(str(repeat_loss)))

        epoch_loss = np.mean(epoch_loss)
        print("\t\t\t|- epoch loss: {}".format(str(epoch_loss)))
        print("\t\t\t|- epoch time: {} [h]".format(str((time.time() - start) / (60 * 60))))

        # evaluate
        model.eval()
        rmse, accuracy = midium_evaluate_AIREX(model, testData, testData_idx)
        model.train()
        log = {'epoch': step, 'train_rmse': epoch_loss, 'test_rmse': rmse}
        logs.append(log)
        print("\t\t\t|- rmse: {}, accuracy: {}".format(str(rmse), str(accuracy)))

        # early stopping
        early_stopping(rmse, model)
        if early_stopping.early_stop:
            print("\t\t\tEarly stopping")
            break

    # load the last checkpoint after early stopping
    model.load_state_dict(torch.load("tmp/checkpoint.pt"))
    rmse = early_stopping.val_loss_min

    # save model
    trial_num = trial.number
    with open("tmp/{}_model.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/{}_log.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        pickle.dump(logs, pl)

    return rmse


def midium_evaluate_AIREX(model, testData, testData_idx):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128
    iteration = 0

    testPath = pickle.load(open("tmp/testPath.pkl", "rb"))
    stationNum = pickle.load(open("{}/fileNum.pkl".format(testPath), "rb"))["station"]

    # for evaluation
    criterion_mse = nn.MSELoss().eval()
    rmse = list()
    accuracy = list()

    for i in range(stationNum):

        result = list()
        result_label = list()
        local_idx = testData_idx[i]

        for batch_i in torch.utils.data.DataLoader(testData[i], batch_size=batch_size, shuffle=False):

            # batch data
            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)
            batch_others_city = batch_others_city.to(device)
            batch_target = batch_target.to(device)

            # predict
            with torch.no_grad():
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, local_idx)

                # add to result
                result.append(y_moe)
                result_label.append(batch_target)

                iteration += len(batch_target)

        with torch.no_grad():
            result = torch.cat(result, dim=0)
            result_label = torch.cat(result_label, dim=0)
            loss = criterion_mse(result, result_label)
            rmse.append(np.sqrt(float(loss.item())))

            result = result.to("cpu")
            result_label = result_label.to("cpu")
            result = list(map(lambda x: x[0], result.data.numpy()))
            result_label = list(map(lambda x: x[0], result_label.data.numpy()))
            accuracy.append(calc_correct(result, result_label) / len(result))

    # evaluation score
    rmse = np.mean(rmse)
    accuracy = np.mean(accuracy)

    return rmse, accuracy

def evaluate_AIREX(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 32
    iteration = 0

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # input dimension
    inputDim = pickle.load(open("datatmp/inputDim.pkl", "rb"))
    cityNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["city"]
    stationNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["station"]

    # model
    model = AIREX(inputDim_local_static=inputDim["local_static"],
                   inputDim_local_seq=inputDim["local_seq"],
                   inputDim_others_static=inputDim["others_static"],
                   inputDim_others_seq=inputDim["others_seq"],
                   cityNum=cityNum,
                   stationNum=stationNum)

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    criterion_mse = nn.MSELoss().eval()
    rmse = list()
    accuracy = list()
    start = time.time()

    for station_id in range(stationNum):

        selectPath = "/test_{}.pkl.bz2".format(str(station_id).zfill(3))
        testData = MyDataset_AIREX(pickle.load(bz2.BZ2File(dataPath + selectPath, 'rb')))
        selectPath = "/test_{}_idx.pkl".format(str(station_id).zfill(3))
        local_idx = pickle.load(open(dataPath + selectPath, 'rb'))

        result = list()
        result_label = list()

        for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

            # batch data
            batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, batch_target = batch_i

            # to GPU
            batch_local_static = batch_local_static.to(device)
            batch_local_seq = batch_local_seq.to(device)
            batch_others_static = batch_others_static.to(device)
            batch_others_seq = batch_others_seq.to(device)
            batch_others_city = batch_others_city.to(device)
            batch_target = batch_target.to(device)

            # predict
            with torch.no_grad():
                y_moe, y_mtl, y_mmd, etp = model(batch_local_static, batch_local_seq, batch_others_static, batch_others_seq, batch_others_city, local_idx)

                # add to result
                result.append(y_moe)
                result_label.append(batch_target)

                iteration += len(batch_target)

        with torch.no_grad():
            result = torch.cat(result, dim=0)
            result_label = torch.cat(result_label, dim=0)
            loss = criterion_mse(result, result_label)
            rmse.append(np.sqrt(float(loss.item())))

            result = result.to("cpu")
            result_label = result_label.to("cpu")
            result = list(map(lambda x: x[0], result.data.numpy()))
            result_label = list(map(lambda x: x[0], result_label.data.numpy()))
            accuracy.append(calc_correct(result, result_label) / len(result))

    print("\t\t\t|- evaluate time: {} [h]".format(str((time.time() - start) / (60 * 60))))

    # evaluation score
    rmse = np.mean(rmse)
    accuracy = np.mean(accuracy)

    return rmse, accuracy

def objective_FNN(trial):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyper parameters for tuning
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    # epochs = trial.suggest_discrete_uniform("epochs", 1, 5, 1)
    # lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    # wd = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # hyper parameters for constance
    hp = pickle.load(open("tmp/hyperParameters.pkl", "rb"))
    batch_size = hp.BATCH_SIZE
    epochs = hp.EPOCHs
    lr = hp.LEARNING_RATE
    wd = hp.WEIGHT_DECAY

    # dataset path
    dataPath = pickle.load(open("tmp/trainPath.pkl", "rb"))
    selector = "/train_000.pkl.bz2"

    # dataset
    local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))

    dataset = list()
    inputDim = 0
    for i in range(len(target)):
        trainData_i = list()
        trainData_i += local_static[i]
        trainData_i += local_seq[i][-1]
        for j in range(len(others_static[i])):
            trainData_i += others_static[i][j]
            trainData_i += others_seq[i][j][-1]
        dataset.append(trainData_i)
        inputDim = len(trainData_i)

    # train and validation data
    trainNum = math.floor(len(dataset)*0.67)
    trainData = MyDataset_FNN(dataset[:trainNum], target[:trainNum])
    validData = MyDataset_FNN(dataset[trainNum:], target[trainNum:])
    del dataset

    # model
    model = FNN(inputDim=inputDim)

    # GPU or CPU
    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # evaluation function
    criterion = nn.MSELoss()

    # initialize the early stopping object
    patience = epochs
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # log
    logs = list()

    # start training
    for step in range(int(epochs)):

        epoch_loss = list()

        for batch_i in torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True):

            print("\t|- batch loss: ", end="")

            # initialize graduation
            optimizer.zero_grad()

            # batch data
            batch_feature, batch_target = batch_i

            # to GPU
            batch_feature = batch_feature.to(device)
            batch_target = batch_target.to(device)

            # predict
            pred = model(batch_feature)

            # calculate loss, back-propagate loss, and step optimizer
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()

            # print a batch loss as RMSE
            batch_loss = np.sqrt(loss.item())
            print("%.10f" % (batch_loss))

            # append batch loss to the list to calculate epoch loss
            epoch_loss.append(batch_loss)

        epoch_loss = np.average(epoch_loss)
        print("\t\t|- epoch %d loss: %.10f" % (step + 1, epoch_loss))

        # validate
        print("\t\t|- validation : ", end="")
        model.eval()
        rmse, accuracy = validate_FNN(model, validData)
        model.train()

        # calculate validation loss
        log = {'epoch': step, 'validation rmse': rmse, 'validation accuracy': accuracy}
        logs.append(log)
        print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

        # early stopping
        early_stopping(rmse, model)
        if early_stopping.early_stop:
            print("\t\tEarly stopping")
            break

    # load the last checkpoint after early stopping
    model.load_state_dict(torch.load("tmp/checkpoint.pt"))
    rmse = early_stopping.val_loss_min

    # save model
    trial_num = trial.number
    with open("tmp/{}_model.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        torch.save(model.state_dict(), pl)

    # save logs
    logs = pd.DataFrame(logs)
    with open("tmp/{}_log.pkl".format(str(trial_num).zfill(4)), "wb") as pl:
        pickle.dump(logs, pl)

    return rmse

def validate_FNN(model, validData):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 2000

    for batch_i in torch.utils.data.DataLoader(validData, batch_size=batch_size, shuffle=False):

        # batch data
        batch_feature, batch_target = batch_i

        # to GPU
        batch_feature = batch_feature.to(device)

        # predict
        pred = model(batch_feature)
        pred = pred.to("cpu")

        # evaluate
        pred = list(map(lambda x: x[0], pred.data.numpy()))
        batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
        result += pred
        result_label += batch_target

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy

def evaluate_FNN(model_state_dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    # dataset
    testData = list()
    labelData = list()
    inputDim = 0
    for idx in range(testNum):
        
        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        local_static, local_seq, others_static, others_seq, target = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))
        for i in range(len(target)):
            testData_i = list()
            testData_i += local_static[i]
            testData_i += local_seq[i][-1]
            for j in range(len(others_static[i])):
                testData_i += others_static[i][j]
                testData_i += others_seq[i][j][-1]
            testData.append(testData_i)
            inputDim = len(testData_i)
        labelData += target
    testData = MyDataset_FNN(testData, labelData)

    # model
    model = FNN(inputDim=inputDim)
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # evaluate mode
    model.eval()

    # for evaluation
    result = list()
    result_label = list()

    batch_size = 2000
    iteration = 0

    for batch_i in torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False):

        # batch data
        batch_feature, batch_target = batch_i

        # to GPU
        batch_feature = batch_feature.to(device)

        # predict
        pred = model(batch_feature)
        pred = pred.to("cpu")

        # evaluate
        pred = list(map(lambda x: x[0], pred.data.numpy()))
        batch_target = list(map(lambda x: x[0], batch_target.data.numpy()))
        result += pred
        result_label += batch_target

        iteration += len(batch_target)
        print("\t|- iteration %d / %d" % (iteration, len(testData)*testNum))

    # evaluation score
    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)
    print("rmse: %.10f, accuracy: %.10f" % (rmse, accuracy))

    return rmse, accuracy

def evaluate_KNN(K):

    # aqi data
    aqiData = pickle.load(open("datatmp/labelData.pkl", "rb"))
    for k, v in aqiData.items():
        aqiData[k] = list(map(lambda x: x[0], v))

    # static data
    staticData = pickle.load(open("datatmp/staticData.pkl", "rb"))

    # station data
    stationData = pickle.load(open("datatmp/stationData.pkl", "rb"))

    # dataset path
    dataPath = pickle.load(open("tmp/testPath.pkl", "rb"))

    # the number which the test dataset was divided into
    testNum = pickle.load(open("{}/fileNum.pkl".format(dataPath), "rb"))["test"]

    # for evaluation
    result = list()
    result_label = list()
    for idx in range(testNum):

        selector = "/test_{}.pkl.bz2".format(str(idx).zfill(3))
        dataset = pickle.load(bz2.BZ2File(dataPath + selector, 'rb'))
        local_static, others_static = dataset[0][0], dataset[2][0]

        # target station
        target_station = [k for k, v in staticData.items() if v == local_static][0]

        # source stations
        source_station = list()
        for staticData_i in others_static:
            staticData_i = staticData_i[:-2]
            source_station.append([k for k, v in staticData.items() if v == staticData_i][0])

        # calculate distance from the local station
        # local station
        lat_target = float(stationData[stationData["sid"] == target_station]["lat"])
        lon_target = float(stationData[stationData["sid"] == target_station]["lon"])

        # distance
        distance = dict()
        for source_station_i in source_station:
            lat_source = float(stationData[stationData["sid"] == source_station_i]["lat"])
            lon_source = float(stationData[stationData["sid"] == source_station_i]["lon"])
            distance[source_station_i] = get_dist_angle(lat1=lat_target, lon1=lon_target, lat2=lat_source, lon2=lon_source)["distance"]

        # get K nearest neighbors
        distance = sorted(distance.items(), key=lambda x: x[1])
        nearest = list(map(lambda x: x[0], distance[:K]))

        # agi data of source cities
        aqiData_source = list()
        for source_station_i in nearest:
            aqiData_source.append(aqiData[source_station_i])

        # aqi data of target city
        aqiData_target = aqiData[target_station]

        # evaluate
        aqiData_source = list(np.mean(np.array(aqiData_source), axis=0))
        result += aqiData_source[1000:3000]
        result_label += aqiData_target[1000:3000]

    rmse = np.sqrt(mean_squared_error(result, result_label))
    accuracy = calc_correct(result, result_label) / len(result)

    return rmse, accuracy
