# -*- coding: utf-8 -*-

# Directory path
import sys
sys.path.append("/home") 

# from python library
import re
import requests
import json
import pandas as pd
import argparse
import overpy
import pickle
from time import sleep

# from my library
from source.utility import citycode
from source.utility import Color
from source.utility import get_grid_id
from source.utility import get_aqi_series
from source.utility import get_meteorology_series
from source.utility import get_road_data
from source.utility import get_poi_data
from source.utility import normalization
from source.utility import data_interpolate
from source.utility import weather_onehot
from source.utility import winddirection_onehot

cities = list(pd.read_csv("rawdata/zheng2015/city.csv")["name_english"])
print(cities)

def station(scale):

    for city in cities:

        # input
        with open("rawdata/zheng2015/station.csv", "r") as infile:

            if scale == "develop":
                scale = 5
            if scale == "small":
                scale = 10
            if scale == "large":
                scale = 10000

            # output
            with open("database/station/station_" + city + ".csv", "w") as outfile:
                outfile.write("sid,lat,lon,did,gid\n")
                pattern = citycode(name=city, scale="station")
                station_counter = 0
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")

                    if re.match(pattern, line[0]):
                        gid = "NULL"
                        line = [line[0], line[3], line[4], line[5], gid]
                        outfile.write(",".join(line) + "\n")
                        station_counter += 1

                    if station_counter >= scale:
                        break

def station_grid():

    for city in cities:

        # input
        with open("rawdata/zheng2015/station.csv", "r") as infile:

            # ll
            with open("database/city/city_"+city+".csv", "r") as cityfile:
                minlat, minlon, maxlat, maxlon = [float(s) for s in cityfile.readlines()[1].strip().split(",")]

                # output
                with open("database/station/station_" + city + ".csv", "w") as outfile:
                    outfile.write("sid,lat,lon,did,gid\n")
                    pattern = citycode(name=city, scale="station")
                    for line in infile.readlines()[1:]:
                        line = line.strip().split(",")
                        if re.match(pattern, line[0]):
                            if minlat < float(line[3]) < maxlat and minlon < float(line[4]) < maxlon:
                                gid = get_grid_id(city=city, lat=float(line[3]), lon=float(line[4]))
                                line = [line[0], line[3], line[4], line[5], gid]
                                outfile.write(",".join(line)+"\n")

def aqi():

    for city in cities:

        # station list
        stationlist = list()
        with open("database/station/station_"+city+".csv", "r") as stationfile:
            for station in stationfile.readlines()[1:]:
                stationlist.append(station.strip().split(",")[0])

        # input
        with open("rawdata/zheng2015/airquality.csv", "r") as infile:

            # output
            with open("database/aqi/aqi_" + city + ".csv", "w") as outfile:
                outfile.write("sid,time,pm25,pm10,no2,co,o3,so2\n")
                pattern = citycode(name=city, scale="station")
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")
                    if re.match(pattern, line[0]):
                        if line[0] in stationlist:
                            outfile.write(",".join(line)+"\n")

def meteorology():

    for city in cities:

        # district list
        districtlist = list()
        with open("database/station/station_"+city+".csv", "r") as stationfile:
            for station in stationfile.readlines()[1:]:
                districtlist.append(station.strip().split(",")[3])

        # input
        with open("rawdata/zheng2015/meteorology.csv", "r") as infile:

            # output
            with open("database/meteorology/meteorology_" + city + ".csv", "w") as outfile:
                outfile.write("did,time,weather,temperature,pressure,humidity,wind_speed,wind_direction\n")
                pattern = citycode(name=city, scale="district")
                for line in infile.readlines()[1:]:
                    line = line.strip().split(",")
                    if re.match(pattern, line[0]):
                        if line[0] in districtlist:
                            outfile.write(",".join(line)+"\n")

def createCategory_tree(subtree, category_dict, category):

    if len(subtree["categories"]) == 0:
        category_dict[subtree["name"]] = category
        return category_dict

    for item in subtree["categories"]:

        category_dict = createCategory_tree(item, category_dict, category)
        category_dict[item["name"]] = category

    return category_dict

def poi(key, secret, radius):

    # create category list
    category_dict = dict()
    poi_counter = dict()
    url = "https://api.foursquare.com/v2/venues/categories"
    params = dict(
        client_id=args.key,
        client_secret=args.secret,
        v="20140401"
    )
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)["response"]
    with open("database/poi/category.csv", "w") as category_file:
        category_file.write("https://developer.foursquare.com/docs/resources/categories\n")
        for item in response["categories"]:
            category_file.write(item["name"]+"\n")
            poi_counter[item["name"]] = 0
            category_dict = createCategory_tree(item, category_dict, item["name"])

    # get poi data
    for city in cities:

        no_poi_stations = list()

        with open("database/poi/poi_"+city+".csv", "w") as outfile:
            outfile.write("sid,{}\n".format(",".join(list(poi_counter.keys()))))

            for line in open("database/station/station_"+city+".csv", "r").readlines()[1:]:

                _poi_list = poi_counter.copy()

                sid, lat, lon, did, gid = line.strip().split(",")

                url = "https://api.foursquare.com/v2/venues/search"
                params = dict(
                    client_id=key,
                    client_secret=secret,
                    intent="browse",
                    ll="{},{}".format(lat, lon),
                    radius=radius, #[m]
                    limit="10000",
                    v="20140401"
                )
                sleep(0.1)
                response = requests.get(url=url, params=params)
                response = json.loads(response.text)

                # check response
                if response["meta"]["code"] == 200:
                    response = response["response"]
                else:
                    print("error {}".format(str(response["metta"]["code"])))
                    exit()

                # count poi
                if len(response["venues"]) == 0:
                    no_poi_stations.append(sid)
                else:
                    for venue in response["venues"]:

                        # select primary category
                        for category in venue["categories"]:
                            if category["primary"] is True:
                                category = category["name"]

                        # select parent category
                        if category in category_dict:
                            _poi_list[category_dict[category]] += 1

                    # write into a file
                    _poi_list = map(str, list(_poi_list.values()))
                    outfile.write("{},{}\n".format(str(sid), ",".join(_poi_list)))

        # drop no poi stations from station file
        if len(no_poi_stations) > 0:
            stations = pd.read_csv("database/station/station_"+city+".csv", dtype=object)
            for _sid in no_poi_stations:
                stations = stations[stations["sid"] != _sid]
            stations.to_csv("database/station/station_"+city+".csv", index=False)

def road(radius):
    for city in cities:
        with open("database/road/road_" + city + ".csv", "w") as outfile:
            outfile.write("sid,motorway,trunk,others\n")

            for line in open("database/station/station_"+city+".csv", "r").readlines()[1:]:

                for i in range(1, 11):

                    try:
                        api = overpy.Overpass()
                        sid, lat, lon, did, gid = line.strip().split(",")
                        result = api.query("way(around:"+str(float(radius))+","+lat+","+lon+");out;")
                        road = dict(
                            motorway=0,
                            trunk=0,
                            others=0,
                            na=0
                        )
                        for way in result.ways:
                            highway = way.tags.get("highway", "n/a")
                            if highway == "n/a":
                                road["na"] += 1
                            elif highway in road:
                                road[highway] += 1
                            else:
                                road["others"] += 1
                        outfile.write(sid+","+str(road["motorway"])+","+str(road["trunk"])+","+str(road["others"])+"\n")
                    except overpy.exception as e:
                        print("error:{e} retry:{i}/10".format(e=e, i=i))
                        sleep(i * 5)
                        fg = False
                    else:
                        fg = True

                    if fg:
                        break

def preprocessing(cities, model_attribute, lstm_data_width, data_length=None):

    '''
    cities (list):
    :param model_attribute:
    :param lstm_data_width:
    :return:
    '''

    print("load data from the DB ... ", end="")

    '''
    station data
    '''
    for city in cities:
        station = pd.read_csv("database/station/station_{}.csv".format(city), dtype=object)

        if city == cities[0]:
            station_raw = station
        else:
            station_raw = pd.concat([station_raw, station], ignore_index=True)

    '''
    road data
    '''
    road_attribute = ["motorway", "trunk", "others"]
    dtype = {att: "float" for att in road_attribute}
    dtype["sid"] = "object"
    for city in cities:
        station = pd.read_csv("database/road/road_{}.csv".format(city), dtype=dtype)
        df = normalization(station[road_attribute])
        station = pd.concat([station.drop(road_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            road_raw = station
        else:
            road_raw = pd.concat([road_raw, station], ignore_index=True)

    '''
    poi data
    '''
    poi_attribute =["Arts & Entertainment", "College & University", "Event",
                    "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                    "Residence", "Shop & Service", "Travel & Transport"]

    dtype = {att: "float" for att in poi_attribute}
    dtype["sid"] = "object"
    for city in cities:
        station = pd.read_csv("database/poi/poi_{}.csv".format(city), dtype=dtype)
        df = normalization(station[poi_attribute])
        station = pd.concat([station.drop(poi_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            poi_raw = station
        else:
            poi_raw = pd.concat([poi_raw, station], ignore_index=True)

    '''
    meteorology data
    '''
    meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
    dtype = {att: "float" for att in meteorology_attribute}
    dtype["did"], dtype["time"] = "object", "object"
    for city in cities:
        station = pd.read_csv("database/meteorology/meteorology_{}.csv".format(city), dtype=dtype)
        meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
        df = normalization(data_interpolate(station[meteorology_attribute]))
        station = pd.concat([station.drop(meteorology_attribute, axis=1), df], axis=1)

        df, columns = weather_onehot(station["weather"])
        station = pd.concat([station.drop(["weather"], axis=1), df], axis=1)
        meteorology_attribute += columns

        df, columns = winddirection_onehot(station["wind_direction"])
        station = pd.concat([station.drop(["wind_direction"], axis=1), df], axis=1)
        meteorology_attribute += columns

        if city == cities[0]:
            meteorology_raw = station
        else:
            meteorology_raw = pd.concat([meteorology_raw, station], ignore_index=True)

    '''
    aqi data
    '''
    aqi_attribute = ["pm25", "pm10", "no2", "co", "o3", "so2"]
    dtype = {att: "float" for att in aqi_attribute}
    dtype["sid"], dtype["time"] = "object", "object"
    for city in cities:
        # for label
        station_label = pd.read_csv("database/aqi/aqi_{}.csv".format(city), dtype=dtype)
        df = data_interpolate(station_label[[model_attribute]])
        station_label = pd.concat([station_label.drop(aqi_attribute, axis=1), df], axis=1)

        # for feature
        station_feature = pd.read_csv("database/aqi/aqi_{}.csv".format(city), dtype=dtype)
        df = normalization(data_interpolate(station_feature[[model_attribute]]))
        station_feature = pd.concat([station_feature.drop(aqi_attribute, axis=1), df], axis=1)

        if city == cities[0]:
            aqi_raw_label = station_label
            aqi_raw_feature = station_feature
        else:
            aqi_raw_label = pd.concat([aqi_raw_label, station_label], ignore_index=True)
            aqi_raw_feature = pd.concat([aqi_raw_feature, station_feature], ignore_index=True)

    print(Color.GREEN + "OK" + Color.END)
    print("make data ... ", end="")

    '''
    make dataset
    '''
    staticData, seqData_m, seqData_a, labelData = dict(), dict(), dict(), dict()
    for sid in list(station_raw["sid"]):

        '''
        static data
            * poi
            * road network data

        format
            X = [poi attributes, road attributes]
        '''
        recode = [float(get_poi_data(data=poi_raw, sid=sid, attribute=att)) for att in poi_attribute]
        staticData[sid] = recode
        recode = [float(get_road_data(data=road_raw, sid=sid, attribute=att)) for att in road_attribute]
        staticData[sid] += recode

        '''
        sequence data
            * meteorological data
            * aqi data

        format
            r_t = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
            R_p = [r_t, ..., r_t+p]
            X = [ R_p, R_p+1, ..., R_n ]

        format
            r_t = ["pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"]
            R_p = [r_t, ..., r_t+p]
            X = [ R_p, R_p+1, ..., R_n ]
        '''
        did = list(station_raw[station_raw["sid"] == sid]["did"])[0]
        meteorology_data = {att: get_meteorology_series(data=meteorology_raw, did=did, attribute=att)
                            for att in meteorology_attribute}
        aqi_data2 = get_aqi_series(data=aqi_raw_feature, sid=sid, attribute=model_attribute)

        if data_length is None:
            data_length = len(meteorology_data[meteorology_attribute[0]])

        seqData_m[sid] = []
        seqData_a[sid] = []
        terminal = data_length
        start = 0
        end = lstm_data_width
        while end <= terminal:
            recode_m_p = []
            recode_a_p = []
            for t in range(start, end):
                recode_m_t = []
                for att in meteorology_attribute:
                    recode_m_t.append(meteorology_data[att][t])
                recode_m_p.append(recode_m_t)
                recode_a_p.append([aqi_data2[t]])
            seqData_m[sid].append(recode_m_p)
            seqData_a[sid].append(recode_a_p)
            start += 1
            end += 1

        '''
        label data
            * aqi data

        format
            aqi = "pm25" or  "pm10" or "no2" or "co" or "o3" or "so2"
            X = [ [aqi_p], [aqi_p+1], ..., [aqi_n] ]
        '''
        aqi_data1 = get_aqi_series(data=aqi_raw_label, sid=sid, attribute=model_attribute)
        labelData[sid] = []
        for t in range(lstm_data_width - 1, data_length):
            labelData[sid].append([aqi_data1[t]])

    # saving
    with open("datatmp/inputDim.pkl", "wb") as fp:
        dc = {"local_static": len(poi_attribute)+len(road_attribute),
              "local_seq": len(meteorology_attribute),
              "others_static": len(poi_attribute)+len(road_attribute) + 2, # + distance and angle
              "others_seq": len(meteorology_attribute) + 1} # + aqi
        pickle.dump(dc, fp)

    with open("datatmp/stationData.pkl", "wb") as fp:
        pickle.dump(station_raw, fp)

    with open("datatmp/staticData.pkl", "wb") as fp:
        pickle.dump(staticData, fp)

    with open("datatmp/meteorologyData.pkl", "wb") as fp:
        pickle.dump(seqData_m, fp)

    with open("datatmp/aqiData.pkl", "wb") as fp:
        pickle.dump(seqData_a, fp)

    with open("datatmp/labelData.pkl", "wb") as fp:
        pickle.dump(labelData, fp)

    print(Color.GREEN + "OK" + Color.END)

if __name__ == "__main__":

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH_ROOT_SRC", type=str, default="src/build.py")
    parser.add_argument("--SCALE", help="please specify large or small, or develop", type=str, default="large")
    parser.add_argument("--RADIUS", help="please set me with meter scale", type=int, default=1000)
    parser.add_argument("--KEY", help="your key to access poi api", type=str, default="TUCJCEQMKNPCTXBKUQ0CA2DF5JOGAYT0EVP1SXOJIOB2NCVH")
    parser.add_argument("--SECRET", help="your secret to access poi api", type=str, default="RDLLQBSZ5UH2FOCNTTL3EANZMLR2VMAYPT5RDS3ADJCXTNGJ")
    parser.add_argument("--ATTRIBUTE", help="inferred attribute", type=str, default="pm25")
    parser.add_argument("--TIME_LENGTH", help="data length of time series data", type=int, default=24*30*4)
    parser.add_argument("--LSTM_DATA_WIDTH", help="data length of LSTM", type=int, default=24)
    args = parser.parse_args()

    '''
    # creating data from raw data or collect it.  
    print("\t|- station data is build ... ", end="")
    station(args.SCALE)
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- poi data is build ... ", end="")
    poi(args.KEY, args.SECRET, args.RADIUS)
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- road network data is build ... ", end="")
    road(args.RADIUS)
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- aqi data is build ... ", end="")
    aqi()
    print(Color.GREEN + "OK" + Color.END)

    print("\t|- meteorological data is build ... ", end="")
    meteorology()
    print(Color.GREEN + "OK" + Color.END)
    '''
    print("\t|- pre-processing built data")
    CITIEs4 = ["BeiJing", "TianJin", "ShenZhen", "GuangZhou"]
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
    preprocessing(CITIEs20, args.ATTRIBUTE, args.LSTM_DATA_WIDTH, args.TIME_LENGTH)
    print(Color.GREEN + "OK" + Color.END)
