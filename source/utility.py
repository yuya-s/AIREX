# -*- coding: utf-8 -*-

# Directory path
import sys
sys.path.append("/home") 

from math import *
from torchvision import transforms
import pandas as pd
import numpy as np
import overpy
import torch
import resource
import pickle
import multiprocessing as mp
import os
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

class MMD_preComputed():

    def __init__(self, city, alpha, data_length=None):

        '''
        make datatmp
        '''

        # station data
        station = pd.read_csv("database/station/station_" + city + ".csv", dtype=object)

        # poi data
        poi_attribute = ["Arts & Entertainment", "College & University", "Event",
                         "Food", "Nightlife Spot", "Outdoors & Recreation", "Professional & Other Places",
                         "Residence", "Shop & Service", "Travel & Transport"]
        dtype = {att: "float" for att in poi_attribute}
        dtype["sid"] = "object"

        poi = pd.read_csv("database/poi/poi_" + city + ".csv", dtype=dtype)
        # df = normalization(poi[poi_attribute])
        # poi = pd.concat([poi.drop(poi_attribute, axis=1), df], axis=1)

        # road network data
        road_attribute = ["motorway", "trunk", "others"]
        dtype = {att: "float" for att in road_attribute}
        dtype["sid"] = "object"

        road = pd.read_csv("database/road/road_" + city + ".csv", dtype=dtype)
        # df = normalization(road[road_attribute])
        # road = pd.concat([road.drop(road_attribute, axis=1), df], axis=1)

        # meteorological data
        meteorology_attribute = ["weather", "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
        dtype = {att: "float" for att in meteorology_attribute}
        dtype["did"], dtype["time"] = "object", "object"

        mete = pd.read_csv("database/meteorology/meteorology_" + city + ".csv", dtype=dtype)
        meteorology_attribute = ["temperature", "pressure", "humidity", "wind_speed"]
        df = data_interpolate(mete[meteorology_attribute])
        #df = normalization(data_interpolate(mete[meteorology_attribute]))
        mete = pd.concat([mete.drop(meteorology_attribute, axis=1), df], axis=1)

        df, columns = weather_onehot(mete["weather"])
        mete = pd.concat([mete.drop(["weather"], axis=1), df], axis=1)
        meteorology_attribute += columns

        df, columns = winddirection_onehot(mete["wind_direction"])
        mete = pd.concat([mete.drop(["wind_direction"], axis=1), df], axis=1)
        meteorology_attribute += columns

        for sid in list(station["sid"]):

            did = list(station[station["sid"] == sid]["did"])[0]
            if data_length is None:
                mete_tmp = mete[mete["did"] == did]
            else:
                mete_tmp = mete[mete["did"] == did][:data_length]
            mete_tmp = mete_tmp.drop(["did"], axis=1).reset_index(drop=True)
            df = pd.DataFrame({"sid": [sid] * len(mete_tmp)})
            mete_tmp = pd.concat([df, mete_tmp], axis=1)

            if sid == list(station["sid"])[0]:
                meteorology = mete_tmp
            else:
                meteorology = pd.concat([meteorology, mete_tmp], axis=0, ignore_index=True)

        mmd_data = pd.merge(meteorology, poi, on="sid")
        mmd_data = pd.merge(mmd_data, road, on="sid")
        mmd_data = mmd_data.drop(["sid", "time"], axis=1)
        mmd_data = mmd_data.values

        print("\t|- {} data is created".format(city))
        with open("tmp/mmdData_" + city + ".pickle", "wb") as pl:
            pickle.dump(mmd_data, pl)

        self.X = mmd_data
        self.n_x = len(mmd_data)
        self.city = city
        self.alpha = alpha
        self.proc = 36

    def __call__(self):

        # single-processing
        # XX = 0
        # for i in range(self.n_x):
        #     for j in range(self.n_x):
        #         if i == j:
        #             continue
        #         XX += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i] - self.X[j], ord=2) ** 2)

        # multi-processing
        pool = mp.Pool(self.proc)
        XX = pool.map(self.xx, range(self.n_x))
        XX = sum(XX)

        print("\t|- {} kernel is computed".format(self.city))
        with open("tmp/kernelScore_" + self.city + ".pickle", "wb") as pl:
            pickle.dump(XX, pl)

    def xx(self, i):

        subtotal = 0
        for j in range(self.n_x):
            if i == j:
                continue
            subtotal += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i] - self.X[j], ord=2) ** 2)

        return subtotal

class MMD:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, source, target, alpha):

        # two samples
        self.X = pickle.load(open("tmp/mmdData_" + source + ".pickle", "rb"))
        self.Y = pickle.load(open("tmp/mmdData_" + target + ".pickle", "rb"))
        self.n_x = len(self.X)
        self.n_y = len(self.Y)

        # kernels
        self.XX = pickle.load(open("tmp/kernelScore_" + source + ".pickle", "rb"))
        self.YY = pickle.load(open("tmp/kernelScore_" + target + ".pickle", "rb"))

        # band with
        self.alpha = alpha

        # The three constants to calculate MMD.
        self.axx = 1. / (self.n_x * (self.n_x - 1))
        self.ayy = 1. / (self.n_y * (self.n_y - 1))
        self.axy = - 2. / (self.n_x * self.n_y)

        # for multiprocessing
        self.proc = 70

    def __call__(self):

        '''
        MMD(X, Y) is

            1/n(n-1) \sum_{a!=b}^{n} k(x_a, x_b)
                + 1/m(m-1) \sum_{c!=d}^{m} k(y_c, y_d)
                    - 2/nm \sum_{a=1}^{n} \sum_{c=1}^{m} k(x_a, y_c)

        for the kernel k.

        The kernel used is

            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},

        for the provided ``alphas``.
        '''

        # multi-processing using pre-computed data
        XX = self.XX
        YY = self.YY
        pool = mp.Pool(self.proc)
        XY = pool.map(self.xy, range(self.n_x))
        XY = sum(XY)

        # single-processing using pre-computed data
        # XX = self.XX
        # YY = self.YY
        # XY = 0
        # for i in range(self.n_x):
        #     for j in range(self.n_y):
        #         XY += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i]-self.Y[j], ord=2) ** 2)

        # multi-processing
        # pool = mp.Pool(self.proc)
        # XX = pool.map(self.xx, range(self.n_x))
        # XX = sum(XX)
        # YY = pool.map(self.yy, range(self.n_y))
        # YY = sum(YY)
        # XY = pool.map(self.xy, range(self.n_x))
        # XY = sum(XY)

        # single-processing
        # XX
        # XX = 0
        # for i in range(self.n_x):
        #     for j in range(self.n_x):
        #         if i == j:
        #             continue
        #         XX += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i]-self.X[j], ord=2) ** 2)
        # # YY
        # YY = 0
        # for i in range(self.n_y):
        #     for j in range(self.n_y):
        #         if i == j:
        #             continue
        #         YY += np.exp(-1 * self.alpha * np.linalg.norm(self.Y[i]-self.Y[j], ord=2) ** 2)
        # # XY
        # XY = 0
        # for i in range(self.n_x):
        #     for j in range(self.n_y):
        #         XY += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i]-self.Y[j], ord=2) ** 2)

        return (self.axx * XX) + (self.ayy * YY) + (self.axy * XY)

    def xx(self, i):

        subtotal = 0
        for j in range(self.n_x):
            if i == j:
                continue
            subtotal += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i]-self.X[j], ord=2) ** 2)

        return subtotal

    def yy(self, i):

        subtotal = 0
        for j in range(self.n_y):
            if i == j:
                continue
            subtotal += np.exp(-1 * self.alpha * np.linalg.norm(self.Y[i]-self.Y[j], ord=2) ** 2)

        return subtotal

    def xy(self, i):

        subtotal = 0
        for j in range(self.n_y):
            subtotal += np.exp(-1 * self.alpha * np.linalg.norm(self.X[i]-self.Y[j], ord=2) ** 2)

        return subtotal

class MyDataset_ADAIN(torch.utils.data.Dataset):

    def __init__(self, data):


        start, end = 1000, 3000

        self.local_static = data[0][start:end]
        self.local_seq = data[1][start:end]
        self.others_static = data[2][start:end]
        self.others_seq = data[3][start:end]
        self.target = data[4][start:end]
        self.data_num = len(data[4][start:end])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        out_local_static = torch.tensor(self.local_static[idx])
        out_local_seq = torch.tensor(self.local_seq[idx])
        out_others_static = torch.tensor(self.others_static[idx])
        out_others_seq = torch.tensor(self.others_seq[idx])
        out_target = torch.tensor(self.target[idx])

        return out_local_static, out_local_seq, out_others_static, out_others_seq, out_target

class MyDataset_AIREX(torch.utils.data.Dataset):

    def __init__(self, data):

        start, end = 1000, 3000

        self.local_static = data[0][start:end]
        self.local_seq = data[1][start:end]
        self.others_static = data[2][start:end]
        self.others_seq = data[3][start:end]
        self.others_city = data[4][start:end]
        self.target = data[5][start:end]
        self.data_num = len(data[5][start:end])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        out_local_static = torch.tensor(self.local_static[idx])
        out_local_seq = torch.tensor(self.local_seq[idx])
        out_others_static = torch.tensor(self.others_static[idx])
        out_others_seq = torch.tensor(self.others_seq[idx])
        out_others_city = torch.tensor(self.others_city[idx])
        out_target = torch.tensor(self.target[idx])

        return out_local_static, out_local_seq, out_others_static, out_others_seq, out_others_city, out_target

class MyDataset_MMD(torch.utils.data.Dataset):

    def __init__(self, data):

        start, end = 1000, 3000

        self.local_static = data[0][start:end]
        self.local_seq = data[1][start:end]
        self.data_num = len(data[0][start:end])

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        out_local_static = torch.tensor(self.local_static[idx])
        out_local_seq = torch.tensor(self.local_seq[idx])

        return out_local_static, out_local_seq

class MyDataset_FNN(torch.utils.data.Dataset):

    def __init__(self, feature, target):
        self.feature = feature
        self.target = target
        self.data_num = len(target)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_feature = torch.tensor(self.feature[idx])
        out_target = torch.tensor(self.target[idx])
        return out_feature, out_target

class Color:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[36m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'

def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    if optimizer_name == optimizer_names[0]:
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)

    return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)

    if activation_name == activation_names[0]:
        activation = F.relu
    else:
        activation = F.elu

    return activation

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'\t\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'\t\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        torch.save(model.state_dict(), 'tmp/checkpoint.pt')
        self.val_loss_min = val_loss

def citycode(name, scale):

    codefile = pd.read_csv("rawdata/zheng2015/city.csv", index_col=2, dtype=object)
    code = codefile.at[name, "city_id"]

    if scale == "city":
        return r"%s$" % (code)
    if scale == "district":
        return r"%s\d{2}$" % (code)
    if scale == "station":
        return r"%s\d{3}$" % (code)


    if scale == "station":
        if name == "beijing":
            return r"001\d{3}$"
        if name == "shenzhen":
            return r"004\d{3}$"
        if name == "tianjin":
            return r"006\d{3}$"
        if name == "guangzhou":
            return r"009\d{3}$"

    if scale == "district":
        if name == "beijing":
            return r"001\d{2}$"
        if name == "shenzhen":
            return r"004\d{2}$"
        if name == "tianjin":
            return r"006\d{2}$"
        if name == "guangzhou":
            return r"009\d{2}$"

    if scale == "city":
        if name == "beijing":
            return r"001$"
        if name == "shenzhen":
            return r"004$"
        if name == "tianjin":
            return r"006$"
        if name == "guangzhou":
            return r"009$"

def aqi_class(value):

    if 0.0 <= value < 51.0:
        return "G"
    elif 51.0 <= value < 101.0:
        return "M"
    elif 101.0 <= value < 151.0:
        return "US"
    elif 151.0 <= value < 201.0:
        return "U"
    elif 201.0 <= value < 301.0:
        return "VU"
    elif 301.0 <= value < 501.0:
        return "H"
    else:
        return "ERROR"

def normalization(df):

    scaler = MinMaxScaler([0, 1])
    scaler.fit(df)
    df_n = scaler.transform(df)
    df_n = pd.DataFrame(df_n, columns=df.columns)

    return df_n

def calc_correct(data, label):

    correct = 0
    for i in range(len(data)):
        data_class = aqi_class(data[i])
        label_class = aqi_class(label[i])
        if data_class == label_class:
            correct += 1

    return correct

def calc_winddirection_onehot(value):

    if value == 9:
        idx = 5
    elif value == 13:
        idx = 6
    elif value == 14:
        idx = 7
    elif value == 23:
        idx = 8
    elif value == 24:
        idx = 9
    else:
        idx = value

    onehot = [0]*10
    onehot[idx] = 1
    return onehot

def winddirection_onehot(df):
    df = df.fillna(0.0)
    df = df.astype("int64", copy=False)
    df = df.values
    df = list(map(lambda x: calc_winddirection_onehot(x), df))
    columns = ["wind_direction_"+str(i).zfill(2) for i in range(10)]
    df = pd.DataFrame(df, columns=columns).astype("float", copy=False)
    return df, columns

def weather_onehot(df):
    df = df.fillna(17.0)
    df = df.astype("int64", copy=False)
    df = df.values
    df = list(map(lambda x: calc_weather_onehot(x), df))
    columns = ["weather_"+str(i).zfill(2) for i in range(18)]
    df = pd.DataFrame(df, columns=columns).astype("float", copy=False)
    return df, columns

def calc_weather_onehot(value):
    onehot = [0]*18
    onehot[value] = 1
    return onehot

def ignore_aqi_error(df):
    return df

def data_interpolate(df):
    return df.interpolate(limit_direction='both')

def get_aqi_series(data, sid, attribute):
    return list(data[data["sid"] == sid][attribute])

def get_meteorology_series(data, did, attribute):
    return list(data[data["did"] == did][attribute])

def get_road_data(data, sid, attribute):
    return float(data[data["sid"] == sid][attribute])

def get_poi_data(data, sid, attribute):
    return float(data[data["sid"] == sid][attribute])

def get_road_over_the_city(city):

    with open("database/city/city_" + city + ".csv", "r") as cityfile:
        minlat, minlon, maxlat, maxlon = cityfile.readlines()[1].strip().split(",")
        api = overpy.Overpass()
        #result = api.query("way(" + minlat + "," + minlon + "," + maxlat + "," + maxlon + ");out;")  # (minlat, minlon, maxlat, maxlon)
        result = api.parse_xml(open("rawdata/osm/large/osm_large_"+city).read())
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

    return road


def get_grid_id(city, lat, lon):

    # load data (gid, minlat, minlon, maxlat, maxlon)
    data = open("database/grid/grid_"+city+".csv", "r").readlines()[1:]
    data = list(map(lambda x: x.strip().split(","), data))
    maxlon = sorted(list(set(map(lambda x: float(x[4]), data))))[-1]
    lon_list = sorted(list(set(map(lambda x: float(x[2]), data))))
    lon_list.append(maxlon)
    maxlat = sorted(list(set(map(lambda x: float(x[3]), data))))[-1]
    lat_list = sorted(list(set(map(lambda x: float(x[1]), data))))
    lat_list.append(maxlat)

    mn = 0
    mx = len(lon_list)-1
    while (mx-mn) > 1:
        half = mn+int((mx-mn)/2)
        if lon_list[half] < lon:
            mn = half
        else:
            mx = half
    data = [x for x in data if x[2] == str(lon_list[mn])]
    data = [x for x in data if x[4] == str(lon_list[mx])]

    mn = 0
    mx = len(lat_list)-1
    while (mx-mn) > 1:
        half = mn+int((mx-mn)/2)
        if lat_list[half] < lat:
            mn = half
        else:
            mx = half
    data = [x for x in data if x[1] == str(lat_list[mn])]
    data = [x for x in data if x[3] == str(lat_list[mx])]

    return data[0][0]

def get_dist_angle(lat1, lon1, lat2, lon2, ellipsoid=None):

    ELLIPSOID_GRS80 = 1  # GRS80
    ELLIPSOID_WGS84 = 2  # WGS84

    GEODETIC_DATUM = {
        ELLIPSOID_GRS80: [
            6378137.0, 
            1 / 298.257222101,  
        ],
        ELLIPSOID_WGS84: [
            6378137.0, 
            1 / 298.257223563,
        ],
    }


    ITERATION_LIMIT = 1000


    if lat1 == lat2 and lon1 == lon2:
        return {
            'distance': 0.0,
            'azimuth1': 0.0,
            'azimuth2': 0.0,
        }


    a, f = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - f) * a

    x1 = radians(lat1)
    x2 = radians(lat2)
    y1 = radians(lon1)
    y2 = radians(lon2)

  
    U1 = atan((1 - f) * tan(x1))
    U2 = atan((1 - f) * tan(x2))

    sinU1 = sin(U1)
    sinU2 = sin(U2)
    cosU1 = cos(U1)
    cosU2 = cos(U2)


    L = y2 - y1


    w = L

  
    for i in range(ITERATION_LIMIT):
        sinw = sin(w)
        cosw = cos(w)
        sinz = sqrt((cosU2 * sinw) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosw) ** 2)
        cosz = sinU1 * sinU2 + cosU1 * cosU2 * cosw
        z = atan2(sinz, cosz)
        sink = cosU1 * cosU2 * sinw / sinz
        cos2k = 1 - sink ** 2
        cos2zm = cosz - 2 * sinU1 * sinU2 / cos2k
        C = f / 16 * cos2k * (4 + f * (4 - 3 * cos2k))
        omega = w
        w = L + (1 - C) * f * sink * (z + C * sinz * (cos2zm + C * cosz * (-1 + 2 * cos2zm ** 2)))


        if abs(w - omega) <= 1e-12:
            break
    else:

        return None


    u2 = cos2k * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    dz = B * sinz * (cos2zm + B / 4 * (cosz * (-1 + 2 * cos2zm ** 2) - B / 6 * cos2zm * (-3 + 4 * sinz ** 2) * (-3 + 4 * cos2zm ** 2)))


    s = b * A * (z - dz)

 
    k1 = atan2(cosU2 * sinw, cosU1 * sinU2 - sinU1 * cosU2 * cosw)
    k2 = atan2(cosU1 * sinw, -sinU1 * cosU2 + cosU1 * sinU2 * cosw) + pi

    if (k1 < 0):
        k1 = k1 + pi * 2.0

    return {
        'distance': s,           
        'azimuth1': degrees(k1),
        'azimuth2': degrees(k2), 
    }