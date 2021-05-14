# -*- coding: utf-8 -*-

# Directory path
import sys
sys.path.append("/home")

# from python library
import pickle
import time
import torch
import optuna
import numpy as np
import pandas as pd

# from my library
from source.func import objective_ADAIN
from source.func import objective_AIREX
from source.func import objective_FNN
from source.func import evaluate_ADAIN
from source.func import evaluate_AIREX
from source.func import evaluate_FNN
from source.func import evaluate_KNN
from source.utility import get_dist_angle

'''
AAAI 2018 reproduce of ADAIN result
'''
def AAAI(TARGET, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("----------------")
        print("* [EXP AAAI18 (ADAIN)]: source/target {}, loop {}".format(TARGET, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/AAAI18/train_{}{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/AAAI18/test_{}{}".format(TARGET, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=args.TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/AAAI18_{}{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/AAAI18_{}{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/AAAI18_{}{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time()-start)/(60*60))))

    with open("result/AAAI18_{}.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))


def train1test19(SOURCE, TARGETs, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("*----------------")
        print("* [EXP train1 test19 (ADAIN)]: source {}, loop {}".format(SOURCE, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train1/train_{}{}".format(SOURCE, SOURCE, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=args.TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Train1{}.pkl".format(SOURCE, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Train{}1.csv".format(SOURCE, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Train{}1.pkl".format(SOURCE, str(loop)), "rb"))

        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Train1/test_{}{}".format(SOURCE, SOURCE, str(loop)), fp)

        # evaluate
        print("* evaluate in {} (loop {})".format(SOURCE, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_tmp.append(rmse)
        accuracy_tmp.append(accuracy)

        for TARGET in TARGETs:

            # save dataset path
            with open("tmp/testPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Train1/test_{}{}".format(SOURCE, TARGET, str(loop)), fp)

            # evaluate
            print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
            rmse, accuracy = evaluate_ADAIN(model_state_dict)
            rmse_tmp.append(rmse)
            accuracy_tmp.append(accuracy)


        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    # to output
    rmse = np.average(np.array(rmse_list), axis=0)
    rmse = list(map(lambda x: str(x), rmse))
    tmp = rmse_list.copy()
    rmse_list = []
    for tmp_i in tmp:
        rmse_list.append(list(map(lambda x: str(x), tmp_i)))

    accuracy = np.average(np.array(accuracy_list), axis=0)
    accuracy = list(map(lambda x: str(x), accuracy))
    tmp = accuracy_list.copy()
    accuracy_list = []
    for tmp_i in tmp:
        accuracy_list.append(list(map(lambda x: str(x), tmp_i)))

    c = pd.read_csv("rawdata/zheng2015/city.csv", index_col="name_english")
    lat_local = c.at[SOURCE, "latitude"]
    lon_local = c.at[SOURCE, "longitude"]
    distance = list()
    for TARGET in TARGETs:
        lat = c.at[TARGET, "latitude"]
        lon = c.at[TARGET, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open("result/{}Train1.csv".format(SOURCE), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), ",".join(rmse_list[loop])))
        result.write("average,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(SOURCE, ",".join(TARGETs)))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), ",".join(accuracy_list[loop])))
        result.write("average,{}\n".format(",".join(accuracy)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def train1test1(SOURCEs, TARGET, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        # to evaluate
        rmse_tmp = list()
        accuracy_tmp = list()

        for SOURCE in SOURCEs:

            print("*----------------")
            print("* [EXP train1 test1 (ADAIN)] : source {}, target {}, loop {}".format(SOURCE, TARGET, str(loop)))

            # save dataset path
            with open("tmp/testPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Test1/test_{}{}".format(TARGET, SOURCE, str(loop)), fp)
            with open("tmp/trainPath.pkl", "wb") as fp:
                pickle.dump("dataset/{}Test1/train_{}{}".format(TARGET, SOURCE, str(loop)), fp)

            # save hyper parameters
            with open("tmp/hyperParameters.pkl", "wb") as fp:
                pickle.dump(args, fp)

            # training & parameter tuning by optuna
            # -- activate function, optimizer, eopchs, batch size
            study = optuna.create_study()
            study.optimize(objective_ADAIN, n_trials=args.TRIAL)

            # save best model
            model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
            pickle.dump(model_state_dict, open("model/{}Test1_{}{}.pkl".format(TARGET, SOURCE, str(loop)), "wb"))

            # save train log
            log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
            log.to_csv("log/{}Test1_{}.csv".format(TARGET, SOURCE, str(loop)), index=False)

            # load best model
            model_state_dict = pickle.load(open("model/{}Test1_{}{}.pkl".format(TARGET, SOURCE, str(loop)), "rb"))

            # evaluate
            print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
            rmse, accuracy = evaluate_ADAIN(model_state_dict)
            rmse_tmp.append(rmse)
            accuracy_tmp.append(accuracy)

        rmse_list.append(rmse_tmp)
        accuracy_list.append(accuracy_tmp)
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    # to output
    rmse = np.average(np.array(rmse_list), axis=0)
    rmse = list(map(lambda x: str(x), rmse))
    tmp = rmse_list.copy()
    rmse_list = []
    for tmp_i in tmp:
        rmse_list.append(list(map(lambda x: str(x), tmp_i)))

    accuracy = np.average(np.array(accuracy_list), axis=0)
    accuracy = list(map(lambda x: str(x), accuracy))
    tmp = accuracy_list.copy()
    accuracy_list = []
    for tmp_i in tmp:
        accuracy_list.append(list(map(lambda x: str(x), tmp_i)))

    c = pd.read_csv("rawdata/zheng2015/city.csv", index_col="name_english")
    lat_local = c.at[TARGET, "latitude"]
    lon_local = c.at[TARGET, "longitude"]
    distance = list()
    for SOURCE in SOURCEs:
        lat = c.at[SOURCE, "latitude"]
        lon = c.at[SOURCE, "longitude"]
        result = get_dist_angle(lat1=lat_local, lon1=lon_local, lat2=lat, lon2=lon)
        distance.append(str(result["distance"]/1000.0))

    with open("result/{}Test1.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(TARGET, ",".join(SOURCEs)))
        for loop in range(len(rmse_list)):
            result.write("exp{},NULL,{}\n".format(str(loop), ",".join(rmse_list[loop])))
        result.write("average,NULL,{}\n".format(",".join(rmse)))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{},{}\n".format(TARGET, ",".join(SOURCEs)))
        for loop in range(len(accuracy_list)):
            result.write("exp{},NULL,{}\n".format(str(loop), ",".join(accuracy_list[loop])))
        result.write("average,NULL,{}\n".format(",".join(accuracy)))
        result.write("--------------------------------------------\n")
        result.write("Distance\n")
        result.write("----------\n")
        result.write("distance,0.0,{}\n".format(",".join(distance)))

def train5test1_adain(TARGET, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("----------------")
        print("* [EXP train5 test1 (ADAIN)] : target {}, loop {}".format(TARGET, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test5/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test5/test{}".format(TARGET, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=args.TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test5_adain_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test5_adain_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test5_adain_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test5_adain.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))


def train5test1_proposal(TARGET, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("----------------")
        print("* [EXP train5 test1 (Proposal)] : target {}, loop {}".format(TARGET, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test5_city/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test5_city/test{}".format(TARGET, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_AIREX, n_trials=args.TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test5_proposal_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test5_proposal_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test5_proposal_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_AIREX(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test5_proposal.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))


def train19test1_adain(TARGET, args):

    # # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):

        start = time.time()
        print("----------------")
        print("* [EXP train19 test1 (ADAIN)]: target {}, loop {}".format(TARGET, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_ADAIN, n_trials=args.TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test19_adain_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test19_adain_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test19_adain_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_ADAIN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19_adain.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))


def train19test1_proposal(TARGET, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")
        print("* [EXP train19 test1 (Proposal)] : target {}, loop {}".format(TARGET, str(loop)))

        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19_city/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19_city/test{}".format(TARGET, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        """
        # training & parameter tuning by optuna: activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_AIREX, n_trials=args.TRIAL)

        # save model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test19_proposal_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test19_proposal_{}.csv".format(TARGET, str(loop)), index=False)
        """

        # load model
        model_state_dict = pickle.load(open("model/{}Test19_proposal_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_AIREX(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19_proposal.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))


def train19test1_fnn(TARGET, args):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")
        print("* [Exp train19 test1 (FNN)] : target {}, loop {}".format(TARGET, str(loop)))

        # save dataset path
        with open("tmp/trainPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/train{}".format(TARGET, str(loop)), fp)
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # save hyper parameters
        with open("tmp/hyperParameters.pkl", "wb") as fp:
            pickle.dump(args, fp)

        # training & parameter tuning by optuna
        # -- activate function, optimizer, eopchs, batch size
        study = optuna.create_study()
        study.optimize(objective_FNN, n_trials=args.TRIAL)

        # save best model
        model_state_dict = torch.load("tmp/{}_model.pkl".format(str(study.best_trial.number).zfill(4)))
        pickle.dump(model_state_dict, open("model/{}Test19_fnn_{}.pkl".format(TARGET, str(loop)), "wb"))

        # save train log
        log = pickle.load(open("tmp/{}_log.pkl".format(str(study.best_trial.number).zfill(4)), "rb"))
        log.to_csv("log/{}Test19_fnn_{}.csv".format(TARGET, str(loop)), index=False)

        # load best model
        model_state_dict = pickle.load(open("model/{}Test19_fnn_{}.pkl".format(TARGET, str(loop)), "rb"))

        # evaluate
        print("* evaluate in {} (loop {})".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_FNN(model_state_dict)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19_fnn.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))


def knn19(TARGET):

    # to evaluate
    rmse_list = list()
    accuracy_list = list()

    for loop in range(1, 4):
        start = time.time()
        print("----------------")

        # save dataset path
        with open("tmp/testPath.pkl", "wb") as fp:
            pickle.dump("dataset/{}Test19/test{}".format(TARGET, str(loop)), fp)

        # evaluate
        print("* [EXP KNN]: target {}, loop {}".format(TARGET, str(loop)))
        rmse, accuracy = evaluate_KNN(K=3)
        rmse_list.append(rmse)
        accuracy_list.append(accuracy)

        # time
        print("time = {} [hour/loop]".format(str((time.time() - start) / (60 * 60))))

    with open("result/{}Test19KNN.csv".format(TARGET), "w") as result:
        result.write("--------------------------------------------\n")
        result.write("RMSE\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(rmse_list)):
            result.write("exp{},{}\n".format(str(loop), str(rmse_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(rmse_list)))))
        result.write("--------------------------------------------\n")
        result.write("Accuracy\n")
        result.write("----------\n")
        result.write("city,{}\n".format(TARGET))
        for loop in range(len(accuracy_list)):
            result.write("exp{},{}\n".format(str(loop), str(accuracy_list[loop])))
        result.write("average,{}\n".format(str(np.average(np.array(accuracy_list)))))

