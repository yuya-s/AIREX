## Requiement
- Python 3.7. We did not test other version
- Pytorch 1.1. We did not test other version
- If you further try other parameter setting, you need to collect raw data that are fully available online. We cannot put raw data due to the size limitation.

## Dataset generation
- python source/build.py
  - (we put the extacted data in database) #extract necessary data from raw data in /rawdata and store extracted data to /database
  - arrange data in /database and store arranged data to /datatmp for creating training data
    - PATH_ROOT_SRC : root of codes. default is src/build.py
	- ATTRIBUTE : target attribute. default is pm25
	- TIME_LENGTH : period of data. default is 24*30*4 (i.e., 4 months)
	- LSTM_DATA_WIDTH : input size of LSTM. default is 24 (i.e., 1 day)

- python source/makedataset.py --DATASET "<dataset_name>" 
  - generate dataset from data in /datatmp 
  - Option (--DATASET "<dataset_name>") 
    - AAAI : for reproducing AAAI18 (ADAIN) experiment
    - Train1 : a single source city to 19 cities
    - Test1 : each source city to a single city
    - Test5 : test for 5 closest cities
    - TestFar : test for far cities (in Beijing and Tenjian, eight cities in the sourthern area and in Shenzhen and Guanzhou, 12 cities in the nothern area)
    - Test19 : Test for all cities

## Experiment
- python source/main.py
  - Specify --EXP "<exp_name>"
    - AAAI18 : reproduce of AAAI18 experiement
    - Train1 : Test ADAIN trained by a single source city and infer 19 cities
    - Test1 : Test ADAIN trained by 19 source cities to a single city
    - Test5_adain : Test ADAIN for 5 closest cities
    - Test19_adain : Test ADAIN for 19 source cities
    - Test19_proposal : Test AIREX for 19 source cities
    - Test19_fnn : Test FNN for 19 source cities
    - Test19_knn : Test KNN for 19 source cities

  - Specify hyperparameters
    - EPOCHs (int) : The number of epochs (Default 200)
    - BATCH_SIZE (int) : Batch size (Default 32)
    - LEARNING_RATE (float) :  Learning rate (Default 0.0005)
    - WEIGHT_DECAY (float) : parameters for L2 regularization (Default 0.0)
    - ALPHA (float) : Weight for L_m  of loss fucntion (Default 0.5)
    - GAMMA (float) : Weight for L_d  of loss fucntion (Default 1.0)
    - ETA (float) : Weigth for regularization of loss fucntion (Default 1.0)

