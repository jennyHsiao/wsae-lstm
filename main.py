# @Author: Jenny Hsiao
# @Date:   2019-09-05T10:11:48+08:00
# @Email:  jenny.hsaio@asmpt.com
# @Filename: main.py
# @Last modified by:   Jenny Hsiao
# @Last modified time: 2019-09-05T10:11:54+08:00


import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.stacked_autoencoder import StackedAutoEncoder
from models.wavelet import waveletSmooth

raw_xl = pd.ExcelFile('data/raw/raw_data.xlsx')


def get_date(ini_date, n_month):
    x = datetime.datetime.strptime(ini_date, '%Y%m%d')

    n_year = int(n_month/12)
    n_month = n_month%12

    if(x.month + n_month) >= 12:
        n_year = n_year+1
        n_month = max(1, (x.month+n_month) - 12)
        nextmonthdate = x.replace(year=x.year+(n_year), month=n_month)

    else:
        nextmonthdate = x.replace(month=x.month+n_month, year=x.year+n_year)

    return nextmonthdate.replace(day=1).strftime('%Y%m%d')

i=0
step_month = 3
for sheet_name in raw_xl.sheet_names[::2]:
    i=0
    '''
        there are 6 sheets in rawdata.xls
        > experiment in order of the sheet name
        > 
    '''

    print("Analysis on ", sheet_name)

    # load raw data
    data_master = raw_xl.parse(sheet_name)

    # normalized scaler
    scaler = MinMaxScaler()
    
    # use 2.5 years of data at a time; 2y for train, 3m for valid, 3m for test
    # the first column is time index
    time_id = data_master.columns[0]

    init_date_time_str= str(data_master[time_id][i])
    timestampStr = get_date(init_date_time_str, 24)
    print("[%s] to [%s]  as  a train set" % (init_date_time_str, timestampStr))
    subset = data_master[ (data_master[time_id]>=int(init_date_time_str))& (data_master[time_id]<int(timestampStr))]
    # normalize
    # normalized based only upon the minimum-maximum values of their corresponding training set
    # in order to eliminamte any prior knowledge of overall scale would occur in real-time prediction
    scaler.fit(subset)
    
    train_set = pd.DataFrame(scaler.transform(subset))

    i = list(train_set.index)[-1]
    print(data_master.shape)
    init_date_time_str = str(data_master[time_id][i+1])
    print(init_date_time_str)
    timestampStr = get_date(init_date_time_str, 3)
    print("[%s] to [%s]  as  a valid set" % (init_date_time_str, timestampStr))
    subset = data_master[ (data_master[time_id]>int(init_date_time_str)) & (data_master[time_id]<int(timestampStr))]
    # normalize
    valid_set = pd.DataFrame(scaler.transform(subset))
    

    i = list(valid_set.index)[-1]
    init_date_time_str = str(data_master[time_id][i+1])
    print(init_date_time_str)
    timestampStr = get_date(init_date_time_str, 3)
    print("[%s] to [%s]  as  a test set" % (init_date_time_str, timestampStr))
    subset = data_master[ (data_master[time_id]>int(init_date_time_str)) & (data_master[time_id]<int(timestampStr))]
    # normalize
    test_set = pd.DataFrame(scaler.transform(subset))


    ################################################################################################################
    #################                                                                              #################
    #################   (1) Wavelet Transform                                                      #################
    #################                                                                              #################
    ################################################################################################################

    # print(test_set)
    # wt_data = train_set.copy()
    # for col in train_set.columns:
    #     print(col, len(waveletSmooth(train_set[col], level=1)))
    #     wt_data[col] = waveletSmooth(train_set[col], level=1)[-len(wt_data[col]):]


    train_set_WT = np.copy(train_set)
    for i in range(train_set.shape[1]):
            train_set_WT[:,i] = waveletSmooth(train_set.values[:,i], level=1)[-len(train_set):]
    print(train_set_WT)

    temp = np.copy(train_set)
    valid_set_WT = np.copy(valid_set)
    for j in range(valid_set.shape[0]):
        #first concatenate train with the latest validation sample
        temp = np.append(temp, np.expand_dims(valid_set.values[j,:], axis=0), axis=0)
        for i in range(valid_set.shape[1]):
            valid_set_WT[j,i] = waveletSmooth(temp[:,i], level=1)[-1]

    temp_train = np.copy(train_set)
    temp_val = np.copy(valid_set)
    temp = np.concatenate((temp_train, temp_val))
    test_set_WT = np.copy(test_set)
    for j in range(test_set.shape[0]):
        #first concatenate train and validate with the latest test sample
        temp = np.append(temp, np.expand_dims(test_set.values[j,:], axis=0), axis=0)
        for i in range(test_set.shape[1]):
            test_set_WT[j,i] = waveletSmooth(temp[:,i], level=1)[-1]

    

    ################################################################################################################
    #################                                                                              #################
    #################   (2) Stacked Autoencoder                                                    #################
    #################                                                                              #################
    ################################################################################################################
    train_data = torch.tensor(train_set_WT).float()
    print("train data shape = %s" % str(train_data.shape))
    
    train_ae = StackedAutoEncoder(train_data, 19)#.cuda()
    train_ae.train()
    print(train_ae(train_data))
    


    break
