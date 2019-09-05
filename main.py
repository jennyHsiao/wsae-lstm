# @Author: Jenny Hsiao
# @Date:   2019-09-05T10:11:48+08:00
# @Email:  jenny.hsaio@asmpt.com
# @Filename: main.py
# @Last modified by:   Jenny Hsiao
# @Last modified time: 2019-09-05T10:11:54+08:00


import datetime
import pandas as pd

raw_xl = pd.ExcelFile('data/rsw/raw_data.xlsx')


def get_date(ini_date, n_month):
    x = datetime.datetime.strptime(ini_date, '%Y%m%d')

    n_year = int(n_month/12)
    n_month = n_month%12

    print(n_year, n_month)

    if(x.month + n_month) >= 12:
        n_year = n_year+1
        n_month = max(1, (x.month+n_month) - 12)
        print(">12", n_year, n_month)
        nextmonthdate = x.replace(year=x.year+(n_year), month=n_month)

    else:
        nextmonthdate = x.replace(month=x.month+n_month, year=x.year+n_year)


    return nextmonthdate.replace(day=1).strftime('%Y%m%d')

i=0
step_month = 3
for sheet_name in raw_xl.sheet_names[::2]:
    i=0
    print("Analysis on ", sheet_name)

    data_master = raw_xl.parse(sheet_name)

    # use 2.5 years of data at a time; 2y for train, 3m for valid, 3m for test
    # the first column is time index
    time_id = data_master.columns[0]

    init_date_time_str= str(data_master[time_id][i])
    timestampStr = get_date(init_date_time_str, 24)
    print("[%s] to [%s]  as  a train set" % (init_date_time_str, timestampStr))
    train_set = data_master[ (data_master[time_id]>=int(init_date_time_str))& (data_master[time_id]<int(timestampStr))]

    i = list(train_set.index)[-1]
    print(data_master.shape)
    init_date_time_str = str(data_master[time_id][i+1])
    print(init_date_time_str)
    timestampStr = get_date(init_date_time_str, 3)
    print("[%s] to [%s]  as  a valid set" % (init_date_time_str, timestampStr))
    valid_set = data_master[ (data_master[time_id]>int(init_date_time_str)) & (data_master[time_id]<int(timestampStr))]

    i = list(valid_set.index)[-1]
    init_date_time_str = str(data_master[time_id][i+1])
    print(init_date_time_str)
    timestampStr = get_date(init_date_time_str, 3)
    print("[%s] to [%s]  as  a test set" % (init_date_time_str, timestampStr))
    vald_set = data_master[ (data_master[time_id]>int(init_date_time_str)) & (data_master[time_id]<int(timestampStr))]
