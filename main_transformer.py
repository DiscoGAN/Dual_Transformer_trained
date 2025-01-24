import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates  
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from garage_Transformer import cal_moving_ave, reframeDF, create_date_dic, final_test
from garage_Transformer import generate_dataset, final_test, test_analysis, generate_prediction_data
from model_base import Transformer_half_base
from model_modify import Transformer_half_modify
from model_weighted import auto_weighted
from scipy.stats import pearsonr


class config():
    def __init__(self):
        self.cycle_num_before = 12
        self.cycle_num_later = 5
        self.moving_length = 30
        self.interval_1 = 30
        self.interval_2 = 30
        self.is_mean = True
        self.is_train = True
        self.train_pro = 1
        self.input_size = 7
        self.batch_size = 256
        self.learning_rate = 0.0001
        self.num_epoch = 6000
        self.space = 730
        self.path_base = './model/transformer_base_6.pt'
        self.path_modify = './model/transformer_modify_6.pt'
        self.path_weight = './model/transformer_aw_6.pt'
        self.data_path = './finaldata.csv'
        self.test_path_1 = './test_2022.csv'
        self.test_path_2 = './test_2023.csv'
        self.test_length = 365*2
        self.start_year = 1981
        self.total_year = 100
        self.predict_year_start = 2022
        self.plot_start_year = self.predict_year_start - 1
        self.predict_long = 2
        self.n = 6


wl_p = config()

scaler = MinMaxScaler(feature_range=(0, 1))
date_original, dataset_original = generate_dataset(wl_p.data_path)
date, dataset = generate_dataset(wl_p.data_path)
np.save('./results/dataset_train.npy', dataset)
date = [datetime.strptime(d, '%Y-%m-%d').date() for d in date]
print(dataset.shape)
dataset = cal_moving_ave(dataset, wl_p.moving_length)
print(dataset.shape)


dataset = reframeDF(dataset, scaler)
date_test_1, dataset_test_1 = generate_dataset(wl_p.test_path_1)
dataset_test_1[:, 1] = dataset_test_1[:, 1] - 273.15
dataset_test_1[:, 2] = dataset_test_1[:, 2] - 273.15
date_test_2, dataset_test_2 = generate_dataset(wl_p.test_path_2)
date_test = np.concatenate((date_test_1, date_test_2), axis=0)
dataset_test = np.concatenate((dataset_test_1, dataset_test_2), axis=0)
np.save('./results/dataset_test.npy', dataset_test)
parameter_dict = {0:[7, 6, 1, 1], 1:[12, 7, 5, 4], 2:[12, 6, 10, 9], 3:[12, 6, 20, 13],\
                  4:[12, 5, 25, 17], 5:[12, 6, 25, 21], 6:[12, 5, 30, 30]}

wl_p.cycle_num_before = parameter_dict[wl_p.n][0]
wl_p.cycle_num_later = parameter_dict[wl_p.n][1]
wl_p.interval_1 = parameter_dict[wl_p.n][2]
wl_p.interval_2 = parameter_dict[wl_p.n][3]
transformer_base = Transformer_half_base(parameter_dict[wl_p.n])
transformer_modify = Transformer_half_modify(parameter_dict[wl_p.n])
aw = auto_weighted(wl_p)
criterion = torch.nn.MSELoss()         
optimizer_1 = torch.optim.Adam(transformer_base.parameters(), lr=wl_p.learning_rate)
optimizer_2 = torch.optim.Adam(transformer_modify.parameters(), lr=wl_p.learning_rate)
optimizer_3 = torch.optim.Adam(aw.parameters(), lr=wl_p.learning_rate)

transformer_base.load_state_dict(torch.load(wl_p.path_base))
transformer_base.eval()
transformer_modify.load_state_dict(torch.load(wl_p.path_modify))
transformer_modify.eval()
aw.load_state_dict(torch.load(wl_p.path_weight))
aw.eval()

wl_p.is_train = False
date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, wl_p)
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)
dataset_new = reframeDF(dataset_new, scaler)
print(dataset_new.shape)


results, targets, weights_1, weights_2 = final_test(dataset_new, wl_p, transformer_base, transformer_modify, aw, criterion, parameter_dict[wl_p.n])
np.save('./results/results.npy', results)
np.save('./results/targets.npy', targets)


error, error_pearson, plot_targets, plot_preds, plot_average, plot_original = test_analysis(date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler)
print(error)