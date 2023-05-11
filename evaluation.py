"""# **Evaluation**

## Functions for evaluation
"""
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Union
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from data_loder import DatasetLoader


class Evaluation():
    def __init__(self,
                 model,
                 model_n: str,
                 model_params: Union[dict, str],
                 data_params: str,
                 data_slice: tuple = (0, 501),
                 file_name: str ='data.csv'
                ):
        
        device = torch.device("cpu") 
        
        self.model = model
        self.model_n = model_n
        self.model = self.model.to(device) 
       
        self.model_params = model_params
        self.data_params = data_params
        self.data_slice = data_slice
        self.file_name = file_name
        
        self._preprocessing_dataset()
        self._forcast()
        self._conf()
        
        
    def _preprocessing_dataset(self):
        ##################
        # load the parameters
        ##################
        with open(self.data_params, 'r') as openfile:
            dicti_d = json.load(openfile)
            
        self.std_Szeged = dicti_d["Szeged"]["std"]
        self.mean_Szeged = dicti_d["Szeged"]["mean"]

        
        
        if isinstance(self.model_params, str):
            with open(self.model_params, 'r') as openfile:
                dicti_m = json.load(openfile)

            self.past_len = dicti_m["node_features"]
            self.target_len = dicti_m["target_len"] 

        elif isinstance(self.model_params, dict):
            self.past_len = self.model_params["node_features"]
            self.target_len = self.model_params["target_len"]
                          

        ##################
        # create datasets: StaticGraphTemporalSignal
        ##################
        if isinstance(self.file_name, str):
            # if test is a csv:
            # the first int(lag) element hasn't a past -> hasn't a forecast
            loader = DatasetLoader(data=self.file_name,
                                   data_params=self.data_params,
                                   data_slice=self.data_slice)

            self.test_data = loader.get_dataset(lags=self.past_len)
        
        ##################
        # load datas and dates
        ##################
        arr = np.loadtxt(self.file_name,  
                          usecols = (np.r_[1:13]), 
                          # np.r_ generates an array of indices
                          delimiter=",")

        place_code = arr[0,:].astype(int)

        Szeged_colum_num = int(np.where(place_code == 2275)[0])  # Szeged node_id is 2275


        start, end = self.data_slice
        value_szeged = arr[start:end, Szeged_colum_num] 

        date = np.loadtxt(self.file_name,
                      dtype='str',
                      usecols = 0,
                      skiprows=1,
                      delimiter=",")

        self.actual_dates = date[start-1:end-1]
        
        ##############
        #### make a date-Szeged_measured table; for all day
        ##############

        self.Table_date_value = pd.DataFrame(data=value_szeged, index =self.actual_dates, columns=['Szeged_measured'])
        
         
        ##############
        #### make a today-target_dates table; only from today
        ##############
        table= np.empty([len(self.actual_dates)-self.target_len, self.target_len], dtype=object)

        for i, date in enumerate(self.actual_dates):
            if i >= self.past_len-1:
                if i < len(self.actual_dates)-self.target_len:
                    table[i] = self.actual_dates[i+1: i+self.target_len+1]


        self.Table_target_dates = pd.DataFrame(data=table, index =self.actual_dates[0:len(self.actual_dates)-self.target_len], columns=['1d', '2d','3d','4d','5d','6d','7d'] )
        
        self.Table_target_dates = self.Table_target_dates.iloc[self.past_len-1:]
        


    def _forcast(self):           
        num_test_days = len(list(self.test_data))
         
        Szeged_hat = np.zeros((num_test_days, self.target_len ))  # columns are the time series of each forecast (more days forcasting)
        Szeged_measured = np.zeros((num_test_days, self.target_len))
        self.t = np.array([])
        
        ################
        # evaluate model
        ################
        h, c = None, None
        for time, snapshot in enumerate(self.test_data):
            y_hat, idx_Szeged = self.model(x=snapshot.x,
                                       edge_index=snapshot.edge_index,
                                       edge_weight=snapshot.edge_attr,
                                       h=h,
                                       c=c)
            

            Szeged_hat[time] = y_hat.detach().numpy().T

            Szeged_measured[time]  = snapshot.y[idx_Szeged, :].detach().numpy()

            self.t = np.append(self.t, time)
        
        ###################
        # restandardization
        ###################
        self.hat = (Szeged_hat * self.std_Szeged) + self.mean_Szeged
        self.measured = np.round((Szeged_measured * self.std_Szeged) + self.mean_Szeged)
        
        self.Table_Szeged_hat = pd.DataFrame(data=self.hat, 
                                             index =pd.to_datetime(self.actual_dates[self.past_len-1:len(self.actual_dates)-self.target_len]), 
                                                                 
                                             columns=['1d', '2d','3d','4d','5d','6d','7d'] )

        self.Table_Szeged_measured =pd.DataFrame(data=self.measured, 
                                                 index =pd.to_datetime(self.actual_dates[self.past_len-1:len(self.actual_dates)-self.target_len]), 
                                                 
                                                 columns=['1d', '2d','3d','4d','5d','6d','7d'] )
        
        
    
    def _conf(self):
        value, hat, measured = self.Table_date_value, self.Table_Szeged_hat, self.Table_Szeged_measured
        
        raw, col = hat.shape
        df = pd.DataFrame()
        for i in range(raw-15):
            df1 = pd.DataFrame([list(abs(hat-measured)[i:i+15].sum()/15)], 
                               index=[hat.index[i+15]], 
                               columns=['1d', '2d','3d','4d','5d','6d','7d'])
            df = pd.concat([df, df1])
            
        self.Table_conf_int = df
        
        
    def plot(self,
             day: int,
             verbose: bool = False):

        self.hat_conf_up =self.Table_Szeged_hat[15:] + self.Table_conf_int
        self.hat_conf_down= self.Table_Szeged_hat[15:]- self.Table_conf_int
        
        self.h = self.Table_Szeged_hat[15:]
        self.m = self.Table_Szeged_measured[15:]
        
        
        col = str(day)+'d'
        self.h[col].plot(label='Forcast' + str(day))
        self.m[col].plot(label='Measured' + str(day))
        plt.fill_between(self.h.index, self.hat_conf_up[col], self.hat_conf_down[col], label='Confidence', alpha=0.2)
        plt.fill_between(self.h.index, self.m[col]+ day*5, self.m[col]- day*5, label='Required', alpha=0.2)
                
        plt.title(str(day) + 'day ahead forcasting')
        #plt.xlabel('Time')
        #plt.ylabel('Water level (cm)')
        plt.legend()
        
        plt.show()
        return plt
    
    def tab(self):
        return self.Table_date_value, self.Table_Szeged_hat, self.Table_Szeged_measured
    
    def errors(self): 
        difference = self.Table_Szeged_hat - self.Table_Szeged_measured
        measured = self.Table_Szeged_measured
        hat = self.Table_Szeged_hat
        
        row, col = difference.shape
        
        MAE = abs(difference).sum()/row
        MRSE = ((difference**2).sum()/row)**0.5
        
        R2 = 1- ((difference**2).sum() / ((measured-measured.mean())**2).sum() )
        W = 1- ((difference**2).sum() / ((abs(hat- measured.mean()) + abs(difference))**2).sum())
        
        with open(str(self.model_n) + "_error.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in MAE.values:
                writer.writerow([i])
            for i in MRSE.values:
                writer.writerow([i])    
            for i in R2.values:
                writer.writerow([i])
            for i in W.values:
                writer.writerow([i])
      
        return MAE,MRSE, R2, W