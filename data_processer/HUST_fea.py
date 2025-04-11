'''
该代码用于将../data/HUST_data,最后一列加入相应rul,以适配模型对寿命预测的需求
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
        

def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

def add_rul(path_csv, path_pkl, root):
    '''
    在path_csv下的csv文件储存的数据的最后一列加入相应rul,以适配模型对寿命预测的需求

    param:
        path_pkl: 原始数据路径
        path_csv: 处理后带特征的路径
        root: 新的数据集保存路径
    '''

    # 将path_pkl下的.pkl文件名拿出来，并排序
    pkl_list = os.listdir(path_pkl)
    pkl_list = sorted(pkl_list, key=lambda x:int(x.split('-')[0])*10 + int(x[-5]))
    pkl_name = []
    for name in pkl_list:
        pkl_name.append(name[:-4])

    for name in pkl_name:
        dict_pkl = load_obj(f'{path_pkl}/{name}')[name]
        data_rul = dict_pkl['rul']      # 每个循环的剩余寿命
        data_pkl = dict_pkl['data']     # 所有循环的数据
        cycle_num = len(data_pkl)

        csv_filename = name + '.csv'
        csv_path =  os.path.join(path_csv,csv_filename)
        if not os.path.isfile(csv_path):
            print(f"文件：{csv_path}不存在")
        
        df_csv = pd.read_csv(csv_path)
        csv_Q = df_csv['capacity']      # 每个循环对应的容量

        Q_pkl = []
        for i in range(1,cycle_num+1):
            cycle_data = data_pkl[i]
            cycle_Q_pkl = cycle_data['Capacity (mAh)'].max() * 0.001
            Q_pkl.append(cycle_Q_pkl)

        csv_rul = []
        for capacity in csv_Q:
            index = np.argmin(np.abs(np.array(Q_pkl) - capacity))
            tmp_rul = data_rul[index+1]
            csv_rul.append(tmp_rul)
        
        if len(csv_rul) != len(csv_Q):
            print('csv_rul长度不一致')

        df_csv['rul'] = csv_rul
        new_csv_path = os.path.join(root,csv_filename)
        df_csv.to_csv(new_csv_path, index=False)
    

if __name__ == '__main__':
    path_csv = 'PINN4SOH/data/HUST data'
    path_pkl = '../bishe_data/raw_data/HUST_data'
    path_data = '../bishe_data/data/HUST_data'
    add_rul(path_csv, path_pkl, path_data)

