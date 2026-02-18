import torch
import time
import Interpolation as Interpolation
from learning import Gyro_train, Acc_train
from Test import test
from dataset import EUROCDataset
from lie_model import bw_func_net, ba_func_net
def main():
    output_dir_gyro = "results/Euroc_master/gyro"
    output_dir_acc = "results/Euroc_master/acc"
    device = "cuda"
    dataset_parameters = {
        'dataset_name': 'EUROC',
        'data_dir': 'data/EUROC', # where are dataset located
        'data_cache_dir': 'data/EUROC', # where to save the preprocessed data
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'val_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy'
        ],
        # time_for_train: 50.0 # 
        'dt': 0.005, # time step
        'percent_for_val': 0.2, # the last 0.2 of the data is used for validation for each sequence
        'loss_window': 16, # size of the loss window CDE-RNN
        'batch_size': 1000, # number of time windows in each batch
        'sg_window_bw': 1,
        'sg_order_bw': 0,
        'sg_window_ba': 1,
        'sg_order_ba': 0,
        }
    dataset_train = EUROCDataset(**dataset_parameters, mode='train',recompute=False)
    dataset_val = EUROCDataset(**dataset_parameters, mode='val')
    dataset_test = EUROCDataset(**dataset_parameters, mode='test')

    #Gyro train
    Gyro_train(dataset_train, dataset_val, output_dir_gyro)
    #Acc train
    Acc_train(dataset_train, dataset_train, output_dir_acc)

    #test
    gyro_model_path = f"{output_dir_gyro}/final_model.pt"
    acc_model_path = f"{output_dir_acc}/final_model.pt"
    bw_func = bw_func_net().to(device)
    bw_func.load_state_dict(torch.load(gyro_model_path, weights_only=True)['func_bw_model'])
    ba_func = ba_func_net().to(device)
    ba_func.load_state_dict(torch.load(acc_model_path, weights_only=True)['func_ba_model'])
    test(dataset_test, device=device, integral_method="euler", bw_func=bw_func, ba_func=ba_func)