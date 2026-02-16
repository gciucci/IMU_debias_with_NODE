import torch
import time
import Interpolation as Interpolation
from learning import Gyro_train, Acc_train
from Test import test_gyro, test_acc
from dataset import EUROCDataset
from lie_model import bw_func_net, ba_func_net
def main():
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
    Gyro_train(dataset_train, dataset_val, "results/Euroc_master")
    #Acc train
    Acc_train(dataset_train, dataset_train,"results/Euroc_master")

    #test
    test_gyro(dataset_test,"{output_dir}/final_model.pt")
    test_acc(dataset_test,"{output_dir}/final_model.pt")






















if __name__ == "__main__":
    main()