import torch
import os
import argparse
import time

import Interpolation as Interpolation
from lie_model import LieModel, bw_func_net, ba_func_net
from dataset import BaseDataset, EUROCDataset
from learning import write_parameters, Gyro_train, Acc_train
from Test import Test

def main():
    parser = argparse.ArgumentParser(description="Training e Testing del modello LieModel su EUROC")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory dei dati')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory per salvare modelli e risultati')
    parser.add_argument('--bw_weights_path', type=str, default='output/bw_weights.pth', help='Percorso per i pesi della rete bias giroscopio')
    parser.add_argument('--ba_weights_path', type=str, default='output/ba_weights.pth', help='Percorso per i pesi della rete bias accelerometro')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo da utilizzare (cuda o cpu)')
    args = parser.parse_args()

    # 1. Caricamento Dataset
    dataset_test = EUROCDataset(args.data_dir, mode="test", percent_for_val=0.2, batch_size=1, loss_window=10, dt=0.01)

    # 2. Training
    # chiamare Gyro_train e Acc_train
    t_start = time.time()
    Gyro_train(dataset_train=dataset_test, dataset_val=dataset_test, output_dir=args.output_dir, bw_func_name="bw_func_net", integral_method="euler", device=args.device, lr=0.005, weight_decay=1e-6, epoch=1801)
    # con i pesi salvati per gyro addestriamo anche acc
    ba_func_name = "ba_func_net"
    ba_func = ba_func_net().to(args.device)
    ba_func.load_state_dict(torch.load(args.ba_weights_path, map_location=args.device))
    Acc_train(dataset_train=dataset_test, dataset_val=dataset_test, output_dir=args.output_dir, ba_func_name=ba_func_name, integral_method="euler", device=args.device, lr=0.005, weight_decay=1e-6, epoch=1801)
    t_end = time.time()
    print(f">>> Training completato in {(t_end - t_start)/60:.2f} minuti")

    # 3. Testing
    Test(dataset_test=dataset_test, output_dir=args.output_dir, bw_weights_path=args.bw_weights_path, ba_weights_path=args.ba_weights_path, device=args.device)


if __name__ == "__main__":
    main()
    