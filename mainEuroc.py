import torch
import time
import os

from dataset import EUROCDataset
from learning import Gyro_train, Acc_train


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "results/euroc_simplified"
    os.makedirs(output_dir, exist_ok=True)

    integral_method = "euler"
    epoch = 1801

    lr_gyro = 5e-3
    wd_gyro = 1e-6
    lr_acc = 5e-3
    wd_acc = 1e-6

    print("=== EUROC Bias Learning ===")
    print("Device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    dataset_params = dict(
        dataset_name="EUROC",
        data_dir="data/EUROC",
        data_cache_dir="data/EUROC",
        train_seqs=[
            "MH_01_easy",
            "MH_03_medium",
            "MH_05_difficult",
            "V1_02_medium",
            "V2_01_easy",
            "V2_03_difficult",
        ],
        val_seqs=[
            "MH_01_easy",
            "MH_03_medium",
            "MH_05_difficult",
            "V1_02_medium",
            "V2_01_easy",
            "V2_03_difficult",
        ],
        dt=0.005,
        percent_for_val=0.2,
        loss_window=16,
        batch_size=1000,
        sg_window_bw=1,
        sg_order_bw=0,
        sg_window_ba=1,
        sg_order_ba=0,
    )

    dataset_train = EUROCDataset(**dataset_params, mode="train", recompute=False)
    dataset_val = EUROCDataset(**dataset_params, mode="val")

    # -------------------------
    # Training
    # -------------------------
    t_start = time.time()

    print("\n--- Training Gyro Bias ---")
    bw_net = Gyro_train(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        output_dir=output_dir,
        bw_func_name="bw_func_net",
        integral_method=integral_method,
        device=device,
        lr=lr_gyro,
        weight_decay=wd_gyro,
        epoch=epoch,
    )

    print("\n--- Training Acc Bias ---")
    ba_net = Acc_train(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        outpur_dir=output_dir,
        bw_model=bw_net,
        ba_model_name="ba_func_net",
        integral_method=integral_method,
        device=device,
        lr=lr_acc,
        weight_decay=wd_acc,
        epoch=epoch,
    )

    t_end = time.time()
    print(f"\nTraining finished in {t_end - t_start:.1f} seconds")


if __name__ == "__main__":
    main()
