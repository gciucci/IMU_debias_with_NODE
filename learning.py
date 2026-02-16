import torch
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import Interpolation as Interpolation
import lie_algebra as Lie
from lie_model import LieModel, bw_func_net, ba_func_net
from dataset import BaseDataset, EUROCDataset, TUMDataset

from SO3diffeq import odeint_SO3


def SO3log_loss(R_pred, R_gt, loss_type = torch.nn.MSELoss()):
   residual = Lie.SO3log(R_pred.transpose(-1,-2) @ R_gt)
   return loss_type(residual, torch.zeros_like(residual))

def Gyro_train(dataset_train: BaseDataset, dataset_val: BaseDataset, output_dir: str,  integral_method = "euler", device = "cuda", lr = 0.005, weight_decay=1e-6, epoch = 1801):
    
    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False} # whether to mask some missing data
    mask_flag = MASKFLAG[dataset_train.dataset_name]
    
    def compute_solution_and_loss(dataset, index):
        """Calcola la soluzione e il loss per un dato dataset e index."""
        (t_gt, X_gt, bwa_gt), t_odeint, coeff, Spline_time, _ = dataset.get_data_for_biasDy(index, device, mask_flag=mask_flag)
        y0, R0 = dataset.construct_init_forodeint(t_gt, X_gt, bwa_gt)
        Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
        model.set_u_func(Spline.evaluate)
        model.set_u_dot_func(Spline.derivative)
        model.set_R0(R0)
        solution, _ = odeint_SO3(model, y0, R0, t_odeint, method=integral_method)
        target = torch.cat([X_gt[...,:3,3], X_gt[...,:3,4]], dim=-1)
        loss = 1e6 * torch.nn.functional.mse_loss(solution[...,7:13], target)
        return solution, loss


    val_freq = 100
    gpu_memory = []
    print(">>> Inizio Training Giroscopio")

    ######### define model #########
    bw_func = bw_func_net().to(device)
    model = LieModel(bias_net_w=bw_func,device=device)
    best_loss = float('inf')
    best_model_state = None
    model = model.to(device)
    model.set_bw_zero = False
    model.set_ba_zero = True
    if model.biasfunc_a is not None:
            for param in model.biasfunc_a.parameters():
                param.requires_grad = False
  
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    for ep in range(epoch):
        optimizer.zero_grad()
        model.train()

        #caricamento dato casuale
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        _, train_loss = compute_solution_and_loss(dataset_train, index)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % val_freq == 0:
            with torch.no_grad(): #quando non faccio training evito di salvare i gradienti
                model.eval()
                val_loss_total = 0
                for idx in range(dataset_val.length()):
                    _, loss = compute_solution_and_loss(dataset_val, idx)
                    val_loss_total += loss.item()
    
            print(
                f"Epoch {ep:04d} | "
                f"train_loss: {train_loss.item():.4e} | "
                f"val_loss: {val_loss_total:.4e}"
            )

            if val_loss_total < best_loss:
                best_loss = val_loss_total
                best_model_state = {
                    'func_bw_model': model.biasfunc_w.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss
                }

        torch.cuda.synchronize()
        gpu_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

    # --- Salva modello finale / miglior modello ---
    save_dict = best_model_state if best_model_state is not None else {
        "func_bw_model": model.biasfunc_w.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": None,
    }

    torch.save(save_dict, f"{output_dir}/final_model.pt")
    print(f"Modello salvato in {output_dir}/final_model.pt")


    return gpu_memory

def Acc_train(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    output_dir: str,
    integral_method="euler",
    device="cuda",
    lr=0.005,
    weight_decay=1e-6,
    epoch=1801,
):
    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False}
    mask_flag = MASKFLAG[dataset_train.dataset_name]

    def compute_solution_and_loss(dataset, index):
        (t_gt, X_gt, bwa_gt), t_odeint, coeff, Spline_time, _ = \
            dataset.get_data_for_biasDy(index, device, mask_flag=mask_flag)

        y0, R0 = dataset.construct_init_forodeint(t_gt, X_gt, bwa_gt)

        Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
        model.set_u_func(Spline.evaluate)
        model.set_u_dot_func(Spline.derivative)
        model.set_R0(R0)

        solution, _ = odeint_SO3(
            model, y0, R0, t_odeint, method=integral_method
        )

        target = torch.cat([X_gt[..., :3, 3], X_gt[..., :3, 4]], dim=-1)
        loss = 1e6 * torch.nn.functional.mse_loss(solution[..., 7:13], target)
        return solution, loss

    val_freq = 100
    gpu_memory = []
    print(">>> Inizio Training Accelerometro")

    # -------- MODEL --------
    ba_func = ba_func_net().to(device)
    model = LieModel(bias_net_a=ba_func, device=device).to(device)

    model.set_bw_zero = True
    model.set_ba_zero = False

    # congela bias gyro
    if model.biasfunc_w is not None:
        for param in model.biasfunc_w.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=300, gamma=0.5
    )

    best_loss = float("inf")
    best_model_state = None

    # -------- TRAIN LOOP --------
    for ep in range(epoch):
        optimizer.zero_grad()
        model.train()

        index = torch.randint(0, dataset_train.length(), (1,)).item()
        _, train_loss = compute_solution_and_loss(dataset_train, index)

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % val_freq == 0:
            model.eval()
            val_loss_total = 0.0

            with torch.no_grad():
                for idx in range(dataset_val.length()):
                    _, loss = compute_solution_and_loss(dataset_val, idx)
                    val_loss_total += loss.item()

            print(
                f"Epoch {ep:04d} | "
                f"train_loss: {train_loss.item():.4e} | "
                f"val_loss: {val_loss_total:.4e}"
            )

            if val_loss_total < best_loss:
                best_loss = val_loss_total
                best_model_state = {
                    "func_ba_model": model.biasfunc_a.state_dict(),
                    "epoch": ep,
                    "best_loss": best_loss,
                }

        torch.cuda.synchronize()
        gpu_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

    # -------- SAVE --------
    save_dict = best_model_state if best_model_state is not None else {
        "func_ba_model": model.biasfunc_a.state_dict(),
        "epoch": epoch,
        "best_loss": None,
    }

    torch.save(save_dict, f"{output_dir}/final_model.pt")
    print(f"Modello salvato in {output_dir}/final_model.pt")

    return gpu_memory
