import torch
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import Interpolation as Interpolation
import lie_algebra as Lie
from lie_model import LieModel, bw_func_net, ba_func_net
from dataset import BaseDataset, EUROCDataset, TUMDataset

from SO3diffeq import odeint_SO3


def write_parameters(type_train: str, dataset: BaseDataset, output_dir: str, network_type: str, lr: float, weight_decay: float, epoch: int):
    if type_train not in ['Gyro', 'Acc']:
        raise ValueError("type should be 'Gyro' or 'Acc'")
    if type_train=='Gyro':
        file_name = "Gyro_parameters.yaml"
    else:
        file_name = "Acc_parameters.yaml"
    
    ## get training parameters and dataset parameters into yaml file
    train_parameters = {
        'dataset_name': dataset.dataset_name,
        'train_type': type_train,
        'network_type': network_type,
        'integral_method': 'euler',
        'loss_windows': dataset._loss_window,
        'lr': lr,
        'weight_decay': weight_decay,
        'epoch': epoch
    }
    dataset_parameters = {
        'dataset_name': dataset.dataset_name,
        'train_seqs': dataset.sequences,
        'sg_window_bw': dataset._sg_window_bw,
        'sg_order_bw': dataset._sg_order_bw,
        'sg_window_ba': dataset._sg_window_ba,
        'sg_order_ba': dataset._sg_order_ba,
    }
    
    output_dir = os.path.join(output_dir, "parameters", file_name)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as file:
        yaml.dump(train_parameters, file)
        yaml.dump(dataset_parameters, file)


def SO3log_loss(R_pred, R_gt, loss_type = torch.nn.MSELoss()):
   residual = Lie.SO3log(R_pred.transpose(-1,-2) @ R_gt)
   return loss_type(residual, torch.zeros_like(residual))

def Gyro_train(dataset_train: BaseDataset, dataset_val: BaseDataset, output_dir: str, bw_func_name: str,  integral_method = "euler", device = "cuda", lr = 0.005, weight_decay=1e-6, epoch = 1801):
    
    print(">>> Inizio Training Giroscopio")

    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False} # whether to mask some missing data
    mask_flag = MASKFLAG[dataset_train.dataset_name]

    bw_net = bw_func_net().to(device)
    model = LieModel(bias_net_w=bw_net,device=device).to(device)
    optimizer = torch.optim.Adam(model.bias_net_w.parameters(), lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter(os.path.join(output_dir, "logs_gyro"))

    for ep in range(epoch):
        optimizer.zero_grad()
        model.train()

        #caricamento dato casuale
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, Spline_time, induces = dataset_train.get_data_for_biasDy(index, device, mask_flag=mask_flag)
        y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
        
        Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
        model.u_func, model.u_dot_func = Spline.evaluate, Spline.derivative

        _, R_sol = odeint_SO3(model, y0_batch, R0_batch, t_odeint, rtol=1e-7, atol=1e-9, method=integral_method)
        
        loss = 1e6 * SO3log_loss(R_sol, X_gt_batch[..., :3, :3])
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Epoch {ep}, Loss: {loss.item():.4f}")
            writer.add_scalar("Gyro_train_loss", loss.item(), ep)
    
    torch.save(bw_net.state_dict(), os.path.join(output_dir, "bw_net_final.pt"))
    return bw_net

def Acc_train(dataset_train: EUROCDataset, dataset_val: EUROCDataset, outpur_dir: str, bw_model: bw_func_net, ba_model_name: str, integral_method = "euler", device = "cuda", lr = 0.005, weight_decay=1e-6, epoch = 1801):
    print(">>> Inizio Training Accelerometro")

    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False} # whether to mask some missing data
    mask_flag = MASKFLAG[dataset_train.dataset_name]
    ba_net = ba_func_net().to(device)
    model = LieModel(bias_net_w=bw_model,bias_net_a=ba_net,device=device).to(device)
    optimizer = torch.optim.Adam(model.bias_net_a.parameters(), lr=lr, weight_decay=weight_decay)  
    writer = SummaryWriter(os.path.join(outpur_dir, "logs_acc"))

    for ep in range(epoch):
        optimizer.zero_grad()
        model.train()
        #caricamento dato casuale
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        (t_gt_batch, X_gt_batch, bwa_gt_batch), t_odeint, coeff, Spline_time, induces = dataset_train.get_data_for_biasDy(index, device, mask_flag=mask_flag)
        y0_batch, R0_batch = dataset_train.construct_init_forodeint(t_gt_batch, X_gt_batch, bwa_gt_batch)
        Spline = Interpolation.CubicHermiteSpline(Spline_time, coeff, device=device)
        model.u_func, model.u_dot_func = Spline.evaluate, Spline.derivative
        sol,_ = odeint_SO3(model, y0_batch, R0_batch, t_odeint, rtol=1e-7, atol=1e-9, method=integral_method)
        loss = 1e6 * torch.nn.MSELoss()(sol[...,7:13], torch.cat([X_gt_batch[...,:3,3], X_gt_batch[...,:3,4]], dim=-1))
        loss.backward()
        optimizer.step()

        if ep % 100 == 0:
            print(f"Epoch {ep}, Loss: {loss.item():.4f}")
            writer.add_scalar("Acc_train_loss", loss.item(), ep)
    torch.save(ba_net.state_dict(), os.path.join(outpur_dir, "ba_net_final.pt"))
    return ba_net
