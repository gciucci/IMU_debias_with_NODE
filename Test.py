import torch
import Interpolation
from SO3diffeq import odeint_SO3
from lie_model import LieModel, bw_func_net, ba_func_net
from dataset import BaseDataset
import lie_algebra as Lie

def test(
    dataset_test:BaseDataset,
    device="cuda",
    integral_method="euler",
    bw_func = bw_func_net,
    ba_func = ba_func_net,
):
    model = LieModel(bias_net_w=bw_func, bias_net_a=ba_func, device=device).to(device)
    model.set_bw_zero = False
    model.set_ba_zero = False
    model.eval()
    print(">>> Modello caricato")

    total_aoe = 0.0
    total_ate = 0.0
    total_samples = 0

    with torch.no_grad():
        for index in range(dataset_test.length()):
            print("Testing:", dataset_test.sequences[index])
            # recupero i dati imu per la sequenza di test
            coeff, time = dataset_test.get_coeff(index)
            spline = Interpolation.CubicHermiteSpline(time, coeff, device)
            # recupero la traiettoria completa (stima bias, stima stato, ground truth stato, ground truth tempo)
            _, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index, device=device)
            # inizializzo R0, bw0, ba0 con i valori della prima misura della sequenza di test
            R0 = X_gt[0,:3,:3].clone().to(device)
            bw0 = dataset_test.get_w_bias(index)[0][0]
            ba0 = dataset_test.get_a_bias(index)[0][0]
            # ensure biases and time are on the correct device
            if not isinstance(bw0, torch.Tensor):
                bw0 = torch.tensor(bw0, device=device)
            else:
                bw0 = bw0.to(device)
            if not isinstance(ba0, torch.Tensor):
                ba0 = torch.tensor(ba0, device=device)
            else:
                ba0 = ba0.to(device)
            # preparo lo stato iniziale y0 con xi inizializzate a 0,
            # t inizializzato al tempo della prima misura, bias iniziali, velocità e posizione iniziali 
            t0 = t_gt[0].unsqueeze(-1).to(device)
            p0 = X_gt[0,:3,3].to(device)
            v0 = X_gt[0,:3,4].to(device)
            y0 = torch.cat([
                torch.zeros(3, device=device),
                t0,
                bw0,
                v0,
                p0,
                ba0,
            ], dim=-1).unsqueeze(0)

            # integrazione numerica
            model.u_func = spline.evaluate
            model.u_dot_func = spline.derivative
            model.set_R0(R0)

            sol_pred, R_pred = odeint_SO3(
                model, y0, R0, t_gt, method=integral_method
            )

            sol_pred = sol_pred.squeeze(1)
            R_pred = R_pred.squeeze(1)

            # ===== Metrics =====

            # --- AOE ---
            R_gt = X_gt[:,:3,:3]
            R_err = torch.matmul(R_gt.transpose(1,2), R_pred)

            log_R = Lie.SO3log(R_err)
            aoe = torch.norm(log_R, dim=1).mean()

            # --- ATE ---
            p_gt = X_gt[:,:3,4]
            p_pred = sol_pred[:,10:13]

            ate = torch.norm(p_pred - p_gt, dim=1).mean()

            n = t_gt.shape[0]
            total_aoe += aoe.item() * n
            total_ate += ate.item() * n
            total_samples += n

    mean_aoe = total_aoe / total_samples
    mean_ate = total_ate / total_samples

    print("\n========== FINAL RESULTS ==========")
    print(f"Mean AOE  (rad): {mean_aoe:.6f}")
    print(f"Mean AOE  (deg): {mean_aoe * 180.0 / torch.pi:.4f}")
    print(f"Mean ATE  (m):   {mean_ate:.6f}")
    print("====================================")

    return mean_aoe, mean_ate