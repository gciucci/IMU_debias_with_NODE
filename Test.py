import torch
import Interpolation
from SO3diffeq import odeint_SO3
from lie_model import LieModel, bw_func_net, ba_func_net
from dataset import BaseDataset


def test_gyro(
    dataset_test: BaseDataset,
    model_path: str,
    device="cuda",
    integral_method="euler",
):
    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False}
    mask_flag = MASKFLAG[dataset_test.dataset_name]

    # -------- LOAD MODEL --------
    bw_func = bw_func_net().to(device)
    model = LieModel(bias_net_w=bw_func, device=device).to(device)

    model.set_bw_zero = False
    model.set_ba_zero = True

    ckpt = torch.load(model_path, map_location=device)
    model.biasfunc_w.load_state_dict(ckpt["func_bw_model"])

    model.eval()
    print(">>> Modello Gyro caricato")

    results = []

    with torch.no_grad():
        for idx in range(dataset_test.length()):
            (t_gt, X_gt, bwa_gt), t_odeint, coeff, spline_time, _ = \
                dataset_test.get_data_for_biasDy(idx, device, mask_flag=mask_flag)

            y0, R0 = dataset_test.construct_init_forodeint(
                t_gt, X_gt, bwa_gt
            )

            spline = Interpolation.CubicHermiteSpline(
                spline_time, coeff, device=device
            )
            model.set_u_func(spline.evaluate)
            model.set_u_dot_func(spline.derivative)
            model.set_R0(R0)

            solution, _ = odeint_SO3(
                model, y0, R0, t_odeint, method=integral_method
            )

            results.append(solution.cpu())

    print(">>> Test Gyro completato")
    return results

def test_acc(
    dataset_test: BaseDataset,
    model_path: str,
    device="cuda",
    integral_method="euler",
):
    MASKFLAG = {"EUROC": False, "TUM": True, "Fetch": False}
    mask_flag = MASKFLAG[dataset_test.dataset_name]

    # -------- LOAD MODEL --------
    ba_func = ba_func_net().to(device)
    model = LieModel(bias_net_a=ba_func, device=device).to(device)

    model.set_bw_zero = True
    model.set_ba_zero = False

    ckpt = torch.load(model_path, map_location=device)
    model.biasfunc_a.load_state_dict(ckpt["func_ba_model"])

    model.eval()
    print(">>> Modello Accelerometro caricato")

    results = []

    with torch.no_grad():
        for idx in range(dataset_test.length()):
            (t_gt, X_gt, bwa_gt), t_odeint, coeff, spline_time, _ = \
                dataset_test.get_data_for_biasDy(idx, device, mask_flag=mask_flag)

            y0, R0 = dataset_test.construct_init_forodeint(
                t_gt, X_gt, bwa_gt
            )

            spline = Interpolation.CubicHermiteSpline(
                spline_time, coeff, device=device
            )
            model.set_u_func(spline.evaluate)
            model.set_u_dot_func(spline.derivative)
            model.set_R0(R0)

            solution, _ = odeint_SO3(
                model, y0, R0, t_odeint, method=integral_method
            )

            results.append(solution.cpu())

    print(">>> Test Accelerometro completato")
    return results