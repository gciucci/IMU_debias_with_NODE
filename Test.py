import torch
import os
import lie_algebra as Lie
import Interpolation as Interpolation
from SO3diffeq.SO3odeint_adj import odeint_adjoint_SO3
from dataset import pdump, pload
from lie_model import LieModel, bw_func_net, ba_func_net

def Test(dataset_test, output_dir, bw_weights_path, ba_weights_path, device="cpu"):
    """
    Carica i pesi salvati e valuta il modello sull'intera traiettoria.
    """
    print(f">>> Inizio Testing su dispositivo: {device}")
    
    # 1. Inizializzazione Reti e Caricamento Pesi
    bw_net = bw_func_net().to(device)
    ba_net = ba_func_net().to(device)
    
    bw_net.load_state_dict(torch.load(bw_weights_path, map_location=device))
    ba_net.load_state_dict(torch.load(ba_weights_path, map_location=device))
    
    # 2. Creazione del Modello Lie (Neural ODE)
    model = LieModel(None, None, bw_net, ba_net, device=device)
    model.eval()
    
    results_dir = os.path.join(output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    # 3. Loop sulle sequenze del dataset
    for i in range(dataset_test.length()):
        seq_name = dataset_test.sequences[i]
        print(f"Elaborazione sequenza: {seq_name}")
        
        # Recupero traiettoria completa e spline
        _, _, X_gt, t_gt = dataset_test.get_full_trajectory(i, device)
        coeff, time = dataset_test.get_coeff(i)
        
        # Setup Condizioni Iniziali (y0)
        # Stato y: [xi(3), t(1), bw(3), v(3), p(3), ba(3)] -> totale 16
        y0 = torch.zeros(1, 16).to(device)
        y0[0, 3] = t_gt[0]                # Tempo iniziale
        y0[0, 7:10] = X_gt[0, :3, 3]      # Velocità iniziale GT
        y0[0, 10:13] = X_gt[0, :3, 4]     # Posizione iniziale GT
        
        # Setup Spline per l'integrazione
        spline = Interpolation.CubicHermiteSpline(time, coeff, device)
        model.u_func = spline.evaluate
        model.u_dot_func = spline.derivative
        
        # Setup Orientamento iniziale
        R0 = X_gt[0, :3, :3]
        model.set_R0(R0)
        
        # 4. Integrazione della Neural ODE
        with torch.no_grad():
            # Il risolutore calcola tutta la traiettoria basandosi sulla fisica + reti bias
            sol_pred, R_pred = odeint_adjoint_SO3(model, y0, R0, t_gt, method='euler')
        
        # 5. Estrazione Risultati (rimuoviamo la dimensione del batch)
        p_pred = sol_pred.squeeze(1)[:, 10:13] # Posizione predetta
        v_pred = sol_pred.squeeze(1)[:, 7:10]  # Velocità predetta
        R_pred = R_pred.squeeze(1)             # Rotazione predetta
        
        # 6. Salvataggio dati per analisi successiva
        res = {
            't': t_gt.cpu(),
            'p_gt': X_gt[:, :3, 4].cpu(),
            'p_pred': p_pred.cpu(),
            'R_gt': X_gt[:, :3, :3].cpu(),
            'R_pred': R_pred.cpu()
        }
        pdump(res, os.path.join(results_dir, f"{seq_name}_results.p"))
        
        # 7. Calcolo Metriche (opzionale)
        # aoe = AOE(R_pred, X_gt[:, :3, :3])
        # print(f"Errore Orientamento (AOE): {aoe:.4f}")

    print(f">>> Test completato. Risultati salvati in: {results_dir}")