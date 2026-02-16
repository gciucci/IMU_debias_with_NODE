import torch
import torch.nn as nn
import lie_algebra as Lie

class bw_func_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128), 
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh(), 
            nn.Linear(128, 9),
        )
        self.net2 = nn.Sequential(
            nn.Linear(9, 512),
            nn.Tanh(),
            nn.Linear(512, 9),
        )
        self.linear2 = nn.Linear(9, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba
            imu_meas: shape (batch_size, 6) where 6 = w, a
            imu_meas_dot: shape (batch_size, 6) where 6 = w_dot, a_dot"""
        x = torch.cat([ y[..., 4:7], imu_meas[...,:3], imu_meas_dot[...,:3]], dim=-1) # order:  bw, w, w_dot
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)
    

    ## ba_func
class ba_func_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256), 
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 9),
        )
        self.net2 = nn.Sequential(
            nn.Linear(9, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 9),
        )
        self.linear2 = nn.Linear(9, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
    def forward(self, y, imu_meas, imu_meas_dot, R0):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba
            imu_meas: shape (batch_size, 6) where 6 = w, a
            imu_meas_dot: shape (batch_size, 6) where 6 = w_dot, a_dot"""
        x = torch.cat([y[..., 13:16], imu_meas[...,3:],imu_meas_dot[...,3:]], dim=-1) # order:  ba, a, a_dot
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)
    
class LieModel(torch.nn.Module):
    def __init__(self, bias_net_w=None, bias_net_a=None, device="cuda"):
        super().__init__()

        self.u_func = None
        self.u_dot_func = None
        self.bias_net_w = bias_net_w
        self.bias_net_a = bias_net_a
        self.device = device
        self.g_const = torch.tensor([0, 0, -9.81]).to(device)
        self.R0 = torch.eye(3).to(device)

    def set_R0(self, R0):
        self.R0 = R0.clone().to(self.device)

    def callback_change_chart(self, y):
        xi = y[..., :3]
        self.set_R0(self.R0 @ Lie.SO3exp(xi)) #aggiornamento R0
        y_new = y.clone()
        y_new[..., :3] = 0 #xi inizializzate a 0
        return y_new

    def __call__(self, t, y):
        """
        Calcola la derivata dello stato y.
        y contiene: [orientamento(3), tempo(1), bias_w(3), velocità(3), posizione(3), bias_a(3)]
        """
        # 1. Recupero dati IMU al tempo attuale
        t_abs = y[..., 3]
        imu_meas = self.u_func(t_abs)
        imu_meas_dot = self.u_dot_func(t_abs)
        
        w_tilde = imu_meas[..., :3] # Giroscopio
        a_tilde = imu_meas[..., 3:] # Accelerometro

        # 2. Calcolo dei Bias tramite Reti Neurali
        bw_dot = (
            self.bias_net_w(y, imu_meas, imu_meas_dot, self.R0)
            if self.bias_net_w is not None
            else torch.zeros_like(y[..., 4:7])
        )

        ba_dot = (
            self.bias_net_a(y, imu_meas, imu_meas_dot, self.R0)
            if self.bias_net_a is not None
            else torch.zeros_like(y[..., 13:16])
        )
        bw = y[..., 4:7]
        ba = y[..., 13:16]

        # 3. Fisica del sistema
        xi_dot = torch.matmul(Lie.SO3rightJacoInv(y[..., :3]), (w_tilde - bw).unsqueeze(-1)).squeeze(-1)
        Rt = self.R0.to(y.device) @ Lie.SO3exp(y[..., :3])
        v_dot = torch.matmul(Rt, (a_tilde - ba).unsqueeze(-1)).squeeze(-1) + self.g_const.to(y.device)
        p_dot = y[..., 7:10]
        
        t_dot = torch.ones_like(t_abs).unsqueeze(-1) # Derivata del tempo (scorre sempre a 1)

        # Uniamo tutto
        return torch.cat([xi_dot, t_dot, bw_dot, v_dot, p_dot, ba_dot], dim=-1)