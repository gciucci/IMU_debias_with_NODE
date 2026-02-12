import torch
import torch.nn as nn
from Third_party.torchdiffeq._impl.misc import _check_inputs, _flat_to_shape
from .SO3dopri8 import SO3Dopri8Solver
from .SO3dopri5 import SO3Dopri5Solver
from .SO3bosh3 import SO3Bosh3Solver
from .SO3fehlberg2 import SO3Fehlberg2
from .SO3adaptive_heun import SO3AdaptiveHeunSolver
from .SO3fixed_grid import SO3Euler, SO3Midpoint, SO3Heun3, SO3RK4
from .lie_util import SO3exp

# =============================================================================
#  odeint_SO3 function
# =============================================================================


SOLVERS = {
    'dopri8': SO3Dopri8Solver,
    'dopri5': SO3Dopri5Solver,
    'bosh3': SO3Bosh3Solver,
    'fehlberg2': SO3Fehlberg2,
    'adaptive_heun': SO3AdaptiveHeunSolver,
    'euler': SO3Euler,
    'midpoint': SO3Midpoint,
    'heun3': SO3Heun3,
    'rk4': SO3RK4,
}

def odeint_SO3(func, y0, R0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """method is 'dopri5' by default"""
    
    original_func = func
    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)
    ### add callback function to change the chart
    try:
        callback = getattr(original_func, 'callback_change_chart')
    except AttributeError:
        callback = lambda x: None
    else:
        pass
    setattr(func, 'callback_change_chart', callback)

    solver = SOLVERS[method](func=func, y0=y0, R0=R0, rtol=rtol, atol=atol, **options) 

    solution, R_sol = solver.integrate(t)
    return solution, R_sol


# =============================================================================
#  Adjoint Method Implementation
# =============================================================================

# Funzione ausiliaria per proiettare il gradiente della Loss (su R) nello spazio tangente (Lie Algebra)
def project_R_grad_to_lie(R, grad_R):
    """
    Calcola il gradiente rispetto a xi (Lie Algebra) dato il gradiente rispetto a R.
    Assume R_new = R * exp(xi) (perturbazione a destra).
    """
    # M = R^T * grad_R
    M = torch.matmul(R.transpose(-1, -2), grad_R)
    
    # Parte antisimmetrica: 0.5 * (M - M^T)
    skew_sym = 0.5 * (M - M.transpose(-1, -2))
    
    # Unskew operation per ottenere il vettore 3D [x, y, z]
    grad_x = skew_sym[..., 2, 1]
    grad_y = skew_sym[..., 0, 2]
    grad_z = skew_sym[..., 1, 0]
    
    return torch.stack([grad_x, grad_y, grad_z], dim=-1)


class OdeintAdjointMethodSO3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, R0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None

        with torch.no_grad():
            ans, R_ans = odeint_SO3(func, y0, R0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)
            if event_fn is None:
                y = ans
                R = R_ans
                ctx.save_for_backward(t, y, R, *adjoint_params)
            else:
                raise NotImplementedError("Event handling not yet fully adapted for SO3 adjoint.")
        
        return ans, R_ans

    @staticmethod
    def backward(ctx, grad_y, grad_R):
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            t, y, R_sol, *adjoint_params = ctx.saved_tensors
            adjoint_params = tuple(adjoint_params)

            # --- SETUP INITIAL STATE FOR BACKWARD ---
            
            # y[-1] shape: (Batch, State_Dim)
            batch_size = y[-1].shape[0]
            
            # Buffer per i gradienti dei parametri
            aug_y0_params = []
            for p in adjoint_params:
                zeros_p = torch.zeros(batch_size, p.numel(), dtype=y[-1].dtype, device=y[-1].device)
                aug_y0_params.append(zeros_p)

            # GESTIONE GRADIENTE R (Se la loss è solo su R, grad_y può essere None)
            if grad_y is None:
                grad_y_final = torch.zeros_like(y[-1])
            else:
                grad_y_final = grad_y[-1].clone()

            # Se esiste un gradiente su R all'ultimo step, lo proiettiamo e sommiamo
            if grad_R is not None:
                grad_xi_final = project_R_grad_to_lie(R_sol[-1], grad_R[-1])
                # Sommiamo alle prime 3 componenti di y (che sono la Lie Algebra xi)
                grad_y_final[..., :3] += grad_xi_final

            # Concateniamo tutto: [y_finale, grad_y_finale, param_grads...]
            aug_y0 = torch.cat([y[-1], grad_y_final] + aug_y0_params, dim=-1)
            
            dim_y = y[-1].shape[-1]
            dim_adj_y = grad_y_final.shape[-1]
            param_shapes = [p.shape for p in adjoint_params]
            param_numels = [p.numel() for p in adjoint_params]

            # --- BACKWARD DYNAMICS ---

            def augmented_dynamics(t, y_aug):
                y = y_aug[..., :dim_y]
                adj_y = y_aug[..., dim_y : dim_y + dim_adj_y]
                
                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)

                    func_eval = func(t if t_requires_grad else t_, y)

                    # Calcolo derivate: d(adj)/dt = - adj * df/dy
                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                
                # Espansione gradienti parametri per il batch
                vjp_params_expanded = []
                for i, (vp, numel) in enumerate(zip(vjp_params, param_numels)):
                    if vp is None:
                        vp_exp = torch.zeros(batch_size, numel, dtype=y.dtype, device=y.device)
                    else:
                        vp_flat = vp.reshape(-1)
                        # Distribuiamo il gradiente totale (somma) su tutto il batch in media
                        # Integramente darà la somma corretta alla fine
                        vp_exp = vp_flat.unsqueeze(0).expand(batch_size, -1) 
                        vp_exp = vp_exp / float(batch_size)
                    vjp_params_expanded.append(vp_exp)

                return torch.cat([func_eval, vjp_y] + vjp_params_expanded, dim=-1)

            if hasattr(func, 'callback_change_chart'):
                augmented_dynamics.callback_change_chart = func.callback_change_chart

            # --- SOLVE ADJOINT ---

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            
            # Integrazione all'indietro passo-passo
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    func_eval = func(t[i], y[i])
                    # Qui usiamo solo la parte di gradiente esplicito su y per il tempo
                    curr_grad_y = grad_y[i] if grad_y is not None else torch.zeros_like(y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(curr_grad_y.reshape(-1))
                    time_vjps[i] = dLd_cur_t

                time_segment = t[i - 1:i + 1].flip(0)
                # IMPORTANTE: Usiamo la rotazione salvata al passo i come riferimento
                curr_R0 = R_sol[i]
                
                aug_sol, _ = odeint_SO3(
                    augmented_dynamics, 
                    aug_y0, 
                    curr_R0, 
                    time_segment,
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                
                aug_y0 = aug_sol[-1]
                
                # --- INIEZIONE GRADIENTI INTERMEDI (Skip Connections) ---
                adj_y_part = aug_y0[..., dim_y : dim_y + dim_adj_y]
                
                # 1. Aggiungi gradiente diretto su y (se esiste, es. loss su velocità/bias)
                if grad_y is not None:
                    adj_y_part = adj_y_part + grad_y[i - 1]
                
                # 2. Aggiungi gradiente proiettato da R (Loss su rotazione)
                if grad_R is not None:
                    grad_xi_step = project_R_grad_to_lie(R_sol[i-1], grad_R[i-1])
                    adj_y_part[..., :3] = adj_y_part[..., :3] + grad_xi_step
                
                # Ricostruiamo lo stato aumentato aggiornato
                aug_y0 = torch.cat([
                    aug_y0[..., :dim_y], 
                    adj_y_part,         
                    aug_y0[..., dim_y + dim_adj_y:]
                ], dim=-1)

            if t_requires_grad:
                time_vjps[0] = 0 

            # --- EXTRACT FINAL GRADIENTS ---
            
            # adj_y contiene il gradiente rispetto allo stato iniziale y0
            adj_y = aug_y0[..., dim_y : dim_y + dim_adj_y]
            adj_params_flat = aug_y0[..., dim_y + dim_adj_y:]
            
            # Ricostruiamo i gradienti dei parametri sommando su tutto il batch
            adj_params_ret = []
            offset = 0
            for i, shape in enumerate(param_shapes):
                numel = param_numels[i]
                grad_p_batch = adj_params_flat[..., offset:offset+numel]
                grad_p = grad_p_batch.sum(dim=0).reshape(shape)
                adj_params_ret.append(grad_p)
                offset += numel

        # Return corretto con None per R0 (4° argomento)
        return (None, None, adj_y, None, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params_ret)


def odeint_adjoint_SO3(func, y0, R0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):
    
    if adjoint_rtol is None: adjoint_rtol = rtol
    if adjoint_atol is None: adjoint_atol = atol
    if adjoint_method is None: adjoint_method = method
    if adjoint_options is None: adjoint_options = options
    
    if adjoint_params is None and isinstance(func, nn.Module):
        adjoint_params = tuple(p for p in func.parameters() if p.requires_grad)
    elif adjoint_params is None:
        adjoint_params = ()

    shapes = None 

    return OdeintAdjointMethodSO3.apply(shapes, func, y0, R0, t, rtol, atol, method, options, event_fn,
                                        adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)