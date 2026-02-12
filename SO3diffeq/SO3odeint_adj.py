import torch
import torch.nn as nn
from Third_party.torchdiffeq._impl.misc import _check_inputs
from Third_party.torchdiffeq import odeint as odeint_euclidean 
from Third_party.torchdiffeq._impl.misc import _check_inputs, _flat_to_shape, _mixed_norm, _all_callback_names, _all_adjoint_callback_names
from BiasDy.lie_algebra import SO3rightJaco, SO3rightJacoInv

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
    setattr(func, 'callback_change_chart', callback)

    solver = SOLVERS[method](func=func, y0=y0, R0=R0, rtol=rtol, atol=atol, **options) 
    solution, R_sol = solver.integrate(t)
    return solution, R_sol

# =============================================================================
#  Adjoint Method Implementation
# =============================================================================

# Aug_state = [adj_t,y,adj_y,*adj_params]

def R_grad_to_xi_grad(R, R_grad):
    M = torch.bmm(R.transpose(-1, -2), R_grad)
    grad_xi_x = M[..., 2, 1] - M[..., 1, 2]
    grad_xi_y = M[..., 0, 2] - M[..., 2, 0]
    grad_xi_z = M[..., 1, 0] - M[..., 0, 1]
    return torch.stack([grad_xi_x, grad_xi_y, grad_xi_z], dim=-1)

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
        
        chart_events = []
        if options is None:
            options = {}
        else:
            options = options.copy()
        options['chart_events'] = chart_events

        with torch.no_grad():
            ans, R_ans = odeint_SO3(func, y0, R0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)
            if event_fn is None:
                ctx.save_for_backward(t, ans, R_ans, *adjoint_params)
                ctx.chart_events = chart_events
            else:
                raise NotImplementedError("Not yet adapted for SO3 adjoint.")
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
            chart_events = ctx.chart_events
            t, y_sol, R_sol, *adjoint_params = ctx.saved_tensors
            adjoint_params = tuple(adjoint_params)

        if grad_y is not None:
            adj_y = grad_y[-1].clone()
        else:
            adj_y = torch.zeros_like(y_sol[-1]) # Inizializza a zero se non c'è gradiente
        if grad_R is not None:
            adj_y[..., :3] += R_grad_to_xi_grad(R_sol[-1], grad_R[-1])
        # definiamo ora aug_state

        aug_state = [torch.zeros((), dtype=y_sol.dtype, device=y_sol.device), y_sol[-1], adj_y]
        aug_state.extend([torch.zeros_like(p) for p in adjoint_params]) # aggiungiamo gli adjoint params

        def aug_dynamics(t, aug_state):
            func_eval = func
            y = aug_state[1]
            adj_y = aug_state[2]
            adj_params = aug_state[3:]

            with torch.set_grad_enabled(True):
                t_ = t.detach()         
                t = t_.requires_grad_(True)
                y = y.detach().requires_grad_(True)
                func_eval = func(t if t_requires_grad else t, y)

             #Workaround for PyTorch bug #39784
            _t = torch.as_strided(t, (), ())  # noqa
            _y = torch.as_strided(y, (), ())  # noqa
            _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

            vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                func_eval, (t, y) + adjoint_params, -adj_y,
                allow_unused=True, retain_graph=True
            )
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
            vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

            return (vjp_t, func_eval, vjp_y, *vjp_params)
        
        # Add adjoint callbacks
        for callback_name, adjoint_callback_name in zip(_all_callback_names, _all_adjoint_callback_names):
            try:
                callback = getattr(func, adjoint_callback_name)
            except AttributeError:
                pass
            else:
                setattr(aug_dynamics, callback_name, callback)
        ##################################
        #       Solve adjoint ODE        #
        # integro all'indietro da T a t_switch_N, poi da t_switch_N a t_switch_(N-1), ..., fino a t0
        # supponiamo che il cambio di carta sia fatto ad ogni passo
        ##################################
        print(chart_events)
        time_events = [event[0] for event in chart_events]
        R_events = [event[1] for event in chart_events]
        xi_events= [event[2] for event in chart_events]

        if t_requires_grad:
            time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
        else:
            time_vjps = None

        for i in range(len(t)-1, 0,-1):
            if t_requires_grad:
                func_eval = func(t[i], y_sol[i])
                dL_dt = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                aug_state[0] -= dL_dt
                time_vjps[i] = dL_dt
            # integra da time_events[i] a time_events[i-1]
            aug_state = odeint_euclidean(
                aug_dynamics, tuple(aug_state),
                t[i - 1:i + 1].flip(0),
                rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
            )
            
            aug_state = [a[1] for a in aug_state]
            aug_state[1] = y_sol[i-1]
            # ora cambiamo carta
            # \lambda_csi_old = J_r(\csi_old)^T \lambda_csi_new * \lambda_csi

            adj_y_vec = aug_state[2]
            grad_xi = adj_y_vec[..., :3].unsqueeze(-1)
            aug_state[1][..., :3] = xi_events[i-1]
            
            adj_y_vec[..., :3] = torch.bmm(SO3rightJaco(xi_events[i-1]).transpose(-1, -2), grad_xi).squeeze(-1)
            aug_state[2] = adj_y_vec
            aug_state[2] += grad_y[i-1]
            aug_state[2][..., :3] += R_grad_to_xi_grad(R_sol[i-1], grad_R[i-1])
            if t_requires_grad:
                time_vjps[0] = aug_state[0]
            adj_y = aug_state[2]
            adj_params = aug_state[3:]


        return (None, None, adj_y, None, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)







def odeint_adjoint_SO3(func, y0, R0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):
    
    if adjoint_rtol is None: adjoint_rtol = rtol
    if adjoint_atol is None: adjoint_atol = atol
    if adjoint_method is None: adjoint_method = method
    if adjoint_options is None: adjoint_options = options
    
    if adjoint_params is None and isinstance(func, torch.nn.Module):
        adjoint_params = tuple(p for p in func.parameters() if p.requires_grad)
    elif adjoint_params is None:
        adjoint_params = ()

    shapes = None 
    return OdeintAdjointMethodSO3.apply(shapes, func, y0, R0, t, rtol, atol, method, options, event_fn,
                                        adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)