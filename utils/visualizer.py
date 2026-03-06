# utils/visualizer.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _fmt_params(mu, nu):
    return f"μ={list(mu)}, ν={list(nu)}"

def _abs_err(u_true, u_pred):
    a = u_true.to_numpy() if hasattr(u_true, "to_numpy") else np.asarray(u_true)
    b = u_pred.to_numpy() if hasattr(u_pred, "to_numpy") else np.asarray(u_pred)
    return np.abs(a - b)

def _force_scientific_colorbars(fig, powerlimits=(-3, 3)):
    try:
        fmt = ScalarFormatter(useMathText=False)
        fmt.set_powerlimits(powerlimits)
        for ax in fig.axes:
            if not ax.images and hasattr(ax, "yaxis"):
                ax.yaxis.set_major_formatter(fmt)
        fig.canvas.draw_idle()
    except Exception:
        pass


def vis_dl_diff(fom, u_true, pod_pred, doddl_pred, mu, nu):
    """
    Two figures using native fom.visualize:
      (1) 2x2: [FOM | POD-DL-ROM; FOM | DOD-DL-ROM] with a single shared colorbar
      (2) 1x2: [|FOM-POD| | |FOM-DODDL|] with a single shared colorbar
    """
    # ---- figure 1: solutions block (shared colorbar) ----
    fields_2x2 = (
        u_true,               pod_pred,
        u_true,               doddl_pred,
    )
    legends_2x2 = (
        f"FOM { _fmt_params(mu, nu) }", "POD-DL-ROM",
        f"FOM { _fmt_params(mu, nu) }", "DOD-DL-ROM",
    )
    fig1 = fom.visualize(
        fields_2x2,
        legend=legends_2x2,
        columns=2,
        separate_colorbars=False,
        block=False
    )
    _force_scientific_colorbars(fig1)

    # ---- figure 2: absolute errors (shared colorbar) ----
    err_pod   = fom.solution_space.from_numpy(_abs_err(u_true, pod_pred))
    err_doddl = fom.solution_space.from_numpy(_abs_err(u_true, doddl_pred))

    fields_err = (err_pod, err_doddl)
    legends_err = ("Abs. error (POD-DL-ROM)", "Abs. error (DOD-DL-ROM)")

    fig2 = fom.visualize(
        fields_err,
        legend=legends_err,
        columns=1,
        separate_colorbars=False,
        block=False
    )
    _force_scientific_colorbars(fig2)


def vis_dod_diff(fom, u_true, dfnn_pred, doddl_pred, mu, nu):
    """
    Two figures:
      (1) 2x2: [FOM | DOD+DFNN; FOM | DOD-DL-ROM] (shared colorbar)
      (2) 1x2: [|FOM-DFNN| | |FOM-DODDL|] (shared colorbar)
    """
    fields_2x2 = (
        u_true,               dfnn_pred,
        u_true,               doddl_pred,
    )
    legends_2x2 = (
        f"FOM { _fmt_params(mu, nu) }", "DOD+DFNN",
        f"FOM { _fmt_params(mu, nu) }", "DOD-DL-ROM",
    )
    fig1 = fom.visualize(
        fields_2x2,
        legend=legends_2x2,
        columns=2,
        separate_colorbars=False,
        block=False
    )
    _force_scientific_colorbars(fig1)

    err_dfnn  = fom.solution_space.from_numpy(_abs_err(u_true, dfnn_pred))
    err_doddl = fom.solution_space.from_numpy(_abs_err(u_true, doddl_pred))

    fields_err = (err_dfnn, err_doddl)
    legends_err = ("Abs. error (DOD+DFNN)", "Abs. error (DOD-DL-ROM)")

    fig2 = fom.visualize(
        fields_err,
        legend=legends_err,
        columns=1,
        separate_colorbars=False,
        block=False
    )
    _force_scientific_colorbars(fig2)


def vis_colora(fom, u_true, colora_pred, mu, nu):
    """
    Two figures:
      (1) 2x2: [FOM | CoLoRA; FOM | CoLoRA] (shared colorbar; repeated to keep layout uniform)
      (2) 1x1: [|FOM-CoLoRA|] (its own shared colorbar trivially)
    """
    fields_2x2 = (
        u_true,               colora_pred,
        u_true,               colora_pred,
    )
    legends_2x2 = (
        f"FOM { _fmt_params(mu, nu) }", "CoLoRA-DL-ROM",
        f"FOM { _fmt_params(mu, nu) }", "CoLoRA-DL-ROM",
    )
    fig1 = fom.visualize(
        fields_2x2,
        legend=legends_2x2,
        columns=2,
        separate_colorbars=False,
        block=False
    )
    _force_scientific_colorbars(fig1)

    err_colora = fom.solution_space.from_numpy(_abs_err(u_true, colora_pred))

    fig2 = fom.visualize(
        (err_colora,),
        legend=("Abs. error (CoLoRA-DL-ROM)",),
        columns=1,
        separate_colorbars=False,
        block=False
    )
    _force_scientific_colorbars(fig2)
