#!/usr/bin/env python3
"""
delu_contract.py
Pure-PyTorch implementation of the DeLU contract designer
(T. Wang et al., NeurIPS 2023).
"""

import math, numpy as np, torch
import time

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop

from problems.online_contract_edgeAI.benchmarks.bandit_solution import create_sys_model, simulate_round, \
    generate_candidate_contracts
from utils import contract_oracle_solver, save_results

# delu_realization.py
# A practical DeLU realization aligned with the NeurIPS-2023 paper:
# - η: ReLU MLP without output bias
# - ζ: Tanh MLP over activation pattern -> output bias
# - ξ(f) = η(f) + ζ(r(f))
# Training: AdamW + OneCycleLR, X/Y standardization, single-batch overfit check
# Inference: gradient-based interior-point (barrier) with sub-argmax early stop

import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# --------------------------- 3.  Demo dataset ---------------------------------
def build_dataset(m, n, cs, N_samples=10_000, seed=0):
    rng = np.random.default_rng(seed)
    p, c, v, _ = cs
    # Sample random contracts in (0, f_max]
    f_max = v.max()
    F_samples = rng.random((N_samples, m)) * f_max
    principal_utils = np.array([simulate_round(f, cs)[2][0] for f in F_samples])
    return F_samples.astype("float32"), principal_utils.astype("float32"), f_max


# -------------------------- 1) DeLU architecture ----------------------------- #
class DeLU(nn.Module):
    """
    η:  ReLU MLP; final η output has no bias.
    ζ:  Tanh MLP mapping activation pattern r(f) (binary mask of η's hidden) to a bias.
    Output: ξ(f) = η(f) + ζ(r(f)).
    """

    def __init__(self, m, h_eta=64, depth_eta=2, h_zeta=512):
        super().__init__()
        self.m = m

        # η-network (no bias at output)
        layers = []
        in_dim = m
        for _ in range(depth_eta):
            layers += [nn.Linear(in_dim, h_eta), nn.ReLU(inplace=True)]
            in_dim = h_eta
        self.eta_body = nn.Sequential(*layers)
        self.eta_out = nn.Linear(h_eta, 1, bias=False)

        # ζ-network: input dim = size of η hidden activations (h_eta), output = scalar bias
        self.z1 = nn.Linear(h_eta, h_zeta)
        self.z2 = nn.Linear(h_zeta, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, f):  # f: (B, m)
        h = self.eta_body(f)  # (B, h_eta)
        eta = self.eta_out(h)  # (B, 1) -- no output bias
        r = (h > 0).float()  # activation pattern (B, h_eta)
        # ζ: bias network
        b = torch.tanh(self.z1(r))
        b = self.z2(b)  # (B, 1)
        return eta + b  # ξ(f)


# -------------------- 2) Training utilities & loop --------------------------- #
def standardize_xy(F_samples, y, eps=1e-8):
    X = torch.as_tensor(F_samples, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)

    X_mean = X.mean(0, keepdim=True)
    X_std = X.std(0, keepdim=True).clamp_min(eps)
    Xn = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std().clamp_min(eps)
    yn = (y - y_mean) / y_std
    return Xn, yn, (X_mean, X_std, y_mean, y_std)


def overfit_one_batch(model, xb, yb, device="cpu", iters=200):
    """Sanity check: must drive normalized MSE well below 1 on a single batch."""
    model.train()
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for i in range(iters):
        pred = model(xb.to(device))
        loss = F.mse_loss(pred, yb.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward();
        opt.step()
        if i % 50 == 0:
            print(f"[sanity] it={i} single-batch loss={float(loss):.4f}")
    return float(loss)


def train_delu(
        F_samples, y, m, *,
        epochs=100, batch_size=512, device="cpu",
        lr=3e-4, max_lr=1e-3, weight_decay=1e-4,
        h_eta=64, depth_eta=2, h_zeta=512,
        do_sanity=True
):
    Xn, yn, stats = standardize_xy(F_samples, y)
    ds = torch.utils.data.TensorDataset(Xn, yn)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DeLU(m, h_eta=h_eta, depth_eta=depth_eta, h_zeta=h_zeta).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(dl))
    sched = OneCycleLR(
        opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, div_factor=max(1.0, max_lr / lr), final_div_factor=1e2
    )

    if do_sanity:
        xb, yb = next(iter(dl))
        final_sanity = overfit_one_batch(model, xb[:min(256, len(xb))], yb[:min(256, len(yb))], device, iters=200)
        print(f"[sanity] final single-batch loss ~ {final_sanity:.4f} (should be << 1.0)")

    model.train()
    for epoch in range(epochs):
        run_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            run_loss += float(loss) * xb.size(0)
        run_loss /= len(ds)
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train_loss(norm MSE): {run_loss:.6f} | lr: {lr_now:.2e}")

    # Attach a predict() that de-normalizes outputs
    X_mean, X_std, y_mean, y_std = stats

    def predict(x_new):
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(x_new, dtype=torch.float32, device=device)
            x = (x - X_mean.to(device)) / X_std.to(device)
            yhat_norm = model(x if x.ndim == 2 else x[None, :])
            yhat = yhat_norm * y_std.to(device) + y_mean.to(device)
            return yhat.squeeze(-1).cpu().numpy()

    model.predict = predict
    model._stats = stats  # for potential reuse
    return model


# ------------------ 3) Gradient-based (barrier) inference -------------------- #
def _eta_preacts(model, x_norm):
    """
    Compute pre-ReLU activations z for each hidden block in eta_body.
    Returns a list of z tensors (one per hidden block) and the final post-ReLU h.
    Assumes eta_body is [Linear, ReLU, Linear, ReLU, ...].
    """
    z_list = []
    h = x_norm
    mods = list(model.eta_body.children())
    i = 0
    while i < len(mods):
        lin = mods[i];
        act = mods[i + 1]
        z = lin(h)  # pre-activation before ReLU
        z_list.append(z)
        h = act(z)  # post-activation
        i += 2
    return z_list, h  # z_list[-1] is the last hidden pre-activation


def _pattern_from_preacts(z):
    """Binary activation pattern r from pre-activations z (last hidden)."""
    return (z > 0).float()


def _signed_margin_penalty(z, r0, margin):
    """
    Enforce staying in the same region as r0:
    if r0_i=1 -> want z_i >= margin; if r0_i=0 -> want z_i <= -margin.
    """
    s = (2.0 * r0 - 1.0)  # {-1, +1}
    return F.relu(margin - s * z).sum()


@torch.no_grad()
def _pull_stats(model, device):
    # Stats saved in training code I provided earlier
    X_mean, X_std, y_mean, y_std = model._stats
    return (X_mean.to(device), X_std.to(device),
            y_mean.to(device), y_std.to(device))


def infer_contract_adam(
        model,
        f_max,
        *,
        restarts=512,
        steps=400,
        lr=1e-1,
        margin=1e-2,  # stability margin for hidden activations
        lam_stab=1e-2,  # weight of stability penalty
        lam_box=0.0,  # optional small penalty if you want to avoid sticking to bounds
        device="cpu"
):
    """
    Optimize contract f directly (box 0<=f<=f_max) using Adam.
    Uses model._stats (X_mean, X_std, y_mean, y_std) to keep inference
    consistent with training normalization.
    """
    model.eval()
    m = model.m
    if np.isscalar(f_max):
        fmax_vec = torch.full((1, m), float(f_max), device=device)
    else:
        fmax_vec = torch.as_tensor(f_max, dtype=torch.float32, device=device).view(1, m)

    X_mean, X_std, y_mean, y_std = _pull_stats(model, device)

    best_f, best_val = None, -1e30

    for ind in range(restarts):
        # random init (away from boundaries a bit)
        f = (0.1 + 0.8 * torch.rand(1, m, device=device)) * fmax_vec
        f = torch.nn.Parameter(f)

        opt = torch.optim.Adam([f], lr=lr)

        # --- At the start of each restart, after creating f ---
        with torch.no_grad():
            x0 = (f - X_mean) / X_std
            z0_list, _ = _eta_preacts(model, x0)
            z0 = z0_list[-1]  # use last hidden layer’s pre-acts
            r0 = _pattern_from_preacts(z0).detach()  # fixed target pattern for this restart

        for t in range(steps):
            # normalize input to match train-time preprocessing
            x_norm = (f - X_mean) / X_std
            # forward: predicted normalized utility
            y_hat_norm = model(x_norm)  # (1,1)
            # de-normalize to get the *real* objective value
            y_hat_real = y_hat_norm * y_std + y_mean  # (1,1)

            # ---- stability penalty ----
            z_list, _ = _eta_preacts(model, x_norm)
            z = z_list[-1]
            stab_pen = _signed_margin_penalty(z, r0, margin)

            # optional: tiny box penalty to avoid living on edges (off by default)
            box_pen = ((f.clamp(min=0.) - f).abs().sum() +
                       ((f - fmax_vec).clamp(min=0.)).sum())

            # maximize y_hat_real -> minimize negative
            loss = -(y_hat_real.mean()) + lam_stab * stab_pen + lam_box * box_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # gradient step on f
            opt.step()
            with torch.no_grad():
                f.clamp_(0.0, float(fmax_vec.max()))

        # evaluate candidate
        with torch.no_grad():
            x_norm = (f - X_mean) / X_std
            y_hat_real = model(x_norm) * y_std + y_mean
            val = float(y_hat_real.squeeze())
            if val > best_val:
                best_val = val
                best_f = f.detach().squeeze(0).cpu().numpy()

        print(f"Current index of restart is {ind}, with principal utility = {val:.6f}")

    return best_f, best_val


def delu_infer_(model, contracts):
    optimal_contract = contracts[0]
    optimal_obj = -np.inf
    model.eval()

    for contract in contracts:
        f = model.predict(contract)
        if f > optimal_obj:
            optimal_contract = contract
            optimal_obj = f

    return optimal_contract, optimal_obj


# ------------------------- 4) End-to-end wrapper ----------------------------- #
def run_delu_training_and_inference(cfg, m, n, N_samples, split_level):
    """
    Uses your existing problem builders:
      - create_sys_model(cfg, m, n)
      - build_dataset(m, n, cs)
      - simulate_round(f, cs)
      - contract_oracle_solver(p, c, v)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cs = create_sys_model(cfg, m, n)
    p, c, v, _ = cs

    # Build dataset & train DeLU
    time_start = time.time()
    F_tr, y_tr, f_max = build_dataset(m, n, cs, N_samples=N_samples, seed=0)
    model = train_delu(
        F_tr, y_tr, m, device=device,
        epochs=100, batch_size=1024,
        lr=3e-4, max_lr=1e-3, weight_decay=1e-4,
        h_eta=32, depth_eta=2, h_zeta=512
    )
    time_end = time.time()
    training_time = time_end-time_start

    # Gradient-based inference (multi-start + sub-argmax)
    time_start_ = time.time()
    f_star, est_val = infer_contract_adam(model, f_max=f_max, restarts=64, steps=400, lr=1e-1, device=device)
    time_end_ = time.time()
    inference_time = time_end_ - time_start_

    # Enumerate-based inference
    # time_start_ = time.time()
    # contracts = generate_candidate_contracts(f_max, m, split_level)
    # f_star, _ = delu_infer_(model, contracts)
    # time_end_ = time.time()
    # inference_time = time_end_ - time_start_

    # Report utilities
    # print("\nLearned contract f* (first 12 entries):", f_star[:12])
    print("Estimated principal utility ξ(f*):", float(model.predict(f_star)))

    return model, f_star, (training_time, inference_time)


def run_delu(cfg, N_samples, split_level: int = 3):
    print("================>Enter into the DeLU algorithm<================")
    # Generate the evaluation dataset
    n_actions_l = cfg.problem.n_actions_l
    m_outcomes_l = cfg.problem.m_outcomes_l
    n_instances = 3

    contract_to_optimal = []
    bonus_prin_u = []
    bonus_agent_u = []
    train_time_l = []
    infer_time_l = []
    for m in m_outcomes_l:
        row_cto = []
        row_bpu = []
        row_bau = []
        row_ttl = []
        row_itl = []
        for n in n_actions_l:
            cto = []
            bpu = []
            bau = []
            ttl = []
            itl = []
            for i in range(n_instances):
                contract_setting = create_sys_model(cfg, m, n)
                p, c, v, _ = contract_setting
                _, inferred_contract, (train_time, infer_time) = run_delu_training_and_inference(cfg, m, n, N_samples, split_level)
                wo_bonus_contract = np.zeros_like(inferred_contract)
                _, _, (principal_u, agent_u) = simulate_round(inferred_contract, contract_setting)
                _, _, (wob_principal_u, wob_agent_u) = simulate_round(wo_bonus_contract, contract_setting)
                oracle_best_prin_u, _ = contract_oracle_solver(p, c, v)
                bonus_improve_prin_u = round((principal_u - wob_principal_u) / wob_principal_u, 4)
                bonus_improve_agent_u = round(agent_u / 4e-5, 4)
                percentage_to_optimal = round(principal_u / oracle_best_prin_u, 4)
                cto.append(percentage_to_optimal)
                bpu.append(bonus_improve_prin_u)
                bau.append(bonus_improve_agent_u)
                ttl.append(train_time)
                itl.append(infer_time)
            row_cto.append(round(np.mean(cto), 4))
            row_bpu.append(round(np.mean(bpu), 4))
            row_bau.append(round(np.mean(bau), 4))
            row_ttl.append(round(np.mean(ttl), 4))
            row_itl.append(round(np.mean(itl), 4))
        contract_to_optimal.append(row_cto)
        bonus_prin_u.append(row_bpu)
        bonus_agent_u.append(row_bau)
        train_time_l.append(row_ttl)
        infer_time_l.append(row_itl)

    save_results("edgeAI_to_optimal_delu.csv", contract_to_optimal)
    save_results("edgeAI_bonus_prin_delu.csv", bonus_prin_u)
    save_results("edgeAI_bonus_agent_delu.csv", bonus_agent_u)
    save_results("edgeAI_train_time_delu.csv", train_time_l)
    save_results("edgeAI_inference_time_delu.csv", infer_time_l)
