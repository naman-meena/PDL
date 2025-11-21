# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 21:30:49 2025

@author: naman, swastik
"""
"""
PDL for AC-OPF (multiple IEEE test cases)
- Keeps printed cost as NaN (you requested cost not used).
- Both optimizers use lr=1e-4.
- Tests cases: 14, 30, 39, 57, 118, 300 (case300 included).
- Stability measures included: angle bounding, bounded dual outputs, gradient clipping,
  milder ALM schedule, per-case input normalization.
"""
import time
from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import torch
import torch.nn as nn
import torch.optim as optim
from pandapower.pypower.makeYbus import makeYbus
import warnings
warnings.filterwarnings("ignore")


# -------------------------
# Utilities: data extraction
# -------------------------
def get_pandapower_data(net, case_name: str) -> Dict:
    """Extract system data from a pandapower network object."""
    print(f"Loading data from pandapower: {case_name}")

    try:
        pp.runpp(net, calculate_voltage_angles=False, enforce_q_limits=False,
                 enforce_v_limits=False, init="flat")
    except Exception:
        # runpp may fail for some networks but _ppc often still exists
        pass

    if not hasattr(net, "_ppc") or net._ppc is None:
        pp.diagnostic(net)

    ppc = net._ppc
    baseMVA = ppc["baseMVA"]
    bus, gen, branch = ppc["bus"], ppc["gen"], ppc["branch"]

    Ybus, _, _ = makeYbus(baseMVA, bus, branch)
    Y_bus = Ybus.toarray()

    n_buses = net.bus.shape[0]
    n_generators = net.gen.shape[0]
    n_loads = net.load.shape[0]

    gen_buses = net.gen.bus.values.tolist()

    slack_bus = int(net.ext_grid.bus.values[0])
    pv_buses = list(set(gen_buses) - {slack_bus})
    all_buses = set(range(n_buses))
    pq_buses = list(all_buses - set(gen_buses))

    # generator voltage setpoints (default 1.0 pu)
    gen_vm = np.ones(n_buses)
    if "vm_pu" in net.gen.columns:
        for i, gb in enumerate(net.gen.bus.values):
            vm_val = net.gen.vm_pu.values[i]
            gen_vm[gb] = vm_val if not pd.isna(vm_val) else 1.0

    # base demands (pu)
    base_P = np.zeros(n_buses)
    base_Q = np.zeros(n_buses)
    for _, ld in net.load.iterrows():
        base_P[int(ld.bus)] += ld.p_mw
        base_Q[int(ld.bus)] += ld.q_mvar
    base_P_pu = base_P / baseMVA
    base_Q_pu = base_Q / baseMVA

    # costs (not used for objective printing; kept for potential future use) (we didnt consider the feasibility domain)
    if hasattr(net, "poly_cost") and net.poly_cost.shape[0] == n_generators:
        costs = net.poly_cost[net.poly_cost.et == "gen"].sort_values(by="element")
        c2 = costs.cp2_eur_per_mw2.values * (baseMVA**2)
        c1 = costs.cp1_eur_per_mw.values * baseMVA
        c0 = costs.cp0_eur.values
    else:
        # placeholder random coefficients (not used)
        c2 = np.random.uniform(0.005, 0.02, max(1, n_generators)) * (baseMVA**2)
        c1 = np.random.uniform(15.0, 35.0, max(1, n_generators)) * baseMVA
        c0 = np.random.uniform(80.0, 160.0, max(1, n_generators))

    gen_limits = {
        "P_min": net.gen.min_p_mw.values / baseMVA,
        "P_max": net.gen.max_p_mw.values / baseMVA,
        "Q_min": net.gen.min_q_mvar.values / baseMVA,
        "Q_max": net.gen.max_q_mvar.values / baseMVA,
    }

    V_min = float(net.bus.min_vm_pu.min())
    V_max = float(net.bus.max_vm_pu.max())

    print(f"Data loaded: {n_buses} buses, {n_generators} generators, {n_loads} loads.")
    print(f"Base MVA: {baseMVA}")
    print(f"Slack bus: {slack_bus}, PV buses: {len(pv_buses)}, PQ buses: {len(pq_buses)}")

    return {
        "n_buses": n_buses,
        "n_generators": n_generators,
        "n_loads": n_loads,
        "gen_buses": gen_buses,
        "slack_bus": slack_bus,
        "pv_buses": pv_buses,
        "pq_buses": pq_buses,
        "gen_voltage_setpoints": gen_vm,
        "Y_bus": Y_bus,
        "base_P_demand_pu": base_P_pu,
        "base_Q_demand_pu": base_Q_pu,
        "cost_coeffs": {"c2": c2, "c1": c1, "c0": c0},
        "gen_limits": gen_limits,
        "V_min": V_min,
        "V_max": V_max,
        "baseMVA": baseMVA,
    }


# -------------------------
# Scenario generation
# -------------------------
def generate_load_scenarios(system_data: Dict, n_samples: int, load_variation: float = 0.3):
    """Create randomized P/Q demand scenarios (p.u.)."""
    n_buses = system_data["n_buses"]
    base_P = system_data["base_P_demand_pu"]
    base_Q = system_data["base_Q_demand_pu"]

    P = np.zeros((n_samples, n_buses))
    Q = np.zeros((n_samples, n_buses))

    for i in range(n_samples):
        lf = 1.0 + np.random.randn() * load_variation
        lf = np.clip(lf, 0.7, 1.3)
        ind = 1.0 + np.random.randn(n_buses) * 0.1
        P[i, :] = np.clip(base_P * lf * ind, 0.0, None)
        Q[i, :] = np.clip(base_Q * lf * ind, 0.0, None)

    return P, Q


# -------------------------
# Neural networks
# -------------------------
class ACPFPrimalNet(nn.Module):
    """Primal network predicting Pg, Qg, V (PQ buses) and angles (n-1)."""

    def __init__(self, n_buses: int, n_generators: int, n_pq_buses: int,
                 hidden_dim: int = 256, max_angle_rad: float = np.pi / 12.0):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.n_pq_buses = n_pq_buses
        self.max_angle_rad = float(max_angle_rad)

        input_dim = 2 * n_buses
        output_dim = 2 * n_generators + n_pq_buses + (n_buses - 1)
        hidden_size = max(64, int(1.2 * output_dim))

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_size),
            nn.ReLU(),
        )

        self.pg_head = nn.Linear(hidden_size, n_generators)
        self.qg_head = nn.Linear(hidden_size, n_generators)
        self.v_head = nn.Linear(hidden_size, n_pq_buses)
        self.theta_head = nn.Linear(hidden_size, n_buses - 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, P_demand, Q_demand, gen_limits, V_min, V_max,
                pv_buses, pq_buses, gen_voltage_setpoints, slack_bus: int = 0):
        device = P_demand.device
        x = torch.cat([P_demand, Q_demand], dim=1)
        h = self.embedding(x)

        # Pg and Qg bounded with sigmoid to [P_min,P_max] and [Q_min,Q_max]
        pg_raw = self.sigmoid(self.pg_head(h))
        P_min = torch.tensor(gen_limits["P_min"], device=device, dtype=torch.float32).unsqueeze(0)
        P_max = torch.tensor(gen_limits["P_max"], device=device, dtype=torch.float32).unsqueeze(0)
        Pg = P_min + pg_raw * (P_max - P_min)

        qg_raw = self.sigmoid(self.qg_head(h))
        Q_min = torch.tensor(gen_limits["Q_min"], device=device, dtype=torch.float32).unsqueeze(0)
        Q_max = torch.tensor(gen_limits["Q_max"], device=device, dtype=torch.float32).unsqueeze(0)
        Qg = Q_min + qg_raw * (Q_max - Q_min)

        # Voltages: set PV and slack to setpoints, predict PQ bus voltages
        batch = P_demand.shape[0]
        V = torch.zeros(batch, self.n_buses, device=device, dtype=torch.float32)
        gen_v_set = torch.tensor(gen_voltage_setpoints, device=device, dtype=torch.float32)
        for b in pv_buses + [slack_bus]:
            V[:, b] = gen_v_set[b]

        if self.n_pq_buses > 0:
            v_pq_raw = self.sigmoid(self.v_head(h))
            V_pq = V_min + v_pq_raw * (V_max - V_min)
            for i, bus in enumerate(pq_buses):
                V[:, bus] = V_pq[:, i]

        # Angles: bounded via tanh to [-max_angle_rad, max_angle_rad]
        theta_raw = self.theta_head(h)
        theta_scaled = torch.tanh(theta_raw) * float(self.max_angle_rad)
        theta = torch.zeros(batch, self.n_buses, device=device, dtype=torch.float32)
        if slack_bus == 0:
            theta[:, 1:] = theta_scaled
        else:
            theta[:, :slack_bus] = theta_scaled[:, :slack_bus]
            theta[:, slack_bus + 1:] = theta_scaled[:, slack_bus:]

        return Pg, Qg, V, theta


class ACPFDualNet(nn.Module):
    """Dual network predicting bounded multipliers (λP, λQ)."""

    def __init__(self, n_buses: int, hidden_dim: int = 256, lambda_scale: float = 0.5):
        super().__init__()
        self.lambda_scale = float(lambda_scale)
        input_dim = 2 * n_buses
        output_dim = 2 * n_buses

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, P_demand, Q_demand):
        x = torch.cat([P_demand, Q_demand], dim=1)
        raw = self.net(x)
        return torch.tanh(raw) * self.lambda_scale


# -------------------------
# PDL training logic
# -------------------------
class PDL_ACPF:
    """Primal-Dual Learner for AC Power Flow violations."""

    def __init__(self, system_data: Dict, device: str = "cpu",
                 rho_init: float = 1.0, rho_max: float = 1e4,
                 alpha: float = 1.2, tau: float = 0.9,
                 lr: float = 1e-4, dual_lambda_scale: float = 0.5,
                 max_angle_rad: float = np.pi / 12.0):
        self.device = torch.device(device)
        self.sys = system_data

        self.n_buses = system_data["n_buses"]
        self.n_generators = system_data["n_generators"]
        self.gen_buses = system_data["gen_buses"]
        self.slack_bus = int(system_data["slack_bus"])
        self.pv_buses = system_data["pv_buses"]
        self.pq_buses = system_data["pq_buses"]
        self.n_pq_buses = len(self.pq_buses)
        self.gen_voltage_setpoints = system_data["gen_voltage_setpoints"]
        self.Y_bus = torch.tensor(system_data["Y_bus"], dtype=torch.complex64).to(self.device)
        self.gen_limits = system_data["gen_limits"]
        self.V_min = system_data["V_min"]
        self.V_max = system_data["V_max"]
        self.baseMVA = system_data["baseMVA"]

        self.rho = float(rho_init)
        self.rho_max = float(rho_max)
        self.alpha = float(alpha)
        self.tau = float(tau)

        # networks
        self.primal_net = ACPFPrimalNet(self.n_buses, self.n_generators, self.n_pq_buses,
                                        max_angle_rad=max_angle_rad).to(self.device)
        self.dual_net = ACPFDualNet(self.n_buses, lambda_scale=dual_lambda_scale).to(self.device)

        # optimizers (lr = 1e-4 per request)
        self.primal_optimizer = optim.Adam(self.primal_net.parameters(), lr=lr)
        self.dual_optimizer = optim.Adam(self.dual_net.parameters(), lr=lr)

        self.history = defaultdict(list)

    def compute_power_balance(self, Pg, Qg, V, theta, P_demand, Q_demand):
        """Return P_viol, Q_viol (pu) for a batch."""
        batch = V.shape[0]
        device = self.device

        P_inj = torch.zeros(batch, self.n_buses, device=device, dtype=torch.float32)
        Q_inj = torch.zeros(batch, self.n_buses, device=device, dtype=torch.float32)
        for idx, bus in enumerate(self.gen_buses):
            P_inj[:, bus] += Pg[:, idx]
            Q_inj[:, bus] += Qg[:, idx]

        P_inj = P_inj - P_demand
        Q_inj = Q_inj - Q_demand

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        V_complex = V * (cos_t + 1j * sin_t)

        I_complex = torch.matmul(V_complex, self.Y_bus.T)
        S_complex = V_complex * torch.conj(I_complex)

        P_calc = S_complex.real
        Q_calc = S_complex.imag

        P_viol = P_calc - P_inj
        Q_viol = Q_calc - Q_inj
        return P_viol, Q_viol

    def primal_loss(self, Pg, Qg, V, theta, P_demand, Q_demand, multipliers):
        P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)
        lambda_P = multipliers[:, : self.n_buses]
        lambda_Q = multipliers[:, self.n_buses :]

        lag = torch.sum(lambda_P * P_viol, dim=1).mean() + torch.sum(lambda_Q * Q_viol, dim=1).mean()
        penalty = (self.rho / 2.0) * (torch.sum(P_viol**2, dim=1).mean() + torch.sum(Q_viol**2, dim=1).mean())
        return lag + penalty

    def dual_loss(self, multipliers, multipliers_old, Pg, Qg, V, theta, P_demand, Q_demand):
        P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)
        lambda_P_target = multipliers_old[:, : self.n_buses] + self.rho * P_viol
        lambda_Q_target = multipliers_old[:, self.n_buses :] + self.rho * Q_viol
        loss = torch.mean((multipliers[:, : self.n_buses] - lambda_P_target) ** 2)
        loss += torch.mean((multipliers[:, self.n_buses :] - lambda_Q_target) ** 2)
        return loss

    def train_epoch(self, P_batch, Q_batch, inner_iters: int = 250):
        """One outer epoch: several inner primal updates and dual updates."""
        primal_losses = []
        costs = []  # intentionally empty -> will be NaN when averaged

        # PRIMAL updates
        for _ in range(inner_iters):
            self.primal_optimizer.zero_grad()
            Pg, Qg, V, theta = self.primal_net(P_batch, Q_batch,
                                              self.gen_limits, self.V_min, self.V_max,
                                              self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus)
            multipliers = self.dual_net(P_batch, Q_batch).detach()
            loss = self.primal_loss(Pg, Qg, V, theta, P_batch, Q_batch, multipliers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.primal_net.parameters(), max_norm=5.0)
            self.primal_optimizer.step()
            primal_losses.append(loss.item())

        # COPY dual net for "old" multipliers
        dual_net_old = ACPFDualNet(self.n_buses, lambda_scale=self.dual_net.lambda_scale).to(self.device)
        dual_net_old.load_state_dict(self.dual_net.state_dict())

        # DUAL updates
        dual_losses = []
        for _ in range(inner_iters):
            self.dual_optimizer.zero_grad()
            Pg, Qg, V, theta = self.primal_net(P_batch, Q_batch,
                                              self.gen_limits, self.V_min, self.V_max,
                                              self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus)
            Pg, Qg, V, theta = Pg.detach(), Qg.detach(), V.detach(), theta.detach()

            multipliers = self.dual_net(P_batch, Q_batch)
            multipliers_old = dual_net_old(P_batch, Q_batch).detach()

            loss = self.dual_loss(multipliers, multipliers_old, Pg, Qg, V, theta, P_batch, Q_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dual_net.parameters(), max_norm=5.0)
            self.dual_optimizer.step()
            dual_losses.append(loss.item())

        # diagnostics
        with torch.no_grad():
            Pg, Qg, V, theta = self.primal_net(P_batch, Q_batch,
                                              self.gen_limits, self.V_min, self.V_max,
                                              self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus)
            P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_batch, Q_batch)
            max_P = torch.max(torch.abs(P_viol)).item()
            max_Q = torch.max(torch.abs(Q_viol)).item()
            max_viol = max(max_P, max_Q)
            mean_viol = (torch.mean(torch.abs(P_viol)) + torch.mean(torch.abs(Q_viol))).item() / 2.0
            slack_v = V[:, self.slack_bus].mean().item()

            if not hasattr(self, "prev_max_viol"):
                self.prev_max_viol = max_viol
            elif max_viol > self.tau * self.prev_max_viol:
                self.rho = min(self.alpha * self.rho, self.rho_max)
            self.prev_max_viol = max_viol

        self.history["primal_loss"].append(float(np.mean(primal_losses)))
        self.history["dual_loss"].append(float(np.mean(dual_losses)))
        self.history["cost"].append(float(np.mean(costs)) if costs else float("nan"))
        self.history["max_P_viol"].append(max_P)
        self.history["max_Q_viol"].append(max_Q)
        self.history["max_viol"].append(max_viol)
        self.history["mean_viol"].append(mean_viol)
        self.history["rho"].append(self.rho)
        self.history["slack_bus_voltage"].append(slack_v)

        return float(np.mean(primal_losses)), max_viol, float(np.mean(costs)) if costs else float("nan")

    def predict(self, P_demand, Q_demand):
        self.primal_net.eval()
        with torch.no_grad():
            Pg, Qg, V, theta = self.primal_net(P_demand, Q_demand,
                                              self.gen_limits, self.V_min, self.V_max,
                                              self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus)
        self.primal_net.train()
        return Pg, Qg, V, theta


# -------------------------
# Plot helpers (kept compact)
# -------------------------
def plot_training_results(history: Dict, case_name: str, slack_bus: int):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"PDL Training Results - {case_name}", fontsize=14)

    axes[0, 0].plot(history["cost"])
    axes[0, 0].set_title("Objective (Cost)")
    axes[0, 1].plot(history["primal_loss"], label="Primal")
    axes[0, 1].plot(history["dual_loss"], label="Dual")
    axes[0, 1].legend(); axes[0, 1].set_yscale("log"); axes[0, 1].set_title("Losses")

    axes[0, 2].plot(history["max_P_viol"], label="Max P"); axes[0, 2].plot(history["max_Q_viol"], label="Max Q")
    axes[0, 2].legend(); axes[0, 2].set_yscale("log"); axes[0, 2].set_title("Max Violations")

    axes[0, 3].plot(history["slack_bus_voltage"]); axes[0, 3].axhline(1.0, linestyle="--")
    axes[0, 3].set_title(f"Slack Bus {slack_bus} Voltage")

    axes[1, 0].plot(history["mean_viol"]); axes[1, 0].set_yscale("log"); axes[1, 0].set_title("Mean Violation")
    axes[1, 1].plot(history["rho"]); axes[1, 1].set_title("Penalty ρ")
    axes[1, 2].plot(history["max_viol"]); axes[1, 2].set_yscale("log"); axes[1, 2].set_title("Overall Max Violation")
    axes[1, 3].hist(history["slack_bus_voltage"], bins=20); axes[1, 3].set_title("Slack V Dist")

    plt.tight_layout()
    return fig


def plot_solution_quality(Pg, Qg, V, theta, P_viol, Q_viol, case_name: str, baseMVA: float, slack_bus: int):
    Pg_np = Pg.cpu().numpy() * baseMVA
    Qg_np = Qg.cpu().numpy() * baseMVA
    V_np = V.cpu().numpy()
    theta_np = np.degrees(theta.cpu().numpy())
    P_viol_np = P_viol.cpu().numpy() * baseMVA
    Q_viol_np = Q_viol.cpu().numpy() * baseMVA

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Solution Quality - {case_name}", fontsize=14)

    # generator powers
    ax = axes[0, 0]; ax.set_title("Generator Pg (samples)"); ax.plot(Pg_np.T, alpha=0.3)
    ax = axes[0, 1]; ax.set_title("Generator Qg (samples)"); ax.plot(Qg_np.T, alpha=0.3)

    ax = axes[0, 2]; ax.set_title("Voltage profile (sampled)"); ax.plot(V_np.T, alpha=0.3); ax.axhline(1.0, linestyle="--")
    ax = axes[0, 3]; ax.set_title(f"Slack Bus {slack_bus} V"); ax.plot(V_np[:, slack_bus], marker="o")

    ax = axes[1, 0]; ax.set_title("Angles (deg)"); ax.plot(theta_np.T, alpha=0.3)
    ax = axes[1, 1]; ax.set_title("|P viol| (MW)"); ax.hist(np.abs(P_viol_np.flatten()), bins=50)
    ax = axes[1, 2]; ax.set_title("|Q viol| (MVAr)"); ax.hist(np.abs(Q_viol_np.flatten()), bins=50)
    axes[1, 3].axis("off")

    plt.tight_layout()
    return fig


# -------------------------
# Runner
# -------------------------
def run_acpf_experiment(case_name: str, system_data: Dict,
                        n_train: int = 25000, n_test: int = 100,
                        max_outer_iters: int = 50, convergence_threshold: float = 1e-2,
                        inner_iters: int = 250, device: str = None) -> Tuple[PDL_ACPF, object, object, Dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f"Running AC-PF PDL Experiment: {case_name}\nDevice: {device}")
    print(f"Buses: {system_data['n_buses']}, Gens: {system_data['n_generators']}, Loads: {system_data['n_loads']}")

    # training data (per-case normalization)
    P_train, Q_train = generate_load_scenarios(system_data, n_train, load_variation=0.3)
    demand_scale = max(1.0, float(np.max(P_train)))
    P_train /= demand_scale
    Q_train /= demand_scale
    P_train_t = torch.tensor(P_train, dtype=torch.float32).to(device)
    Q_train_t = torch.tensor(Q_train, dtype=torch.float32).to(device)

    pdl = PDL_ACPF(system_data, device=device, rho_init=1.0, rho_max=1e4,
                   alpha=1.2, tau=0.9, lr=1e-4, dual_lambda_scale=0.5,
                   max_angle_rad=np.pi / 12.0)

    start_time = time.time()
    k = 0
    max_viol = float("inf")

    print(f"Training until max violation < {convergence_threshold} (or max {max_outer_iters} outer its)")
    while max_viol > convergence_threshold and k < max_outer_iters:
        _, max_viol, cost = pdl.train_epoch(P_train_t, Q_train_t, inner_iters=inner_iters)

        # sample slack voltage
        with torch.no_grad():
            _, _, V_samp, _ = pdl.primal_net(P_train_t[:100], Q_train_t[:100],
                                            pdl.gen_limits, pdl.V_min, pdl.V_max,
                                            pdl.pv_buses, pdl.pq_buses, pdl.gen_voltage_setpoints, pdl.slack_bus)
            slack_v = V_samp[:, pdl.slack_bus].mean().item()

        print(f"Iter {k + 1:3d}: Cost={cost}, MaxViol={max_viol:.6f} pu, SlackV={slack_v:.4f} pu, rho={pdl.rho:.2f}")
        k += 1

    training_time = time.time() - start_time
    print(f"Training stopped after {k} iterations ({training_time:.1f}s).")

    # test (scale test inputs same way)
    P_test, Q_test = generate_load_scenarios(system_data, n_test, load_variation=0.3)
    P_test /= demand_scale
    Q_test /= demand_scale
    P_test_t = torch.tensor(P_test, dtype=torch.float32).to(device)
    Q_test_t = torch.tensor(Q_test, dtype=torch.float32).to(device)

    start_inf = time.time()
    Pg, Qg, V, theta = pdl.predict(P_test_t, Q_test_t)
    inference_time = time.time() - start_inf

    with torch.no_grad():
        P_viol, Q_viol = pdl.compute_power_balance(Pg, Qg, V, theta, P_test_t, Q_test_t)

    max_P_viol_pu = float(torch.max(torch.abs(P_viol)).item())
    max_Q_viol_pu = float(torch.max(torch.abs(Q_viol)).item())
    mean_P_viol_pu = float(torch.mean(torch.abs(P_viol)).item())
    mean_Q_viol_pu = float(torch.mean(torch.abs(Q_viol)).item())
    max_viol_pu = max(max_P_viol_pu, max_Q_viol_pu)

    slack_Vs = V[:, pdl.slack_bus].cpu().numpy()
    slack_stats = {
        "mean": float(np.mean(slack_Vs)),
        "std": float(np.std(slack_Vs)),
        "min": float(np.min(slack_Vs)),
        "max": float(np.max(slack_Vs)),
    }

    baseMVA = pdl.baseMVA
    Pg_total_pu = float(torch.sum(Pg, dim=1).mean().item())
    Pg_total_mw = Pg_total_pu * baseMVA

    results = {
        "training_time": training_time,
        "inference_time_per_sample": inference_time / max(1, len(P_test)),
        "max_P_viol_pu": max_P_viol_pu,
        "max_Q_viol_pu": max_Q_viol_pu,
        "max_viol_pu": max_viol_pu,
        "mean_P_viol_pu": mean_P_viol_pu,
        "mean_Q_viol_pu": mean_Q_viol_pu,
        "V_mean": float(torch.mean(V).item()),
        "V_std": float(torch.std(V).item()),
        "V_min": float(torch.min(V).item()),
        "V_max": float(torch.max(V).item()),
        "slack_bus": pdl.slack_bus,
        "slack_V_mean": slack_stats["mean"],
        "slack_V_std": slack_stats["std"],
        "slack_V_min": slack_stats["min"],
        "slack_V_max": slack_stats["max"],
        "Pg_total_mw": Pg_total_mw,
        "baseMVA": baseMVA,
    }

    # plotting
    fig_train = plot_training_results(pdl.history, case_name, pdl.slack_bus)
    fig_sol = plot_solution_quality(Pg, Qg, V, theta, P_viol, Q_viol, case_name, baseMVA, pdl.slack_bus)

    return pdl, fig_train, fig_sol, results


# -------------------------
# Comparison printing
# -------------------------
def print_comparison_table(results_dict: Dict):
    rows = []
    for name, res in results_dict.items():
        rows.append({
            "Case": name,
            "Slack Bus": res["slack_bus"],
            "Slack V (pu)": f"{res['slack_V_mean']:.4f}±{res['slack_V_std']:.4f}",
            "Max Viol (pu)": f"{res['max_viol_pu']:.6f}",
            "Max P (MW)": f"{res['max_P_viol_pu'] * res['baseMVA']:.4f}",
            "Max Q (MVAr)": f"{res['max_Q_viol_pu'] * res['baseMVA']:.4f}",
            "Train (s)": f"{res['training_time']:.1f}",
            "Infer (ms)": f"{res['inference_time_per_sample']*1000:.4f}",
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))


# -------------------------
# Main: run multiple cases
# -------------------------
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PDL FOR AC-OPTIMAL POWER FLOW: MULTIPLE IEEE CASES (CLEANED)")
    print("lr=1e-4, cost printed as NaN, cases include case300")
    print("=" * 80)

    case_numbers = [14, 30, 39, 57, 118, 300]
    all_results = {}
    all_models = {}
    all_figs = {}

    for num in case_numbers:
        func_name = f"case{num}"
        try:
            net = getattr(nw, func_name)()
        except Exception as e:
            print(f"Unable to load {func_name}: {e} -> skipping")
            continue

        system_data = get_pandapower_data(net, func_name)
        label = f"Case {num}"
        try:
            model, fig1, fig2, results = run_acpf_experiment(
                case_name=f"IEEE {label}",
                system_data=system_data,
                n_train=25000,
                n_test=100,
                max_outer_iters=50,
                convergence_threshold=1e-2,
                inner_iters=250,
                device=None,  # auto-detect
            )
            all_results[label] = results
            all_models[label] = model
            all_figs[label] = (fig1, fig2)
        except Exception as e:
            print(f"Error running experiment for {label}: {e}")

    if all_results:
        print("\nSummary of results:")
        print_comparison_table(all_results)
        # produce a combined comparison figure if desired
        plt.show()

    print("\nAll experiments completed.")
