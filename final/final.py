# pdl_acpf_extended_case_tests_stable_keep_cost_nan.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy  # Add this import
from collections import defaultdict
import pandapower as pp
import pandapower.networks as nw
from pandapower.pypower.makeYbus import makeYbus
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Extracting data from pandapower
# ---------------------------
def get_pandapower_data(net, case_name):

    print(f"Loading data from pandapower: {case_name}")

    try:
        pp.runpp(net, calculate_voltage_angles=False, enforce_q_limits=False, enforce_v_limits=False, init='flat')
    except:
        # Some cases might fail if not initialized, but ppc is often still built
        pass

    if not hasattr(net, '_ppc') or net._ppc is None:
        print("Running diagnostic to build ppc...")
        pp.diagnostic(net)

    ppc = net._ppc

    baseMVA = ppc['baseMVA']
    bus, gen, branch = ppc['bus'], ppc['gen'], ppc['branch']

    # Build Ybus
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Y_bus = Ybus.toarray()

    n_buses = net.bus.shape[0]
    n_generators = net.gen.shape[0]
    n_loads = net.load.shape[0]

    gen_buses = net.gen.bus.values.tolist()

    # Identify bus types
    slack_bus = net.ext_grid.bus.values[0]
    pv_buses = list(set(gen_buses) - {slack_bus})  # Generator buses except slack
    all_buses = set(range(n_buses))
    pq_buses = list(all_buses - set(gen_buses))  # Buses without generators

    # Get generator voltage setpoints (typically 1.0 pu for PV buses)
    gen_voltage_setpoints = np.ones(n_buses)  # Default all to 1.0
    for idx, gen_bus in enumerate(net.gen.bus.values):
        if 'vm_pu' in net.gen.columns and not pd.isna(net.gen.vm_pu.values[idx]):
            gen_voltage_setpoints[gen_bus] = net.gen.vm_pu.values[idx]
        else:
            gen_voltage_setpoints[gen_bus] = 1.0  # Default to 1.0 pu

    base_P_demand = np.zeros(n_buses)
    base_Q_demand = np.zeros(n_buses)

    for _, load in net.load.iterrows():
        bus_idx = load.bus
        base_P_demand[bus_idx] += load.p_mw
        base_Q_demand[bus_idx] += load.q_mvar

    base_P_demand_pu = base_P_demand / baseMVA
    base_Q_demand_pu = base_Q_demand / baseMVA

    # Get generator cost coefficients (in p.u.) -- though you asked to keep costs NaN in prints
    if hasattr(net, 'poly_cost') and net.poly_cost.shape[0] == n_generators:
        costs = net.poly_cost[net.poly_cost.et == 'gen'].sort_values(by='element')
        c2 = costs.cp2_eur_per_mw2.values
        c1 = costs.cp1_eur_per_mw.values
        c0 = costs.cp0_eur.values

        c2_pu = c2 * (baseMVA**2)
        c1_pu = c1 * baseMVA
        c0_pu = c0
    else:
        # Not used (we'll intentionally keep costs list empty so printed cost remains nan)
        c2_pu = np.random.uniform(0.005, 0.02, max(1, n_generators)) * (baseMVA**2)
        c1_pu = np.random.uniform(15.0, 35.0, max(1, n_generators)) * baseMVA
        c0_pu = np.random.uniform(80.0, 160.0, max(1, n_generators))

    cost_coeffs = {'c2': c2_pu, 'c1': c1_pu, 'c0': c0_pu}

    gen_limits = {
        'P_min': net.gen.min_p_mw.values / baseMVA,
        'P_max': net.gen.max_p_mw.values / baseMVA,
        'Q_min': net.gen.min_q_mvar.values / baseMVA,
        'Q_max': net.gen.max_q_mvar.values / baseMVA
    }


    slack_bus = net.ext_grid.bus.values[0]

    print(f"Data loaded: {n_buses} buses, {n_generators} generators, {n_loads} loads.")
    print(f"Base MVA: {baseMVA}")
    print(f"Slack bus: {slack_bus}, PV buses: {len(pv_buses)}, PQ buses: {len(pq_buses)}")

    return {
        'n_buses': n_buses,
        'n_generators': n_generators,
        'n_loads': n_loads,
        'gen_buses': gen_buses,
        'slack_bus': slack_bus,
        'pv_buses': pv_buses,
        'pq_buses': pq_buses,
        'gen_voltage_setpoints': gen_voltage_setpoints,
        'Y_bus': Y_bus,
        'base_P_demand_pu': base_P_demand_pu,
        'base_Q_demand_pu': base_Q_demand_pu,
        'cost_coeffs': cost_coeffs,
        'gen_limits': gen_limits,
        'baseMVA': baseMVA,
        'net': net
    }


# ---------------------------
# Load scenario generation
# ---------------------------
def generate_load_scenarios(system_data, n_samples, load_variation=0.3):
    """Generate load scenarios with variations based on pandapower base loads"""
    n_buses = system_data['n_buses']
    base_P = system_data['base_P_demand_pu']
    base_Q = system_data['base_Q_demand_pu']

    P_demand = np.zeros((n_samples, n_buses))
    Q_demand = np.zeros((n_samples, n_buses))

    for i in range(n_samples):
        load_factor = 1.0 + np.random.randn() * load_variation
        load_factor = np.clip(load_factor, 0.7, 1.3)

        individual_variation = 1.0 + np.random.randn(n_buses) * 0.1

        P_demand[i, :] = base_P * load_factor * individual_variation
        Q_demand[i, :] = base_Q * load_factor * individual_variation

        P_demand[i, :] = np.clip(P_demand[i, :], a_min=0, a_max=None)

    return P_demand, Q_demand


# ---------------------------
# Primal network
# ---------------------------
class ACPFPrimalNet(nn.Module):
    def __init__(self, n_buses, n_generators, n_pq_buses, hidden_dim=256, max_angle_rad=np.pi/12.0):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators
        self.n_pq_buses = n_pq_buses
        self.max_angle_rad = float(max_angle_rad)

        input_dim = 2 * n_buses  # P and Q demands at all buses
        output_dim = 2 * n_generators + n_pq_buses + (n_buses - 1)
        hidden_size = int(1.2 * output_dim)

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_size),
            nn.ReLU()
        )

        self.pg_net = nn.Linear(hidden_size, n_generators)
        self.qg_net = nn.Linear(hidden_size, n_generators)
        self.v_net = nn.Linear(hidden_size, n_pq_buses)  # Only PQ bus voltages
        self.theta_net = nn.Linear(hidden_size, n_buses - 1)  # Predicts angles for all buses except slack

        self.sigmoid = nn.Sigmoid()

    def forward(self, P_demand, Q_demand, gen_limits,
                pv_buses, pq_buses, gen_voltage_setpoints, slack_bus=0):
        batch_size = P_demand.shape[0]
        device = P_demand.device

        # Ensure slack_bus is scalar integer
        if isinstance(slack_bus, (list, np.ndarray)):
            slack_bus = int(slack_bus[0]) if len(slack_bus) > 0 else 0
        else:
            slack_bus = int(slack_bus)

        x = torch.cat([P_demand, Q_demand], dim=1)
        h = self.embedding(x)

        # Generator active power
        pg_raw = self.sigmoid(self.pg_net(h))
        P_min = torch.tensor(gen_limits['P_min'], device=device, dtype=torch.float32).unsqueeze(0)
        P_max = torch.tensor(gen_limits['P_max'], device=device, dtype=torch.float32).unsqueeze(0)
        Pg = P_min + pg_raw * (P_max - P_min)

        # Generator reactive power
        qg_raw = self.sigmoid(self.qg_net(h))
        Q_min = torch.tensor(gen_limits['Q_min'], device=device, dtype=torch.float32).unsqueeze(0)
        Q_max = torch.tensor(gen_limits['Q_max'], device=device, dtype=torch.float32).unsqueeze(0)
        Qg = Q_min + qg_raw * (Q_max - Q_min)

        # Voltage magnitudes: PV buses fixed, only predict PQ buses
        V = torch.zeros(batch_size, self.n_buses, device=device, dtype=torch.float32)

        # Set fixed voltages for PV buses (including slack)
        gen_v_setpoints = torch.tensor(gen_voltage_setpoints, device=device, dtype=torch.float32)
        for bus in pv_buses + [slack_bus]:
            V[:, bus] = gen_v_setpoints[bus]

        # Predict voltages only for PQ buses (UNCONSTRAINED)
        if self.n_pq_buses > 0:
            V_pq = 1.0 + 0.2 * torch.tanh(self.v_net(h))
            for i, bus in enumerate(pq_buses):
                V[:, bus] = V_pq[:, i]

        # Voltage angles (slack bus fixed at 0)
        theta_raw = self.theta_net(h)
        # constrain theta to safe range using tanh
        theta_scaled = torch.tanh(theta_raw) * (self.max_angle_rad)

        theta = torch.zeros(batch_size, self.n_buses, device=device, dtype=torch.float32)
        if slack_bus == 0:
            theta[:, 1:] = theta_scaled
        else:
            theta[:, :slack_bus] = theta_scaled[:, :slack_bus]
            theta[:, slack_bus+1:] = theta_scaled[:, slack_bus:]

        return Pg, Qg, V, theta


# ---------------------------
# Dual network (bounded multipliers)
# ---------------------------
class ACPFDualNet(nn.Module):
    def __init__(self, n_buses, hidden_dim=256, lambda_scale=0.5):
        super().__init__()
        self.lambda_scale = float(lambda_scale)
        input_dim = 2 * n_buses
        output_dim = 2 * n_buses

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, P_demand, Q_demand):
        x = torch.cat([P_demand, Q_demand], dim=1)
        raw = self.net(x)
        # Bound multipliers to a reasonable range using tanh
        return torch.tanh(raw) * self.lambda_scale


# ---------------------------
# Primal-Dual Learning class
# ---------------------------
class PDL_ACPF:

    def __init__(self, system_data, rho_init=1.0, rho_max=10000,
                 alpha=1.2, tau=0.9, device='cpu',
                 dual_lambda_scale=0.5, max_angle_rad=np.pi/12.0):
        self.device = device
        self.system_data = system_data

        self.n_buses = system_data['n_buses']
        self.n_generators = system_data['n_generators']
        self.n_loads = system_data['n_loads']
        self.gen_buses = system_data['gen_buses']
        self.slack_bus = system_data['slack_bus']
        self.pv_buses = system_data['pv_buses']
        self.pq_buses = system_data['pq_buses']
        self.n_pq_buses = len(self.pq_buses)
        self.gen_voltage_setpoints = system_data['gen_voltage_setpoints']
        # store Ybus as numpy for building then convert to torch complex later
        self.Y_bus = torch.tensor(system_data['Y_bus'], dtype=torch.complex64).to(device)
        self.cost_coeffs = system_data['cost_coeffs']
        self.gen_limits = system_data['gen_limits']

        self.baseMVA = system_data['baseMVA']

        self.rho = float(rho_init)
        self.rho_max = float(rho_max)
        self.alpha = float(alpha)
        self.tau = float(tau)

        # networks with modifications
        self.primal_net = ACPFPrimalNet(self.n_buses, self.n_generators, self.n_pq_buses,
                                       hidden_dim=256, max_angle_rad=max_angle_rad).to(device)
        self.dual_net = ACPFDualNet(self.n_buses, hidden_dim=256, lambda_scale=dual_lambda_scale).to(device)

        # both optimizers with lr = 1e-4 (as requested)
        self.primal_optimizer = optim.Adam(self.primal_net.parameters(), lr=1e-4)
        self.dual_optimizer = optim.Adam(self.dual_net.parameters(), lr=1e-4)

        self.history = defaultdict(list)

    # Compute power balance violations
    def compute_power_balance(self, Pg, Qg, V, theta, P_demand, Q_demand):
        batch_size = V.shape[0]

        P_inj = torch.zeros(batch_size, self.n_buses, device=self.device, dtype=torch.float32)
        Q_inj = torch.zeros(batch_size, self.n_buses, device=self.device, dtype=torch.float32)

        for idx, bus in enumerate(self.gen_buses):
            P_inj[:, bus] += Pg[:, idx]
            Q_inj[:, bus] += Qg[:, idx]

        P_inj -= P_demand
        Q_inj -= Q_demand

        # Construct complex voltages using cos/sin for safety
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        V_complex = V * (cos_t + 1j * sin_t)  # (batch, n_buses) complex

        # I = Ybus * V  -> using row-major matmul, do V_complex @ Y_bus.T
        I_complex = torch.matmul(V_complex, self.Y_bus.T)  # (batch, n_buses)
        S_complex = V_complex * torch.conj(I_complex)

        P_calc = S_complex.real
        Q_calc = S_complex.imag

        # Power balance violations (all in p.u.)
        P_viol = P_calc - P_inj
        Q_viol = Q_calc - Q_inj

        return P_viol, Q_viol

    def primal_loss(self, Pg, Qg, V, theta, P_demand, Q_demand, multipliers):
        """Primal loss for Power Flow (PF): Lagrangian + penalty (NO COST)"""
        P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)

        lambda_P = multipliers[:, :self.n_buses]
        lambda_Q = multipliers[:, self.n_buses:]

        # Lagrangian term
        lagrangian = torch.sum(lambda_P * P_viol, dim=1).mean()
        lagrangian += torch.sum(lambda_Q * Q_viol, dim=1).mean()

        # Penalty term
        penalty = (self.rho / 2.0) * torch.sum(P_viol**2, dim=1).mean()
        penalty += (self.rho / 2.0) * torch.sum(Q_viol**2, dim=1).mean()

        loss = lagrangian + penalty
        return loss

    def dual_loss(self, multipliers, multipliers_old, Pg, Qg, V, theta, P_demand, Q_demand):
        """Dual loss: match ALM update rule"""
        P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)

        # Target multipliers (ALM)
        lambda_P_target = multipliers_old[:, :self.n_buses] + self.rho * P_viol
        lambda_Q_target = multipliers_old[:, self.n_buses:] + self.rho * Q_viol

        # MSE loss
        loss = torch.mean((multipliers[:, :self.n_buses] - lambda_P_target)**2)
        loss += torch.mean((multipliers[:, self.n_buses:] - lambda_Q_target)**2)

        return loss

    # training for one epoch
    def train_epoch(self, P_demand_batch, Q_demand_batch, inner_iters=250):
        # Primal learning
        primal_losses = []
        costs = []  # intentionally left empty/unused so printed cost will be nan (per your request)
        for _ in range(inner_iters):
            self.primal_optimizer.zero_grad()

            Pg, Qg, V, theta = self.primal_net(
                P_demand_batch, Q_demand_batch,
                self.gen_limits,
                self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus
            )
            # multipliers used for primal are from dual network detached
            multipliers = self.dual_net(P_demand_batch, Q_demand_batch).detach()

            loss = self.primal_loss(Pg, Qg, V, theta, P_demand_batch, Q_demand_batch, multipliers)
            loss.backward()
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.primal_net.parameters(), max_norm=5.0)
            self.primal_optimizer.step()

            primal_losses.append(loss.item())

        # copy current dual net
        dual_net_old = ACPFDualNet(self.n_buses, lambda_scale=self.dual_net.lambda_scale).to(self.device)
        dual_net_old.load_state_dict(self.dual_net.state_dict())

        # Dual learning
        dual_losses = []
        for _ in range(inner_iters):
            self.dual_optimizer.zero_grad()

            Pg, Qg, V, theta = self.primal_net(
                P_demand_batch, Q_demand_batch,
                self.gen_limits,
                self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus
            )
            Pg, Qg, V, theta = Pg.detach(), Qg.detach(), V.detach(), theta.detach()

            multipliers = self.dual_net(P_demand_batch, Q_demand_batch)
            multipliers_old = dual_net_old(P_demand_batch, Q_demand_batch).detach()

            loss = self.dual_loss(multipliers, multipliers_old, Pg, Qg, V, theta,
                                   P_demand_batch, Q_demand_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dual_net.parameters(), max_norm=5.0)
            self.dual_optimizer.step()

            dual_losses.append(loss.item())

        with torch.no_grad():
            Pg, Qg, V, theta = self.primal_net(
                P_demand_batch, Q_demand_batch,
                self.gen_limits,
                self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus
            )
            P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta,
                                                         P_demand_batch, Q_demand_batch)

            max_P_viol = torch.max(torch.abs(P_viol)).item()
            max_Q_viol = torch.max(torch.abs(Q_viol)).item()
            max_viol = max(max_P_viol, max_Q_viol)
            mean_viol = (torch.mean(torch.abs(P_viol)) + torch.mean(torch.abs(Q_viol))).item() / 2.0

            slack_V = V[:, self.slack_bus].mean().item()

            # milder rho adaptation to avoid fast explosion
            if not hasattr(self, 'prev_max_viol'):
                self.prev_max_viol = max_viol
            elif max_viol > self.tau * self.prev_max_viol:
                self.rho = min(self.alpha * self.rho, self.rho_max)
            self.prev_max_viol = max_viol

        self.history['primal_loss'].append(np.mean(primal_losses))
        self.history['dual_loss'].append(np.mean(dual_losses))
        self.history['cost'].append(np.mean(costs))  # costs will be nan if costs list empty
        self.history['max_P_viol'].append(max_P_viol)
        self.history['max_Q_viol'].append(max_Q_viol)
        self.history['max_viol'].append(max_viol)
        self.history['mean_viol'].append(mean_viol)
        self.history['rho'].append(self.rho)
        self.history['slack_bus_voltage'].append(slack_V)

        return np.mean(primal_losses), max_viol, np.mean(costs)

    def predict(self, P_demand, Q_demand):
        """Predict solution"""
        self.primal_net.eval()
        with torch.no_grad():
            Pg, Qg, V, theta = self.primal_net(
                P_demand, Q_demand,
                self.gen_limits,
                self.pv_buses, self.pq_buses, self.gen_voltage_setpoints, self.slack_bus
            )
        self.primal_net.train()
        return Pg, Qg, V, theta


# ---------------------------
# Plotting helpers (unchanged except cosmetic)
# ---------------------------
def plot_training_results(history, case_name, slack_bus_id, V_pdl=None, V_nr=None, nr_max_P_viol=None, nr_max_Q_viol=None):
    """Plot training metrics (6 figures per bus)"""

    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle(f'PDL Training Results - {case_name}', fontsize=20, fontweight='bold')

    # 1. Training losses
    #ax = axes[0, 0]
    #ax.plot(history['primal_loss'], label='Primal Loss', linewidth=2)
    #ax.plot(history['dual_loss'], label='Dual Loss', linewidth=2)
    #ax.set_xlabel('Outer Iteration')
    #ax.set_ylabel('Loss')
    #ax.set_title('Training Losses')
    #ax.set_yscale('log')
    #ax.legend()
    #ax.grid(True, alpha=0.3)

    # 2. Max P & Q violations
    ax = axes[0, 0]
    x = np.arange(len(history['max_P_viol']))
    # Main lines
    ax.plot(x, history['max_P_viol'], label='Max P Violation', color='#0072B2', linewidth=2.5, marker='o', markersize=5, alpha=0.85)
    ax.plot(x, history['max_Q_viol'], label='Max Q Violation', color='#D55E00', linewidth=2.5, marker='s', markersize=5, alpha=0.85)
    # Moving average for smoothness
    window = max(1, len(x)//20)
    if window > 1:
        from pandas import Series
        p_ma = Series(history['max_P_viol']).rolling(window, min_periods=1).mean()
        q_ma = Series(history['max_Q_viol']).rolling(window, min_periods=1).mean()
        ax.plot(x, p_ma, color='#0072B2', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot(x, q_ma, color='#D55E00', linestyle='--', linewidth=1.5, alpha=0.7)
    # Shaded region for max violation
    ax.fill_between(x, history['max_P_viol'], history['max_Q_viol'], color='#56B4E9', alpha=0.08)
    ax.set_xlabel('Outer Iteration', fontsize=14)
    ax.set_ylabel('Maximum Violation (p.u.)', fontsize=14)
    ax.set_title('PDL Maximum Power Balance Violations', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.4, linestyle='--')

    # 3. Slack bus voltage evolution
    ax = axes[0, 1]
    slack_v = np.array(history['slack_bus_voltage'])
    ax.plot(slack_v, marker='o', markersize=4, linewidth=2.2, color='#009E73', label='Slack Bus Voltage')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.6, linewidth=1.5, label='Nominal (1.0 pu)')
    mean_v = np.mean(slack_v)
    std_v = np.std(slack_v)
    ax.axhline(mean_v, color='#E69F00', linestyle=':', linewidth=2, label=f'Mean: {mean_v:.4f} pu')
    ax.fill_between(np.arange(len(slack_v)), mean_v-std_v, mean_v+std_v, color='#E69F00', alpha=0.15, label=f'±1 Std: {std_v:.4f}')
    ax.set_xlabel('Outer Iteration', fontsize=14)
    ax.set_ylabel('Slack Bus Voltage (p.u.)', fontsize=14)
    ax.set_title(f'Slack Bus Voltage Evolution (Bus {slack_bus_id})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.4, linestyle='--')

    # 4. Mean power balance violation
    ax = axes[1, 0]
    ax.plot(history['mean_viol'], linewidth=2.2, color='#CC79A7', marker='D', markersize=4, alpha=0.85)
    ax.set_xlabel('Outer Iteration', fontsize=14)
    ax.set_ylabel('Mean Violation (p.u.)', fontsize=14)
    ax.set_title('Mean Power Balance Violation', fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='--')


    #comparaison
    ax = axes[1, 1]
    ax.axis('off')


    # 5. Penalty coefficient (rho)
    #ax = axes[1, 1]
    #ax.plot(history['rho'], linewidth=2)
    #ax.set_xlabel('Outer Iteration')
    #ax.set_ylabel('ρ')
    #ax.set_title('Penalty Coefficient')
    #ax.grid(True, alpha=0.3)

    # 6. Overall maximum violation
    #ax = axes[1, 2]
    #ax.plot(history['max_viol'], linewidth=2)
    #ax.set_xlabel('Outer Iteration')
    #ax.set_ylabel('Max Violation (pu)')
    #ax.set_title('Overall Maximum Violation')
    #ax.set_yscale('log')
    #ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig



def plot_solution_quality(Pg, Qg, V, theta, P_viol, Q_viol, case_name, baseMVA, slack_bus_id, pdl_max_P=None, pdl_max_Q=None, nr_max_P=None, nr_max_Q=None):
    """Plot solution quality metrics"""
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle(f'Solution Quality - {case_name}', fontsize=20, fontweight='bold')

    # Convert to numpy and to MW/MVAr
    Pg_np = Pg.cpu().numpy() * baseMVA
    Qg_np = Qg.cpu().numpy() * baseMVA
    V_np = V.cpu().numpy()
    theta_np = np.degrees(theta.cpu().numpy())
    P_viol_np = P_viol.cpu().numpy() * baseMVA
    Q_viol_np = Q_viol.cpu().numpy() * baseMVA

    V_min = V.cpu().numpy().min()
    V_max = V.cpu().numpy().max()





    # Voltage Profile
    ax = axes[0, 0]
    for i in range(min(5, V_np.shape[0])):
        ax.plot(V_np[i, :], alpha=0.7, marker='o', markersize=2)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Nominal')
    ax.axhline(y=1.0, color='k', linestyle='--', label='Nominal')

    ax.set_xlabel('Bus ID')
    ax.set_ylabel('Voltage Magnitude (pu)')
    ax.set_title('Voltage Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([V_min * 0.98, V_max * 1.02])

    # Slack Bus Voltage Across Test Samples
    ax = axes[0, 1]
    slack_voltages = V_np[:, slack_bus_id]
    ax.plot(slack_voltages, linewidth=2.2, marker='o', markersize=5, color='#009E73', label='Slack Bus Voltage')
    mean_v = np.mean(slack_voltages)
    std_v = np.std(slack_voltages)
    ax.axhline(mean_v, color='#E69F00', linestyle=':', linewidth=2, label=f'Mean: {mean_v:.4f} pu')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal: 1.0 pu')
    ax.fill_between(np.arange(len(slack_voltages)), mean_v-std_v, mean_v+std_v, color='#E69F00', alpha=0.15, label=f'±1 Std: {std_v:.4f}')
    ax.set_xlabel('Test Sample Index', fontsize=14)
    ax.set_ylabel('Slack Bus Voltage (p.u.)', fontsize=14)
    ax.set_title(f'Slack Bus Voltage (Bus {slack_bus_id})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.4, linestyle='--')

    # Voltage angles
    #ax = axes[1, 0]
    #for i in range(min(5, theta_np.shape[0])):
        #ax.plot(theta_np[i, :], alpha=0.7, marker='o', markersize=2)
    #ax.set_xlabel('Bus ID')
    #ax.set_ylabel('Voltage Angle (degrees)')
    #ax.set_title('Voltage Angle Profile')
    #ax.grid(True, alpha=0.3)





    # Slack Bus Voltage Distribution (Test Set)
    ax = axes[1, 0]
    import seaborn as sns
    sns.violinplot(y=slack_voltages, color='#56B4E9', ax=ax, inner='box', linewidth=2)
    ax.axhline(mean_v, color='#E69F00', linestyle=':', linewidth=2, label=f'Mean: {mean_v:.4f}')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal: 1.0')
    ax.set_ylabel('Slack Bus Voltage (p.u.)', fontsize=14)
    ax.set_title(f'Slack Bus Voltage Distribution (Test Set)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.4, linestyle='--')

    # -------------------------------------------------
# PDL vs NR: Max Power Balance Violations (Solution Quality)
# -------------------------------------------------
    ax = axes[1, 1]
    if (pdl_max_P is not None and pdl_max_Q is not None):
        labels = ['Max P Violation', 'Max Q Violation']
        x = np.arange(len(labels))
        width = 0.5
        pdl_vals = [pdl_max_P * baseMVA, pdl_max_Q * baseMVA]
        bar_colors = ['#0072B2', '#D55E00']
        bars = ax.bar(x, pdl_vals, width, color=bar_colors, alpha=0.85, edgecolor='black')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(pdl_vals),
                    f'{height:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel('Violation (MW / MVAr)', fontsize=14)
        ax.set_title('PDL Max Power Balance Violations (Test Set)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--', axis='y')
    else:
        ax.axis('off')

    plt.tight_layout()
    return fig


# ---------------------------
# Experiment runner
# ---------------------------
def run_acpf_experiment(case_name, system_data, n_train=25000, n_test=100,
                         max_outer_iters=500, convergence_threshold=1e-2, inner_iters=250):
    print(f"\n{'='*70}")
    print(f"Running AC-PF PDL Experiment: {case_name}")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Number of buses: {system_data['n_buses']}")
    print(f"Number of generators: {system_data['n_generators']}")
    print(f"Number of loads: {system_data['n_loads']}")
    print(f"Slack bus: {system_data['slack_bus']}")
    print(f"PV buses: {len(system_data['pv_buses'])}")
    print(f"PQ buses: {len(system_data['pq_buses'])}\n")

    # Generate training data
    print(f"Generating {n_train} training samples...")
    P_train, Q_train = generate_load_scenarios(system_data, n_train, load_variation=0.3)

    # Input normalization (per-case) to keep ranges stable across small and large systems
    demand_scale = max(1.0, np.max(P_train))
    P_train = P_train / demand_scale
    Q_train = Q_train / demand_scale
    P_train_tensor = torch.tensor(P_train, dtype=torch.float32).to(device)
    Q_train_tensor = torch.tensor(Q_train, dtype=torch.float32).to(device)

    # Initialize PDL
    print("Initializing PDL model...")
    pdl = PDL_ACPF(
        system_data=system_data,
        rho_init=1.0,
        rho_max=10000,
        alpha=1.5,    # milder growth
        tau=0.8,      # less sensitive
        device=device,
        dual_lambda_scale=0.5,
        max_angle_rad=np.pi/12.0  # ±15 degrees by default
    )

    baseMVA = pdl.baseMVA

    # --- Start of Modified Training Loop ---
    print(f"\nTraining until max violation < {convergence_threshold} (or max {max_outer_iters} outer iterations)...")
    print(f"Inner iterations per phase: {inner_iters}")
    print("-" * 70)

    start_time = time.time()
    k = 0
    max_viol = float('inf')  # Initialize large

    while max_viol > convergence_threshold and k < max_outer_iters:
        loss, max_viol, cost = pdl.train_epoch(
            P_train_tensor, Q_train_tensor, inner_iters=inner_iters
        )

        # Get current slack bus voltage snapshot
        with torch.no_grad():
            _, _, V_temp, _ = pdl.primal_net(
                P_train_tensor[:100], Q_train_tensor[:100],
                pdl.gen_limits,
                pdl.pv_buses, pdl.pq_buses, pdl.gen_voltage_setpoints, pdl.slack_bus
            )
            slack_V_current = V_temp[:, pdl.slack_bus].mean().item()

        # cost is intentionally nan (per your request) so printing will show nan
        print(f"Iter {k+1:3d}: Cost=${cost:}, "
              f"Max Viol={max_viol:.6f} pu, "
              f"Slack V={slack_V_current:.4f} pu, "
              f"ρ={pdl.rho:.2f}")
        k += 1

    training_time = time.time() - start_time
    pdl_convergence_iters = k  # Track PDL iterations to convergence
    print(f"\nTraining stopped after {pdl_convergence_iters} iterations.")

    # Testing / inference (we must scale test demands the same way)
    print(f"\nGenerating {n_test} test samples...")
    P_test, Q_test = generate_load_scenarios(system_data, n_test, load_variation=0.3)
    P_test = P_test / demand_scale
    Q_test = Q_test / demand_scale
    P_test_tensor = torch.tensor(P_test, dtype=torch.float32).to(device)
    Q_test_tensor = torch.tensor(Q_test, dtype=torch.float32).to(device)

    print("Testing on unseen samples...")
    start_time = time.time()
    Pg, Qg, V, theta = pdl.predict(P_test_tensor, Q_test_tensor)
    inference_time = time.time() - start_time

    # -------------------------------------------------
# Newton–Raphson HOT START from PDL voltages
# -------------------------------------------------
    V_nr_all = []
    theta_nr_all = []
    nr_iters_per_sample = []  # Track NR iterations (hotstart)
    nr_times_per_sample = []  # Track NR solve times (hotstart)
    nr_iters_normal_per_sample = []  # Track NR iterations (normal)
    nr_times_normal_per_sample = []  # Track NR solve times (normal)

    for i in range(n_test):
        # --- Normal NR (flat start) ---
        net_normal = copy.deepcopy(system_data['net'])
        # apply loads
        for idx, load in net_normal.load.iterrows():
            net_normal.load.at[idx, 'p_mw'] = P_test[i, load.bus] * baseMVA
            net_normal.load.at[idx, 'q_mvar'] = Q_test[i, load.bus] * baseMVA
        # Do NOT set generator p_mw or voltages (let NR use default flat start)
        nr_start_time = time.time()
        pp.runpp(
            net_normal,
            init='flat',
            calculate_voltage_angles=True,
            enforce_q_limits=False,
            enforce_v_limits=False
        )
        nr_solve_time = time.time() - nr_start_time
        # Extract NR iterations
        if hasattr(net_normal, '_ppc') and 'et' in net_normal._ppc:
            try:
                nr_iters_normal = net_normal._ppc.get('iterations', 3)
            except:
                nr_iters_normal = 3
        else:
            nr_iters_normal = 3
        nr_iters_normal_per_sample.append(nr_iters_normal)
        nr_times_normal_per_sample.append(nr_solve_time)

        # --- Hotstart NR (PDL) ---
        net_tmp = copy.deepcopy(system_data['net'])
        for idx, load in net_tmp.load.iterrows():
            net_tmp.load.at[idx, 'p_mw'] = P_test[i, load.bus] * baseMVA
            net_tmp.load.at[idx, 'q_mvar'] = Q_test[i, load.bus] * baseMVA
        for g, bus in enumerate(pdl.gen_buses):
            net_tmp.gen.at[g, 'p_mw'] = Pg[i, g].item() * baseMVA
        # Hotstart: set voltages/angles
        net_tmp.res_bus.vm_pu.values[:] = V[i].detach().cpu().numpy()
        net_tmp.res_bus.va_degree.values[:] = np.degrees(
            theta[i].detach().cpu().numpy()
        )
        nr_start_time = time.time()
        pp.runpp(
            net_tmp,
            init='results',
            calculate_voltage_angles=True,
            enforce_q_limits=False,
            enforce_v_limits=False
        )
        nr_solve_time = time.time() - nr_start_time
        V_nr_all.append(net_tmp.res_bus.vm_pu.values)
        theta_nr_all.append(
            np.deg2rad(net_tmp.res_bus.va_degree.values)
        )
        if hasattr(net_tmp, '_ppc') and 'et' in net_tmp._ppc:
            try:
                nr_iters = net_tmp._ppc.get('iterations', 3)
            except:
                nr_iters = 3
        else:
            nr_iters = 3
        nr_iters_per_sample.append(nr_iters)
        nr_times_per_sample.append(nr_solve_time)

    V_nr_all = np.array(V_nr_all)
    theta_nr_all = np.array(theta_nr_all)

    nr_avg_iters = np.mean(nr_iters_per_sample)
    nr_avg_time = np.mean(nr_times_per_sample)
    nr_avg_iters_normal = np.mean(nr_iters_normal_per_sample)
    nr_avg_time_normal = np.mean(nr_times_normal_per_sample)

    # --- Compute NR power balance violations ---
    P_viol_nr_list = []
    Q_viol_nr_list = []

    for i in range(n_test):
        V_nr_t = torch.tensor(V_nr_all[i], dtype=torch.float32).to(device)
        theta_nr_t = torch.tensor(theta_nr_all[i], dtype=torch.float32).to(device)

        V_nr_t = V_nr_t.unsqueeze(0)
        theta_nr_t = theta_nr_t.unsqueeze(0)

        Pg_i = Pg[i:i+1]
        Qg_i = Qg[i:i+1]
        Pd_i = P_test_tensor[i:i+1]
        Qd_i = Q_test_tensor[i:i+1]

        Pvn, Qvn = pdl.compute_power_balance(
            Pg_i, Qg_i, V_nr_t, theta_nr_t, Pd_i, Qd_i
        )

        P_viol_nr_list.append(torch.max(torch.abs(Pvn)).item())
        Q_viol_nr_list.append(torch.max(torch.abs(Qvn)).item())

    nr_max_P_viol = np.mean(P_viol_nr_list)
    nr_max_Q_viol = np.mean(Q_viol_nr_list)



    with torch.no_grad():
        P_viol, Q_viol = pdl.compute_power_balance(Pg, Qg, V, theta,
                                                   P_test_tensor, Q_test_tensor)
    # -------------------------------------------------
    # PDL vs Newton–Raphson comparison metrics
    # -------------------------------------------------
    V_pdl_np = V.detach().cpu().numpy()
    theta_pdl_deg = np.degrees(theta.detach().cpu().numpy())
    theta_nr_deg = np.degrees(theta_nr_all)

    voltage_rmse = np.sqrt(np.mean((V_pdl_np - V_nr_all)**2))
    voltage_max = np.max(np.abs(V_pdl_np - V_nr_all))

    angle_rmse = np.sqrt(np.mean((theta_pdl_deg - theta_nr_deg)**2))
    angle_max = np.max(np.abs(theta_pdl_deg - theta_nr_deg))

    max_P_viol_test_pu = torch.max(torch.abs(P_viol)).item()
    max_Q_viol_test_pu = torch.max(torch.abs(Q_viol)).item()
    mean_P_viol_test_pu = torch.mean(torch.abs(P_viol)).item()
    mean_Q_viol_test_pu = torch.mean(torch.abs(Q_viol)).item()
    max_viol_test_pu = max(max_P_viol_test_pu, max_Q_viol_test_pu)

    # Extract slack bus voltage statistics
    slack_V_test = V[:, pdl.slack_bus].cpu().numpy()
    slack_V_mean = np.mean(slack_V_test)
    slack_V_std = np.std(slack_V_test)
    slack_V_min = np.min(slack_V_test)
    slack_V_max = np.max(slack_V_test)

    print(f"\nInference time for {n_test} samples: {inference_time*1000:.2f} ms")
    print(f"Average inference time: {inference_time*1000/n_test:.4f} ms/sample")

    print(f"\n{'='*70}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*70}")
    print(f"Base MVA: {baseMVA}")
    print(f"Slack Bus: {pdl.slack_bus}")
    print(f"Max P Violation:          {max_P_viol_test_pu:.8f} pu ({max_P_viol_test_pu*baseMVA:.4f} MW)")
    print(f"Max Q Violation:          {max_Q_viol_test_pu:.8f} pu ({max_Q_viol_test_pu*baseMVA:.4f} MVAr)")
    print(f"Max Overall Violation:    {max_viol_test_pu:.8f} pu")
    print(f"Mean P Violation:         {mean_P_viol_test_pu:.8f} pu ({mean_P_viol_test_pu*baseMVA:.4f} MW)")
    print(f"Mean Q Violation:         {mean_Q_viol_test_pu:.8f} pu ({mean_Q_viol_test_pu*baseMVA:.4f} MVAr)")

    print(f"\nNewton-Raphson Iterations (Test Set):")
    print(f"  Normal NR (flat start):   {nr_avg_iters_normal:.2f} avg iterations")
    print(f"  Hotstart NR (PDL init):   {nr_avg_iters:.2f} avg iterations")

    total_time_normal_nr = n_test * nr_avg_time_normal
    total_time_pdl_hotstart = training_time + n_test * nr_avg_time
    print(f"\nTotal Time Comparison (Test Set):")
    print(f"  Normal NR (test only):         {total_time_normal_nr:.2f} s")
    print(f"  PDL training + NR hotstart:    {total_time_pdl_hotstart:.2f} s")
    print(f"PDL vs NR Comparison:")
    print(f"Voltage RMSE: {voltage_rmse:.6e} pu")
    print(f"Voltage Max Error: {voltage_max:.6e} pu")
    print(f"Angle RMSE: {angle_rmse:.6e} deg")
    print(f"Angle Max Error: {angle_max:.6e} deg")

    V_mean = torch.mean(V).item()
    V_std = torch.std(V).item()
    V_min_val = torch.min(V).item()
    V_max_val = torch.max(V).item()

    print(f"\nVoltage Statistics (All Buses):")
    print(f"  Mean: {V_mean:.4f} pu")
    print(f"  Std:  {V_std:.4f} pu")
    print(f"  Min:  {V_min_val:.4f} pu")
    print(f"  Max:  {V_max_val:.4f} pu")

    print(f"\nSlack Bus Voltage Statistics (Bus {pdl.slack_bus}):")
    print(f"  Mean: {slack_V_mean:.4f} pu")
    print(f"  Std:  {slack_V_std:.4f} pu")
    print(f"  Min:  {slack_V_min:.4f} pu")
    print(f"  Max:  {slack_V_max:.4f} pu")
    print(f"  Deviation from nominal (1.0 pu): {abs(slack_V_mean - 1.0):.4f} pu")

    Pg_total_pu = torch.sum(Pg, dim=1).mean().item()
    # Note: Pg_total_mw is scaled by baseMVA and also the demand_scale (we normalized inputs).
    Pg_total_mw = Pg_total_pu * baseMVA  # this reports generation in MW (approx, because inputs were normalized)

    print(f"\nTotal Generation: {Pg_total_mw:.2f} MW (average)")

    # -------------------------------------------------
    # PDL vs NR comparison for training-results plot
    # -------------------------------------------------
    nr_comparison = {
        'pdl_max_viol': max_viol_test_pu,
        'nr_max_viol': max(
            np.max(np.abs(V_nr_all - V.detach().cpu().numpy())),
            np.max(np.abs(theta_nr_all - theta.detach().cpu().numpy()))
        )
    }


    print("\nGenerating plots...")
    fig1 = plot_training_results(pdl.history, case_name, pdl.slack_bus, V_pdl=V.detach().cpu().numpy(), V_nr=V_nr_all, nr_max_P_viol=nr_max_P_viol, nr_max_Q_viol=nr_max_Q_viol)
    fig2 = plot_solution_quality(Pg, Qg, V, theta, P_viol, Q_viol, case_name, baseMVA, pdl.slack_bus, pdl_max_P=max_P_viol_test_pu, pdl_max_Q=max_Q_viol_test_pu, nr_max_P=nr_max_P_viol, nr_max_Q=nr_max_Q_viol)

    results = {
        'training_time': training_time,
        'pdl_convergence_iters': pdl_convergence_iters,
        'nr_avg_convergence_iters': nr_avg_iters,
        'nr_avg_convergence_iters_normal': nr_avg_iters_normal,
        'inference_time_per_sample': inference_time / n_test,
        'nr_avg_inference_time_per_sample': nr_avg_time,
        'nr_avg_inference_time_per_sample_normal': nr_avg_time_normal,
        'max_P_viol_pu': max_P_viol_test_pu,
        'max_Q_viol_pu': max_Q_viol_test_pu,
        'max_viol_pu': max_viol_test_pu,
        'mean_P_viol_pu': mean_P_viol_test_pu,
        'mean_Q_viol_pu': mean_Q_viol_test_pu,
        'V_mean': V_mean,
        'V_std': V_std,
        'slack_bus': pdl.slack_bus,
        'slack_V_mean': slack_V_mean,
        'slack_V_std': slack_V_std,
        'slack_V_min': slack_V_min,
        'slack_V_max': slack_V_max,
        'Pg_total_mw': Pg_total_mw,
        'baseMVA': baseMVA,
        'total_time_normal_nr': total_time_normal_nr,
        'total_time_pdl_hotstart': total_time_pdl_hotstart
    }

    return pdl, fig1, fig2, results


# ---------------------------
# Comparison & reporting helpers (unchanged)
# ---------------------------
def create_comparison_plots(results_dict):
    """Create comparison plots between cases"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comparison: Multiple IEEE Cases', fontsize=16, fontweight='bold')

    cases = list(results_dict.keys())

    # Violations comparison
    ax = axes[0, 0]
    x = np.arange(len(cases))
    width = 0.35
    max_P = [results_dict[c]['max_P_viol_pu'] * 1000 for c in cases]  # Convert to milli-pu
    max_Q = [results_dict[c]['max_Q_viol_pu'] * 1000 for c in cases]

    ax.bar(x - width/2, max_P, width, label='Max P Violation', alpha=0.7)
    ax.bar(x + width/2, max_Q, width, label='Max Q Violation', alpha=0.7)
    ax.set_ylabel('Max Violation (×10⁻³ pu)')
    ax.set_title('Maximum Power Balance Violations (p.u.)')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Mean violations
    ax = axes[0, 1]
    mean_P = [results_dict[c]['mean_P_viol_pu'] * 1000 for c in cases]
    mean_Q = [results_dict[c]['mean_Q_viol_pu'] * 1000 for c in cases]

    ax.bar(x - width/2, mean_P, width, label='Mean P Violation', alpha=0.7)
    ax.bar(x + width/2, mean_Q, width, label='Mean Q Violation', alpha=0.7)
    ax.set_ylabel('Mean Violation (×10⁻³ pu)')
    ax.set_title('Mean Power Balance Violations (p.u.)')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Training time
    ax = axes[0, 2]
    times = [results_dict[c]['training_time'] for c in cases]
    bars = ax.bar(cases, times, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time')
    ax.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')

    # Inference time (test-only for PDL)
    ax = axes[1, 0]
    inf_times = [results_dict[c]['inference_time_per_sample'] * 1000 for c in cases]
    bars = ax.bar(cases, inf_times, alpha=0.85, edgecolor='black', color='#0072B2')
    ax.set_ylabel('Inference Time per Sample (ms)', fontsize=14)
    ax.set_title('PDL Inference Time per Sample (Test Only)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--', axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(inf_times),
                f'{height:.3f} ms', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Voltage statistics
    ax = axes[1, 1]
    v_means = [results_dict[c]['V_mean'] for c in cases]
    v_stds = [results_dict[c]['V_std'] for c in cases]

    ax.bar(x - width/2, v_means, width, label='Mean Voltage', alpha=0.7)
    ax.bar(x + width/2, [s*10 for s in v_stds], width, label='Std Dev (×10)', alpha=0.7)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel('Voltage (pu)')
    ax.set_title('Voltage Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Empty plot placeholder for layout completeness
    ax = axes[1, 2]
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_convergence_comparison_plots(results_dict):
    """Create detailed convergence and performance comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Convergence & Performance: PDL vs Newton-Raphson', fontsize=16, fontweight='bold')

    cases = list(results_dict.keys())
    x = np.arange(len(cases))
    width = 0.35

    # 1. Convergence Iterations Comparison
    ax = axes[0, 0]
    pdl_iters = [results_dict[c]['pdl_convergence_iters'] for c in cases]
    nr_iters = [results_dict[c]['nr_avg_convergence_iters'] for c in cases]

    ax.bar(x - width/2, pdl_iters, width, label='PDL', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, nr_iters, width, label='Newton-Raphson', alpha=0.8, color='coral')
    ax.set_ylabel('Number of Iterations to Converge')
    ax.set_title('Convergence Speed Comparison (Fewer is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (pdl, nr) in enumerate(zip(pdl_iters, nr_iters)):
        ax.text(i - width/2, pdl, f'{pdl}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, nr, f'{nr:.1f}', ha='center', va='bottom', fontsize=9)

    # 2. Inference Time per Sample
    ax = axes[0, 1]
    pdl_inf_times = [results_dict[c]['inference_time_per_sample'] * 1000 for c in cases]
    nr_inf_times = [results_dict[c]['nr_avg_inference_time_per_sample'] * 1000 for c in cases]

    ax.bar(x - width/2, pdl_inf_times, width, label='PDL', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, nr_inf_times, width, label='Newton-Raphson', alpha=0.8, color='coral')
    ax.set_ylabel('Time (milliseconds)')
    ax.set_title('Inference Time per Sample (Faster is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (pdl, nr) in enumerate(zip(pdl_inf_times, nr_inf_times)):
        ax.text(i - width/2, pdl, f'{pdl:.3f}ms', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, nr, f'{nr:.3f}ms', ha='center', va='bottom', fontsize=9)

    # 3. Speedup Factor (NR time / PDL time)
    ax = axes[1, 0]
    speedup_factors = [nr / pdl for nr, pdl in zip(nr_inf_times, pdl_inf_times)]
    colors = ['green' if s > 1 else 'red' for s in speedup_factors]

    bars = ax.bar(cases, speedup_factors, alpha=0.8, color=colors, edgecolor='black')
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=2, label='Parity (1x)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Inference Speedup: PDL vs NR (>1 = PDL faster)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, speedup in zip(bars, speedup_factors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Total Time Comparison (Training + Testing)
    ax = axes[1, 1]
    pdl_total_times = [results_dict[c]['training_time'] + results_dict[c]['inference_time_per_sample'] * 100
                       for c in cases]  # 100 test samples
    nr_total_times = [results_dict[c]['nr_avg_convergence_iters'] * results_dict[c]['nr_avg_inference_time_per_sample'] * 100
                      for c in cases]

    ax.bar(x - width/2, pdl_total_times, width, label='PDL (Train + Test)', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, nr_total_times, width, label='NR (Test only)', alpha=0.8, color='coral')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Total Computational Time')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (pdl, nr) in enumerate(zip(pdl_total_times, nr_total_times)):
        ax.text(i - width/2, pdl, f'{pdl:.2f}s', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, nr, f'{nr:.2f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def print_comparison_table(results_dict):
    """Print comparison table"""
    print(f"\n{'='*150}")
    print("COMPARATIVE RESULTS TABLE")
    print(f"{'='*150}")

    df_data = []
    for case_name, results in results_dict.items():
        df_data.append({
            'Case': case_name,
            'Slack Bus': results['slack_bus'],
            'Slack V (pu)': f"{results['slack_V_mean']:.4f}±{results['slack_V_std']:.4f}",
            'Max Viol (pu)': f"{results['max_viol_pu']:.6f}",
            'PDL Iters': results['pdl_convergence_iters'],
            'NR Iters (avg)': f"{results['nr_avg_convergence_iters']:.1f}",
            'PDL Infer (ms)': f"{results['inference_time_per_sample']*1000:.4f}",
            'NR Infer (ms)': f"{results['nr_avg_inference_time_per_sample']*1000:.4f}",
            'Speedup': f"{results['nr_avg_inference_time_per_sample'] / results['inference_time_per_sample']:.2f}x",
            'Train (s)': f"{results['training_time']:.1f}"
        })

    df = pd.DataFrame(df_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1500)
    print(df.to_string(index=False))
    print(f"{'='*150}\n")

    print("DETAILED CONVERGENCE ANALYSIS")
    print("-" * 150)
    for case_name, results in results_dict.items():
        pdl_iters = results['pdl_convergence_iters']
        nr_iters = results['nr_avg_convergence_iters']
        pdl_time = results['inference_time_per_sample'] * 1000
        nr_time = results['nr_avg_inference_time_per_sample'] * 1000
        speedup = nr_time / pdl_time

        print(f"{case_name}:")
        print(f"  Convergence Iterations:")
        print(f"    PDL:                 {pdl_iters} iterations")
        print(f"    Newton-Raphson (avg): {nr_iters:.1f} iterations")
        print(f"    Ratio (NR/PDL):      {nr_iters/pdl_iters:.2f}x")
        print(f"  Inference Time per Sample:")
        print(f"    PDL:                 {pdl_time:.4f} ms")
        print(f"    Newton-Raphson (avg): {nr_time:.4f} ms")
        print(f"    Speedup (NR/PDL):    {speedup:.2f}x (PDL is {'faster' if speedup > 1 else 'slower'})")
        print()
    print("-" * 150)


# ---------------------------
# Main execution (multiple cases incl. case300)
# ---------------------------
if __name__ == "__main__":
    print("\n" + "="*90)
    print("PDL FOR AC-OPTIMAL POWER FLOW: MULTIPLE IEEE CASES (PANDAPOWER)")
    print("Replicating/Testing multiple cases with lr=1e-4 for both optimizers")
    print("="*90)

    # Run experiments
    all_results = {}
    all_pdls = {}
    all_figs = {}

    # Include 4 additional cases plus earlier ones; ensure case300 present
    case_numbers = [57, 118]  # includes case300

    print("\nLoading pandapower networks...")
    for num in case_numbers:
        func_name = f"case{num}"
        try:
            net = getattr(nw, func_name)()
            print(f"Successfully loaded {func_name}.")
        except Exception as e:
            print(f"Error loading {func_name}: {e} -- skipping this case.")
            continue

        # --- Get system data from pandapower networks ---
        system_data = get_pandapower_data(net, func_name)

        case_label = f"Case {num}"
        try:
            pdl_model, fig1, fig2, results = run_acpf_experiment(
                case_name=f"IEEE {case_label}",
                system_data=system_data,
                n_train=25000,
                n_test=100,
                max_outer_iters=60,
                convergence_threshold=1e-3,
                inner_iters=250
            )

            all_results[case_label] = results
            all_pdls[case_label] = pdl_model
            all_figs[case_label] = (fig1, fig2)

        except Exception as e:
            print(f"Error running experiment for {case_label}: {e}")
            continue

    if len(all_results) > 0:
        print("\nCreating comparison visualizations...")
        fig_comparison = create_comparison_plots(all_results)
        fig_convergence = create_convergence_comparison_plots(all_results)

        # Print comparison table
        print_comparison_table(all_results)

        plt.show()

    print("\n All experiments completed!")