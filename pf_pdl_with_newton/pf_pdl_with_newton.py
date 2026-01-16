import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import pandapower as pp
import pandapower.networks as nw
from pandapower.pypower.makeYbus import makeYbus
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 1: DATA EXTRACTION AND PREPROCESSING
# ============================================================================

def get_pandapower_data(net, case_name):
    """Extracting data from pandapower"""
    
    print(f"Loading data from pandapower: {case_name}")

    try:
        pp.runpp(net, calculate_voltage_angles=False, enforce_q_limits=False, enforce_v_limits=False, init='flat')
    except:
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

    base_P_demand = np.zeros(n_buses)
    base_Q_demand = np.zeros(n_buses)

    for _, load in net.load.iterrows():
        bus_idx = load.bus
        base_P_demand[bus_idx] += load.p_mw
        base_Q_demand[bus_idx] += load.q_mvar

    base_P_demand_pu = base_P_demand / baseMVA
    base_Q_demand_pu = base_Q_demand / baseMVA

    # Get generator cost coefficients
    if net.poly_cost.shape[0] == n_generators:
        costs = net.poly_cost[net.poly_cost.et == 'gen'].sort_values(by='element')
        c2 = costs.cp2_eur_per_mw2.values
        c1 = costs.cp1_eur_per_mw.values
        c0 = costs.cp0_eur.values

        c2_pu = c2 * (baseMVA**2)
        c1_pu = c1 * baseMVA
        c0_pu = c0

    else:
        print("Warning: Cost data not found or incomplete. Using random costs.")
        c2_pu = np.random.uniform(0.005, 0.02, n_generators) * (baseMVA**2)
        c1_pu = np.random.uniform(15.0, 35.0, n_generators) * baseMVA
        c0_pu = np.random.uniform(80.0, 160.0, n_generators)

    cost_coeffs = {'c2': c2_pu, 'c1': c1_pu, 'c0': c0_pu}

    gen_limits = {
        'P_min': net.gen.min_p_mw.values / baseMVA,
        'P_max': net.gen.max_p_mw.values / baseMVA,
        'Q_min': net.gen.min_q_mvar.values / baseMVA,
        'Q_max': net.gen.max_q_mvar.values / baseMVA
    }

    V_min = net.bus.min_vm_pu.min()
    V_max = net.bus.max_vm_pu.max()

    slack_bus = net.ext_grid.bus.values[0]

    print(f"Data loaded: {n_buses} buses, {n_generators} generators, {n_loads} loads.")
    print(f"Base MVA: {baseMVA}")

    return {
        'n_buses': n_buses,
        'n_generators': n_generators,
        'n_loads': n_loads,
        'gen_buses': gen_buses,
        'Y_bus': Y_bus,
        'base_P_demand_pu': base_P_demand_pu,
        'base_Q_demand_pu': base_Q_demand_pu,
        'cost_coeffs': cost_coeffs,
        'gen_limits': gen_limits,
        'V_min': V_min,
        'V_max': V_max,
        'slack_bus': slack_bus,
        'baseMVA': baseMVA
    }


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


# ============================================================================
# SECTION 2: NEWTON-RAPHSON POWER FLOW SOLVER
# ============================================================================

class NewtonRaphsonPF:
    """Newton-Raphson Power Flow Solver"""
    
    def __init__(self, Y_bus, slack_bus, gen_buses, gen_limits, baseMVA):
        self.Y_bus = Y_bus  # Admittance matrix (complex, not tensor)
        self.slack_bus = slack_bus
        self.gen_buses = gen_buses
        self.gen_limits = gen_limits
        self.baseMVA = baseMVA
        self.n_buses = Y_bus.shape[0]
        
    def solve(self, P_demand, Q_demand, Pg_pdl, Qg_pdl, V_init, theta_init, max_iterations=20, tolerance=1e-6):
        """
        Solve power flow using Newton-Raphson method with SAME generator dispatch as PDL.
        
        Args:
            P_demand: Active power demand (pu) - shape (n_buses,)
            Q_demand: Reactive power demand (pu) - shape (n_buses,)
            Pg_pdl: Generator active power from PDL (pu) - shape (n_gen,)
            Qg_pdl: Generator reactive power from PDL (pu) - shape (n_gen,)
            V_init: Initial voltage magnitudes (pu) - shape (n_buses,)
            theta_init: Initial voltage angles (rad) - shape (n_buses,)
            max_iterations: Maximum iterations for NR
            tolerance: Convergence tolerance
            
        Returns:
            V_nr, theta_nr: Converged voltage magnitudes and angles
        """
        
        V = V_init.copy()
        theta = theta_init.copy()
        
        # Compute net power injections at each bus (generation - demand)
        P_net = np.zeros(self.n_buses)
        Q_net = np.zeros(self.n_buses)
        
        # Add generator injections
        for idx, bus in enumerate(self.gen_buses):
            P_net[bus] += Pg_pdl[idx]
            Q_net[bus] += Qg_pdl[idx]
        
        # Subtract load demands
        P_net -= P_demand
        Q_net -= Q_demand
        
        # Identify non-slack buses for equations
        # All buses except slack need P balance
        # Only PQ buses (non-generator, non-slack) need Q balance
        pv_buses = np.array([b for b in self.gen_buses if b != self.slack_bus])
        pq_buses = np.array([b for b in range(self.n_buses) 
                            if b not in self.gen_buses and b != self.slack_bus])
        
        for iteration in range(max_iterations):
            # Compute power flow from current voltage solution
            V_complex = V * np.exp(1j * theta)
            I = self.Y_bus @ V_complex
            S = V_complex * np.conj(I)
            P_calc = S.real
            Q_calc = S.imag
            
            # Compute mismatches (specified net injection - calculated)
            P_mismatch = P_net - P_calc
            Q_mismatch = Q_net - Q_calc
            
            # Check convergence on non-slack buses
            P_check = np.concatenate([P_mismatch[pv_buses], P_mismatch[pq_buses]]) if len(pv_buses) > 0 else P_mismatch[pq_buses]
            Q_check = Q_mismatch[pq_buses]
            
            if len(P_check) > 0 and len(Q_check) > 0:
                max_mismatch = np.max(np.abs(np.concatenate([P_check, Q_check])))
            elif len(P_check) > 0:
                max_mismatch = np.max(np.abs(P_check))
            else:
                max_mismatch = np.max(np.abs(Q_check)) if len(Q_check) > 0 else 0
            
            if max_mismatch < tolerance:
                break
            
            # Build Jacobian matrix
            J = self._build_jacobian(V, theta, self.Y_bus, pv_buses, pq_buses)
            
            # Build mismatch vector
            if len(pv_buses) > 0:
                mismatch = np.concatenate([P_mismatch[pv_buses], P_mismatch[pq_buses], Q_mismatch[pq_buses]])
            else:
                mismatch = np.concatenate([P_mismatch[pq_buses], Q_mismatch[pq_buses]])
            
            # Solve linear system J * dx = mismatch
            try:
                dx = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                # Use least squares if singular
                dx = np.linalg.lstsq(J, mismatch, rcond=None)[0]
            
            # Update state variables
            idx = 0
            
            # Update angles for PV buses (non-slack generators)
            for bus in pv_buses:
                theta[bus] += dx[idx]
                idx += 1
            
            # Update angles for PQ buses
            for bus in pq_buses:
                theta[bus] += dx[idx]
                idx += 1
            
            # Update voltage magnitudes for PQ buses only
            for bus in pq_buses:
                V[bus] += dx[idx]
                V[bus] = np.clip(V[bus], 0.5, 1.5)  # Safety voltage limits
                idx += 1
        
        # Ensure slack bus angle remains at reference (0)
        theta[self.slack_bus] = 0.0
        
        return V, theta
    
    def _build_jacobian(self, V, theta, Y_bus, pv_buses, pq_buses):
        """Build Jacobian matrix for Newton-Raphson"""
        n_pv = len(pv_buses)
        n_pq = len(pq_buses)
        n_vars = n_pv + 2 * n_pq
        
        J = np.zeros((n_vars, n_vars))
        
        V_complex = V * np.exp(1j * theta)
        
        # Compute derivatives
        for i, bus_i in enumerate(pv_buses):
            # dP_i/dtheta_j for PV buses
            for j, bus_j in enumerate(pv_buses):
                if bus_i == bus_j:
                    term = 0
                    for k in range(self.n_buses):
                        Yk = np.abs(Y_bus[bus_i, k])
                        term += Yk * V[k] * np.sin(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[i, j] = -V[bus_i] * term
                else:
                    J[i, j] = V[bus_i] * np.abs(Y_bus[bus_i, bus_j]) * V[bus_j] * \
                               np.sin(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dP_i/dtheta_j for PQ buses
            for j, bus_j in enumerate(pq_buses):
                if bus_i == bus_j:
                    term = 0
                    for k in range(self.n_buses):
                        Yk = np.abs(Y_bus[bus_i, k])
                        term += Yk * V[k] * np.sin(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[i, n_pv + j] = -V[bus_i] * term
                else:
                    J[i, n_pv + j] = V[bus_i] * np.abs(Y_bus[bus_i, bus_j]) * V[bus_j] * \
                                     np.sin(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dP_i/dV_j for PQ buses
            for j, bus_j in enumerate(pq_buses):
                if bus_i == bus_j:
                    term = 2 * V[bus_i] * np.abs(Y_bus[bus_i, bus_i]) * np.cos(np.angle(Y_bus[bus_i, bus_i]))
                    for k in range(self.n_buses):
                        if k != bus_i:
                            term += np.abs(Y_bus[bus_i, k]) * np.cos(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[i, n_pv + n_pq + j] = term
                else:
                    J[i, n_pv + n_pq + j] = np.abs(Y_bus[bus_i, bus_j]) * \
                                            np.cos(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
        
        # Equations for PQ buses
        for i, bus_i in enumerate(pq_buses):
            # dP_i/dtheta_j
            for j, bus_j in enumerate(pv_buses):
                J[n_pv + i, j] = V[bus_i] * np.abs(Y_bus[bus_i, bus_j]) * V[bus_j] * \
                                  np.sin(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dP_i/dtheta_j for PQ buses
            for j, bus_j in enumerate(pq_buses):
                if bus_i == bus_j:
                    term = 0
                    for k in range(self.n_buses):
                        Yk = np.abs(Y_bus[bus_i, k])
                        term += Yk * V[k] * np.sin(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[n_pv + i, n_pv + j] = -V[bus_i] * term
                else:
                    J[n_pv + i, n_pv + j] = V[bus_i] * np.abs(Y_bus[bus_i, bus_j]) * V[bus_j] * \
                                            np.sin(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dP_i/dV_j for PQ buses
            for j, bus_j in enumerate(pq_buses):
                if bus_i == bus_j:
                    term = 2 * V[bus_i] * np.abs(Y_bus[bus_i, bus_i]) * np.cos(np.angle(Y_bus[bus_i, bus_i]))
                    for k in range(self.n_buses):
                        if k != bus_i:
                            term += np.abs(Y_bus[bus_i, k]) * np.cos(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[n_pv + i, n_pv + n_pq + j] = term
                else:
                    J[n_pv + i, n_pv + n_pq + j] = np.abs(Y_bus[bus_i, bus_j]) * \
                                                    np.cos(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dQ_i/dtheta_j for PV buses
            for j, bus_j in enumerate(pv_buses):
                J[n_pv + n_pq + i, j] = -V[bus_i] * np.abs(Y_bus[bus_i, bus_j]) * V[bus_j] * \
                                         np.cos(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dQ_i/dtheta_j for PQ buses
            for j, bus_j in enumerate(pq_buses):
                if bus_i == bus_j:
                    term = 0
                    for k in range(self.n_buses):
                        Yk = np.abs(Y_bus[bus_i, k])
                        term += Yk * V[k] * np.cos(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[n_pv + n_pq + i, n_pv + j] = V[bus_i] * term
                else:
                    J[n_pv + n_pq + i, n_pv + j] = -V[bus_i] * np.abs(Y_bus[bus_i, bus_j]) * V[bus_j] * \
                                                    np.cos(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
            
            # dQ_i/dV_j for PQ buses
            for j, bus_j in enumerate(pq_buses):
                if bus_i == bus_j:
                    term = -2 * V[bus_i] * np.abs(Y_bus[bus_i, bus_i]) * np.sin(np.angle(Y_bus[bus_i, bus_i]))
                    for k in range(self.n_buses):
                        if k != bus_i:
                            term += np.abs(Y_bus[bus_i, k]) * np.sin(theta[bus_i] - theta[k] - np.angle(Y_bus[bus_i, k]))
                    J[n_pv + n_pq + i, n_pv + n_pq + j] = term
                else:
                    J[n_pv + n_pq + i, n_pv + n_pq + j] = np.abs(Y_bus[bus_i, bus_j]) * \
                                                          np.sin(theta[bus_i] - theta[bus_j] - np.angle(Y_bus[bus_i, bus_j]))
        
        return J


# ============================================================================
# SECTION 3: PDL NETWORKS
# ============================================================================

class ACPFPrimalNet(nn.Module):
    def __init__(self, n_buses, n_generators, hidden_dim=256):
        super().__init__()
        self.n_buses = n_buses
        self.n_generators = n_generators

        input_dim = 2 * n_buses
        output_dim = 2 * n_generators + n_buses + (n_buses - 1)

        hidden_size = int(1.2 * output_dim)

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_size),
            nn.ReLU()
        )

        self.pg_net = nn.Linear(hidden_size, n_generators)
        self.qg_net = nn.Linear(hidden_size, n_generators)
        self.v_net = nn.Linear(hidden_size, n_buses)
        self.theta_net = nn.Linear(hidden_size, n_buses - 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, P_demand, Q_demand, gen_limits, V_min, V_max, slack_bus=0):
        batch_size = P_demand.shape[0]
        device = P_demand.device

        x = torch.cat([P_demand, Q_demand], dim=1)
        h = self.embedding(x)

        pg_raw = self.sigmoid(self.pg_net(h))
        P_min = torch.tensor(gen_limits['P_min'], device=device, dtype=torch.float32).unsqueeze(0)
        P_max = torch.tensor(gen_limits['P_max'], device=device, dtype=torch.float32).unsqueeze(0)
        Pg = P_min + pg_raw * (P_max - P_min)

        qg_raw = self.sigmoid(self.qg_net(h))
        Q_min = torch.tensor(gen_limits['Q_min'], device=device, dtype=torch.float32).unsqueeze(0)
        Q_max = torch.tensor(gen_limits['Q_max'], device=device, dtype=torch.float32).unsqueeze(0)
        Qg = Q_min + qg_raw * (Q_max - Q_min)

        v_raw = self.sigmoid(self.v_net(h))
        V = V_min + v_raw * (V_max - V_min)

        theta = torch.zeros(batch_size, self.n_buses, device=device)
        theta_raw = self.theta_net(h)
        if slack_bus == 0:
            theta[:, 1:] = theta_raw
        else:
            theta[:, :slack_bus] = theta_raw[:, :slack_bus]
            theta[:, slack_bus+1:] = theta_raw[:, slack_bus:]

        return Pg, Qg, V, theta


class ACPFDualNet(nn.Module):
    def __init__(self, n_buses, hidden_dim=256):
        super().__init__()

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
        return self.net(x)


# ============================================================================
# SECTION 4: PDL CLASS
# ============================================================================

class PDL_ACPF:

    def __init__(self, system_data, rho_init=1.0, rho_max=10000,
                 alpha=2.0, tau=0.5, device='cpu'):
        self.device = device
        self.system_data = system_data

        self.n_buses = system_data['n_buses']
        self.n_generators = system_data['n_generators']
        self.n_loads = system_data['n_loads']
        self.gen_buses = system_data['gen_buses']
        self.Y_bus = torch.tensor(system_data['Y_bus'], dtype=torch.complex64).to(device)
        self.cost_coeffs = system_data['cost_coeffs']
        self.gen_limits = system_data['gen_limits']
        self.V_min = system_data['V_min']
        self.V_max = system_data['V_max']
        self.slack_bus = system_data['slack_bus']
        self.baseMVA = system_data['baseMVA']

        self.rho = rho_init
        self.rho_max = rho_max
        self.alpha = alpha
        self.tau = tau

        self.primal_net = ACPFPrimalNet(self.n_buses, self.n_generators).to(device)
        self.dual_net = ACPFDualNet(self.n_buses).to(device)

        self.primal_optimizer = optim.Adam(self.primal_net.parameters(), lr=2e-4)
        self.dual_optimizer = optim.Adam(self.dual_net.parameters(), lr=2e-4)

        self.history = defaultdict(list)

    def compute_power_balance(self, Pg, Qg, V, theta, P_demand, Q_demand):
        batch_size = V.shape[0]

        P_inj = torch.zeros(batch_size, self.n_buses, device=self.device, dtype=torch.float32)
        Q_inj = torch.zeros(batch_size, self.n_buses, device=self.device, dtype=torch.float32)

        for idx, bus in enumerate(self.gen_buses):
            P_inj[:, bus] += Pg[:, idx]
            Q_inj[:, bus] += Qg[:, idx]

        P_inj -= P_demand
        Q_inj -= Q_demand

        V_complex = V * torch.exp(1j * theta)

        I_complex = torch.matmul(V_complex, self.Y_bus.T)
        S_complex = V_complex * torch.conj(I_complex)

        P_calc = S_complex.real
        Q_calc = S_complex.imag

        P_viol = P_calc - P_inj
        Q_viol = Q_calc - Q_inj

        return P_viol, Q_viol

    def primal_loss(self, Pg, Qg, V, theta, P_demand, Q_demand, multipliers):
        """Primal loss for Power Flow (PF): Lagrangian + penalty (NO COST)"""
        
        P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)

        lambda_P = multipliers[:, :self.n_buses]
        lambda_Q = multipliers[:, self.n_buses:]

        lagrangian = torch.sum(lambda_P * P_viol, dim=1).mean()
        lagrangian += torch.sum(lambda_Q * Q_viol, dim=1).mean()

        penalty = (self.rho / 2) * torch.sum(P_viol**2, dim=1).mean()
        penalty += (self.rho / 2) * torch.sum(Q_viol**2, dim=1).mean()

        loss = lagrangian + penalty

        return loss

    def dual_loss(self, multipliers, multipliers_old, Pg, Qg, V, theta, P_demand, Q_demand):
        """Dual loss: match ALM update rule"""
        P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)

        lambda_P_target = multipliers_old[:, :self.n_buses] + self.rho * P_viol
        lambda_Q_target = multipliers_old[:, self.n_buses:] + self.rho * Q_viol

        loss = torch.mean((multipliers[:, :self.n_buses] - lambda_P_target)**2)
        loss += torch.mean((multipliers[:, self.n_buses:] - lambda_Q_target)**2)

        return loss

    def train_epoch(self, P_demand_batch, Q_demand_batch, inner_iters=250):

        primal_losses = []
        for _ in range(inner_iters):
            self.primal_optimizer.zero_grad()

            Pg, Qg, V, theta = self.primal_net(
                P_demand_batch, Q_demand_batch,
                self.gen_limits, self.V_min, self.V_max, self.slack_bus
            )
            multipliers = self.dual_net(P_demand_batch, Q_demand_batch).detach()

            loss = self.primal_loss(Pg, Qg, V, theta, P_demand_batch, Q_demand_batch, multipliers)
            loss.backward()
            self.primal_optimizer.step()

            primal_losses.append(loss.item())

        dual_net_old = ACPFDualNet(self.n_buses).to(self.device)
        dual_net_old.load_state_dict(self.dual_net.state_dict())

        dual_losses = []
        for _ in range(inner_iters):
            self.dual_optimizer.zero_grad()

            Pg, Qg, V, theta = self.primal_net(
                P_demand_batch, Q_demand_batch,
                self.gen_limits, self.V_min, self.V_max, self.slack_bus
            )
            Pg, Qg, V, theta = Pg.detach(), Qg.detach(), V.detach(), theta.detach()

            multipliers = self.dual_net(P_demand_batch, Q_demand_batch)
            multipliers_old = dual_net_old(P_demand_batch, Q_demand_batch).detach()

            loss = self.dual_loss(multipliers, multipliers_old, Pg, Qg, V, theta,
                                   P_demand_batch, Q_demand_batch)
            loss.backward()
            self.dual_optimizer.step()

            dual_losses.append(loss.item())

        with torch.no_grad():
            Pg, Qg, V, theta = self.primal_net(
                P_demand_batch, Q_demand_batch,
                self.gen_limits, self.V_min, self.V_max, self.slack_bus
            )
            P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta,
                                                         P_demand_batch, Q_demand_batch)

            max_P_viol = torch.max(torch.abs(P_viol)).item()
            max_Q_viol = torch.max(torch.abs(Q_viol)).item()
            max_viol = max(max_P_viol, max_Q_viol)
            mean_viol = (torch.mean(torch.abs(P_viol)) + torch.mean(torch.abs(Q_viol))).item() / 2

            if not hasattr(self, 'prev_max_viol'):
                self.prev_max_viol = max_viol
            elif max_viol > self.tau * self.prev_max_viol:
                self.rho = min(self.alpha * self.rho, self.rho_max)
            self.prev_max_viol = max_viol

        self.history['primal_loss'].append(np.mean(primal_losses))
        self.history['dual_loss'].append(np.mean(dual_losses))
        self.history['max_P_viol'].append(max_P_viol)
        self.history['max_Q_viol'].append(max_Q_viol)
        self.history['max_viol'].append(max_viol)
        self.history['mean_viol'].append(mean_viol)
        self.history['rho'].append(self.rho)

        return np.mean(primal_losses), max_viol

    def predict(self, P_demand, Q_demand):
        """Predict solution"""
        self.primal_net.eval()
        with torch.no_grad():
            Pg, Qg, V, theta = self.primal_net(
                P_demand, Q_demand,
                self.gen_limits, self.V_min, self.V_max, self.slack_bus
            )
        self.primal_net.train()
        return Pg, Qg, V, theta


# ============================================================================
# SECTION 5: COMPARISON PLOTS
# ============================================================================

def plot_voltage_comparison(V_pdl, V_nr, case_name):
    """Plot voltage magnitude comparison between PDL and Newton-Raphson"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Voltage Magnitude Comparison: PDL vs Newton-Raphson - {case_name}', 
                 fontsize=14, fontweight='bold')

    V_pdl_np = V_pdl.cpu().numpy() if isinstance(V_pdl, torch.Tensor) else V_pdl
    
    # Average voltages across samples
    V_pdl_mean = V_pdl_np.mean(axis=0)
    V_nr_mean = V_nr.mean(axis=0)
    
    n_buses = len(V_pdl_mean)
    
    # Plot 1: Voltage profiles comparison
    ax = axes[0, 0]
    ax.plot(range(n_buses), V_pdl_mean, 'o-', label='PDL', linewidth=2, markersize=4)
    ax.plot(range(n_buses), V_nr_mean, 's-', label='Newton-Raphson', linewidth=2, markersize=4)
    ax.set_xlabel('Bus ID')
    ax.set_ylabel('Voltage Magnitude (pu)')
    ax.set_title('Average Voltage Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Voltage difference (error)
    ax = axes[0, 1]
    voltage_diff = np.abs(V_pdl_mean - V_nr_mean)
    ax.bar(range(n_buses), voltage_diff, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Bus ID')
    ax.set_ylabel('Absolute Voltage Difference (pu)')
    ax.set_title('Voltage Magnitude Error: |V_PDL - V_NR|')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution of voltage errors
    ax = axes[1, 0]
    all_errors = np.abs(V_pdl_np - V_nr).flatten()
    ax.hist(all_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=np.mean(all_errors), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(all_errors):.6f} pu')
    ax.axvline(x=np.max(all_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Max: {np.max(all_errors):.6f} pu')
    ax.set_xlabel('Voltage Error (pu)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Voltage Magnitude Errors')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot
    ax = axes[1, 1]
    ax.scatter(V_nr_mean, V_pdl_mean, s=50, alpha=0.6, edgecolor='black')
    min_v = min(V_nr_mean.min(), V_pdl_mean.min())
    max_v = max(V_nr_mean.max(), V_pdl_mean.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, label='Perfect Match')
    ax.set_xlabel('Newton-Raphson Voltage (pu)')
    ax.set_ylabel('PDL Voltage (pu)')
    ax.set_title('Scatter: Voltage Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_angle_comparison(theta_pdl, theta_nr, case_name, slack_bus):
    """Plot voltage angle comparison between PDL and Newton-Raphson"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Voltage Angle Comparison: PDL vs Newton-Raphson - {case_name}', 
                 fontsize=14, fontweight='bold')

    theta_pdl_np = theta_pdl.cpu().numpy() if isinstance(theta_pdl, torch.Tensor) else theta_pdl
    theta_pdl_deg = np.degrees(theta_pdl_np)
    theta_nr_deg = np.degrees(theta_nr)
    
    # Average angles across samples
    theta_pdl_mean = theta_pdl_deg.mean(axis=0)
    theta_nr_mean = theta_nr_deg.mean(axis=0)
    
    n_buses = len(theta_pdl_mean)
    
    # Plot 1: Angle profiles comparison
    ax = axes[0, 0]
    ax.plot(range(n_buses), theta_pdl_mean, 'o-', label='PDL', linewidth=2, markersize=4)
    ax.plot(range(n_buses), theta_nr_mean, 's-', label='Newton-Raphson', linewidth=2, markersize=4)
    ax.set_xlabel('Bus ID')
    ax.set_ylabel('Voltage Angle (degrees)')
    ax.set_title('Average Voltage Angle Profile Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Angle difference (error)
    ax = axes[0, 1]
    angle_diff = np.abs(theta_pdl_mean - theta_nr_mean)
    ax.bar(range(n_buses), angle_diff, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Bus ID')
    ax.set_ylabel('Absolute Angle Difference (degrees)')
    ax.set_title('Voltage Angle Error: |θ_PDL - θ_NR|')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution of angle errors
    ax = axes[1, 0]
    all_errors = np.abs(theta_pdl_deg - theta_nr_deg).flatten()
    ax.hist(all_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=np.mean(all_errors), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_errors):.4f}°')
    ax.axvline(x=np.max(all_errors), color='orange', linestyle='--', linewidth=2,
               label=f'Max: {np.max(all_errors):.4f}°')
    ax.set_xlabel('Angle Error (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Voltage Angle Errors')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot
    ax = axes[1, 1]
    ax.scatter(theta_nr_mean, theta_pdl_mean, s=50, alpha=0.6, edgecolor='black')
    min_a = min(theta_nr_mean.min(), theta_pdl_mean.min())
    max_a = max(theta_nr_mean.max(), theta_pdl_mean.max())
    ax.plot([min_a, max_a], [min_a, max_a], 'k--', linewidth=2, label='Perfect Match')
    ax.set_xlabel('Newton-Raphson Angle (degrees)')
    ax.set_ylabel('PDL Angle (degrees)')
    ax.set_title('Scatter: Angle Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_combined_comparison(V_pdl, theta_pdl, V_nr, theta_nr, case_name, baseMVA, slack_bus):
    """Create combined comparison metrics plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'PDL vs Newton-Raphson: Detailed Comparison - {case_name}', 
                 fontsize=14, fontweight='bold')

    V_pdl_np = V_pdl.cpu().numpy() if isinstance(V_pdl, torch.Tensor) else V_pdl
    theta_pdl_np = theta_pdl.cpu().numpy() if isinstance(theta_pdl, torch.Tensor) else theta_pdl
    
    V_pdl_mean = V_pdl_np.mean(axis=0)
    V_nr_mean = V_nr.mean(axis=0)
    theta_pdl_deg = np.degrees(theta_pdl_np)
    theta_nr_deg = np.degrees(theta_nr)
    
    n_buses = V_pdl_np.shape[1]
    n_samples = V_pdl_np.shape[0]
    
    # Voltage statistics
    ax = axes[0, 0]
    voltage_rmse = np.sqrt(np.mean((V_pdl_np - V_nr)**2))
    voltage_mae = np.mean(np.abs(V_pdl_np - V_nr))
    
    ax.bar(['RMSE', 'MAE'], [voltage_rmse, voltage_mae], color=['blue', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Error (pu)')
    ax.set_title('Voltage Magnitude Error Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([voltage_rmse, voltage_mae]):
        ax.text(i, v, f'{v:.6f}', ha='center', va='bottom')
    
    # Angle statistics
    ax = axes[0, 1]
    angle_rmse = np.sqrt(np.mean((theta_pdl_deg - theta_nr_deg)**2))
    angle_mae = np.mean(np.abs(theta_pdl_deg - theta_nr_deg))
    
    ax.bar(['RMSE', 'MAE'], [angle_rmse, angle_mae], color=['purple', 'green'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Voltage Angle Error Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([angle_rmse, angle_mae]):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Voltage per bus comparison
    ax = axes[0, 2]
    bus_v_errors = np.abs(V_pdl_mean - V_nr_mean)
    top_buses = np.argsort(bus_v_errors)[-10:]
    ax.barh(range(len(top_buses)), bus_v_errors[top_buses], alpha=0.7, color='red', edgecolor='black')
    ax.set_yticks(range(len(top_buses)))
    ax.set_yticklabels([f'Bus {b}' for b in top_buses])
    ax.set_xlabel('Voltage Error (pu)')
    ax.set_title('Top 10 Buses with Highest Voltage Error')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Angle per bus comparison
    ax = axes[1, 0]
    theta_pdl_mean = theta_pdl_deg.mean(axis=0)
    theta_nr_mean = theta_nr_deg.mean(axis=0)
    bus_a_errors = np.abs(theta_pdl_mean - theta_nr_mean)
    top_buses_a = np.argsort(bus_a_errors)[-10:]
    ax.barh(range(len(top_buses_a)), bus_a_errors[top_buses_a], alpha=0.7, color='purple', edgecolor='black')
    ax.set_yticks(range(len(top_buses_a)))
    ax.set_yticklabels([f'Bus {b}' for b in top_buses_a])
    ax.set_xlabel('Angle Error (degrees)')
    ax.set_title('Top 10 Buses with Highest Angle Error')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Slack bus voltage comparison
    ax = axes[1, 1]
    slack_V_pdl = V_pdl_np[:, slack_bus]
    slack_V_nr = V_nr[:, slack_bus]
    ax.plot(range(n_samples), slack_V_pdl, 'o-', label='PDL', alpha=0.6, markersize=3)
    ax.plot(range(n_samples), slack_V_nr, 's-', label='Newton-Raphson', alpha=0.6, markersize=3)
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Voltage (pu)')
    ax.set_title(f'Slack Bus (Bus {slack_bus}) Voltage Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Voltage range comparison
    ax = axes[1, 2]
    pdl_std = V_pdl_np.std(axis=0)
    nr_std = V_nr.std(axis=0)
    x = np.arange(min(20, n_buses))
    width = 0.35
    ax.bar(x - width/2, pdl_std[:min(20, n_buses)], width, label='PDL Std', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, nr_std[:min(20, n_buses)], width, label='NR Std', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Bus ID')
    ax.set_ylabel('Standard Deviation (pu)')
    ax.set_title('Voltage Variability (First 20 Buses)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 6: MAIN EXPERIMENT
# ============================================================================

def run_comparison_experiment(case_name, system_data, n_test=100, max_outer_iters=50, 
                             convergence_threshold=1e-2, inner_iters=250):
    """Run PDL training and comparison with Newton-Raphson"""
    
    print(f"\n{'='*80}")
    print(f"Running PDL vs Newton-Raphson Comparison: {case_name}")
    print(f"{'='*80}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Number of buses: {system_data['n_buses']}")
    print(f"Number of generators: {system_data['n_generators']}\n")

    # Generate training data
    print(f"Generating 25000 training samples...")
    P_train, Q_train = generate_load_scenarios(system_data, 25000, load_variation=0.3)
    P_train_tensor = torch.tensor(P_train, dtype=torch.float32).to(device)
    Q_train_tensor = torch.tensor(Q_train, dtype=torch.float32).to(device)

    # Initialize and train PDL
    print("Training PDL model...")
    pdl = PDL_ACPF(
        system_data=system_data,
        rho_init=1.0,
        rho_max=10000,
        alpha=2.0,
        tau=0.5,
        device=device
    )

    start_time = time.time()
    k = 0
    max_viol = float('inf')

    while max_viol > convergence_threshold and k < max_outer_iters:
        loss, max_viol = pdl.train_epoch(P_train_tensor, Q_train_tensor, inner_iters=inner_iters)
        if (k + 1) % 10 == 0:
            print(f"Iter {k+1:3d}: Max Viol={max_viol:.6f} pu, ρ={pdl.rho:.2f}")
        k += 1

    training_time = time.time() - start_time
    print(f"PDL training completed in {training_time:.2f}s after {k} iterations.\n")

    # Generate test data
    print(f"Generating {n_test} test samples...")
    P_test, Q_test = generate_load_scenarios(system_data, n_test, load_variation=0.3)
    P_test_tensor = torch.tensor(P_test, dtype=torch.float32).to(device)
    Q_test_tensor = torch.tensor(Q_test, dtype=torch.float32).to(device)

    # PDL predictions
    print("Running PDL inference...")
    start_time = time.time()
    Pg_pdl, Qg_pdl, V_pdl, theta_pdl = pdl.predict(P_test_tensor, Q_test_tensor)
    pdl_time = time.time() - start_time

    # Newton-Raphson solutions
    print("Running Newton-Raphson solver (20 iterations)...")
    nr_solver = NewtonRaphsonPF(
        system_data['Y_bus'],
        system_data['slack_bus'],
        system_data['gen_buses'],
        system_data['gen_limits'],
        system_data['baseMVA']
    )

    V_nr_all = []
    theta_nr_all = []

    start_time = time.time()
    for i in range(n_test):
        # Get PDL solution - use same generator dispatch
        Pg_pdl_sample = Pg_pdl[i].detach().cpu().numpy()
        Qg_pdl_sample = Qg_pdl[i].detach().cpu().numpy()
        
        # Initialize NR with flat start (standard practice)
        V_init = np.ones(system_data['n_buses'])
        V_init[system_data['slack_bus']] = 1.0  # Slack bus voltage
        # Set generator bus voltages to nominal
        for gen_bus in system_data['gen_buses']:
            V_init[gen_bus] = 1.0
        
        theta_init = np.zeros(system_data['n_buses'])

        # Solve with Newton-Raphson using SAME generator dispatch as PDL
        V_nr, theta_nr = nr_solver.solve(
            P_test[i],
            Q_test[i],
            Pg_pdl_sample,  # Use PDL's generator dispatch
            Qg_pdl_sample,  # Use PDL's generator dispatch
            V_init,
            theta_init,
            max_iterations=20
        )

        V_nr_all.append(V_nr)
        theta_nr_all.append(theta_nr)

    nr_time = time.time() - start_time

    V_nr_all = np.array(V_nr_all)
    theta_nr_all = np.array(theta_nr_all)

    print(f"Newton-Raphson completed in {nr_time:.4f}s ({nr_time/n_test*1000:.4f}ms per sample)\n")

    # Compute comparison metrics
    V_pdl_np = V_pdl.detach().cpu().numpy()
    theta_pdl_np = theta_pdl.detach().cpu().numpy()

    voltage_rmse = np.sqrt(np.mean((V_pdl_np - V_nr_all)**2))
    voltage_mae = np.mean(np.abs(V_pdl_np - V_nr_all))
    voltage_max_error = np.max(np.abs(V_pdl_np - V_nr_all))

    theta_pdl_deg = np.degrees(theta_pdl_np)
    theta_nr_deg = np.degrees(theta_nr_all)
    angle_rmse = np.sqrt(np.mean((theta_pdl_deg - theta_nr_deg)**2))
    angle_mae = np.mean(np.abs(theta_pdl_deg - theta_nr_deg))
    angle_max_error = np.max(np.abs(theta_pdl_deg - theta_nr_deg))

    # Print results
    print(f"{'='*80}")
    print(f"COMPARISON RESULTS: {case_name}")
    print(f"{'='*80}")
    print(f"\nVoltage Magnitude Errors:")
    print(f"  RMSE: {voltage_rmse:.8f} pu")
    print(f"  MAE:  {voltage_mae:.8f} pu")
    print(f"  Max:  {voltage_max_error:.8f} pu")

    print(f"\nVoltage Angle Errors:")
    print(f"  RMSE: {angle_rmse:.6f} degrees")
    print(f"  MAE:  {angle_mae:.6f} degrees")
    print(f"  Max:  {angle_max_error:.6f} degrees")

    print(f"\nTiming:")
    print(f"  PDL training time:  {training_time:.2f}s")
    print(f"  PDL inference time: {pdl_time*1000:.2f}ms ({pdl_time/n_test*1000:.4f}ms/sample)")
    print(f"  NR solving time:    {nr_time*1000:.2f}ms ({nr_time/n_test*1000:.4f}ms/sample)")
    print(f"{'='*80}\n")

    # Create comparison plots
    print("Generating comparison plots...")
    fig1 = plot_voltage_comparison(V_pdl, V_nr_all, case_name)
    fig2 = plot_angle_comparison(theta_pdl, theta_nr_all, case_name, system_data['slack_bus'])
    fig3 = plot_combined_comparison(V_pdl, theta_pdl, V_nr_all, theta_nr_all, case_name,
                                   system_data['baseMVA'], system_data['slack_bus'])

    results = {
        'case_name': case_name,
        'voltage_rmse': voltage_rmse,
        'voltage_mae': voltage_mae,
        'voltage_max_error': voltage_max_error,
        'angle_rmse': angle_rmse,
        'angle_mae': angle_mae,
        'angle_max_error': angle_max_error,
        'training_time': training_time,
        'pdl_inference_time': pdl_time,
        'nr_solving_time': nr_time,
        'V_pdl': V_pdl_np,
        'V_nr': V_nr_all,
        'theta_pdl': theta_pdl_deg,
        'theta_nr': theta_nr_deg
    }

    return pdl, fig1, fig2, fig3, results


def print_summary_table(results_list):
    """Print summary table for multiple cases"""
    print(f"\n{'='*100}")
    print("SUMMARY: PDL vs NEWTON-RAPHSON COMPARISON")
    print(f"{'='*100}\n")

    data = []
    for results in results_list:
        data.append({
            'Case': results['case_name'],
            'V RMSE (pu)': f"{results['voltage_rmse']:.8f}",
            'V MAE (pu)': f"{results['voltage_mae']:.8f}",
            'V Max (pu)': f"{results['voltage_max_error']:.8f}",
            'θ RMSE (°)': f"{results['angle_rmse']:.6f}",
            'θ MAE (°)': f"{results['angle_mae']:.6f}",
            'θ Max (°)': f"{results['angle_max_error']:.6f}",
            'PDL Time (ms)': f"{results['pdl_inference_time']*1000:.2f}",
            'NR Time (ms)': f"{results['nr_solving_time']*1000:.2f}"
        })

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    print("\n" + "="*90)
    print("PDL WITH NEWTON-RAPHSON COMPARISON: IEEE CASE 57 AND CASE 118")
    print("="*90)

    # Load pandapower networks
    print("\nLoading pandapower networks...")
    try:
        net_57 = nw.case57()
        net_118 = nw.case118()
        print("Successfully loaded case57 and case118.")
    except Exception as e:
        print(f"Error loading pandapower networks: {e}")
        exit()

    # Get system data
    system_data_57 = get_pandapower_data(net_57, "case57")
    system_data_118 = get_pandapower_data(net_118, "case118")

    all_results = []

    # Run comparison for Case 57
    print("\n" + "="*90)
    print("EXPERIMENT 1: IEEE CASE 57 - PDL vs NEWTON-RAPHSON")
    print("="*90)
    pdl_57, fig1_57, fig2_57, fig3_57, results_57 = run_comparison_experiment(
        case_name="IEEE Case 57",
        system_data=system_data_57,
        n_test=50,
        max_outer_iters=50,
        convergence_threshold=1e-2,
        inner_iters=250
    )
    all_results.append(results_57)

    # Run comparison for Case 118
    print("\n" + "="*90)
    print("EXPERIMENT 2: IEEE CASE 118 - PDL vs NEWTON-RAPHSON")
    print("="*90)
    pdl_118, fig1_118, fig2_118, fig3_118, results_118 = run_comparison_experiment(
        case_name="IEEE Case 118",
        system_data=system_data_118,
        n_test=50,
        max_outer_iters=50,
        convergence_threshold=1e-2,
        inner_iters=250
    )
    all_results.append(results_118)

    # Print summary
    print_summary_table(all_results)

    plt.show()

    print("\nAll experiments completed successfully!")
