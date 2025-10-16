    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import time
    from collections import defaultdict

    import pandapower as pp
    import pandapower.networks as nw
    from pandapower.pypower.makeYbus import makeYbus
    import warnings
    warnings.filterwarnings("ignore")


    def get_pandapower_data(net, case_name):
        """
        Extracts system data from a pandapower network object
        and formats it for the PDL model.
        """
        print(f"Loading data from pandapower: {case_name}")


        try:
            pp.runpp(net, calculate_voltage_angles=False, enforce_q_limits=False, enforce_v_limits=False, init='flat')
        except:
            # Some cases might fail if not initialized, but ppc is often still built
            pass


        if not hasattr(net, '_ppc') or net._ppc is None:
            print("Running diagnostic to build ppc...")
            pp.diagnostic(net)

        # Now, net._ppc should exist
        ppc = net._ppc


        baseMVA = ppc['baseMVA']
        bus, gen, branch = ppc['bus'], ppc['gen'], ppc['branch']

        # Build Ybus
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        Y_bus = Ybus.toarray()

        # --- Get system dimensions ---
        n_buses = net.bus.shape[0]
        n_generators = net.gen.shape[0]
        n_loads = net.load.shape[0]


        gen_buses = net.gen.bus.values.tolist()

        base_P_demand = np.zeros(n_buses)
        base_Q_demand = np.zeros(n_buses)

        # Sum loads at each bus
        for _, load in net.load.iterrows():
            bus_idx = load.bus
            base_P_demand[bus_idx] += load.p_mw
            base_Q_demand[bus_idx] += load.q_mvar

        base_P_demand_pu = base_P_demand / baseMVA
        base_Q_demand_pu = base_Q_demand / baseMVA

        if net.poly_cost.shape[0] == n_generators:
            costs = net.poly_cost[net.poly_cost.et == 'gen'].sort_values(by='element')
            c2 = costs.cp2_eur_per_mw2.values
            c1 = costs.cp1_eur_per_mw.values
            c0 = costs.cp0_eur.values


            c2_pu = c2 * (baseMVA**2)
            c1_pu = c1 * baseMVA
            c0_pu = c0

        else:
            # Fallback if cost data is missing
            print("Warning: Cost data not found or incomplete. Using random costs.")
            c2_pu = np.random.uniform(0.005, 0.02, n_generators) * (baseMVA**2)
            c1_pu = np.random.uniform(15.0, 35.0, n_generators) * baseMVA
            c0_pu = np.random.uniform(80.0, 160.0, n_generators)

        cost_coeffs = {'c2': c2_pu, 'c1': c1_pu, 'c0': c0_pu}

        # --- Get generator limits (in p.u.) ---
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
        # Get the base demand vectors (shape n_buses)
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


    class ACOPFPrimalNet(nn.Module):
        """Primal network for AC-OPF"""
        def __init__(self, n_buses, n_generators, hidden_dim=256):
            super().__init__()
            self.n_buses = n_buses
            self.n_generators = n_generators

            input_dim = 2 * n_buses  # P and Q demands at all buses
            output_dim = 2 * n_generators + n_buses + (n_buses - 1)  # Pg, Qg, V (all buses), theta (n-1)

            hidden_size = int(1.2 * output_dim)


            self.embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_size),
                nn.ReLU()
            )


            self.pg_net = nn.Linear(hidden_size, n_generators)
            self.qg_net = nn.Linear(hidden_size, n_generators)
            self.v_net = nn.Linear(hidden_size, n_buses)  # Predicts voltage for ALL buses including slack
            self.theta_net = nn.Linear(hidden_size, n_buses - 1)  # Predicts angles for all buses except slack

            self.sigmoid = nn.Sigmoid()

        def forward(self, P_demand, Q_demand, gen_limits, V_min, V_max, slack_bus=0):
            batch_size = P_demand.shape[0]
            device = P_demand.device

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

            # Voltage magnitudes for ALL buses (network learns slack bus voltage too)
            v_raw = self.sigmoid(self.v_net(h))
            V = V_min + v_raw * (V_max - V_min)

            # Voltage angles (only slack bus angle is fixed at 0)
            theta = torch.zeros(batch_size, self.n_buses, device=device)
            theta_raw = self.theta_net(h)
            if slack_bus == 0:
                # Slack at bus 0, predict angles for buses 1 to n-1
                theta[:, 1:] = theta_raw
            else:
                # Slack at arbitrary bus, predict all other angles
                theta[:, :slack_bus] = theta_raw[:, :slack_bus]
                theta[:, slack_bus+1:] = theta_raw[:, slack_bus:]

            return Pg, Qg, V, theta


    class ACOPFDualNet(nn.Module):
        """Dual network for AC-OPF"""
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



    class PDL_ACOPF:
        """Primal-Dual Learning for AC-OPF"""

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

            self.primal_net = ACOPFPrimalNet(self.n_buses, self.n_generators).to(device)
            self.dual_net = ACOPFDualNet(self.n_buses).to(device)

            self.primal_optimizer = optim.Adam(self.primal_net.parameters(), lr=2e-4)
            self.dual_optimizer = optim.Adam(self.dual_net.parameters(), lr=2e-4)

            self.history = defaultdict(list)

        def compute_power_balance(self, Pg, Qg, V, theta, P_demand, Q_demand):
            """Compute power balance violations"""
            batch_size = V.shape[0]

            P_inj = torch.zeros(batch_size, self.n_buses, device=self.device, dtype=torch.float32)
            Q_inj = torch.zeros(batch_size, self.n_buses, device=self.device, dtype=torch.float32)


            for idx, bus in enumerate(self.gen_buses):
                P_inj[:, bus] += Pg[:, idx]
                Q_inj[:, bus] += Qg[:, idx]

            # Subtract demand (P_demand, Q_demand are p.u.)
            P_inj -= P_demand
            Q_inj -= Q_demand

            # Compute actual power flow using AC equations
            V_complex = V * torch.exp(1j * theta)

            I_complex = torch.matmul(V_complex, self.Y_bus.T) # (batch, n_buses) * (n_buses, n_buses) -> (batch, n_buses)
            S_complex = V_complex * torch.conj(I_complex)

            P_calc = S_complex.real
            Q_calc = S_complex.imag

            # Power balance violations (all in p.u.)
            P_viol = P_calc - P_inj
            Q_viol = Q_calc - Q_inj

            return P_viol, Q_viol

        def compute_cost(self, Pg):
            """Compute generation cost (using p.u. cost coefficients)"""
            c2 = torch.tensor(self.cost_coeffs['c2'], device=self.device, dtype=torch.float32).unsqueeze(0)
            c1 = torch.tensor(self.cost_coeffs['c1'], device=self.device, dtype=torch.float32).unsqueeze(0)
            c0 = torch.tensor(self.cost_coeffs['c0'], device=self.device, dtype=torch.float32).unsqueeze(0)

            # Pg is already in p.u.
            cost = torch.sum(c2 * Pg**2 + c1 * Pg + c0, dim=1)
            return cost.mean()

        def primal_loss(self, Pg, Qg, V, theta, P_demand, Q_demand, multipliers):
            """Primal loss for Power Flow (PF): Lagrangian + penalty (NO COST)"""
            # Generation cost (still computed for logging, but not used in loss)
            cost = self.compute_cost(Pg)

            # Power balance violations
            P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)

            # Multipliers
            lambda_P = multipliers[:, :self.n_buses]
            lambda_Q = multipliers[:, self.n_buses:]

            # Lagrangian term
            lagrangian = torch.sum(lambda_P * P_viol, dim=1).mean()
            lagrangian += torch.sum(lambda_Q * Q_viol, dim=1).mean()

            # Penalty term
            penalty = (self.rho / 2) * torch.sum(P_viol**2, dim=1).mean()
            penalty += (self.rho / 2) * torch.sum(Q_viol**2, dim=1).mean()


            loss = lagrangian + penalty
            # ---

            return loss, cost

        def dual_loss(self, multipliers, multipliers_old, Pg, Qg, V, theta, P_demand, Q_demand):
            """Dual loss: match ALM update rule"""
            P_viol, Q_viol = self.compute_power_balance(Pg, Qg, V, theta, P_demand, Q_demand)

            # Target multipliers
            lambda_P_target = multipliers_old[:, :self.n_buses] + self.rho * P_viol
            lambda_Q_target = multipliers_old[:, self.n_buses:] + self.rho * Q_viol

            # MSE loss
            loss = torch.mean((multipliers[:, :self.n_buses] - lambda_P_target)**2)
            loss += torch.mean((multipliers[:, self.n_buses:] - lambda_Q_target)**2)

            return loss

        def train_epoch(self, P_demand_batch, Q_demand_batch, inner_iters=250):
            """Train for one outer iteration"""

            # Primal learning
            primal_losses = []
            costs = []
            for _ in range(inner_iters):
                self.primal_optimizer.zero_grad()

                Pg, Qg, V, theta = self.primal_net(
                    P_demand_batch, Q_demand_batch,
                    self.gen_limits, self.V_min, self.V_max, self.slack_bus
                )
                multipliers = self.dual_net(P_demand_batch, Q_demand_batch).detach()

                loss, cost = self.primal_loss(Pg, Qg, V, theta, P_demand_batch, Q_demand_batch, multipliers)
                loss.backward()
                self.primal_optimizer.step()

                primal_losses.append(loss.item())
                costs.append(cost.item())

            dual_net_old = ACOPFDualNet(self.n_buses).to(self.device)
            dual_net_old.load_state_dict(self.dual_net.state_dict())

            # Dual learning
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
                
                # Track slack bus voltage
                slack_V = V[:, self.slack_bus].mean().item()

                # Update rho
                if not hasattr(self, 'prev_max_viol'):
                    self.prev_max_viol = max_viol
                elif max_viol > self.tau * self.prev_max_viol:
                    self.rho = min(self.alpha * self.rho, self.rho_max)
                self.prev_max_viol = max_viol

            # Store history
            self.history['primal_loss'].append(np.mean(primal_losses))
            self.history['dual_loss'].append(np.mean(dual_losses))
            self.history['cost'].append(np.mean(costs))
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
                    self.gen_limits, self.V_min, self.V_max, self.slack_bus
                )
            self.primal_net.train()
            return Pg, Qg, V, theta



    def plot_training_results(history, case_name, slack_bus_id):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'PDL Training Results - {case_name}', fontsize=16, fontweight='bold')

        # Cost convergence
        ax = axes[0, 0]
        ax.plot(history['cost'], linewidth=2, color='darkgreen')
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Generation Cost ($)')
        ax.set_title('Objective Function (Cost)')
        ax.grid(True, alpha=0.3)

        # Losses
        ax = axes[0, 1]
        ax.plot(history['primal_loss'], label='Primal Loss', linewidth=2)
        ax.plot(history['dual_loss'], label='Dual Loss', linewidth=2)
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Violations (P and Q)
        ax = axes[0, 2]
        ax.plot(history['max_P_viol'], label='Max P Violation', linewidth=2, color='red')
        ax.plot(history['max_Q_viol'], label='Max Q Violation', linewidth=2, color='blue')
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Violation (pu)')
        ax.set_title('Maximum Power Balance Violations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Slack Bus Voltage Evolution
        ax = axes[0, 3]
        ax.plot(history['slack_bus_voltage'], linewidth=2, color='darkblue', marker='o', markersize=3)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Nominal (1.0 pu)')
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Voltage (pu)')
        ax.set_title(f'Slack Bus (Bus {slack_bus_id}) Voltage Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mean violation
        ax = axes[1, 0]
        ax.plot(history['mean_viol'], linewidth=2, color='purple')
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Mean Violation (pu)')
        ax.set_title('Mean Power Balance Violation')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Rho evolution
        ax = axes[1, 1]
        ax.plot(history['rho'], linewidth=2, color='orange')
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('ρ')
        ax.set_title('Penalty Coefficient')
        ax.grid(True, alpha=0.3)

        # Max violation (total)
        ax = axes[1, 2]
        ax.plot(history['max_viol'], linewidth=2, color='darkred')
        ax.set_xlabel('Outer Iteration')
        ax.set_ylabel('Max Violation (pu)')
        ax.set_title('Overall Maximum Violation')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Slack Bus Voltage Statistics
        ax = axes[1, 3]
        slack_voltages = history['slack_bus_voltage']
        ax.hist(slack_voltages, bins=20, alpha=0.7, color='darkblue', edgecolor='black')
        ax.axvline(x=np.mean(slack_voltages), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(slack_voltages):.4f}')
        ax.axvline(x=1.0, color='k', linestyle=':', linewidth=2, label='Nominal: 1.0')
        ax.set_xlabel('Voltage (pu)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Slack Bus Voltage Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


    def plot_solution_quality(Pg, Qg, V, theta, P_viol, Q_viol, case_name, baseMVA, slack_bus_id):
        """Plot solution quality metrics"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Solution Quality - {case_name}', fontsize=16, fontweight='bold')

        # Convert to numpy and to MW/MVAr
        Pg_np = Pg.cpu().numpy() * baseMVA
        Qg_np = Qg.cpu().numpy() * baseMVA
        V_np = V.cpu().numpy()
        theta_np = np.degrees(theta.cpu().numpy())
        P_viol_np = P_viol.cpu().numpy() * baseMVA
        Q_viol_np = Q_viol.cpu().numpy() * baseMVA

        V_min = V.cpu().numpy().min()
        V_max = V.cpu().numpy().max()

        # Generator active power
        ax = axes[0, 0]
        for i in range(min(5, Pg_np.shape[0])):
            ax.plot(Pg_np[i, :], alpha=0.7, marker='o', markersize=4)
        ax.set_xlabel('Generator ID')
        ax.set_ylabel('Active Power (MW)')
        ax.set_title('Generator Active Power Setpoints')
        ax.grid(True, alpha=0.3)

        # Generator reactive power
        ax = axes[0, 1]
        for i in range(min(5, Qg_np.shape[0])):
            ax.plot(Qg_np[i, :], alpha=0.7, marker='o', markersize=4)
        ax.set_xlabel('Generator ID')
        ax.set_ylabel('Reactive Power (MVAr)')
        ax.set_title('Generator Reactive Power Setpoints')
        ax.grid(True, alpha=0.3)

        # Voltage Profile
        ax = axes[0, 2]
        for i in range(min(5, V_np.shape[0])):
            ax.plot(V_np[i, :], alpha=0.7, marker='o', markersize=2)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Nominal')
        ax.axhline(y=V_min, color='r', linestyle=':', alpha=0.5, label=f'V_min {V_min:.3f}')
        ax.axhline(y=V_max, color='r', linestyle=':', alpha=0.5, label=f'V_max {V_max:.3f}')
        ax.set_xlabel('Bus ID')
        ax.set_ylabel('Voltage Magnitude (pu)')
        ax.set_title('Voltage Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([V_min * 0.98, V_max * 1.02])

        # Slack Bus Voltage Across Test Samples
        ax = axes[0, 3]
        slack_voltages = V_np[:, slack_bus_id]
        ax.plot(slack_voltages, linewidth=2, color='darkblue', marker='o', markersize=4)
        ax.axhline(y=np.mean(slack_voltages), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(slack_voltages):.4f} pu')
        ax.axhline(y=1.0, color='k', linestyle=':', linewidth=2, label='Nominal: 1.0 pu')
        ax.set_xlabel('Test Sample ID')
        ax.set_ylabel('Voltage (pu)')
        ax.set_title(f'Slack Bus (Bus {slack_bus_id}) Voltage')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Voltage angles
        ax = axes[1, 0]
        for i in range(min(5, theta_np.shape[0])):
            ax.plot(theta_np[i, :], alpha=0.7, marker='o', markersize=2)
        ax.set_xlabel('Bus ID')
        ax.set_ylabel('Voltage Angle (degrees)')
        ax.set_title('Voltage Angle Profile')
        ax.grid(True, alpha=0.3)

        # P violation distribution
        ax = axes[1, 1]
        ax.hist(np.abs(P_viol_np.flatten()), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.set_xlabel('|P Violation| (MW)')
        ax.set_ylabel('Frequency')
        ax.set_title('Active Power Violation Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Q violation distribution
        ax = axes[1, 2]
        ax.hist(np.abs(Q_viol_np.flatten()), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('|Q Violation| (MVAr)')
        ax.set_ylabel('Frequency')
        ax.set_title('Reactive Power Violation Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Slack Bus Voltage Distribution (Test Set)
        ax = axes[1, 3]
        ax.hist(slack_voltages, bins=20, alpha=0.7, color='darkblue', edgecolor='black')
        ax.axvline(x=np.mean(slack_voltages), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(slack_voltages):.4f}')
        ax.axvline(x=1.0, color='k', linestyle=':', linewidth=2, label='Nominal: 1.0')
        ax.set_xlabel('Voltage (pu)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Slack Bus Voltage Distribution (Test Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig



    def run_acopf_experiment(case_name, system_data, n_train=25000, n_test=100,
                            max_outer_iters=500, convergence_threshold=1e-2, inner_iters=250):
        """Run AC-OPF experiment"""
        print(f"\n{'='*70}")
        print(f"Running AC-OPF PDL Experiment: {case_name}")
        print(f"{'='*70}\n")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        print(f"Number of buses: {system_data['n_buses']}")
        print(f"Number of generators: {system_data['n_generators']}")
        print(f"Number of loads: {system_data['n_loads']}")
        print(f"Slack bus: {system_data['slack_bus']}\n")

        # Generate training data
        print(f"Generating {n_train} training samples...")
        P_train, Q_train = generate_load_scenarios(system_data, n_train, load_variation=0.3)
        P_train_tensor = torch.tensor(P_train, dtype=torch.float32).to(device)
        Q_train_tensor = torch.tensor(Q_train, dtype=torch.float32).to(device)

        # Initialize PDL
        print("Initializing PDL model...")
        pdl = PDL_ACOPF(
            system_data=system_data,
            rho_init=1.0,
            rho_max=10000,
            alpha=2.0,
            tau=0.5,
            device=device
        )

        baseMVA = pdl.baseMVA

        # --- Start of Modified Training Loop ---
        # The training will now run until the maximum violation (P or Q)
        # falls below the convergence_threshold, or until max_outer_iters is reached.
        print(f"\nTraining until max violation < {convergence_threshold} (or max {max_outer_iters} outer iterations)...")
        print(f"Inner iterations per phase: {inner_iters}")
        print("-" * 70)

        start_time = time.time()
        k = 0
        max_viol = float('inf') # Initialize with a large value to start the loop

        while max_viol > convergence_threshold and k < max_outer_iters:
            loss, max_viol, cost = pdl.train_epoch(
                P_train_tensor, Q_train_tensor, inner_iters=inner_iters
            )
            
            # Get current slack bus voltage
            with torch.no_grad():
                _, _, V_temp, _ = pdl.primal_net(
                    P_train_tensor[:100], Q_train_tensor[:100],
                    pdl.gen_limits, pdl.V_min, pdl.V_max, pdl.slack_bus
                )
                slack_V_current = V_temp[:, pdl.slack_bus].mean().item()

            print(f"Iter {k+1:3d}: Cost=${cost:,.2f}, "
                f"Max Viol={max_viol:.6f} pu, "
                f"Slack V={slack_V_current:.4f} pu, "
                f"ρ={pdl.rho:.2f}")
            k += 1

        training_time = time.time() - start_time
        print(f"\nTraining stopped after {k} iterations.")


        print(f"\nGenerating {n_test} test samples...")
        P_test, Q_test = generate_load_scenarios(system_data, n_test, load_variation=0.3)
        P_test_tensor = torch.tensor(P_test, dtype=torch.float32).to(device)
        Q_test_tensor = torch.tensor(Q_test, dtype=torch.float32).to(device)

        print("Testing on unseen samples...")
        start_time = time.time()
        Pg, Qg, V, theta = pdl.predict(P_test_tensor, Q_test_tensor)
        inference_time = time.time() - start_time

        with torch.no_grad():
            P_viol, Q_viol = pdl.compute_power_balance(Pg, Qg, V, theta,
                                                    P_test_tensor, Q_test_tensor)
            test_cost = pdl.compute_cost(Pg)

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
        print(f"Average Generation Cost:  ${test_cost.item():,.2f}")
        print(f"Max P Violation:          {max_P_viol_test_pu:.8f} pu ({max_P_viol_test_pu*baseMVA:.4f} MW)")
        print(f"Max Q Violation:          {max_Q_viol_test_pu:.8f} pu ({max_Q_viol_test_pu*baseMVA:.4f} MVAr)")
        print(f"Max Overall Violation:    {max_viol_test_pu:.8f} pu")
        print(f"Mean P Violation:         {mean_P_viol_test_pu:.8f} pu ({mean_P_viol_test_pu*baseMVA:.4f} MW)")
        print(f"Mean Q Violation:         {mean_Q_viol_test_pu:.8f} pu ({mean_Q_viol_test_pu*baseMVA:.4f} MVAr)")

        V_mean = torch.mean(V).item()
        V_std = torch.std(V).item()
        V_min_val = torch.min(V).item()
        V_max_val = torch.max(V).item()

        print(f"\nVoltage Statistics (All Buses):")
        print(f"  Mean: {V_mean:.4f} pu")
        print(f"  Std:  {V_std:.4f} pu")
        print(f"  Min:  {V_min_val:.4f} pu (System limit: {pdl.V_min:.4f})")
        print(f"  Max:  {V_max_val:.4f} pu (System limit: {pdl.V_max:.4f})")

        print(f"\nSlack Bus Voltage Statistics (Bus {pdl.slack_bus}):")
        print(f"  Mean: {slack_V_mean:.4f} pu")
        print(f"  Std:  {slack_V_std:.4f} pu")
        print(f"  Min:  {slack_V_min:.4f} pu")
        print(f"  Max:  {slack_V_max:.4f} pu")
        print(f"  Deviation from nominal (1.0 pu): {abs(slack_V_mean - 1.0):.4f} pu")

        Pg_total_pu = torch.sum(Pg, dim=1).mean().item()
        print(f"\nTotal Generation: {Pg_total_pu * baseMVA:.2f} MW (average)")

        print("\nGenerating plots...")
        fig1 = plot_training_results(pdl.history, case_name, pdl.slack_bus)
        fig2 = plot_solution_quality(Pg, Qg, V, theta, P_viol, Q_viol, case_name, baseMVA, pdl.slack_bus)

        results = {
            'training_time': training_time,
            'inference_time_per_sample': inference_time / n_test,
            'max_P_viol_pu': max_P_viol_test_pu,
            'max_Q_viol_pu': max_Q_viol_test_pu,
            'max_viol_pu': max_viol_test_pu,
            'mean_P_viol_pu': mean_P_viol_test_pu,
            'mean_Q_viol_pu': mean_Q_viol_test_pu,
            'cost': test_cost.item(),
            'V_mean': V_mean,
            'V_std': V_std,
            'V_min': V_min_val,
            'V_max': V_max_val,
            'slack_bus': pdl.slack_bus,
            'slack_V_mean': slack_V_mean,
            'slack_V_std': slack_V_std,
            'slack_V_min': slack_V_min,
            'slack_V_max': slack_V_max,
            'Pg_total_mw': Pg_total_pu * baseMVA,
            'baseMVA': baseMVA
        }

        return pdl, fig1, fig2, results


    def create_comparison_plots(results_dict):
        """Create comparison plots between cases"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Comparison: Case 57 vs Case 118', fontsize=16, fontweight='bold')

        cases = list(results_dict.keys())

        # Violations comparison
        ax = axes[0, 0]
        x = np.arange(len(cases))
        width = 0.35
        max_P = [results_dict[c]['max_P_viol_pu'] * 1000 for c in cases]  # Convert to milli-pu
        max_Q = [results_dict[c]['max_Q_viol_pu'] * 1000 for c in cases]

        ax.bar(x - width/2, max_P, width, label='Max P Violation', color='red', alpha=0.7)
        ax.bar(x + width/2, max_Q, width, label='Max Q Violation', color='blue', alpha=0.7)
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

        ax.bar(x - width/2, mean_P, width, label='Mean P Violation', color='red', alpha=0.7)
        ax.bar(x + width/2, mean_Q, width, label='Mean Q Violation', color='blue', alpha=0.7)
        ax.set_ylabel('Mean Violation (×10⁻³ pu)')
        ax.set_title('Mean Power Balance Violations (p.u.)')
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Generation cost
        ax = axes[0, 2]
        costs = [results_dict[c]['cost'] for c in cases]
        bars = ax.bar(cases, costs, color='green', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Generation Cost ($)')
        ax.set_title('Average Generation Cost')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom')

        # Training time
        ax = axes[1, 0]
        times = [results_dict[c]['training_time'] for c in cases]
        bars = ax.bar(cases, times, color='orange', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom')

        # Inference time
        ax = axes[1, 1]
        inf_times = [results_dict[c]['inference_time_per_sample'] * 1000 for c in cases]
        bars = ax.bar(cases, inf_times, color='purple', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Time (milliseconds)')
        ax.set_title('Inference Time per Sample')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}ms', ha='center', va='bottom')

        # Voltage statistics
        ax = axes[1, 2]
        v_means = [results_dict[c]['V_mean'] for c in cases]
        v_stds = [results_dict[c]['V_std'] for c in cases]

        ax.bar(x - width/2, v_means, width, label='Mean Voltage', color='darkgreen', alpha=0.7)
        ax.bar(x + width/2, [s*10 for s in v_stds], width, label='Std Dev (×10)', color='lightgreen', alpha=0.7)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Voltage (pu)')
        ax.set_title('Voltage Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(cases)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def print_comparison_table(results_dict):
        """Print comparison table"""
        print(f"\n{'='*110}")
        print("COMPARATIVE RESULTS TABLE")
        print(f"{'='*110}")

        df_data = []
        for case_name, results in results_dict.items():
            df_data.append({
                'Case': case_name,
                'Slack Bus': results['slack_bus'],
                'Slack V (pu)': f"{results['slack_V_mean']:.4f}±{results['slack_V_std']:.4f}",
                'Cost ($)': f"{results['cost']:,.2f}",
                'Max Viol (pu)': f"{results['max_viol_pu']:.6f}",
                'Max P (MW)': f"{results['max_P_viol_pu'] * results['baseMVA']:.4f}",
                'Max Q (MVAr)': f"{results['max_Q_viol_pu'] * results['baseMVA']:.4f}",
                'Train (s)': f"{results['training_time']:.1f}",
                'Infer (ms)': f"{results['inference_time_per_sample']*1000:.4f}"
            })

        df = pd.DataFrame(df_data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        print(f"{'='*110}\n")

        print("SLACK BUS VOLTAGE ANALYSIS")
        print("-" * 110)
        for case_name, results in results_dict.items():
            deviation = abs(results['slack_V_mean'] - 1.0)
            print(f"{case_name}:")
            print(f"  Slack Bus ID: {results['slack_bus']}")
            print(f"  Mean Voltage: {results['slack_V_mean']:.4f} pu")
            print(f"  Std Deviation: {results['slack_V_std']:.4f} pu")
            print(f"  Range: [{results['slack_V_min']:.4f}, {results['slack_V_max']:.4f}] pu")
            print(f"  Deviation from nominal (1.0 pu): {deviation:.4f} pu ({deviation*100:.2f}%)")
            print()
        print("-" * 110)

        print("\nCOMPARISON WITH PAPER RESULTS (Table 3)")
        print("-" * 110)
        print("Paper Results (PDL):")
        print("  Case 57:  Gap(%)=0.21(0.02), Max viol=0.01(0.00)")
        print("  Case 118: Gap(%)=0.73(0.12), Max viol=0.04(0.01)")
        print("\nOur Implementation:")
        for case_name, results in results_dict.items():
            print(f"  {case_name}: Max viol={results['max_viol_pu']:.4f} pu")
        print("-" * 110)

    if __name__ == "__main__":
        print("\n" + "="*90)
        print("PDL FOR AC-OPTIMAL POWER FLOW: LARGE-SCALE POWER SYSTEM (PANDAPOWER)")
        print("Replicating Results from Park & Van Hentenryck (2022)")
        print("="*90)

        # Run experiments
        all_results = {}
        all_pdls = {}
        all_figs = {}

        # --- Load pandapower network ---
        print("\nLoading pandapower network...")
        try:
            # Try to load case2000 or case1354 (Polish network) or case2869 (PEGASE)
            try:
                net = nw.case2869pegase()
                case_name = "case2869pegase"
                print(f"Successfully loaded {case_name} (2869 buses).")
            except:
                try:
                    net = nw.case2000()
                    case_name = "case2000"
                    print(f"Successfully loaded {case_name} (2000 buses).")
                except:
                    try:
                        net = nw.case1354pegase()
                        case_name = "case1354pegase"
                        print(f"Successfully loaded {case_name} (1354 buses).")
                    except:
                        try:
                            net = nw.case1888rte()
                            case_name = "case1888rte"
                            print(f"Successfully loaded {case_name} (1888 buses).")
                        except:
                            # Fallback to case300
                            net = nw.case300()
                            case_name = "case300"
                            print(f"Loaded fallback {case_name} (300 buses).")
        except Exception as e:
            print(f"Error loading pandapower network: {e}")
            print("Please ensure 'pandapower' is installed correctly.")
            exit()

        # --- Get system data from pandapower network ---
        system_data = get_pandapower_data(net, case_name)

        # --- Run single experiment ---
        print("\n" + "="*90)
        print(f"EXPERIMENT: {case_name.upper()}")
        print("="*90)
        pdl, fig1, fig2, results = run_acopf_experiment(
            case_name=f"IEEE {case_name}",
            system_data=system_data,
            n_train=50000,  # Increased training samples for larger system
            n_test=200,     # Increased test samples
            max_outer_iters=100,  # More iterations for convergence
            convergence_threshold=1e-2,
            inner_iters=250
        )
        
        all_results[case_name] = results
        all_pdls[case_name] = pdl
        all_figs[case_name] = (fig1, fig2)

        # Print detailed results
        print("\n" + "="*110)
        print("DETAILED RESULTS")
        print("="*110)
        
        print(f"\nCase: {case_name}")
        print(f"  Number of Buses: {system_data['n_buses']}")
        print(f"  Number of Generators: {system_data['n_generators']}")
        print(f"  Number of Loads: {system_data['n_loads']}")
        print(f"  Base MVA: {results['baseMVA']}")
        print(f"  Slack Bus: {results['slack_bus']}")
        
        print(f"\nTraining Performance:")
        print(f"  Training Time: {results['training_time']:.1f} seconds")
        print(f"  Inference Time per Sample: {results['inference_time_per_sample']*1000:.4f} ms")
        
        print(f"\nTest Set Performance:")
        print(f"  Average Generation Cost: ${results['cost']:,.2f}")
        print(f"  Max P Violation: {results['max_P_viol_pu']:.8f} pu ({results['max_P_viol_pu']*results['baseMVA']:.4f} MW)")
        print(f"  Max Q Violation: {results['max_Q_viol_pu']:.8f} pu ({results['max_Q_viol_pu']*results['baseMVA']:.4f} MVAr)")
        print(f"  Max Overall Violation: {results['max_viol_pu']:.8f} pu")
        print(f"  Mean P Violation: {results['mean_P_viol_pu']:.8f} pu ({results['mean_P_viol_pu']*results['baseMVA']:.4f} MW)")
        print(f"  Mean Q Violation: {results['mean_Q_viol_pu']:.8f} pu ({results['mean_Q_viol_pu']*results['baseMVA']:.4f} MVAr)")
        
        print(f"\nVoltage Statistics (All Buses):")
        print(f"  Mean: {results['V_mean']:.4f} pu")
        print(f"  Std: {results['V_std']:.4f} pu")
        print(f"  Min: {results['V_min']:.4f} pu")
        print(f"  Max: {results['V_max']:.4f} pu")
        
        print(f"\nSlack Bus Voltage Statistics (Bus {results['slack_bus']}):")
        print(f"  Mean: {results['slack_V_mean']:.4f} pu")
        print(f"  Std: {results['slack_V_std']:.4f} pu")
        print(f"  Min: {results['slack_V_min']:.4f} pu")
        print(f"  Max: {results['slack_V_max']:.4f} pu")
        deviation = abs(results['slack_V_mean'] - 1.0)
        print(f"  Deviation from nominal (1.0 pu): {deviation:.4f} pu ({deviation*100:.2f}%)")
        
        print(f"\nTotal Generation: {results['Pg_total_mw']:.2f} MW (average)")
        
        print("="*110)

        # Final summary
        print("\n" + "="*90)
        print("SUMMARY")
        print("="*90)
        print(f"\nExperiment completed for {case_name}:")
        print(f"  System Size: {system_data['n_buses']} buses, {system_data['n_generators']} generators")
        print(f"  Training completed in {results['training_time']:.1f} seconds")
        print(f"  Maximum constraint violation: {results['max_viol_pu']:.6f} pu")
        print(f"  Slack bus voltage learned: {results['slack_V_mean']:.4f} ± {results['slack_V_std']:.4f} pu")
        print(f"  Average inference time: {results['inference_time_per_sample']*1000:.4f} ms/sample")
        
        print("\nKey Findings:")
        print("1. PDL successfully loaded large-scale real network data from pandapower")
        print("2. Model trained on p.u. values from a standard AC-OPF case")
        print("3. Constraint violations are minimized through primal-dual learning")
        print("4. Neural network learns to predict slack bus voltage dynamically")
        print(f"5. Inference is extremely fast (< {results['inference_time_per_sample']*1000:.2f} ms per instance)")
        
        print("\nNote: This is a simplified implementation. For exact paper replication:")
        print("- Implement all AC-OPF constraints (line limits, angle differences)")
        print("- The current model only enforces power balance and generator/voltage limits")
        print("- Use exact hyperparameters from paper")
        print("- Run multiple random seeds and report statistics")
        print("="*90)

        plt.show()

        print(f"\n✓ Experiment completed successfully for {case_name}!")