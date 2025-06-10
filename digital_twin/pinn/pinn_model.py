import torch
import torch.nn as nn


class PINN_GasTurbine(nn.Module):
    def __init__(self, nn_arch, name="pinn_gas_turbine", **kwargs):
        super(PINN_GasTurbine, self).__init__(**kwargs)
        self.nn_arch = nn_arch

        layers_list = []
        input_size = nn_arch["num_inputs"]
        for neurons in nn_arch["hidden_units"]:
            layers_list.append(nn.Linear(input_size, neurons))
            if nn_arch["activation"] == "tanh":
                layers_list.append(nn.Tanh())
            elif nn_arch["activation"] == "relu":
                layers_list.append(nn.ReLU())
            # Add other activations if needed
            input_size = neurons

        layers_list.append(nn.Linear(input_size, nn_arch["num_outputs"]))
        self.network = nn.Sequential(*layers_list)

        self.lambda_data = torch.tensor(1.0, dtype=torch.float32)
        self.lambda_physics = torch.tensor(1.0, dtype=torch.float32)

        # Normalization parameters (means and stds for all inputs and outputs)
        self.input_means_tensor = None
        self.input_stds_tensor = None
        self.output_means_tensor = None
        self.output_stds_tensor = None

        # For hard constraints, defining an epsilon
        self.hard_constraint_epsilon = 1e-3

    def build_scaler(self, df_train_input, df_train_output):
        # Assuming df_train_input.columns and df_train_output.columns provide the correct order of features
        self.input_means_tensor = torch.tensor(
            [df_train_input[col].mean() for col in df_train_input.columns],
            dtype=torch.float32,
        )
        self.input_stds_tensor = torch.tensor(
            [df_train_input[col].std() for col in df_train_input.columns],
            dtype=torch.float32,
        )

        self.output_means_tensor = torch.tensor(
            [df_train_output[col].mean() for col in df_train_output.columns],
            dtype=torch.float32,
        )
        self.output_stds_tensor = torch.tensor(
            [df_train_output[col].std() for col in df_train_output.columns],
            dtype=torch.float32,
        )

    def forward(self, inputs_norm_concat):
        # inputs_norm_concat are the normalized inputs to the network
        raw_outputs_norm = self.network(inputs_norm_concat)

        # Example of a hard constraint: T2_denormalized > T1_denormalized + epsilon
        # This requires denormalizing parts of input and output, applying constraint, then re-normalizing.
        # Assumes:
        # - T1 is the 0th input feature.
        # - T2 is the 0th output feature.

        # Ensure scalers are built
        if self.input_means_tensor is None or self.output_means_tensor is None:
            # This case should ideally not happen if build_scaler is called before training/inference
            return raw_outputs_norm

        # Move tensors to the correct device
        current_device = inputs_norm_concat.device
        input_means_dev = self.input_means_tensor.to(current_device)
        input_stds_dev = self.input_stds_tensor.to(current_device)
        output_means_dev = self.output_means_tensor.to(current_device)
        output_stds_dev = self.output_stds_tensor.to(current_device)

        # Denormalize T1 from input
        t1_norm = inputs_norm_concat[:, 0:1]  # T1 is the first input feature
        t1_denorm = t1_norm * (input_stds_dev[0] + 1e-8) + input_means_dev[0]

        # Denormalize T2_raw from output
        t2_raw_norm = raw_outputs_norm[:, 0:1]  # T2 is the first output feature
        t2_raw_denorm = t2_raw_norm * (output_stds_dev[0] + 1e-8) + output_means_dev[0]

        # Apply hard constraint: T2_denorm = T1_denorm + ReLU(T2_raw_denorm - T1_denorm) + epsilon
        # This ensures T2_denorm >= T1_denorm + epsilon
        t2_enforced_denorm = (
            t1_denorm
            + torch.relu(t2_raw_denorm - t1_denorm)
            + self.hard_constraint_epsilon
        )

        # Re-normalize T2_enforced
        t2_enforced_norm = (t2_enforced_denorm - output_means_dev[0]) / (
            output_stds_dev[0] + 1e-8
        )

        # Create a clone of raw_outputs_norm to modify
        modified_outputs_norm = raw_outputs_norm.clone()
        modified_outputs_norm[:, 0:1] = t2_enforced_norm  # Replace T2 prediction

        return modified_outputs_norm

    def normalize_inputs(self, inputs_denorm_concat):
        if self.input_means_tensor is None:
            raise ValueError("Input scalers not built. Call build_scaler first.")
        means = self.input_means_tensor.to(inputs_denorm_concat.device)
        stds = self.input_stds_tensor.to(inputs_denorm_concat.device)
        return (inputs_denorm_concat - means) / (stds + 1e-8)

    def denormalize_outputs(self, outputs_norm_concat):
        if self.output_means_tensor is None:
            raise ValueError("Output scalers not built. Call build_scaler first.")
        means = self.output_means_tensor.to(outputs_norm_concat.device)
        stds = self.output_stds_tensor.to(outputs_norm_concat.device)
        return outputs_norm_concat * (stds + 1e-8) + means

    def physics_loss(
        self, inputs_norm_concat, outputs_denorm_concat, inputs_denorm_concat
    ):
        # Denormalize T1 and P1 from inputs_norm_concat for physics calculations
        # Assumes T1 is input feature 0, P1 is input feature 1
        t1_norm = inputs_norm_concat[:, 0:1]
        p1_norm = inputs_norm_concat[:, 1:2]

        # Move tensors to the correct device
        current_device = inputs_norm_concat.device
        input_means_dev = self.input_means_tensor.to(current_device)
        input_stds_dev = self.input_stds_tensor.to(current_device)

        t1_input_denorm = t1_norm * (input_stds_dev[0] + 1e-8) + input_means_dev[0]
        p1_input_denorm = p1_norm * (input_stds_dev[1] + 1e-8) + input_means_dev[1]

        # Get mdot_f from denormalized inputs (Fuel Flow is 3rd column, index 2)
        mdot_f_lph = inputs_denorm_concat[:, 2:3]
        # Convert L/hr to kg/s, assuming jet fuel density of ~0.8 kg/L
        mdot_f_kgs = mdot_f_lph * 0.8 / 3600

        # Split denormalized outputs
        # Order: T2, P2, T3, P3, T4, P4, T5, P5, Thrust, mdot_a
        (
            t2_pred,
            p2_pred,
            t3_pred,
            p3_pred,
            t4_pred,
            p4_pred,
            t5_pred,
            p5_pred,
            thrust_pred,
            mdot_a_pred,
        ) = torch.split(outputs_denorm_concat, 1, dim=1)

        # Convert to Kelvin and Pascals for thermodynamic equations
        t1_k = t1_input_denorm + 273.15
        t2_k_pred = t2_pred + 273.15
        t3_k_pred = t3_pred + 273.15
        t4_k_pred = t4_pred + 273.15
        t5_k_pred = t5_pred + 273.15

        p1_pa_input = p1_input_denorm * 1000
        p2_pa_pred = p2_pred * 1000
        p3_pa_pred = p3_pred * 1000
        p4_pa_pred = p4_pred * 1000
        p5_pa_pred = p5_pred * 1000

        # Physical constants
        eta_c = 0.85  # Compressor efficiency
        gamma_a = 1.4  # Specific heat ratio for air
        eta_t = 0.90  # Turbine efficiency
        gamma_g = 1.33  # Specific heat ratio for gas
        k_loss_comb = 0.97  # Pressure loss factor in combustor
        cp_a = 1005  # J/kgK
        cp_g = 1148  # J/kgK for gas
        q_hv = 43.1e6  # J/kg

        # Residual for Compressor Temperature-Pressure relationship
        pressure_ratio_comp = p2_pa_pred / (p1_pa_input + 1e-8)
        pressure_ratio_comp = torch.clamp(pressure_ratio_comp, min=1e-6)
        term_comp = torch.pow(pressure_ratio_comp, (gamma_a - 1) / gamma_a)
        res_T2_P2 = t2_k_pred - t1_k * (1 + (1 / eta_c) * (term_comp - 1))

        # Residual for Combustor Pressure Drop
        res_P3 = p3_pa_pred - k_loss_comb * p2_pa_pred

        # Residual for Turbine Temperature-Pressure relationship
        pressure_ratio_turb = p4_pa_pred / (p3_pa_pred + 1e-8)
        pressure_ratio_turb = torch.clamp(pressure_ratio_turb, min=1e-6, max=1.0)
        term_turb = torch.pow(pressure_ratio_turb, (gamma_g - 1) / gamma_g)
        res_T4_P4 = t4_k_pred - t3_k_pred * (1 - eta_t * (1 - term_turb))

        # Residual for Nozzle/Exhaust Temperature (assuming T5 approx T4)
        res_T5 = t5_k_pred - t4_k_pred

        # Power balance loss
        w_comp = mdot_a_pred * cp_a * (t2_k_pred - t1_k)
        w_turb = (mdot_a_pred + mdot_f_kgs) * cp_g * (t3_k_pred - t4_k_pred)
        res_power = w_comp - w_turb

        # Combustor energy balance loss
        q_fuel = mdot_f_kgs * q_hv
        q_added = (mdot_a_pred + mdot_f_kgs) * cp_g * (t3_k_pred - t2_k_pred)
        res_comb = q_fuel - q_added

        loss_T2_P2 = torch.mean(torch.square(res_T2_P2))
        loss_P3 = torch.mean(torch.square(res_P3))
        loss_T4_P4 = torch.mean(torch.square(res_T4_P4))
        loss_T5 = torch.mean(torch.square(res_T5))
        loss_power = torch.mean(torch.square(res_power))
        loss_comb = torch.mean(torch.square(res_comb))

        # Soft Constraint losses (penalize violations)
        # Note: T2 > T1 is now a hard constraint in forward(), so this soft constraint can be removed or kept as a small regularizer
        # loss_T2_gt_T1 = torch.mean(torch.square(torch.relu(t1_k - t2_k_pred))) # T2 should be > T1
        loss_P2_gt_P1 = torch.mean(
            torch.square(torch.relu(p1_pa_input - p2_pa_pred))
        )  # P2 should be > P1
        loss_T3_gt_T2 = torch.mean(
            torch.square(torch.relu(t2_k_pred - t3_k_pred))
        )  # T3 should be > T2
        loss_P3_lt_P2 = torch.mean(
            torch.square(torch.relu(p3_pa_pred - p2_pa_pred))
        )  # P3 should be < P2
        loss_T4_lt_T3 = torch.mean(
            torch.square(torch.relu(t4_k_pred - t3_k_pred))
        )  # T4 should be < T3
        loss_P4_lt_P3 = torch.mean(
            torch.square(torch.relu(p4_pa_pred - p3_pa_pred))
        )  # P4 should be < P3
        loss_P5_lt_P4 = torch.mean(
            torch.square(torch.relu(p5_pa_pred - p4_pa_pred))
        )  # P5 should be < P4

        total_physics_loss = (
            loss_T2_P2
            + loss_P3
            + loss_T4_P4
            + loss_T5
            + loss_power
            + loss_comb
            + loss_P2_gt_P1
            + loss_T3_gt_T2
            + loss_P3_lt_P2
            + loss_T4_lt_T3
            + loss_P4_lt_P3
            + loss_P5_lt_P4
        )

        return total_physics_loss
