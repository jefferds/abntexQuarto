import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from pinn_model import PINN_GasTurbine

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = {
    "Time (sec)": [
        16459.861,
        16460.076,
        16487.754,
        16487.955,
        16488.174,
        16488.300,
        16488.500,
        16488.700,
    ],
    "T1 (C)": [23.940, 23.943, 24.135, 24.144, 24.142, 24.140, 24.130, 24.120],
    "T2 (C)": [
        105.142,
        105.160,
        104.511,
        104.498,
        104.479,
        104.450,
        104.400,
        104.350,
    ],
    "T3 (C)": [
        568.238,
        568.534,
        568.616,
        568.492,
        568.953,
        569.000,
        569.100,
        569.200,
    ],
    "T4 (C)": [
        563.108,
        563.129,
        562.486,
        562.507,
        562.535,
        562.550,
        562.600,
        562.650,
    ],
    "T5 (C)": [
        573.720,
        573.632,
        573.565,
        573.660,
        573.781,
        573.800,
        573.850,
        573.900,
    ],
    "P1 (kPa)": [0.384, 0.372, 0.396, 0.395, 0.400, 0.401, 0.402, 0.403],
    "P2 (kPa)": [56.987, 56.956, 56.984, 56.987, 56.989, 57.000, 57.010, 57.020],
    "P3 (kPa)": [56.680, 56.642, 56.700, 56.627, 56.668, 56.670, 56.680, 56.690],
    "P4 (kPa)": [6.195, 6.184, 6.254, 6.200, 6.240, 6.250, 6.260, 6.270],
    "P5 (kPa)": [0.507, 0.501, 0.499, 0.498, 0.514, 0.515, 0.516, 0.517],
    "Fuel Flow (L/hr)": [
        13.631,
        13.631,
        13.621,
        13.628,
        13.640,
        13.642,
        13.643,
        13.644,
    ],
    "N1 (RPM)": [
        50093.726,
        50096.756,
        50079.372,
        50080.917,
        50080.363,
        50080.000,
        50079.500,
        50079.000,
    ],
    "Thrust (N)": [0.993, np.nan, 1.438, 1.624, 2.589, 2.600, 2.610, 2.620],
}
df = pd.DataFrame(data)
df.dropna(subset=["Thrust (N)"], inplace=True)

input_features = ["T1 (C)", "P1 (kPa)", "Fuel Flow (L/hr)", "N1 (RPM)"]
output_features = [
    "T2 (C)",
    "P2 (kPa)",
    "T3 (C)",
    "P3 (kPa)",
    "T4 (C)",
    "P4 (kPa)",
    "T5 (C)",
    "P5 (kPa)",
    "Thrust (N)",
]

df_input = df[input_features]
df_output = df[output_features]

num_train_samples = int(len(df) * 0.8)
if num_train_samples == 0 and len(df) > 0:
    num_train_samples = 1
elif len(df) == 0:
    print("DataFrame is empty. Cannot proceed with training.")
    exit()

df_train_input = df_input[:num_train_samples]
df_train_output = df_output[:num_train_samples]
df_test_input = df_input[num_train_samples:]
df_test_output = df_output[num_train_samples:]

X_train_pd = df_train_input
y_train_pd = df_train_output

X_train = torch.tensor(df_train_input.values, dtype=torch.float32).to(device)
y_train = torch.tensor(df_train_output.values, dtype=torch.float32).to(device)
X_test = torch.tensor(df_test_input.values, dtype=torch.float32).to(device)
y_test = torch.tensor(df_test_output.values, dtype=torch.float32).to(device)

nn_config = {
    "hidden_units": [128, 128, 128],
    "activation": "tanh",
    "num_outputs": len(output_features),
    "num_inputs": len(input_features),
}

pinn_model = PINN_GasTurbine(nn_arch=nn_config).to(device)
pinn_model.build_scaler(X_train_pd, y_train_pd)

pinn_model.lambda_data = torch.tensor(1.0, dtype=torch.float32).to(device)
pinn_model.lambda_physics = torch.tensor(0.1, dtype=torch.float32).to(
    device
)  # Tune this

optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
criterion_data = nn.MSELoss()  # For data loss

print("Starting training...")
if X_train.shape[0] > 0:
    epochs = 100  # Increased epochs for better convergence demonstration
    batch_size = 2
    if X_train.shape[0] < batch_size:  # Adjust batch size if dataset is too small
        batch_size = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_phys_loss = 0

        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i : i + batch_size]
            batch_x, batch_y_true = X_train[indices], y_train[indices]

            # Normalize inputs for the model
            T1_batch, P1_batch, FF_batch, N1_batch = torch.split(batch_x, 1, dim=1)
            batch_x_norm = pinn_model.normalize_inputs(
                T1_batch, P1_batch, FF_batch, N1_batch
            )

            # Model prediction (normalized)
            batch_y_pred_norm = pinn_model(batch_x_norm)

            # Denormalize predictions for physics loss and comparison with true (denormalized) y
            batch_y_pred_denorm = pinn_model.denormalize_outputs(batch_y_pred_norm)

            # Data Loss (comparing denormalized prediction with denormalized true y)
            # Or, normalize true y and compare with normalized prediction
            batch_y_true_norm = (batch_y_true - pinn_model.output_means.to(device)) / (
                pinn_model.output_stds.to(device) + 1e-8
            )
            data_loss = criterion_data(batch_y_pred_norm, batch_y_true_norm)

            # Physics Loss
            phys_loss = pinn_model.physics_loss(batch_x_norm, batch_y_pred_denorm)

            total_loss = (
                pinn_model.lambda_data * data_loss
                + pinn_model.lambda_physics * phys_loss
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * batch_x.size(0)  # Weighted by batch size
            epoch_data_loss += data_loss.item() * batch_x.size(0)
            epoch_phys_loss += phys_loss.item() * batch_x.size(0)

        epoch_loss /= X_train.size(0)
        epoch_data_loss /= X_train.size(0)
        epoch_phys_loss /= X_train.size(0)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Total Loss: {epoch_loss:.4f}, Data Loss: {epoch_data_loss:.4f}, Physics Loss: {epoch_phys_loss:.4f}"
            )
    print("Training finished.")

    if X_test.shape[0] > 0:
        print("\nEvaluating on test data...")
        pinn_model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients for evaluation
            T1_test, P1_test, FF_test, N1_test = torch.split(X_test, 1, dim=1)
            X_test_norm = pinn_model.normalize_inputs(
                T1_test, P1_test, FF_test, N1_test
            )

            y_pred_norm_test = pinn_model(X_test_norm)
            y_pred_denorm_test = pinn_model.denormalize_outputs(y_pred_norm_test)

            y_test_norm = (y_test - pinn_model.output_means.to(device)) / (
                pinn_model.output_stds.to(device) + 1e-8
            )
            test_data_loss = criterion_data(y_pred_norm_test, y_test_norm)
            test_phys_loss = pinn_model.physics_loss(X_test_norm, y_pred_denorm_test)
            test_total_loss = (
                pinn_model.lambda_data * test_data_loss
                + pinn_model.lambda_physics * test_phys_loss
            )

            print(
                f"Test Total Loss: {test_total_loss.item():.4f}, Test Data Loss: {test_data_loss.item():.4f}, Test Physics Loss: {test_phys_loss.item():.4f}"
            )

            print("\nSample predictions (denormalized):")
            for i in range(min(5, len(y_pred_denorm_test))):
                print(
                    f"Input: {X_test[i].cpu().numpy()}, Predicted: {y_pred_denorm_test[i].cpu().numpy()}, Actual: {y_test[i].cpu().numpy()}"
                )
    else:
        print("No test data to evaluate or predict.")

    # Example of predicting with new, unseen data
    # new_data_point_np = np.array([[24.0, 0.39, 13.6, 50000]], dtype=np.float32)
    # new_data_point = torch.tensor(new_data_point_np, dtype=torch.float32).to(device)
    # T1_new, P1_new, FF_new, N1_new = torch.split(new_data_point, 1, dim=1)
    # new_data_point_norm = pinn_model.normalize_inputs(T1_new, P1_new, FF_new, N1_new)
    # pinn_model.eval()
    # with torch.no_grad():
    #     new_prediction_norm = pinn_model(new_data_point_norm)
    #     new_prediction_denorm = pinn_model.denormalize_outputs(new_prediction_norm)
    # print("\nPrediction for new data point (denormalized):", new_prediction_denorm.cpu().numpy())

else:
    print("No training data available after processing.")
