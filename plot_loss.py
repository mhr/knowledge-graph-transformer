# Kernel has been reset, re-import all necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import re

# Define file paths
file_transformer = "geoml_transformer_results.txt"
file_kg_transformer = "geoml_kgtransformer_results.txt"

# Re-define the function to parse training logs
def parse_epoch_loss_data(filepath):
    epochs, train_losses, val_losses = [], [], []
    with open(filepath, "r") as file:
        for line in file:
            match = re.search(r"Epoch (\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
    return np.array(epochs), np.array(train_losses), np.array(val_losses)

# Load data
epochs1, train_loss1, val_loss1 = parse_epoch_loss_data(file_transformer)
epochs2, train_loss2, val_loss2 = parse_epoch_loss_data(file_kg_transformer)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(epochs1, train_loss1, label="Transformer Train", color="navy", linestyle='-')
plt.plot(epochs1, val_loss1, label="Transformer Val", color="navy", linestyle='--')
plt.plot(epochs2, train_loss2, label="KG Transformer Train", color="crimson", linestyle='-')
plt.plot(epochs2, val_loss2, label="KG Transformer Val", color="crimson", linestyle='--')

# Reference line at ln(2)
ln2 = np.log(2)
plt.axhline(y=ln2, color='gray', linestyle=':', label='ln(2)')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()