import torch
import matplotlib.pyplot as plt
import os


history_path = "./logs/training_history.pt"

if not os.path.exists(history_path):
    raise FileNotFoundError(
        f"No training history found at {history_path}. Train until at least step 100 first."
    )

history = torch.load(history_path, map_location="cpu")

train_steps = history["train_steps"]
train_losses = history["train_losses"]

validation_steps = history["validation_steps"]
validation_losses = history["validation_losses"]

if not train_steps or not validation_steps:
    raise ValueError("Training history exists, but it does not contain any saved losses yet.")


plt.figure(figsize=(10, 6))

plt.plot(train_steps, train_losses,
         label="Train Loss",
         linewidth=2)

plt.plot(validation_steps, validation_losses,
         label="Validation Loss",
         linewidth=2)

plt.xlabel("Training Step")
plt.ylabel("Cross Entropy Loss")
plt.title("Lumiere Training")
plt.grid(True)
plt.legend()

plt.show()
