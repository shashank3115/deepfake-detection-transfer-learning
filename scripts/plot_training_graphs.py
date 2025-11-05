import matplotlib.pyplot as plt

# Replace with your real logged metrics from train.py (from console output)
epochs = [1, 2, 3, 4, 5]
train_loss = [0.3726, 0.1654, 0.0962, 0.0824, 0.0431]
val_loss = [0.2309, 0.2580, 0.2154, 0.6079, 0.3119]
val_acc = [0.900, 0.890, 0.910, 0.810, 0.895]

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Train Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('outputs/loss_curve.png')
plt.show()

# Plot Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_acc, marker='o', color='green', label='Validation Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.0)
plt.legend()
plt.grid(True)
plt.savefig('outputs/accuracy_curve.png')
plt.show()

print("✅ Saved loss_curve.png and accuracy_curve.png in outputs/")
