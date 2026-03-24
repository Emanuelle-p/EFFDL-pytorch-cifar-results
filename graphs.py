import torch
import matplotlib.pyplot as plt

# 1. Load the saved data
checkpoint = torch.load('./checkpoint/training_log.pth')
history = checkpoint['history']

epochs = history['epoch']
train_loss = history['train_loss']
test_loss = history['test_loss']
train_acc = history['train_acc']
test_acc = history['test_acc']

# 2. Create the Epoch x Loss plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', color='blue')
plt.plot(epochs, test_loss, label='Test Loss', color='red')
plt.title('Loss Curve per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('epoch_vs_loss.png') # Save the plot as an image
plt.show()

# 3. Create the Epoch x Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
plt.plot(epochs, test_acc, label='Test Accuracy', color='orange')
plt.title('Accuracy Curve per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('epoch_vs_accuracy.png') # Save the plot as an image
plt.show()