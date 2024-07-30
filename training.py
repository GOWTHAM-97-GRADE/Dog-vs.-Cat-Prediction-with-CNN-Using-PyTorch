import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from cnn_model import ConvNeuralNet
from dataset_creation import CustomDataset
from transforms import all_transforms

# DataLoader parameters
batch_size = 10
num_classes = 2
learning_rate = 0.001
num_epochs = 10

# Dataset and DataLoader
full_dataset = CustomDataset(root_dir='./train', annotation_file='train_csv.csv', transform=all_transforms)
train_size = int(0.7 * len(full_dataset))
test_size = int(0.2 * len(full_dataset))
validate_size = len(full_dataset) - train_size - test_size
train_dataset, test_dataset, validate_dataset = random_split(full_dataset, [train_size, test_size, validate_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True)

# Check a sample image
sample_image, _ = train_dataset[100]
plt.imshow(sample_image.permute(1, 2, 0).numpy())  # Convert tensor to numpy array for visualization
plt.show()

# Model, criterion, and optimizer
model = ConvNeuralNet(num_classes)
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader: 
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    acc /= count
    print(f"Epoch {epoch+1}: Test accuracy {acc*100:.4f}%")

# Validation accuracy
model.eval()
val_acc = 0
val_count = 0
with torch.no_grad():
    for inputs, labels in validate_loader:
        y_pred = model(inputs)
        val_acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        val_count += len(labels)
val_acc /= val_count
print(f"Validation accuracy {val_acc*100:.2f}%")

# Save model
torch.save(model.state_dict(), "CatvsDoggo.pth")
