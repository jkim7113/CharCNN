import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from model import CharCNN, EarlyStopper
import preprocess

#torch.manual_seed(123) 
torch.backends.cudnn.deterministic = True # for reproducibility
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WordDataset(Dataset):
    def __init__(self, words, levels):
        self.words = words
        self.levels = levels
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, i):
        return self.words[i], self.levels[i]
    
learning_rate = 1e-3
num_epochs = 300
batch_size = 512
val_split = 0.2

filepath = "./data/I159729-refined.csv"
save_path = "./model/charCNN.pt"

# Load training and validation data
x, y = preprocess.load_data(filepath)
x = preprocess.encode_data(x)

print(x.shape)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split)

train_dataset = WordDataset(x, y)
val_dataset = WordDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = CharCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

early_stop = EarlyStopper()
train_loss = []
val_loss = []

for epoch in range(num_epochs):
    model.train()
    step_loss = []
    val_step_loss = []

    # Train model on training data 
    for words, levels in train_loader:
        # Forward pass
        words = words.to(device).float()
        levels = levels.to(device)

        outputs = model(words)
        loss = criterion(outputs, levels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_loss.append(loss.item())
    
    train_loss.append(np.array(step_loss).mean())

    # Evaluate model performance on validation datas
    model.eval()
    with torch.no_grad():
        for words, levels in val_loader:
            words = words.to(device).float()
            levels = levels.to(device)

            outputs = model(words)
            loss = criterion(outputs, levels)
            val_step_loss.append(loss.item())

    val_loss.append(np.array(val_step_loss).mean())
    if early_stop(train_loss[-1]):     
        print(f'Early stopped at Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss[-1]:.5f}, Validation Loss: {val_loss[-1]:.5f}')     
        num_epochs = epoch + 1
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss[-1]:.5f}, Validation Loss: {val_loss[-1]:.5f}')

epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_loss, "green", label="Training loss")
plt.plot(epochs, val_loss, "blue", label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Training complete. Saving model...")
torch.save(model, save_path)




