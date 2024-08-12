import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, dropout=0.5, num_filters=64):
        super(CharCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=27, out_channels=num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
        ).apply(self.init_weights)

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
        ).apply(self.init_weights)

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
        ).apply(self.init_weights)

        self.fc1 = nn.Sequential(
            nn.Linear(320, 256),
            nn.ELU(),
            nn.Dropout(dropout)
        ).apply(self.init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Dropout(dropout)
        ).apply(self.init_weights)

        self.output = nn.Linear(256, 1).apply(self.init_weights)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight) 
            if module.bias != None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.zeros_(module.bias)
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x).squeeze(1)
        x = self.sigmoid(x)
        return x

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
