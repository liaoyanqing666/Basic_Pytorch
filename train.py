import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GRUNetwork
from data import SimpleDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 0.0001
epochs = 20
batch_size = 1

train_dataset = SimpleDataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = SimpleDataset(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = GRUNetwork(input_size=10, linear_hidden_size=20, gru_hidden_size=30, num_layers=3, dropout=0.1).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losest_test_loss = float('inf')
for epoch in range(epochs):
    total_train_loss = 0
    total_test_loss = 0

    model.train()
    for train_x1, train_x2, labels in train_dataloader:
        train_x1, train_x2, labels = train_x1.to(device), train_x2.to(device), labels.to(device)

        train_outputs = model(train_x1, train_x2)
        trian_loss = criterion(train_outputs, labels)

        optimizer.zero_grad()
        trian_loss.backward()
        optimizer.step()

        total_train_loss += trian_loss.item() * train_x1.size(0)

    model.eval()
    with torch.no_grad():
        for test_x1, test_x2, labels in test_dataloader:
            test_x1, test_x2, labels = test_x1.to(device), test_x2.to(device), labels.to(device)

            test_outputs = model(test_x1, test_x2)
            test_loss = criterion(test_outputs, labels)

            total_test_loss += test_loss.item() * test_x1.size(0)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_train_loss/len(train_dataset)}, Test Loss: {total_test_loss/len(test_dataset)}, Lowest Test Loss: {losest_test_loss}')

    if total_test_loss < losest_test_loss:
        losest_test_loss = total_test_loss
        torch.save(model.state_dict(), 'simple_model.pth')
