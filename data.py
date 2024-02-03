import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # x1:[L-1, D], x2:[L-1, D], label:[L-1, 2]
        sample = self.data[index]
        x1 = torch.tensor(sample['x1'], dtype=torch.float32) 
        x2 = torch.tensor(sample['x2'], dtype=torch.float32) 
        label = torch.tensor(sample['label'], dtype=torch.float32) 

        return x1, x2, label

if __name__ == "__main__":
    # Example usage with dummy data
    dummy_data = [
        {'x1': [[1.0]], 'x2': [[1.0]], 'label': [[1, 0]},
        # Add more samples as needed
    ]

    dataset = SimpleDataset(dummy_data)
    print(len(dataset))
    print(dataset[0])
