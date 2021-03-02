import torch
from torchvision import datasets, transforms

train_set = datasets.MNIST("data", train=True, download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))
test_set = datasets.MNIST("data", train=False, download=True,
                    transform=transforms.Compose([transforms.ToTensor()]))

batch_size = 50

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                           shuffle=False)
