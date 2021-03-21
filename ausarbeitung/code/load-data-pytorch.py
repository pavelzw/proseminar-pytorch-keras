import torch
from torchvision import datasets, transforms

train_set = datasets.MNIST(root="data", train=True, download=True,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), torch.flatten]))
test_set = datasets.MNIST(root="data", train=False, download=True,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), torch.flatten]))

batch_size = 50

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                           shuffle=False)
