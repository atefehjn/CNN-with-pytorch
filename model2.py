import torch
from torch import nn

class CNN(nn.Module):
    '''
    Convolutional Neural Network.
    '''
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(32,64,(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(64,16,(3,3)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.seq4 = nn.Sequential(
            nn.Linear(22*22*16,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = torch.flatten(x, 1)
        x = self.seq4(x)

        return x
