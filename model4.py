import torch
from torch import nn

class CNN(nn.Module):
    '''
    Convolutional Neural Network.
    '''
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(1,32,(5,5)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(32,16,(5,5)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Linear(20*20*16,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = torch.flatten(x, 1)
        x = self.seq3(x)

        return x
