import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torchviz import make_dot
import random
import numpy as np
from TrainVal import train,validate
# from model1 import CNN
# from model2 import CNN
# from model3 import CNN
# from model4 import CNN
from model5 import CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Validation dataset
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = CNN()
model.to(device)
print(model)
summary(model,(1,28,28))
example_input = torch.randn(1, 1, 28, 28).to(device)
from fvcore.nn import flop_count_table,FlopCountAnalysis
flops = FlopCountAnalysis(model,example_input)
flops.total()
print(flop_count_table(flops))

num_epochs = 10
learning_rate = 0.001
best_acc = 0
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    train_acc , train_loss = train(model, train_loader, learning_rate,criterion,optimizer)
    val_acc , val_loss = validate(model, val_loader,learning_rate,criterion,optimizer)
    if val_acc>best_acc:
       torch.save(model, 'best-model.pt')
       torch.save(model.state_dict(), 'best-model-parameters.pt')
       best_acc = val_acc
       print(f"best model is in epoch {epoch+1}")
