import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,train_loader,learning_rate,criterion,optimizer):
    train_loss = 0.0
    total_correct=0
    model.train()
    start_time = time.time()
    for batch_idx, (x, label) in enumerate(train_loader):
          x, label = x.to(device) ,label.to(device)
          optimizer.zero_grad()
          out = model(x)
          loss = criterion(out, label)
          loss.backward()
          optimizer.step()
          predicted_label = torch.max(out, 1)[1]  # Use [1] to get the indices
          correct = (predicted_label == label).sum().item()
          total = label.size(0)
          train_loss += loss.item()
    end_time = time.time()
    accuracy = correct / total
    avg_train_loss = train_loss/len(train_loader)
    train_time = end_time - start_time
    print('train accuracy:',accuracy," | train loss: ", avg_train_loss, " | train time", train_time)
    return accuracy,avg_train_loss

def validate(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(val_loader):
            x, label = x.to(device), label.to(device)
            out = model(x)
            loss = criterion(out, label)
            val_loss += loss.item()

            # Calculate accuracy
            predicted_label = torch.max(out, 1)[1]  # Use [1] to get the indices
            correct = (predicted_label == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)
    end_time = time.time()
    # Calculate average loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    accuracy = total_correct / total_samples
    validation_time = end_time - start_time
    print('Validation accuracy:',accuracy," | validation loss: ", avg_val_loss, " | validation time", validation_time)
    return accuracy,avg_val_loss
