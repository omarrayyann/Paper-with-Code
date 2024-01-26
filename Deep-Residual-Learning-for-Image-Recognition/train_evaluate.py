import numpy as np
import pandas as pd
import torch

def evaluate(model,data_loader,device):
  total = 0
  correct = 0
  with torch.no_grad():
    model.eval()
    for inputs, labels, in data_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      total += len(inputs)
      correct += (torch.max(outputs,1)[1] == labels).sum().item()
    error = (total-correct)/total
    print(f'Error: {error}%')
  return error

def train(model, epochs, train_loader, test_loader, criterion, optimizer, RESULTS_PATH, MODEL_PATH):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.train()
  running_loss = 0.0
  cols       = ['epoch', 'train_loss', 'train_err', 'test_err']
  results_df = pd.DataFrame(columns=cols).set_index('epoch')
  best_test_err = 1.0
  print('Epoch \tBatch \tNLLLoss_Train')
  for epoch in range(epochs):
    model.train()
    for inputs,labels, in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    model.eval()
    train_err = evaluate(model,train_loader,device)
    test_err  = evaluate(model,test_loader,device)
    results_df.loc[epoch] = [running_loss/len(train_loader), train_err, test_err]
    results_df.to_csv(RESULTS_PATH)
    print(f'train_err: {train_err} test_err: {test_err}')
    if MODEL_PATH and (test_err < best_test_err):
      torch.save(model.state_dict(), MODEL_PATH)
      best_test_err = test_err
  print('Finished Training')
  return model


      