import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models.resnets import ResNet_18
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    labels = []
    preds = []
    with torch.no_grad():
        for x, y in dataloader:
            labels.append(y.item())
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)

            preds.append(nn.Softmax(dim=1)(pred).detach().cpu().numpy()[0])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    preds = np.squeeze(np.array(preds))
    accuracy = accuracy_score(labels, np.argmax(preds, axis=1))
    print('accuracy', accuracy)
    one_hot_labels = pd.get_dummies(labels).values
    auc = roc_auc_score(one_hot_labels, preds)
    print('auc', auc)
    return accuracy, auc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    print(args)
    config = OmegaConf.load(args.config)
    print(config)
    base_path = os.path.dirname(os.path.abspath(__file__))

    y_hats = torch.load(os.path.join(base_path,config.data.y_hat))
    labels = torch.load(os.path.join(base_path,config.data.labels))
    dataset = TensorDataset(y_hats, labels)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.is_available(), device)

    model = ResNet_18(image_channels=192, num_classes=config.test.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
 
    #load pretrained model
    pretrain_ckpt = config.model.get('pretrain', None)
    if not pretrain_ckpt is None:
        state_dict = torch.load(os.path.join(base_path, pretrain_ckpt), map_location=device)
        model.load_state_dict(state_dict)
        model.train(False)
        accuracy, auc = test_loop(test_dataloader, model, loss_fn, device)
    else:
        print('No pretrain model')
    
    
    
if __name__ == '__main__':
    main()