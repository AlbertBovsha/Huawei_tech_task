import argparse
from omegaconf import OmegaConf
import logging
import matplotlib.pyplot as plt

import os
from os.path import join
import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader,random_split

from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T

from models.resnets import ResNet_18
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd


def save_checkpoint(epoch, model, checkpoints_folder):
    model_out_path = os.path.join(checkpoints_folder, 'epoch_{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_out_path)
    logging.info('Checkpoint for %d epoch is saved at %s' % (epoch, model_out_path))

def init_experiment(experiments_root='experiments', prefix=''):
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = f'{prefix}_{time}' if prefix else time
    experiments_path = os.path.join(experiments_root, experiment_name)

    checkpoints_folder = os.path.join(experiments_path, 'checkpoints')
    os.makedirs(checkpoints_folder)

    logs_folder = os.path.join(experiments_path, 'logs') 
    os.makedirs(logs_folder)

    return checkpoints_folder, logs_folder

def set_logger(log_path):
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s: %(message)s')



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

    checkpoints_folder, logs_folder = init_experiment(os.path.join(base_path,config.output.experiments_root), 
                                                        config.output.experiment_name)

    
    # logging
    set_logger(os.path.join(logs_folder, 'log'))
    logging.info(config)
    logging.info('start experiment')
 

    y_hats = torch.load(os.path.join(base_path, config.data.y_hat))
    labels = torch.load(os.path.join(base_path, config.data.labels))
    dataset = TensorDataset(y_hats, labels)
    train_size = int(0.8 * len(dataset))  
    test_size = len(dataset) - train_size
    
    #Разделение датасета на тренировочное и тестовое подмножества
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #Инициализация двух DataLoader для тренировочного и тестового наборов
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    #создание модели
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(torch.cuda.is_available(), device)

    model = ResNet_18(image_channels=192, num_classes=config.train.num_classes).to(device)
    
    #оптимизатор и лосс функция
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

    scheduler = StepLR(optimizer, step_size=config.train.step_size, gamma=config.train.lr_gamma)
    
    TrainLoss=[]
    for epoch in tqdm(range(config.train.epochs)):
        totalTrainLoss = 0.0
        
        model.train(True)
        for (i, (x, y)) in tqdm(enumerate(train_dataloader)):
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalTrainLoss += loss
            if i%20==0:
                print("Loss/train",loss.item(),epoch)            
        logging.info('totaltrainloss = %s',str(totalTrainLoss.item()))
        logging.info('lr = %s', str(scheduler.get_last_lr()[0]))
        TrainLoss.append(totalTrainLoss.item())
        scheduler.step()
        
        model.train(False)
        if epoch % config.train.valid_every == 0:
            accuracy, auc = test_loop(test_dataloader, model, loss_fn, device)
            print(accuracy, auc)

            tuple_for_text = (epoch, config.train.epochs, str(accuracy))
            logging.info('validation %d/%d epoch: accuracy = %s' %  tuple_for_text)
            tuple_for_text = (epoch, config.train.epochs, str(auc))
            logging.info('validation %d/%d epoch: auc = %s' %  tuple_for_text)
        
        if epoch % config.train.save_every == 0:
            save_checkpoint(epoch, model, checkpoints_folder)
    
    fig,axes=plt.subplots(figsize=(15,25))
    axes.plot(TrainLoss)
    axes.set_xlabel('epoch')
    axes.set_ylabel('totalTrainLoss')
    plt.show()

    accuracy, auc = test_loop(test_dataloader, model, loss_fn, device)
    save_checkpoint(epoch, model, checkpoints_folder)
    
    
    
if __name__ == '__main__':
    main()