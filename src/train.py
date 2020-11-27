import torch
from model.resnet import *
import numpy as np
from tqdm import tqdm
import os
import sys
import importlib
import shutil
import json
from data_prep.dataLoader import *
import importlib
from pathlib import Path
from model import pointcloudnet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))


def main():

    # Default parameters 
    batch_size = 24
    epochs = 80
    learning_rate = 0.0001 # 10^-5
    decay_rate = 1e-4
    resnet = resnet18(pretrained=True)

    # Hyper Parameters 
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    TRAIN_DATASET = dataLoader()
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=False, num_workers=4)
    #MODEL = importlib.import_module(pointcloudnet)

    network_model = pointcloudnet.pointcloudnet(layers=[2, 2, 2, 2, 2, 2])
    loss_function = pointcloudnet.get_loss()

    optimizer = torch.optim.Adam(
        network_model.parameters(),
        lr = learning_rate,
        betas = (0.9, 0.999),
        eps = 1e-08,
        weight_decay = decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    # Training 
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
            inputPtCld, transformVec, targetCld = data
            print(something)
            



if __name__ == "__main__":
    main()


