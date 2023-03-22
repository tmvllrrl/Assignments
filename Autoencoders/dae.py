import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__(encoder, decoder)
        
        self.encoder = encoder
        self.decoder = decoder
    
    # In PyTorch Lightning, you define the behavior for each training step by overriding this method
    def training_step(self, batch, batch_idx):
        x, y = batch # x is the image batch, and y is the label batch. Is y necessary to use?
        
        z = self.encoder(x) # z is the latent representation discussed in the lecture
        x_hat = self.decoder(z) # x_hat is the reconstructed image
        
        loss = F.mse_loss(x_hat, x)
        
        self.log("loss", loss) # This is used later for saving the best model
        return loss
    
    # This results in the epoch loss being printed out at the end of an epoch. This is a utility func from PyTorch Lightning
    def training_epoch_end(self, outputs):
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        print(f"\nEPOCH TRAIN LOSS: {loss}")

    # In Keras, the optimizer is given in the .compile() function; however, in PyTorch Lightning, the optimizer is defined here
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train_dae(dae, train_loader):
    # Defines what model checkpoint to save during training
    dae_ckpt_callback = ModelCheckpoint(
        save_top_k=1, # For this exercise, we only save the best 1 model during training
        monitor="loss",
        mode="min",
        filename="best-dae"
    )

    # Defining various aspects about the training such as
    trainer = pl.Trainer(
        max_epochs=50, # Only doing 5 epochs as you get the jist pretty quickly      
        deterministic=True, # This flag just ensures reproducibility
        callbacks=dae_ckpt_callback,
        log_every_n_steps=1
    ) 

    # Defining
    trainer.fit(
        model=dae, 
        train_dataloaders=train_loader,
    )