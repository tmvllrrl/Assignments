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

class DenoisingAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        noise_x = torch.tensor(x) # Making a copy of x and labelling it noise_x
        
        for i, ind_x in enumerate(x):
            noise_x[i] = add_noise(ind_x) # Using add_noise function to create and add noisy images to batch
        
        z = self.encoder(noise_x) # z is the latent representation discussed in the lecture
        x_hat = self.decoder(z) # x_hat is the reconstructed image
        
        loss = F.mse_loss(x_hat, x)
        
        self.log("loss", loss)
        return loss
    
    def training_epoch_end(self, outputs):
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        print(f"\nEPOCH TRAIN LOSS: {loss}")

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