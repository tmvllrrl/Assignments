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

seed_everything(42, workers=True)

class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        z, mu, var = self.encoder(x) # Notice that you now have 3 variables being returned here
        x_hat = self.decoder(z) # x_hat is the reconstructed image
        
        kl_div = 0.5 * torch.sum(-1 - var + mu.pow(2) + var.exp())
        loss = F.binary_cross_entropy(x_hat, x) + kl_div
        
        self.log("loss", loss)
        return loss
    
    def training_epoch_end(self, outputs):
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        print(f"\nEPOCH TRAIN LOSS: {loss}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train_vae(vae, train_loader):
    vae_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss",
        mode="min",
        filename="vae_best"
    )
   
    trainer = pl.Trainer(
        max_epochs=20,     
        deterministic=True,
        callbacks=vae_ckpt_callback,
        log_every_n_steps=1
    ) 

     # train model
    trainer.fit(
        model=vae, 
        train_dataloaders=train_loader,
    )