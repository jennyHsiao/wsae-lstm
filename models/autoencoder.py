import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable

# torch.manual_seed(1)    # reproducible

class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden=10, lr=0.001, weight_decay=0.0):#lr=0.0001):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden        
        self.weight_decay = weight_decay
        self.lr = lr
        self.build_model()
    # end constructor

    def build_model(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_hidden),
            torch.nn.ELU())
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_in),
            torch.nn.ELU())
        self.criterion = torch.nn.MSELoss()        
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x):
        
        y = self.encoder(x)

        if self.training:
            x_reconstruct = self.decoder(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.decoder(x)



    # def forward(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return encoded, decoded

    # def fit(self, X, n_epoch=10, batch_size=64, en_shuffle=True):
    #     for epoch in range(n_epoch):
    #         if en_shuffle:
    #             print("Data Shuffled")
    #             X = sklearn.utils.shuffle(X)
    #         for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
    #             inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
    #             outputs, sparsity_loss = self.forward(inputs)

    #             loss = self.loss(outputs, inputs)                
    #             self.optimizer.zero_grad()                             # clear gradients for this training step
    #             loss.backward()                                        # backpropagation, compute gradients
    #             self.optimizer.step()                                  # apply gradients
    #             if local_step % 50 == 0:
    #                 print ("Epoch %d/%d | Step %d/%d | train loss: %.4f | l1 loss: %.4f | sparsity loss: %.4f"
    #                        %(epoch+1, n_epoch, local_step, len(X)//batch_size,
    #                          loss.data[0], l1_loss.data[0], sparsity_loss.data[0]))

    # def gen_batch(self, arr, batch_size):
    #     for i in range(0, len(arr), batch_size):
    #         yield arr[i : i+batch_size]
