import torch.optim as optim
import math
import util
import torch
import numpy as np


class Trainer_D():
    def __init__(self, model, lrate, lrate2, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        ignored_params = list(map(id, model.stae.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        self.optimizer = optim.Adam([{'params': base_params},
                                     {'params': model.stae.parameters(), 'lr': lrate2}], lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, eps=1e-8, verbose=True)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level +=1
        if self.cl:
            loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = self.loss(predict, real, 0.0)
        # loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        self.iter += 1
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse


class AETrainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, input):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = input[:, :1, :, :]
        predict = self.scaler.inverse_transform(output)
        real = self.scaler.inverse_transform(real)
        #print('--------------predict-------------------')
        #print(predict)
        #print('--------------real-------------------')
        #print(real)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        self.iter += 1
        return loss.item(), mape, rmse

    def eval(self, input):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        # output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = input[:, :1, :, :]
        predict = self.scaler.inverse_transform(output)
        real = self.scaler.inverse_transform(real)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse


class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
