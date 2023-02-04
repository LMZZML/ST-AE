from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from net import MultiHeadAttention_G, ae_gated_selfatt


class ae_mlp(nn.Module):
    def __init__(self, seq_len, num_nodes, kernel_list, dropout, act):
        super(ae_mlp, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.enc_mlps = nn.ModuleList()
        self.dec_mlps = nn.ModuleList()
        self.len = len(kernel_list)
        self.dropout = dropout
        self.act = act

        self.enc_mlps.append(nn.Linear(seq_len*num_nodes, kernel_list[0]))
        for i in range(len(kernel_list)-1):
            self.enc_mlps.append(nn.Linear(kernel_list[i], kernel_list[i+1]))
            self.dec_mlps.append(nn.Linear(kernel_list[-i-1], kernel_list[-i-2]))
        self.dec_mlps.append(nn.Linear(kernel_list[0], seq_len*num_nodes))

        for mlps in self.enc_mlps:
            nn.init.eye_(mlps.weight)
            nn.init.constant_(mlps.bias, 0.0)
        for mlps in self.dec_mlps:
            nn.init.eye_(mlps.weight)
            nn.init.constant_(mlps.bias, 0.0)

    def ae_mlp_enc(self, inputs):
        inputs = inputs[:, :1, :, :]
        W = inputs.flatten(1)
        # print("W: "+str(W.shape))
        for i in range(self.len):
            if self.act:
                # W = F.leaky_relu(self.enc_mlps[i](W))
                W = torch.tanh(self.enc_mlps[i](W))
            else:
                W = self.enc_mlps[i](W)
            # W = self.enc_mlps[i](W)
            if self.dropout != 0:
                W = F.dropout(W, self.dropout, self.training)
        return W

    def ae_mlp_dec(self, Y):
        """
            :param inputs: (batch_size, history_length, seq_num, 1)
            :return: (batch_size, history_length, seq_num, 1)
            """
        shape = [Y.shape[0], 1, self.num_nodes, self.seq_len]
        for i in range(self.len):
            if self.act:
                # Y = F.leaky_relu(self.dec_mlps[i](Y))
                Y = torch.tanh(self.dec_mlps[i](Y))
            else:
                Y = self.dec_mlps[i](Y)
            if self.dropout != 0:
                Y = F.dropout(Y, self.dropout, self.training)
        output = torch.reshape(Y, shape)
        return output

    def forward(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)
        :return: (batch_size, 1, node_num, predict_length)
        """
        Y = self.ae_mlp_enc(inputs)
        output = self.ae_mlp_dec(Y)
        # print("output: ", output.shape)
        return output


class mlp_predict(nn.Module):
    def __init__(self, train_len=12, num_nodes=207, horizon=12, kernel_list=None, hid_dim=2, dropout=0.01, act=False, fixed=False,
                 att_heads=None, load_model=False, model_path=None, device='cuda:0'):
        super(mlp_predict, self).__init__()
        self.train_len = train_len
        self.horizon = horizon
        self.hid_dim = hid_dim

        self.ae_mlp = ae_mlp(seq_len=train_len, num_nodes=num_nodes, kernel_list=kernel_list, dropout=dropout, act=act)
        if fixed is True:
            self.ae_mlp.requires_grad_(False)

        if load_model:
            checkpoint_train = torch.load(model_path, map_location=device)
            model_dict_enc = self.ae_mlp.state_dict()
            checkpoint_enc = {k: v for k, v in checkpoint_train.items() if k in model_dict_enc.keys()}
            model_dict_enc.update(checkpoint_enc)
            print(model_dict_enc.keys())
            self.ae_mlp.load_state_dict(model_dict_enc)

        self.attention = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.att_heads = att_heads
        if att_heads is not None:
            for i in att_heads:
                # self.attention.append(SelfAttention(num_attention_heads=i, input_size=num_nodes*self.hid_dim, hidden_size=num_nodes*self.hid_dim, hidden_dropout_prob=0.3))
                self.attention.append(MultiHeadAttention_G(model_dim=num_nodes*self.hid_dim, hid_dim=num_nodes*self.hid_dim, num_heads=i, dropout=0.3))
                # self.attention.append(ae_gated_selfatt(num_node=num_nodes, seq_len=hid_dim, num_heads=i, dropout=0.1))
        else:
            for i in range(3):
                self.fc.append(nn.Linear(num_nodes*self.hid_dim, num_nodes*self.hid_dim))
                nn.init.eye_(self.fc[i].weight)
                nn.init.constant_(self.fc[i].bias, 0.0)

    def forward(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)
        :return: (batch_size, predict_length, node_num, 1)
        """
        history_hidden = self.ae_mlp.ae_mlp_enc(inputs).unsqueeze(1)
        if self.att_heads is not None:
            for i in range(len(self.att_heads)):
                # history_hidden = self.attention[i](history_hidden)
                history_hidden = self.attention[i](history_hidden, history_hidden, history_hidden)
        else:
            shape = history_hidden.shape
            history_hidden = history_hidden.flatten(1)
            for i in range(3):
                history_hidden = torch.tanh(self.fc[i](history_hidden))
                history_hidden = F.dropout(history_hidden, 0.1, self.training)
            history_hidden = torch.reshape(history_hidden, shape)
        predict_output = self.ae_mlp.ae_mlp_dec(history_hidden.squeeze(1))

        return predict_output



