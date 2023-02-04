from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from net import gated_selfatt_T


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class final_ae_stcnn(nn.Module):
    def __init__(self, num_nodes, seq_len, node_dim, blocks=4, layers=2, kernel_size=2, dropout=0.5,
                 channels=32, out_dim=2, device='cuda:0'):
        super(final_ae_stcnn, self).__init__()
        self.num_nodes = num_nodes
        self.blocks = blocks
        self.layers = layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.device = device

        self.filter_convs = nn.ModuleList()
        self.filter_convs_trans = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gate_convs_trans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.residual_convs_trans = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.skip_convs_trans = nn.ModuleList()
        self.res_map = nn.ModuleList()
        self.res_map_trans = nn.ModuleList()
        self.skip_map = nn.ModuleList()
        self.skip_map_trans = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.bn_trans = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv_trans = nn.ModuleList()

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_dim).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec1_trans = nn.Parameter(torch.randn(num_nodes, node_dim).to(device), requires_grad=True).to(device)
        self.nodevec2_trans = nn.Parameter(torch.randn(node_dim, num_nodes).to(device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.start_conv_trans = nn.ConvTranspose2d(channels, 1, kernel_size=(1, 1))

        receptive_field = 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size),dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.filter_convs_trans.append(nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs_trans.append(nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))
                self.residual_convs_trans.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))  # focus on this "trans"
                self.skip_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))
                self.skip_convs_trans.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))
                self.res_map.append(nn.Linear(in_features=receptive_field+kernel_size, out_features=receptive_field+new_dilation-1))
                self.res_map_trans.append(nn.Linear(in_features=receptive_field+new_dilation-1, out_features=receptive_field+kernel_size))
                self.skip_map.append(nn.Linear(in_features=receptive_field+kernel_size, out_features=receptive_field+new_dilation-1))
                self.skip_map_trans.append(nn.Linear(in_features=receptive_field+new_dilation-1, out_features=receptive_field+kernel_size))
                self.bn.append(nn.BatchNorm2d(channels))
                self.bn_trans.append(nn.BatchNorm2d(channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(gcn(channels, channels, dropout, support_len=1))
                self.gconv_trans.append(gcn(channels, channels, dropout, support_len=1))

        self.end_conv = nn.Conv2d(in_channels=channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.end_conv_trans = nn.ConvTranspose2d(in_channels=out_dim, out_channels=channels, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field


    def encoder(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)
        :return: (batch_size, 1, node_num, predict_length)
        """
        inputs = inputs[:, :1, :, :]

        support = [torch.eye(self.num_nodes, self.num_nodes).to(self.device)+F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)]

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)
        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = self.skip_map[self.blocks * self.layers-i-1](skip)
            except:
                skip = 0
            skip = s + skip

            # graph conv
            x = self.gconv[i](x, support)
            # residual connect
            x = x + self.res_map[self.blocks * self.layers-i-1](residual)

            x = self.bn[i](x)

        x = skip
        x = self.end_conv(x)

        return x

    def decoder(self, inputs):
        support_trans = [torch.eye(self.num_nodes, self.num_nodes).to(self.device)+F.softmax(F.relu(torch.mm(self.nodevec1_trans, self.nodevec2_trans)), dim=1)]

        y = inputs
        y = self.end_conv_trans(y)
        skip = 0
        for i in reversed(range(self.blocks * self.layers)):
            residual = y
            # dilated convolution
            filter = self.filter_convs_trans[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs_trans[i](residual)
            gate = torch.sigmoid(gate)
            y = filter * gate

            # parametrized skip connection
            s = y
            s = self.skip_convs_trans[i](s)
            try:
                skip = self.skip_map_trans[self.blocks * self.layers-i-1](skip)
            except:
                skip = 0
            skip = s + skip

            y = self.gconv_trans[i](y, support_trans)

            # residual connect (trans)
            pad_r = self.res_map_trans[self.blocks * self.layers-i-1](residual)
            y = y + pad_r

            y = self.bn_trans[i](y)

        y = self.start_conv_trans(skip)

        output = y
        out_len = y.size(3)
        if out_len > self.seq_len:
            output = output[:, :, :, -self.seq_len:]

        return output

    def forward(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)
        :return: (batch_size, 1, node_num, predict_length)
        """
        hidden_state = self.encoder(inputs)
        output = self.decoder(hidden_state)
        return output


class stae_predict(nn.Module):
    def __init__(self, train_len=12, num_nodes=207, node_dim=10, horizon=12, device='cuda:0', dropout=0.3,
                 blocks=4, layers=2, kernel_size=2, channels=32, out_dim=2, att_heads=None, load_model=False,
                 model_path=None, fixed=False, map_func='att'):
        super(stae_predict, self).__init__()
        self.train_len = train_len
        self.horizon = horizon
        self.map_func = map_func
        self.dropout = dropout
        self.fc_layers = att_heads

        self.stae = final_ae_stcnn(num_nodes=num_nodes, seq_len=train_len, node_dim=node_dim,
                             blocks=blocks, layers=layers, kernel_size=kernel_size, dropout=dropout,
                             channels=channels, out_dim=out_dim, device=device).to(device)
        if fixed:
            print('------The autoencoder is fixed!--------')
            self.stae.requires_grad_(False)

        if load_model:
            checkpoint_train = torch.load(model_path, map_location=device)
            model_dict_enc = self.stae.state_dict()
            checkpoint_enc = {k: v for k, v in checkpoint_train.items() if k in model_dict_enc.keys()}
            model_dict_enc.update(checkpoint_enc)
            print(model_dict_enc.keys())
            self.stae.load_state_dict(model_dict_enc)

        self.attention = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.att_heads = att_heads
        if self.map_func == 'att':
            for i in att_heads:
                self.attention.append(gated_selfatt_T(num_node=num_nodes, seq_len=out_dim, num_heads=i, dropout=dropout))
        elif self.map_func == 'fc':
            for i in range(self.fc_layers):
                self.fc.append(nn.Linear((int)(num_nodes * out_dim / 2**i), (int)(num_nodes * out_dim / 2**(i+1))))
            for i in reversed(range(self.fc_layers)):
                self.fc.append(nn.Linear((int)(num_nodes * out_dim / 2**(i+1)), (int)(num_nodes * out_dim / 2**i)))
            for i in range(2*self.fc_layers):
                print("w_size: ", self.fc[i].weight.shape)
                nn.init.eye_(self.fc[i].weight)
                nn.init.constant_(self.fc[i].bias, 0.0)


    def forward(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)
        :return: (batch_size, predict_length, node_num, 1)
        """
        history_hidden = self.stae.encoder(inputs)       # [batch_size, out_dim, node_num, 1]

        if self.map_func == 'att':
            predict_hidden = history_hidden.transpose(1,3)
            for i in range(len(self.att_heads)):
                predict_hidden = self.attention[i](predict_hidden)
            predict_hidden = predict_hidden.transpose(1,3)
            # res
            predict_hidden = predict_hidden + history_hidden
        elif self.map_func == 'fc':
            shape = history_hidden.shape
            predict_hidden = history_hidden.flatten(start_dim=1)
            for i in range(2*self.fc_layers):
                predict_hidden = self.fc[i](predict_hidden)
                predict_hidden = F.dropout(predict_hidden, self.dropout, self.training)
            predict_hidden = predict_hidden.reshape(shape)
            predict_hidden = predict_hidden + history_hidden
        elif self.map_func == 'lstm':
            predict_hidden = history_hidden.squeeze().transpose(0,1)
            predict_hidden = self.start_lin(predict_hidden)
            predict_hidden, _ = self.lstm(predict_hidden)
            predict_hidden = self.end_lin(predict_hidden)
            predict_hidden = predict_hidden.transpose(0,1).unsqueeze(-1)
        predict_output = self.stae.decoder(predict_hidden)
        output = predict_output[:, :, :, -self.horizon:]

        return output
