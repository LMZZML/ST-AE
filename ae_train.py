import torch
import numpy as np
import random
import argparse
import time
from util import *
from trainer import *
from stae_model import final_ae_stcnn


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/PEMS08/', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--load_model', type=str_to_bool, default=False, help='adj type')

# metr-la:207; pems-bay:325; PEMS03:358; PEMS04:307; PEMS08:170
parser.add_argument('--num_nodes', type=int, default=170, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--node_dim', type=int, default=10, help='dim of nodes')
parser.add_argument('--blocks', type=int, default=4, help='dim of nodes')
parser.add_argument('--layers', type=int, default=2, help='dim of nodes')
parser.add_argument('--kernel_size', type=int, default=2, help='dim of nodes')

parser.add_argument('--residual_channels', type=int, default=32, help='dim of nodes')
parser.add_argument('--out_dim', type=int, default=2, help='dim of nodes')

parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')
parser.add_argument('--step_size1', type=int, default=800, help='step_size')
parser.add_argument('--step_size2', type=int, default=150, help='step_size')

parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--print_every', type=int, default=100, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save_ae/PEMS08_stae_test_', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()
torch.set_num_threads(3)


def main(runid):
    device = torch.device(args.device)
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    model = final_ae_stcnn(num_nodes=args.num_nodes, seq_len=args.seq_in_len, node_dim=args.node_dim,
                           blocks=args.blocks, layers=args.layers, kernel_size=args.kernel_size, dropout=args.dropout,
                           channels=args.residual_channels, out_dim=args.out_dim, device=device)
    if args.load_model:
        # model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        save_model = torch.load(args.model_path, map_location=args.device)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = AETrainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler,
                     device, args.cl)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    minl = 1e5
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)                     # trainx.shape: [batch_size, dim, node_num, seq_length]
            metrics = engine.train(trainx)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            #testx = torch.cat([testx, testy], 0)
            metrics = engine.eval(testx)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)

        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth")
            minl = mvalid_loss

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) + ".pth"))

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # valid data
    outputs = []
    realy = torch.Tensor(dataloader['x_val']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    # realy = realy[:, :, :, 0]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)#.transpose(1,3)
            # preds = preds.transpose(1,3)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    pred = scaler.inverse_transform(yhat)
    realy = scaler.inverse_transform(realy)
    print("pred: " + str(pred.shape))
    print("realy: " + str(realy.shape))
    vmae, vmape, vrmse, vpcc = metric(pred, realy)

    # test data
    outputs = []
    realy = torch.Tensor(dataloader['x_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    # realy = realy[:, :, :, 0]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)#.transpose(1, 3)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = scaler.inverse_transform(realy[:, :, i])
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    return vmae, vmape, vrmse, mae, mape, rmse


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []

    # load data
    dataloader, scaler = [], []

    for i in range(1):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    print('\n\nResults for 10 runs\n\n')
    # valid data
    print('valid\tMAE\t\tRMSE\tMAPE\tPCC')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    print('\n')

    # test data
    log = 'All step mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(mae), np.mean(rmse), np.mean(mape)))
    print('\n')