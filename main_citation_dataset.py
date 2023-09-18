import dgl
import numpy as np
import os
import random
import time
import glob
import json, argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

"""
    IMPORT CUSTOM MODULES/METHODS
"""
from data.data import load_data
from nets.load_net import cluster_model

"""
    GPU Setup
"""

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

"""
    VIEWING MODEL CONFIG AND NUM PARAMETERS
"""

def view_model_param(net_params):
    model = cluster_model(net_params['model'], net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model.parameters())
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    
    print('Model Name: {}\nTotal Parameters: {}\n'.format(net_params['model'], total_param))
    return total_param

"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    start_time = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    g = dataset.graph
    features =  g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']

    net_params['total_param'] = view_model_param(net_params)
    debug = net_params['debug']

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(train_mask))
    print("Validation Graphs: ", len(val_mask))
    print("Test Graphs: ", len(test_mask))
    print("Number of Classes: ", dataset.num_classes)
    print(net_params)

    model = cluster_model(net_params['model'], net_params)
    model = model.to(device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=params['lr_reduce_factor'],
                                                        patience=params['lr_schedule_patience'],
                                                        verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs, epoch_test_accs = [], [], []
    epoch_train_f1s, epoch_val_f1s, epoch_test_f1s = [], [], []
    epoch_train_nmis, epoch_val_nmis, epoch_test_nmis = [], [], []
    epoch_train_aris, epoch_val_aris, epoch_test_aris = [], [], []

    #import train and evaluate functions
    from train.train_citation_dataset import train_epoch, evaluate_network

    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                start_ = time.time()

                epoch_train_loss, epoch_train_evals = train_epoch(model, optimizer, device, dataset.graph)
                epoch_val_loss, epoch_val_evals = evaluate_network(model, device, dataset.graph)
                _, epoch_test_evals = evaluate_network(model, device, dataset.graph) 

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_evals[0])
                epoch_val_accs.append(epoch_val_evals[0])
                epoch_test_accs.append(epoch_test_evals[0])
                epoch_train_f1s.append(epoch_train_evals[1])
                epoch_val_f1s.append(epoch_val_evals[1])
                epoch_test_f1s.append(epoch_test_evals[1])
                epoch_train_nmis.append(epoch_train_evals[2])
                epoch_val_nmis.append(epoch_val_evals[2])
                epoch_test_nmis.append(epoch_test_evals[2])
                epoch_train_aris.append(epoch_train_evals[3])
                epoch_val_aris.append(epoch_val_evals[3])
                epoch_test_aris.append(epoch_test_evals[3])

                if not debug:
                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_evals[0], epoch)
                    writer.add_scalar('val/_acc', epoch_val_evals[0], epoch)
                    writer.add_scalar('test/_acc', epoch_test_evals[0], epoch)
                    writer.add_scalar('train/_f1', epoch_train_evals[1], epoch)
                    writer.add_scalar('val/_f1', epoch_val_evals[1], epoch)
                    writer.add_scalar('test/_f1', epoch_test_evals[1], epoch)
                    writer.add_scalar('train/_nmi', epoch_train_evals[2], epoch)
                    writer.add_scalar('val/_nmi', epoch_val_evals[2], epoch)
                    writer.add_scalar('test/_nmi', epoch_test_evals[2], epoch)
                    writer.add_scalar('train/_ari', epoch_train_evals[3], epoch)
                    writer.add_scalar('val/_ari', epoch_val_evals[3], epoch)
                    writer.add_scalar('test/_ari', epoch_test_evals[3], epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    t.set_postfix(time=time.time()-start_, lr=optimizer.param_groups[0]['lr'],
                                    train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                    train_acc=epoch_train_evals[0], val_acc=epoch_val_evals[0],
                                    test_acc=epoch_test_evals[0], train_f1=epoch_train_evals[1],
                                    val_f1=epoch_val_evals[1], test_f1=epoch_test_evals[1],
                                    train_nmi=epoch_train_evals[2], val_nmi=epoch_val_evals[2],
                                    test_nmi=epoch_test_evals[2], train_ari=epoch_train_evals[3],
                                    val_ari=epoch_val_evals[3], test_ari=epoch_test_evals[3])
                    
                    per_epoch_time.append(time.time()-start_)

                    # Saving checkpoint
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch-1:
                            os.remove(file)
                
                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - start_time > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    # Return test and metrics at best val metric
    best_val_epoch = np.argmax(np.array(epoch_val_accs))

    test_acc = epoch_test_accs[best_val_epoch]
    test_f1 = epoch_test_f1s[best_val_epoch]
    test_nmi = epoch_test_nmis[best_val_epoch]
    test_ari = epoch_test_aris[best_val_epoch]
    train_acc = epoch_train_accs[best_val_epoch]
    train_f1 = epoch_train_f1s[best_val_epoch]
    train_nmi = epoch_train_nmis[best_val_epoch]
    train_ari = epoch_train_aris[best_val_epoch]

    print("Test Performance: acc {:.4f}, f1 {:.4f}, nmi {:.4f}, ari {:.4f}".format(test_acc, test_f1, test_nmi, test_ari))
    print("Train Performance: acc {:.4f}, f1 {:.4f}, nmi {:.4f}, ari {:.4f}".format(train_acc, train_f1, train_nmi, train_ari))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("Total Time (Epochs): {:.4f}".format(time.time()-start_time))
    print("Avg Time per Epoch: {:.4f}".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {}, \nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n\n
        Test Performance: acc {:.4f}, f1 {:.4f}, nmi {:.4f}, ari {:.4f}\n\n
        Train Performance: acc {:.4f}, f1 {:.4f}, nmi {:.4f}, ari {:.4f}\n\n
        Convergence Time (Epochs): {:.4f}\n\nTotal Time (Epochs): {:.4f}\n\nAvg Time per Epoch: {:.4f}\n\n\n\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'], 
            test_acc, test_f1, test_nmi, test_ari, 
            train_acc, train_f1, train_nmi, train_ari, 
            epoch, time.time()-start_time, np.mean(per_epoch_time)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file name, json file', required=True)
    parser.add_argument('--gpu_id', help='gpu id', default=None)
    parser.add_argument('--model', help='model name')
    parser.add_argument('--dataset', help='dataset name')
    parser.add_argument('--out_dir', help='output directory name')
    parser.add_argument('--seed', help='random seed')
    parser.add_argument('--epochs', help='number of epochs')
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--max_time', help="Please give a value for max_time")

    # Model details
    parser.add_argument('--n_layers', help="Please give a value for GCN layers")
    parser.add_argument('--hidden_dim', help="Please give a value for GT_hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for GT_out_dim")
    parser.add_argument('--residual', help="Please give a value for readout")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = load_data(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    # model parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    if args.n_layers is not None:
        net_params['n_layers'] = int(args.n_layers)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)

    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False

    # SBM

    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[1]
    net_params['n_classes'] = dataset.num_classes

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()