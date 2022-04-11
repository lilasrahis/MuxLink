import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from torch.utils.data import DataLoader
sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *


parser = argparse.ArgumentParser(description='MUXLink Attack')
parser.add_argument('--file-name', default=None, help='Dataset file name')
parser.add_argument('--train-name', default=None, help='Positive training links. i.e., the obervable wires in the circuit.')
parser.add_argument('--testneg-name', default=None, help='Negative test links')
parser.add_argument('--test-name', default=None, help='Positive test links')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
# model settings
parser.add_argument('--hop', default=1, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the final model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
args.hop = int(args.hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# check whether train and test links are provided
train_pos, test_pos,test_neg,link_pos = None, None, None,None

if args.train_name is not None:
    args.train_dir = os.path.join(args.file_dir, './data/{}/{}'.format(args.file_name,args.train_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    train_pos = (train_idx[:, 0], train_idx[:, 1])

if args.testneg_name is not None:
    args.test_dir = os.path.join(args.file_dir, './data/{}/{}'.format(args.file_name,args.testneg_name))
    testneg_idx = np.loadtxt(args.test_dir, dtype=int)
    test_neg = (testneg_idx[:, 0], testneg_idx[:, 1])

if args.test_name is not None:
    args.test_dir = os.path.join(args.file_dir, './data/{}/{}'.format(args.file_name,args.test_name))
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    test_pos = (test_idx[:, 0], test_idx[:, 1])

#Build the network
feat=[]
count=[]
feats_test = np.loadtxt('./data/{}/feat.txt'.format(args.file_name), dtype='float32')
count = np.loadtxt('./data/{}/count.txt'.format(args.file_name))
arr1inds = count.argsort()
sorted_count = count[arr1inds[0::]]
attributes = feats_test[arr1inds[0::]]
assert (args.train_name is not None), "Must provide train links"
args.data_name='links'

max_idx = max(np.max(train_idx), np.max(test_idx))
net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])),shape=(max_idx+1, max_idx+1))
net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

A = net.copy()  # the observed network
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
if test_neg is not None:
    A[test_neg[0], test_neg[1]] = 0  # mask test links
    A[test_neg[1], test_neg[0]] = 0  # mask test links
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

# sample train links.
train_pos, train_neg = sample_neg(
        net,
        train_pos=train_pos,
        test_neg=test_neg,
        test_pos=test_pos,
        max_train_num=args.max_train_num,
    )

'''Train and apply classifier'''

node_information = attributes
if args.only_predict:  # no need to use negatives
    _, test_graphs, max_n_label ,min_n_label= links2subgraphs(
        A,
        None,
        None,
        test_pos, 
        None,
        args.hop,
        node_information,
        args.no_parallel,
    )

    print('# test: %d' % (len(test_graphs)))
else:
    train_graphs, test_graphs, max_n_label,min_n_label = links2subgraphs(
        A,
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        args.hop,
        node_information,
        args.no_parallel,
    )

    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
# GNN configurations
if args.only_predict:
    with open('./data/{}/{}_hyper.pkl'.format(args.file_name,args.data_name), 'rb') as hyperparameters_name:
        saved_cmd_args = pickle.load(hyperparameters_name)
    for key, value in vars(saved_cmd_args).items(): # replace with saved cmd_args
        vars(cmd_args)[key] = value
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    model_name = './data/{}/{}_model.pth'.format(args.file_name,args.data_name)
    classifier.load_state_dict(torch.load(model_name))
    classifier.eval()
    predictions = []
    batch_graph = []
    print(str(cmd_args.batch_size))
    for i, graph in enumerate(test_graphs):
        batch_graph.append(graph)
        if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
            predictions.append(classifier(batch_graph)[0][:, 1].exp().cpu().detach())
            batch_graph = []
    predictions = torch.cat(predictions, 0).unsqueeze(1).numpy()
    test_idx_and_pred = np.concatenate([test_idx, predictions], 1)
    pred_name = './data/{}/'.format(args.file_name) + args.test_name.split('.')[0] +'_'+str(args.hop)+'_' +'_pred.txt'
    np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%d', '%1.2f'])
    print('Predictions for {} are saved in {}'.format(args.test_name, pred_name))
    exit()


cmd_args.printAUC = True
cmd_args.num_epochs = 100
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6;
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])


cmd_args.latent_dim = [ 32, 32, 32, 1] 
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.learning_rate = 1e-4
if (min_n_label<0):
    min_n_label=-3
cmd_args.max_n_label=max_n_label
cmd_args.feat_dim = max_n_label + 1 +(min_n_label*-1)

cmd_args.attr_dim = node_information.shape[1]
classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()
optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)
random.shuffle(train_graphs)
val_num = int(0.1 * len(train_graphs)) # 10% of training set is used for validation
val_graphs = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]
train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size )
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0

    classifier.eval()
    val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0

print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
    best_epoch, test_loss[0], test_loss[1], test_loss[2]))

if args.save_model:
    model_name = './data/{}/{}_model.pth'.format(args.file_name,args.data_name)
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    hyper_name = './data/{}/{}_hyper.pkl'.format(args.file_name,args.data_name)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters to {}...'.format(hyper_name))

with open('acc_results.txt', 'a+') as f:
    f.write(str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(test_loss[2]) + '\n')

