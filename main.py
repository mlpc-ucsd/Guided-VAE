from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import pickle

from tqdm import tqdm
import random
from model import unGuidedVAE, suGuidedVAE, Classifier
from dataset import CelebA
from torchvision import transforms as T
from torch.utils.data import DataLoader
from train import train_unsupervised, train_supervised
from test import test_unsupervised, test_supervised

def arg_parse():
    parser = argparse.ArgumentParser(description='Guided VAE')
    parser.add_argument('--batch-size', '-b', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=128, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id to use')
    parser.add_argument('--nz', type=int, default=10,
                        help='bottleneck size')
    parser.add_argument('--output', default='output',
                        help='output directory for results')
    parser.add_argument('--dataroot', default='data',
                        help='root directory for dataset')
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CelebA'],
                        help='dataset to train')
    parser.add_argument('--cls', default='200.0', type=float,
                        help='classification error weight for supervised Guided-VAE')
    parser.add_argument('--selected_attrs', default=['Smiling'], nargs='+',
                        help='selected attrs in CelebA training')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='number of workers for dataloader')
    parser.add_argument('--test_interval', default=1, type=int,
                        help='interval for testing')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--lr_c', default=1e-4, type=float,
                        help='classifier learning rate(in supervised version)')
    parser.add_argument('--weight_decay_c', default=1e-4, type=float,
                        help='classifier weight decay(in supervised version)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    device = torch.device("cuda:{}".format(args.gpu))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    torch.manual_seed(1024)

    if args.dataset == 'MNIST':
        model = unGuidedVAE(args.nz).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        dataset_train = datasets.MNIST(args.dataroot, train=True, download=True, transform=transforms.ToTensor())
        dataset_test = datasets.MNIST(args.dataroot, train=False, download=False, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=False, worker_init_fn=0)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
        for epoch in range(1, args.epochs + 1):
            train_unsupervised(epoch, model, optimizer, train_loader, device)
            if epoch % args.test_interval == 0:
                test_unsupervised(epoch, model, test_loader, args.output, device)
        torch.save(model.state_dict(), args.output + '/unGuidedVAE_MNIST.pth')

    elif args.dataset == 'CelebA':
        model = suGuidedVAE().to(device)
        model_c = model_c = Classifier().to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_c = optim.Adam(model_c.parameters(), lr=args.lr, weight_decay=args.weight_decay_c)
        transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(64),
            T.ToTensor(),
        ])
        data_path = os.path.join(args.dataroot, 'celeba/images')
        attr_path = os.path.join(args.dataroot, 'celeba/list_attr_celeba.txt')
        train_dataset = CelebA(data_path, attr_path, args.selected_attrs, transform, 'train')
        test_dataset = CelebA(data_path, attr_path, args.selected_attrs, transform, 'test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        
        for epoch in range(1, args.epochs + 1):
            train_supervised(epoch, model, model_c, optimizer, optimizer_c, train_loader, args.cls, device)
            if epoch % args.test_interval == 0:
                test_supervised(epoch, model, model_c, test_loader, args.output, device)
