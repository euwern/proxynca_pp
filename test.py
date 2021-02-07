import logging
import dataset
import utils

import os

import torch
import numpy as np
import argparse
import json
import random

parser = argparse.ArgumentParser(description='Training ProxyNCA++')
parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='config.json')
parser.add_argument('--embedding-size', default=2048, type=int, dest='sz_embedding')
parser.add_argument('--batch-size', default=32, type=int, dest='sz_batch')
parser.add_argument('--epochs', default=40, type=int, dest='nb_epochs')
parser.add_argument('--log-filename', default='example')
parser.add_argument('--workers', default=16, type=int, dest='nb_workers')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mode', default='train', choices=['train', 'trainval', 'test'],
                    help='train with train data or train with trainval')
parser.add_argument('--lr_steps', default=[1000], nargs='+', type=int)
parser.add_argument('--source_dir', default='', type=str)
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--eval_nmi', default=False, action='store_true')
parser.add_argument('--recall', default=[1, 2, 4, 8], nargs='+', type=int)
parser.add_argument('--init_eval', default=False, action='store_true')
parser.add_argument('--no_warmup', default=False, action='store_true')
parser.add_argument('--apex', default=False, action='store_true')
parser.add_argument('--warmup_k', default=5, type=int)
parser.add_argument('--model_path', default='', type=str)
parser.add_argument('--version', type=str)

args = parser.parse_args()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)  # set random seed for all gpus

curr_fn = os.path.basename(args.config).split(".")[0]

out_results_fn = "log/%s_%s_%s_%s.json" % (args.dataset, curr_fn, args.mode, args.version)

config = utils.load_config(args.config)

dataset_config = utils.load_config('dataset/config.json')

args.nb_epochs = config['nb_epochs']
args.sz_batch = config['sz_batch']
args.sz_embedding = config['sz_embedding']

transform_key = 'transform_parameters'
if 'transform_key' in config.keys():
    transform_key = config['transform_key']

args.log_filename = '%s_%s_%s_%s' % (args.dataset, curr_fn, args.mode, args.version)

if args.mode == 'test':
    args.log_filename = args.log_filename.replace('test', 'trainval')

feat = config['model']['type']()
feat.eval()
in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
feat.train()
emb = torch.nn.Linear(in_sz, args.sz_embedding)
model = torch.nn.Sequential(feat, emb)

if not args.apex:
    model = torch.nn.DataParallel(model)
model = model.cuda()


def save_best_checkpoint(model):
    torch.save(model.state_dict(), 'results/' + args.log_filename + '.pt')


def load_best_checkpoint(model):
    model.load_state_dict(torch.load('results/' + args.log_filename + '.pt'))
    model = model.cuda()
    return model


if args.mode == 'trainval':
    train_results_fn = "log/%s_%s_%s_%s.json" % (args.dataset, curr_fn, 'train', args.version)
    if os.path.exists(train_results_fn):
        with open(train_results_fn, 'r') as f:
            train_results = json.load(f)
        args.lr_steps = train_results['lr_steps']
        best_epoch = train_results['best_epoch']

train_transform = dataset.utils.make_transform(
    **dataset_config[transform_key]
)

results = {}

dl_ev = torch.utils.data.DataLoader(
    dataset.load(
        name=args.dataset,
        root=dataset_config['dataset'][args.dataset]['root'],
        source=dataset_config['dataset'][args.dataset]['source'],
        classes=dataset_config['dataset'][args.dataset]['classes']['eval'],
        transform=dataset.utils.make_transform(
            **dataset_config[transform_key],
            is_train=False
        )
    ),
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
    # pin_memory = True
)

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)

if args.mode == 'train':
    tr_dataset = dataset.load(
        name=args.dataset,
        root=dataset_config['dataset'][args.dataset]['root'],
        source=dataset_config['dataset'][args.dataset]['source'],
        classes=dataset_config['dataset'][args.dataset]['classes']['train'],
        transform=train_transform
    )
elif args.mode == 'trainval' or args.mode == 'test':
    tr_dataset = dataset.load(
        name=args.dataset,
        root=dataset_config['dataset'][args.dataset]['root'],
        source=dataset_config['dataset'][args.dataset]['source'],
        classes=dataset_config['dataset'][args.dataset]['classes']['trainval'],
        transform=train_transform
    )

num_class_per_batch = config['num_class_per_batch']
num_gradcum = config['num_gradcum']
is_random_sampler = config['is_random_sampler']
if is_random_sampler:
    batch_sampler = dataset.utils.RandomBatchSampler(tr_dataset.ys, args.sz_batch, True, num_class_per_batch,
                                                     num_gradcum)
else:

    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch,
                                                       int(args.sz_batch / num_class_per_batch))

dl_tr = torch.utils.data.DataLoader(
    tr_dataset,
    batch_sampler=batch_sampler,
    num_workers=args.nb_workers,
    # pin_memory = True
)

with torch.no_grad():
    logging.info("**Evaluating...**")
    model = load_best_checkpoint(model)

    best_test_nmi, (best_test_r1, best_test_r2, best_test_r4, best_test_r8) = utils.evaluate(model, dl_ev,
                                                                                             args.eval_nmi,
                                                                                             args.recall)
    # logging.info('Best test r8: %s', str(best_test_r8))

    results['NMI'] = best_test_nmi
    results['R1'] = best_test_r1
    results['R2'] = best_test_r2
    results['R4'] = best_test_r4
    results['R8'] = best_test_r8

with open(out_results_fn, 'w') as outfile:
    json.dump(results, outfile)
