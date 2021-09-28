import logging

from tqdm import tqdm

import dataset
import utils

import os

import torch
import numpy as np
import time
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
parser.add_argument('--version', type=str)
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

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus
# torch.backends.cudnn.enabled = False

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('log'):
    os.makedirs('log')

curr_fn = os.path.basename(args.config).split(".")[0]

out_results_fn = "log/%s_%s_%s_%s.json" % (args.dataset, curr_fn, args.mode, args.version)

config = utils.load_config(args.config)

dataset_config = utils.load_config('dataset/config.json')


if args.source_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['source'])
    dataset_config['dataset'][args.dataset]['source'] = os.path.join(args.source_dir, bs_name)
if args.root_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['root'])
    dataset_config['dataset'][args.dataset]['root'] = os.path.join(args.root_dir, bs_name)

if args.apex:
    from apex import amp

# set NMI or recall accordingly depending on dataset. note for cub and cars R=1,2,4,8
if args.mode == 'trainval' or args.mode == 'test':
    if args.dataset == 'sop' or args.dataset == 'sop_h5':
        args.recall = [1, 10, 100, 1000]
    elif 'cub' in args.dataset or 'cars' in args.dataset:
        args.eval_nmi = True

args.nb_epochs = config['nb_epochs']
args.sz_batch = config['sz_batch']
args.sz_embedding = config['sz_embedding']
if 'warmup_k' in config:
    args.warmup_k = config['warmup_k']

transform_key = 'transform_parameters'
if 'transform_key' in config.keys():
    transform_key = config['transform_key']

args.log_filename = '%s_%s_%s_%s' % (args.dataset, curr_fn, args.mode, args.version)
if args.mode == 'test':
    args.log_filename = args.log_filename.replace('test', 'trainval')

best_epoch = args.nb_epochs

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

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)

logging.info('best_epoch {}'.format(best_epoch))

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
    num_workers=args.nb_workers
)

print("===")
if args.mode == 'train':
    dl_val = torch.utils.data.DataLoader(
        dataset.load(
            name=args.dataset,
            root=dataset_config['dataset'][args.dataset]['root'],
            source=dataset_config['dataset'][args.dataset]['source'],
            classes=dataset_config['dataset'][args.dataset]['classes']['val'],
            transform=dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers
    )
elif args.mode == 'trainval' or args.mode == 'test':
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
        num_workers=args.nb_workers
    )

criterion = config['criterion']['type'](
    nb_classes=dl_tr.dataset.nb_classes(),
    sz_embed=args.sz_embedding,
    **config['criterion']['args']
).cuda()

opt_warmup = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            'lr': 0
        },
        {
            **{'params': list(emb.parameters()
                              )
               },
            **config['opt']['args']['embedding']

        },

        {
            **{'params': criterion.parameters()}
            ,
            **config['opt']['args']['proxynca']

        },

    ],
    **config['opt']['args']['base']
)

opt = config['opt']['type'](
    [
        {
            **{'params': list(feat.parameters()
                              )
               },
            **config['opt']['args']['backbone']
        },
        {
            **{'params': list(emb.parameters()
                              )
               },
            **config['opt']['args']['embedding']
        },

        {
            **{'params': criterion.parameters()},
            **config['opt']['args']['proxynca']
        },

    ],
    **config['opt']['args']['base']
)

if args.apex:
    [model, criterion], [opt, opt_warmup] = amp.initialize([model, criterion], [opt, opt_warmup], opt_level='O1')
    model = torch.nn.DataParallel(model)

if args.mode == 'train' and args.model_path != '':
    model.load_state_dict(torch.load('results/' + args.model_path, map_location=torch.device('cpu')))
    model = model.cuda()

if args.mode == 'test':
    with torch.no_grad():
        logging.info("**Evaluating...(test mode)**")
        model = load_best_checkpoint(model)

        utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)

    exit()

if args.mode == 'train':
    scheduler = config['lr_scheduler']['type'](
        opt, **config['lr_scheduler']['args']
    )
elif args.mode == 'trainval':
    scheduler = config['lr_scheduler2']['type'](
        opt,
        milestones=args.lr_steps,
        gamma=0.1
        # opt, **config['lr_scheduler2']['args']
    )

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()

it = 0

best_val_hmean = 0
best_val_nmi = 0
best_val_epoch = 0
best_val_r1 = 0
best_test_nmi = 0
best_test_r1 = 0
best_test_r2 = 0
best_test_r5 = 0
best_test_r8 = 0
best_tnmi = 0


def batch_lbl_stats(y):
    print(torch.unique(y))
    kk = torch.unique(y)
    kk_c = torch.zeros(kk.size(0))
    for kx in range(kk.size(0)):
        for jx in range(y.size(0)):
            if y[jx] == kk[kx]:
                kk_c[kx] += 1


def get_centers(dl_tr):
    c_centers = torch.zeros(dl_tr.dataset.nb_classes(), args.sz_embedding).cuda()
    n_centers = torch.zeros(dl_tr.dataset.nb_classes()).cuda()
    for ct, (x, y, _) in enumerate(dl_tr):
        with torch.no_grad():
            m = model(x.cuda())
        for ix in range(m.size(0)):
            c_centers[y] += m[ix]
            n_centers[y] += 1
    for ix in range(n_centers.size(0)):
        c_centers[ix] = c_centers[ix] / n_centers[ix]

    return c_centers


prev_lr = opt.param_groups[0]['lr']
lr_steps = []

print(len(dl_tr))

if not args.no_warmup:
    # warm up training for 5 epochs
    logging.info("**warm up for %d epochs.**" % args.warmup_k)
    for e in range(0, args.warmup_k):
        for ct, (x, y, _) in tqdm(enumerate(dl_tr)):
            opt_warmup.zero_grad()
            m = model(x.cuda())
            loss = criterion(m, y.cuda())
            if args.apex:
                with amp.scale_loss(loss, opt_warmup) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_warmup.step()
        logging.info('warm up ends in %d epochs' % (args.warmup_k - e))

for e in range(0, args.nb_epochs):
    # if args.mode == 'trainval':
    #    scheduler.step(e)

    if args.mode == 'train':
        curr_lr = opt.param_groups[0]['lr']
        print(prev_lr, curr_lr)
        if curr_lr != prev_lr:
            prev_lr = curr_lr
            lr_steps.append(e)

    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    tnmi = []

    opt.zero_grad()
    for ct, (x, y, _) in tqdm(enumerate(dl_tr)):
        it += 1

        m = model(x.cuda())

        loss1 = criterion(m, y.cuda())
        loss = loss1

        if args.apex:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())

        if (ct + 1) % 1 == 0:
            opt.step()
            opt.zero_grad()

    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch))

    if args.mode == 'trainval' and e % 4 == 0:
        val_loss = 0

        val_losses_per_epoch = []
        for ct, (x, y, _) in tqdm(enumerate(dl_ev)):
            with torch.no_grad():
                m = model(x.cuda())
                loss_val = criterion(m, y.cuda())
                val_losses_per_epoch.append(loss_val.data.cpu().numpy())

        val_loss = np.mean(val_losses_per_epoch)

    print('it: {}'.format(it))
    print(opt)
    logging.info(
        "Epoch: {}, loss: {:.3f}, val loss: {:.3f} ,time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            val_loss,
            time_per_epoch_2 - time_per_epoch_1
        )
    )

    model.losses = losses
    model.current_epoch = e

    if e == best_epoch:
        break

    if args.mode == 'train':
        with torch.no_grad():
            logging.info("**Validation...**")
            nmi, recall = utils.evaluate(model, dl_val, args.eval_nmi, args.recall)

        chmean = (2 * nmi * recall[0]) / (nmi + recall[0])

        scheduler.step(chmean)

        if chmean > best_val_hmean:
            best_val_hmean = chmean
            best_val_nmi = nmi
            best_val_r1 = recall[0]
            best_val_r2 = recall[1]
            best_val_r4 = recall[2]
            best_val_r8 = recall[3]
            best_val_epoch = e
            best_tnmi = torch.Tensor(tnmi).mean()

        if e == (args.nb_epochs - 1):
            # saving last epoch
            results['last_NMI'] = nmi
            results['last_hmean'] = chmean
            results['best_epoch'] = best_val_epoch
            results['last_R1'] = recall[0]
            results['last_R2'] = recall[1]
            results['last_R4'] = recall[2]
            results['last_R8'] = recall[3]

            # saving best epoch
            results['best_NMI'] = best_val_nmi
            results['best_hmean'] = best_val_hmean
            results['best_R1'] = best_val_r1
            results['best_R2'] = best_val_r2
            results['best_R4'] = best_val_r4
            results['best_R8'] = best_val_r8

        logging.info('Best val epoch: %s', str(best_val_epoch))
        logging.info('Best val hmean: %s', str(best_val_hmean))
        logging.info('Best val nmi: %s', str(best_val_nmi))
        logging.info('Best val r1: %s', str(best_val_r1))
        logging.info(str(lr_steps))

    if args.mode == 'trainval':
        scheduler.step(e)

if args.mode == 'trainval':
    save_best_checkpoint(model)

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

if args.mode == 'train':
    print('lr_steps', lr_steps)
    results['lr_steps'] = lr_steps

with open(out_results_fn, 'w') as outfile:
    json.dump(results, outfile)

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
