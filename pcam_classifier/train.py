import torch
import torch.nn as nn
from torch.distributions import normal
import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler
import utils
from tqdm import tqdm
import os
import argparse
from torchvision import datasets
import time
import random
import math
import Network
import wandb
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default='pcam', help='dataset')
parser.add_argument('--pretrain_dataset', type=str, default='pcam', help='dataset')
parser.add_argument('--run_num', type=int, default=1)
parser.add_argument('--data_percent', type=float, default=100)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--model', choices=['resnet34'], default='resnet34')
parser.add_argument('--init_cond', choices=['random', 'imagenet', 'jigsaw', 'rotation'], default='imagenet')
parser.add_argument('--batch_size',type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_steps', default=[5], nargs='+', type=int)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--save_epoch', type=int, default=25)
parser.add_argument('--tqdm_off', action='store_true', default=False)
parser.add_argument('--dataset_path', type=str, default='/mnt/datasets/pcam/')

#Parse input arguments
args = parser.parse_args()
print('batch size is {}'.format(args.batch_size))

if args.init_cond in ['jigsaw', 'rotation']:
    save_path = 'results/%s%s_%s%s_%s' % (args.dataset, str(args.data_percent), args.init_cond, args.pretrain_dataset, str(args.run_num))
else:
    save_path = 'results/%s%s_%s_%s' % (args.dataset, str(args.data_percent), args.init_cond, str(args.run_num))


save_path = save_path + '/' + args.model

if not os.path.exists(save_path):
    os.makedirs(save_path)

#Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# #Initialize weights and biases for experiment tracking
if not args.is_test:
    wandb_id = wandb.util.generate_id()
    text_file= open("wandb_id.txt", "w")
    text_file.write(wandb_id)

else:
    text_file = open("wandb_id.txt", "r+")
    wandb_id=text_file.read()

project_name='MIDL2021_finetune_{}'.format(args.data_percent)

if args.init_cond in ['imagenet', 'random']:
    group_name = '{}{} Baseline-{}'.format(str(args.data_percent), args.dataset, args.init_cond, args.pretrain_dataset)
else:
    group_name='{}{} SSL-{}-{}'.format(str(args.data_percent), args.dataset, args.init_cond, args.pretrain_dataset)


wandb.init(
project=project_name,
group=group_name,
name='Initialization-{} {} - Run {}'.format(args.init_cond, args.data_percent, args.run_num),
id=wandb_id,
reinit='True',
resume='allow',
# mode='dryrun',
config={
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "run_number": args.run_num,
    "data_percent": args.data_percent,
    "pretrain_dataset": args.pretrain_dataset,
    "dataset": args.dataset,
    "init_cond": args.init_cond,
    })

config = wandb.config


if args.tqdm_off:
    def nop(it, *a, **k):
        return it
    tqdm = nop


dataset = utils.__dict__['ImageDataset_hdf5']

def save_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%s/checkpoint_%d_%d.pth' % (save_path, args.seed, epoch))

def save_best_checkpoint(epoch, acc):
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%s/checkpoint_best_%d.pth' % (save_path, args.seed))

    #Make file recording best epoch for training
    print('updating best epoch {}'.format(epoch))
    best_epoch_file = open(save_path+"/best_epoch.txt", "w")
    best_epoch_file.write("Best epoch is {} with a testing accuracy of {}".format(epoch, acc))
    best_epoch_file.close()


def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint[0])
    opt.load_state_dict(checkpoint[1])

def compute_acc(class_out, targets):
    preds = torch.max(class_out, 1)[1]
    softmax  = torch.exp(class_out[0])
    pos = 0; 
    for ix in range(preds.size(0)):
        if preds[ix] == targets[ix]:
            pos = pos + 1
    accuracy = pos / preds.size(0) * 100

    return accuracy

def train():
    model.train()
    avg_loss = 0
    avg_acc = 0
    avg_real_acc = 0
    avg_fake_acc = 0
    count = 0
    for _, (data, target) in enumerate(tqdm(train_data_loader)):
        opt.zero_grad()
        data, target  = data.cuda(), target.long().cuda()
        out = model(data)
        loss = ent_loss(out, target)

        loss.backward()
        opt.step()
        avg_loss = avg_loss + loss.item()
        curr_acc = compute_acc(out.data, target.data)
        avg_acc = avg_acc + curr_acc
        count = count + 1
    avg_loss = avg_loss / count
    avg_acc = avg_acc / count
    print('Epoch: %d; Loss: %f; Acc: %.2f; ' % (epoch, avg_loss, avg_acc))
    loss_logger.log(str(avg_loss))
    acc_logger.log(str(avg_acc))

    # Log train accuracy in wandb
    wandb.log({
        "Train Accuracy": avg_acc
    })

    return avg_loss

def test():
    print('Testing')
    model.eval()
  
    pos=0; total=0;
    prediction_list = []
    groundtruth_list = []
    for _, (data, target) in enumerate(tqdm(test_data_loader)):
        data, target  = data.cuda(), target.long().cuda()
        with torch.no_grad():
            out = model(data)
        pred = torch.max(out, out.dim() - 1)[1]
        pos = pos + torch.eq(pred.cpu().long(), target.data.cpu().long()).sum().item()
        
        groundtruth_list += target.data.tolist()
        prediction_list += out[:,1].tolist()

        total = total + data.size(0)
    acc = pos * 1.0 / total * 100
    print('Acc: %.2f' % acc)

    if args.is_test:
        #Log test accuracy in wandb
        wandb.log({
            "Test Accuracy": pos * 1.0 / total * 100
        })

        wandb.join()

    else:
        #Log validation accuracy in wandb
        wandb.log({
            "Validation Accuracy": pos * 1.0 / total * 100
        })

    return acc

def create_model():
    if args.init_cond == 'jigsaw':

        pretrain_path = '../SS_pretrain/results/jigsaw/' + args.pretrain_dataset + str(
            args.run_num) + '/checkpoint_best_1111.pth'
        #
        if not args.is_test:
            print('loading in model from {}'.format(pretrain_path))

        model = Network.Model_LoadJigsaw(pretrain_path)

    if args.init_cond == 'rotation':

        pretrain_path = '../SS_pretrain/results/rotation/' + args.pretrain_dataset + str(
            args.run_num) + '/checkpoint_best_1111.pth'
        if not args.is_test:
            print('loading in model from {}'.format(pretrain_path))

        model = Network.Model_LoadRotation(pretrain_path)

    if args.init_cond == 'imagenet':
        model = Network.Model_imagenet()

    if args.init_cond == 'random':
        model = Network.Model_random()

    model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = lr_scheduler.MultiStepLR(opt, milestones=args.lr_steps, gamma=0.1)

    return model, opt, sch

    return acc


if not args.is_test:
    train_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, mode="train", data_percent=args.data_percent, init_cond=args.init_cond), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(
        dataset(dataset_path=args.dataset_path, mode="valid", data_percent=args.data_percent,init_cond=args.init_cond),
        batch_size=args.batch_size, num_workers=4)
else:
    test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, mode='test', data_percent=100, init_cond=args.init_cond), batch_size=args.batch_size, num_workers=4)

model, opt, sch = create_model()

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not args.is_test:
    loss_logger = utils.TextLogger('loss', '{}/loss_{}.log'.format(save_path, args.seed))
    acc_logger = utils.TextLogger('acc', '{}/acc_{}.log'.format(save_path, args.seed))
    test_acc_logger = utils.TextLogger('test_acc', '{}/test_acc_{}.log'.format(save_path, args.seed))

ent_loss = nn.CrossEntropyLoss().cuda()
epoch = 1
if args.load_epoch != -1:
    epoch = args.load_epoch + 1
    load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.load_epoch))
   
if not args.is_test:
    best_acc = 0
    while True:
        loss = train()
        print(opt.param_groups[0]['lr'])
        sch.step(epoch)
        acc = test()

        test_acc_logger.log(str(acc))

        if epoch % args.save_epoch == 0:
            save_checkpoint()

        if acc > best_acc:
            best_acc = acc
            save_best_checkpoint(epoch, acc)

        if epoch == args.max_epochs:
            break

        epoch += 1
else:
    print('Loading and evaluating model from {}'.format(save_path))
    best_epoch_file = open(save_path + "/best_epoch.txt", "r")
    print(best_epoch_file.read())
    best_epoch_file.close()
    load_checkpoint('%s/checkpoint_best_%d.pth' % (save_path, args.seed))


    test()
    wandb.finish()
