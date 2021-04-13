import torch
import torch.nn as nn
from torch.distributions import normal
import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler
import utils
import Network
from tqdm import tqdm
import os
import argparse
import datetime
from torchvision import datasets
import time
import random
import numpy as np
import string
import math
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--is_test', action='store_true', default=False)
parser.add_argument('--ss_task', choices=['jigsaw', 'rotation'], default='jigsaw')
parser.add_argument('--run_num', type=int, default=1)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--model', choices=['resnet34'], default='resnet34')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_steps', default=[5], nargs='+', type=int)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--save_epoch', type=int, default=25)
parser.add_argument('--tqdm_off', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='pcam')
parser.add_argument('--dataset_path', type=str, default='/mnt/data/kupfersk/StrengthInDiversity/')


args = parser.parse_args()

#Set all random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


if args.tqdm_off:
    def nop(it, *a, **k):
        return it
    tqdm = nop

#Set up dataloader
dataset = utils.__dict__['ImageDataset']
args.dataset_path = args.dataset_path + args.dataset  +'/'

#Check that input for SSL task is correct
if args.ss_task not in (['rotation', 'jigsaw']):
    raise ValueError('SSL task provided is not valid. Currently the only supported tasks are rotation and jigsaw')

# #Initialize weights and biases for experiment tracking
if not args.is_test:
    wandb_id = wandb.util.generate_id()
    uid_file= open("wandb_id.txt", "w")
    uid_file.write(wandb_id)

else:
    text_file = open("wandb_id.txt", "r+")
    wandb_id=text_file.read()

wandb.init(
project='MIDL2021_SSL_Pretraining',
group='{}-SSL-{}'.format(args.ss_task, args.dataset),
name='{}-SSL - {} Run {}'.format(args.ss_task, args.dataset, args.run_num),
id=wandb_id,
reinit='True',
resume='allow',
# mode='dryrun',
config={
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "run_number": args.run_num,
    "SSL_task": args.ss_task,
    "pretraining_dataset": args.dataset
    })
config = wandb.config

def save_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%scheckpoint_%d_%d.pth' % (save_path, args.seed, epoch))

def save_best_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%scheckpoint_best_%d.pth' % (save_path, args.seed))

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

    count = 0
    for _, (data, target, image) in enumerate(tqdm(train_data_loader)):
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
    for _, (data, target, image) in enumerate(tqdm(test_data_loader)):

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

    #Log validation or test accuracy
    if args.is_test:
        # Log test accuracy in wandb
        wandb.log({
            "Test Accuracy": pos * 1.0 / total * 100
        })

        wandb.join()

    else:
        # Log validation accuracy in wandb
        wandb.log({
            "Validation Accuracy": pos * 1.0 / total * 100
        })

    return acc

def create_model(num_classes):
    if args.ss_task == 'jigsaw':
        if num_classes not in [10, 100, 1000]:
            raise ValueError('For Jigsaw SSL number of classes must be equal to 10, 100, or 1000')

        # Create a model for jigsaw SSL task
        Model = Network.__dict__['Model_Jigsaw']
        model = Model(num_classes=num_classes)


    if args.ss_task == 'rotation':
        #Create a model for rotation SSL task
        if num_classes != 4:
            raise ValueError('For Rotation SSL number of classes must be equal to 4')

        Model = Network.__dict__['Model_Rotation']
        model = Model(num_classes=4)

    model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = lr_scheduler.MultiStepLR(opt, milestones=args.lr_steps, gamma=0.1)
    return model, opt, sch

if __name__ == "__main__":

    #set up directory to save files
    save_path = 'results/{}/{}{}/'.format(args.ss_task, args.dataset, str(args.run_num))
    print(save_path)

    if not os.path.exists(save_path):
        print(save_path)
        os.makedirs(save_path)


    '''Train data loader returns:
     1) For jigsaw SSL task (tiles, labels, image)
     2) For rotation SSL task (data, labels, original image)'''

    #Load in data for training/validation or testing
    if 'train' in args.mode:
        train_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=True, is_test=False,dataset=args.dataset, num_classes=args.num_classes,  ss_task=args.ss_task), batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=False, is_test=False,dataset=args.dataset,num_classes=args.num_classes, ss_task=args.ss_task), batch_size=args.batch_size, num_workers=4)

    else:
        test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=False, is_test=True, dataset=args.dataset,num_classes=args.num_classes, ss_task=args.ss_task),batch_size=args.batch_size, num_workers=4)


    #Create the model
    model, opt, sch = create_model(num_classes=args.num_classes)

    if not args.mode =='test':
        loss_logger = utils.TextLogger('loss', '{}/loss_{}.log'.format(save_path, args.seed))
        acc_logger = utils.TextLogger('acc', '{}/acc_{}.log'.format(save_path, args.seed))
        test_acc_logger = utils.TextLogger('test_acc', '{}/test_acc_{}.log'.format(save_path, args.seed))

    ent_loss = nn.CrossEntropyLoss().cuda()
    epoch = 1

    if args.load_epoch != -1:
        epoch = args.load_epoch + 1
        load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.seed, args.load_epoch))

    if not args.mode == 'test':
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
                save_best_checkpoint()

            if epoch == args.max_epochs or acc>=85:
                break

            epoch += 1

    else:
        print(save_path)
        load_checkpoint(save_path)
        acc = test()
