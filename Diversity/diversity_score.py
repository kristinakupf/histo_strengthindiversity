import torch
import utils
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.hub import load_state_dict_from_url
import torchvision
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--run_num', type=int, default=1)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--model', choices=['resnet34'], default='resnet34')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_steps', default=[5], nargs='+', type=int)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--load_epoch', type=int, default=-1)
parser.add_argument('--save_epoch', type=int, default=1)
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

dataset = utils.__dict__['ImageDataset']
args.dataset_path = args.dataset_path + args.dataset  +'/'

''' Code snippet borrowed from https://github.com/uoguelph-mlrg/instance_selection_for_gans'''
def GaussianModel(embeddings):
    gmm = GaussianMixture(n_components=1, reg_covar=1e-05)
    gmm.fit(embeddings)

    log_likelihood = gmm.score_samples(embeddings)
    return log_likelihood

class BaseImageEmbedding(nn.Module):
    def __init__(self, model, img_res=224):
        super().__init__()

        self.model = model
        self.img_res = img_res
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)


    def forward(self, x):
        # expects image in range [-1, 1]
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std


        if (x.shape[2] != self.img_res) or (x.shape[3] != self.img_res):
            x = F.interpolate(x,
                              size=(self.img_res, self.img_res),
                              mode='bilinear',
                              align_corners=True)

        x = self.model(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Inceptionv3Embedding(BaseImageEmbedding):
    def __init__(self):
        model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        model.fc = Identity()
        super().__init__(model, img_res=299)


class ResNextWSL(BaseImageEmbedding):
    def __init__(self, d=8):
        model = torch.hub.load('facebookresearch/WSL-Images',
                               'resnext101_32x{}d_wsl'.format(d))
        model.fc = Identity()
        super().__init__(model)


class SwAVEmbedding(BaseImageEmbedding):
    def __init__(self):
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
        model.fc = Identity()
        super().__init__(model)


def get_embedder(embedding):
    if embedding == 'inceptionv3':
        embedder = Inceptionv3Embedding().eval().cuda()
    elif embedding == 'resnextwsl':
        embedder = ResNextWSL().eval().cuda()
    elif embedding == 'swav':
        embedder = SwAVEmbedding().eval().cuda()

    if torch.cuda.current_device() > 1:
        embedder = nn.DataParallel(embedder)
    return embedder


def get_embeddings_from_loader(dataloader, embedder, verbose=False):
    embeddings = []

    with torch.no_grad():
        if verbose:
            dataloader = tqdm(dataloader, desc='Extracting embeddings')
        for data in dataloader:
            if len(data) == 3:
                tiles, label, images = data
                images = images.cuda()
            else:
                images = data.cuda()

    embed = embedder(images)
    embeddings.append(embed.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings



def calculate_diversity_score(dataloader, embedding='resnextwsl'):
    '''Calculate diversity of dataset.
​
    Args:
    -----
    dataloader: torch.dataloader
        Dataloader which returns either images or image/label pairs
    embedding: str
        Which pretrained network to use for embedding. Options are 'inceptionv3', 'resnextwsl', and 'swav'
    '''
    embedder = get_embedder(embedding)
    embeddings = get_embeddings_from_loader(dataloader, embedder, verbose=True)
    embeddings = embeddings.detach().numpy()
    likelihood = GaussianModel(embeddings)
    return np.mean(likelihood)


if __name__ == "__main__":

    #set up directory to save files
    save_path = 'results/'+args.dataset+str(args.run_num)+'/'
    print(save_path)

    if not os.path.exists(save_path):
        print(save_path)
        os.makedirs(save_path)

    #Load in data for training/validation or testing
    if 'train' in args.mode:
        train_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=True, is_test=False,dataset=args.dataset, num_classes=args.num_classes), batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=False, is_test=False,dataset=args.dataset,num_classes=args.num_classes ), batch_size=args.batch_size, num_workers=4)

    else:
        test_data_loader = torch.utils.data.DataLoader(dataset(dataset_path=args.dataset_path, train=False, is_test=True, dataset=args.dataset,num_classes=args.num_classes), batch_size=args.batch_size, num_workers=4)

diversity_score = calculate_diversity_score(train_data_loader)
print('diversity_score is {}'.format(diversity_score))