import matplotlib.pyplot as plt
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Resize, RandomCrop, ToTensor, Compose
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from util import get_hole, get_mask, crop
from generator import Generator
from discriminator import Discriminator
from tqdm import tqdm
import random

MIN_HOLEW, MAX_HOLEW = 96, 128
MIN_HOLEH, MAX_HOLEH = 96, 128
EPOCH_G = 10
EPOCH_D = 1000
EPOCH_M = 1000
batch_size = 16
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

tsfm = Compose([Resize(256), RandomCrop((256, 256)), ToTensor()])

training_data = datasets.CelebA(
    root="../data",
    split='train',
    download=False,
    transform=tsfm
)
test_data = datasets.CelebA(
    root="../data",
    split='test',
    download=False,
    transform=tsfm
)

# calculating mean pixel value of the training set
mpv = torch.tensor((0.50925811, 0.42336759, 0.37791181)).view(1, 3, 1, 1).to(device)
# mpv = np.zeros((3,))
# for x in training_data:
#    r = x[0][0]
#    g = x[0][1]
#    b = x[0][2]
#   mpv += (torch.mean(r), torch.mean(g), torch.mean(b))
# mpv /= len(training_data)

def collate_fn(batch):
    batch = torch.cat([sample[0].unsqueeze(0) for sample in batch], dim=0)
    return batch

train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

generator = Generator().to(device)
optimizer = optim.Adadelta(generator.parameters())
discriminator = Discriminator().to(device)
alpha = 0.5
OptimizerD = optim.Adadelta(discriminator.parameters())
DPATH = None
BCEloss = nn.BCELoss()
if DPATH is not None:
  checkpoint = torch.load(DPATH)
  discriminator.load_state_dict(checkpoint['model_state_dict'])
  OptimizerD.load_state_dict(checkpoint['optimizer_state_dict'])

GPATH = "../model/gen-mutual-20-loss13.787893346045166.pth"
if GPATH is not None:
  checkpoint = torch.load(GPATH)
  generator.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
generator.eval()
with torch.no_grad():
  x = next(iter(test_dataloader)).to(device)
  shape = (x.shape[0], 1, x.shape[2], x.shape[3])
  hole = get_hole((random.randint(MIN_HOLEW, MAX_HOLEW),
                    random.randint(MIN_HOLEH, MAX_HOLEH)))
  mask = get_mask(shape, hole).to(device)
  x = x - x * mask + mpv * mask
  input = torch.cat((x, mask), dim=1)
  out = generator(input)
  imgs = torch.cat((x.cpu(), out.cpu()), dim=0)
  save_image(imgs, "../result/test2.jpg", nrow=len(x))