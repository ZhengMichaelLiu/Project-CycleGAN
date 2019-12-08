#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim

from CycleGAN_discriminator import Discriminator
from CycleGAN_generator import Generator

from load_images import ImageDataset
from PIL import Image
import itertools
import time


# Turn on the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# define parameters
lambda_val = 10
learning_rate = 0.0002
EPOCH = 200


# custom weights initialization called on G and D
# Reference: https://github.com/pytorch/examples/blob/master/dcgan/main.py#L131
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

# define and initialize the model and put to GPU

# G: X -> Y
G_X = Generator(3, 3, 9)
G_X.cuda()
G_X.apply(weights_init)

# F: Y -> X
G_Y = Generator(3, 3, 9)
G_Y.cuda()
G_Y.apply(weights_init)

# DX aims to distinguish between images x and translated images F(y);
DX = Discriminator(3)
DX.cuda()
DX.apply(weights_init)

# DY aims to discriminate between y and G(x).
DY = Discriminator(3)
DY.cuda()
DY.apply(weights_init)

# set up optimizer
optimizer_G = optim.Adam(itertools.chain(G_X.parameters(), G_Y.parameters()), lr = learning_rate, betas=(0.5, 0.999))
optimizer_DX = optim.Adam(DX.parameters(), lr = learning_rate, betas=(0.5, 0.999))
optimizer_DY = optim.Adam(DY.parameters(), lr = learning_rate, betas=(0.5, 0.999))

# Define loss function
criterionIDT = nn.L1Loss()
criterionGAN = nn.MSELoss()
criterionCYC = nn.L1Loss()

# variable holders
# Reference: https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train
Tensor = torch.cuda.FloatTensor
input_X = Tensor(1, 3, 256, 256)
input_Y = Tensor(1, 3, 256, 256)
target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)


# load data
use_dataset = 'horse2zebra/'
transforms_ = [transforms.Resize(int(256 * 1.1), Image.BICUBIC), transforms.RandomCrop(256), 
               transforms.RandomHorizontalFlip(),transforms.ToTensor()]
               # transforms.Normalize(mean = (0.5,),std = (0.5,))]
dataloader = DataLoader(ImageDataset(use_dataset, transforms_ = transforms_, unaligned = True), batch_size = 1, shuffle = True, num_workers = 4)


# For plot loss values
generator_loss_list = []
discriminator_loss_list = []

# Train the CycleGAN model
for epoch in range(1, EPOCH + 1):
    now = time.time()
    print("Epoch: ", epoch)
    
    generator_loss = 0.0
    discriminator_loss = 0.0
    
    # update learning rate
    # For both discriminator and generator
    if epoch <= 100:
        learning_rate = 0.0002
        for group in optimizer_G.param_groups:
            group['lr'] = learning_rate
        for group in optimizer_DX.param_groups:
            group['lr'] = learning_rate
        for group in optimizer_DY.param_groups:
            group['lr'] = learning_rate
            
    elif epoch > 100:
        learning_rate = learning_rate - 0.0002 / (EPOCH - 100)
        for group in optimizer_G.param_groups:
            group['lr'] = learning_rate
        for group in optimizer_DX.param_groups:
            group['lr'] = learning_rate
        for group in optimizer_DY.param_groups:
            group['lr'] = learning_rate
    print("Current epoch learning rate: ", optimizer_DX.param_groups[0]['lr'])
    
    counter = 0
    # load data. x_real, y_real
    for i, batch in enumerate(dataloader):
        counter += 1
        # set up input data to gpu
        x_real = Variable(input_X.copy_(batch['A']))
        y_real = Variable(input_Y.copy_(batch['B']))

        # forward pass
        # y_fake: G:X->Y; x_cyc: F:Y->X
        y_fake = G_X(x_real)
        x_cyc = G_Y(y_fake)
        
        # x_fake: F:Y->X; y_cyc: G:X->Y
        x_fake = G_Y(y_real)
        y_cyc = G_X(x_fake)
        
        # Update G_X and G_Y
        # freeze discriminators
        for param in DX.parameters():
            param.requires_grad = False
        for param in DY.parameters():
            param.requires_grad = False
        
        # =================================================================
        # zero out previous gradient of generators
        optimizer_G.zero_grad()
        
        # Calculate loss of G
        # Identity loss, in original code, this is optional
        x_no_change = G_Y(x_real)
        loss_X_idt = criterionIDT(x_no_change, x_real) * lambda_val * 0.5
        
        y_no_change = G_X(y_real)
        loss_Y_idt = criterionIDT(y_no_change, y_real) * lambda_val * 0.5
        
        # train the G to minimize Ex~pdata(x)[(D(G(x)) - 1) ^ 2]
        loss_G_X = criterionGAN(DY(y_fake), target_real)
        loss_G_Y = criterionGAN(DX(x_fake), target_real)
        
        # Cycle Loss
        loss_X_cyc = criterionCYC(x_real, x_cyc) * lambda_val
        loss_Y_cyc = criterionCYC(y_real, y_cyc) * lambda_val

        loss_G = loss_X_idt + loss_Y_idt + loss_G_X + loss_G_Y + loss_X_cyc + loss_Y_cyc

        loss_G.backward()
        
        # Update G
        optimizer_G.step()
        
        generator_loss += loss_G.item()
        
        # =================================================================
        
        # Update DX and DY
        # unfreeze discriminators
        for param in DX.parameters():
            param.requires_grad = True
        for param in DY.parameters():
            param.requires_grad = True
            
        # Calculate loss of D
        # train the D to minimize Ey~pdata(y)[(D(y) - 1) ^ 2] + Ex~pdata(x)[D(G(x)) ^ 2]
        
        # zero out previous gradient of discriminators
        optimizer_DX.zero_grad()
        loss_dx_real = criterionGAN(DX(x_real), target_real)
        loss_dx_fake = criterionGAN(DX(x_fake.detach()), target_fake)
        loss_DX = (loss_dx_real + loss_dx_fake) * 0.5
        loss_DX.backward()
        optimizer_DX.step()
        
        optimizer_DY.zero_grad()
        loss_dy_real = criterionGAN(DY(y_real), target_real)
        loss_dy_fake = criterionGAN(DY(y_fake.detach()), target_fake)
        loss_DY = (loss_dy_real + loss_dy_fake) * 0.5
        loss_DY.backward()
        optimizer_DY.step()
        
        discriminator_loss += loss_DX.item() + loss_DY.item()
        
        if counter % 100 == 0:
            torch.save(G_X.state_dict(), 'data/state_gx.pth')
            torch.save(G_Y.state_dict(), 'data/state_gy.pth')
            torch.save(DX.state_dict(), 'data/state_dx.pth')
            torch.save(DY.state_dict(), 'data/state_dy.pth')

            save_x_real_y_fake_x_cyc = torch.cat((x_real, y_fake, x_cyc), -1)[0]
            save_y_real_x_fake_y_cyc = torch.cat((y_real, x_fake, y_cyc), -1)[0]

            save_image(save_x_real_y_fake_x_cyc, 'train_data_generate/' + use_dataset + '/epoch-' + str(epoch) + '-' + str(counter) + 'x.jpg')
            save_image(save_y_real_x_fake_y_cyc, 'train_data_generate/' + use_dataset + '/epoch-' + str(epoch) + '-' + str(counter) + 'y.jpg')
        
    later = time.time()
    print("Current epoch elapsed time: ", later - now)
    print("Current epoch generator loss: ", generator_loss / counter, "Current epoch discriminator loss: ", discriminator_loss / counter)
    generator_loss_list.append(generator_loss / counter)
    discriminator_loss_list.append(discriminator_loss / counter)
        


# 12.2 evening - 12.3 morning   13  hr
# 12.3 evening                   4  hr
# 12.3 evening - 12.4 morning   14  hr
# 12.4 12:04pm - 2:30pm        2.5  hr
# 12.4 5:50pm - 9:50pm           4  hr
# 12.4 10:30pm - 12.6 12:00pm 37.5  hr
# 12.6 12:00pm - 12.8  6:00am   42  hr

# Plot loss values vs. epoch num
plt.plot(generator_loss_list)
plt.title('Generator Loss Plot of Each Epoch')
plt.xlabel('Epoch number')
plt.ylabel('Generator Loss')
plt.show()

plt.plot(discriminator_loss_list)
plt.title('Discriminator Loss Plot of Each Epoch')
plt.xlabel('Epoch number')
plt.ylabel('Discriminator Loss')
plt.show()

# Save models
torch.save(G_X, 'gx.pth')
torch.save(G_Y, 'gy.pth')
torch.save(DX, 'dx.pth')
torch.save(DY, 'dy.pth')
