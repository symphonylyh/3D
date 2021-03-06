import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

import time

def main():

    RESUME = False

    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = bce_loss(d0,labels_v)
        loss1 = bce_loss(d1,labels_v)
        loss2 = bce_loss(d2,labels_v)
        loss3 = bce_loss(d3,labels_v)
        loss4 = bce_loss(d4,labels_v)
        loss5 = bce_loss(d5,labels_v)
        loss6 = bce_loss(d6,labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

        return loss0, loss


    # ------- 2. set the directory of training dataset --------

    model_name = 'u2net' #'u2netp'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    #tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
    #tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)
    tra_image_dir = os.path.join('ROCK', 'images' + os.sep)
    tra_label_dir = os.path.join('ROCK', 'masks' + os.sep)

    image_ext = '.JPG'
    label_ext = '.png'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    epoch_num = 100#100000
    batch_size_train = 8 #12
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16, persistent_workers=True)

    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        print('CUDA available')
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- resume training if needed ----
    checkpoint_path = model_dir + model_name+"_epoch_1.pth"
    if RESUME:
        start_epoch = load_ckpt(net, optimizer, checkpoint_path)
    else:
        start_epoch = 0
    print("Starting from epoch {}".format(start_epoch))

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations
    save_epoch_frq = 10 # save the model every 10 epochs

    for epoch in range(start_epoch, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):

            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                #torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

        if (epoch + 1) % save_epoch_frq == 0:
            checkpoint = {
                'epoch': epoch + 1, # epoch is completed, should start from epoch+1 later on
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            checkpoint_path = model_dir + model_name+"_epoch_%d.pth" % (epoch+1)
            print("Saving model checkpoint...")
            save_ckpt(checkpoint, checkpoint_path)
            net.train()  # resume training

        if (epoch + 1) == epoch_num:
            torch.save(net.state_dict(), model_dir + model_name + os.sep + 'u2net_rock.pth') # final model should be saved without optimizer

def save_ckpt(state, checkpoint_path):
    torch.save(state, checkpoint_path)

def load_ckpt(model, optimizer, checkpoint_path): # model, optimizer, scheduler are global variables
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

if __name__ == '__main__':
    main() # on Windows, if not wrapped in main(), can use multi-workers for dataloader!!!
