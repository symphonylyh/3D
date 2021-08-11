import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    
    pb_np = np.array(imo)
    
    raw = Image.fromarray(image).convert('RGB')
    white = Image.new('RGB', (image.shape[1], image.shape[0]), color=(255,255,255))
    mask = imo.convert('1')
    masked = Image.composite(raw, white, mask)
    
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

#    montage = Image.new('RGB', (3*image.shape[1], image.shape[0]))
#    montage.paste(raw, (0*image.shape[1],0))
#    montage.paste(imo, (1*image.shape[1],0))
#    montage.paste(masked, (2*image.shape[1],0))
#    montage.save(d_dir+imidx+'_montage.png') # image montage
    imo.save(d_dir+imidx+'.png') # object mask
    #masked.save(d_dir+imidx+'.png') # masked image
    
    # manually generate GIF in PIL (in another script that loads all the images, not in this function)
    # suppose image_list contains all the images (highly recommended to resize it otherwise it's very large)
    # image_list[0].save('test.gif', save_all=True, append_images=image_list[1:], optimize=True, loop=0)

def main(root_path, rock_folder):

    # --------- 1. get image path and name ---------
    model_folder_name = 'u2net' # 'u2netp'
    model_name='u2net_rock' 

    #image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')	
    #prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    image_dir = os.path.join(root_path, rock_folder)
    prediction_dir = os.path.join(root_path, rock_folder, 'masks' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_folder_name, model_name + '.pth')
    img_name_list = glob.glob(image_dir + os.sep + '*.JPG')

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,num_workers=1)

    # --------- 3. model define ---------
    if(model_folder_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_folder_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    root_path = 'H:\RockScan\Ballast'
    rock_folders = ['1', '41', '81']
    for rock_folder in rock_folders:
        main(root_path, rock_folder)
