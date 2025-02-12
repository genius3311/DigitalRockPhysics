import time
from MBHASR.model import HAMSR3D,SRCNN3D,SRResNet3D,RDN3D,SAFMN3D
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from dataset import SRDataset
from util import *
import tifffile
import torch
from loss import *
import torch.nn.functional as F

batch_size = 1
workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/json_data'
test_dataset = SRDataset(data_folder=dataloader, split='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=workers, pin_memory=False)

if __name__ == '__main__':

    # checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/carbonate_HAMSR3D.pth'
    checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/sandstone_SAFMN3D_noSE.pth'
    # checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/sandstone_SRCNN3D.pth'
    # checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/carbonate_SRCNN3D.pth'
    checkpoint = torch.load(checkpoint, map_location='cpu')
    # generator = SRResNet3D.SRResNet()
    # generator = HAMSR3D.HAMSR3D()
    # generator = SRCNN3D.Net(layers_num=22)
    # generator = RDN3D.RDN3D()
    generator = SAFMN3D.SAFMN(dim=64,n_blocks=12)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['model'])
    generator.eval()
    model = generator

    start = time.time()
    test_loss = []
    test_psnr = []
    inter_psnr = []

    L1Loss = nn.L1Loss().to(device)
    FFT = FFTLoss().to(device)
    # criterion = nn.MSELoss().to(device)
    for i, (lr_img, hr_img) in enumerate(test_loader):
        lr_img = lr_img.unsqueeze(1).to(device)/255.
        hr_img = hr_img.unsqueeze(1).to(device)/255.
        lr_img = lr_img.float()

        with torch.no_grad():

            sr_img = model(lr_img)
            loss = L1Loss(sr_img, hr_img) + 0.05 * FFT(sr_img, hr_img)
            # loss = criterion(sr_img, hr_img)
            psnr = PSNR(hr_img.cpu().detach().numpy(), sr_img.cpu().detach().numpy())
            test_loss.append(loss.item())
            test_psnr.append(psnr)


            sr_img = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)*255
            tifffile.imwrite('result/sand_sr/' + 'SAFMN1_%d.tiff' % i, sr_img)
            # hr_img = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)*255
            # tifffile.imwrite('result/carbon_hr/' + 'real_%d.tiff' % i, hr_img)

            # 插值法得到sr
            sr_inter=F.interpolate(lr_img,scale_factor=4,mode='nearest')
            # 损失值
            # loss_inter=criterion(hr_img,sr_inter)
            # psnr_inter=PSNR(hr_img.cpu().detach().numpy(),sr_inter.cpu().detach().numpy())
            # inter_psnr.append(psnr_inter)
            # sr_inter=sr_inter.cpu().detach().numpy().squeeze(0).squeeze(0)
            # tifffile.imwrite('result/sand_sr/' + 'Inter_%d.tiff' % i, sr_inter )



    column_loss = ['test_loss']
    column_psnr = ['test_psnr']
    # column_psnr_inter=['inter_psnr']
    #
    test_l = pd.DataFrame(columns=column_loss, data=test_loss)
    test_p = pd.DataFrame(columns=column_psnr, data=test_psnr)
    # test_p_i = pd.DataFrame(columns=column_psnr_inter,data=inter_psnr)
    #
    # test_l.to_csv('result/carbon_csv/loss_SAFMN_nose_carbon.csv')
    # test_p.to_csv('result/carbon_csv/psnr_SAFMN_nose_carbon.csv')
    # test_p_i.to_csv('result/sand_csv/psnr_Inter3D_sand.csv')

    print('用时  {:.3f} 秒'.format(time.time() - start))
