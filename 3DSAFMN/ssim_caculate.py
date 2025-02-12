import time
from MBHASR.model import HAMSR3D,SRCNN3D,SRResNet3D,RDN3D,SAFMN3D
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from dataset import SRDataset
import torch
import SSIM
from torch.autograd import Variable
import torch.nn.functional as F

batch_size = 1
workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/json_data'
test_dataset = SRDataset(data_folder=dataloader, split='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=workers, pin_memory=False)

if __name__ == '__main__':

    # HAMSR3D_checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/carbonate_HAMSR3D.pth'
    # HAMSR3D_checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/carbonate_SRCNN3D.pth'
    # HAMSR3D_checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/sandstone_HAMSR3D.pth'
    HAMSR3D_checkpoint = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/sandstone_SAFMN3D_noSE.pth'
    checkpoint = torch.load(HAMSR3D_checkpoint, map_location='cpu')
    # generator = HAMSR3D.HAMSR3D()
    # generator = SRCNN3D.Net(layers_num=22)
    # generator = SRResNet3D.SRResNet()
    # generator = RDN3D.RDN3D()
    generator = SAFMN3D.SAFMN(dim=64,n_blocks=12)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['model'])
    generator.eval()
    model = generator

    start = time.time()
    test_ssim = []
    inter_ssim = []
    criterion=nn.MSELoss().to(device)
    for i, (lr_img, hr_img) in enumerate(test_loader):
        lr_img = lr_img.unsqueeze(1).to(device)/255.
        hr_img = hr_img.unsqueeze(1).to(device)/255.
        lr_img = lr_img.float()

        hr_img = Variable(hr_img, requires_grad=False)
        lr_img = Variable(lr_img, requires_grad=False)

        with torch.no_grad():
            sr_img = model(lr_img)
            ssim = SSIM.ssim3D(hr_img,sr_img)
            ssim = ssim.cpu().detach().numpy()
            test_ssim.append(ssim)

            # 插值法计算SSIM
            # sr_inter = F.interpolate(lr_img,scale_factor=4,mode='nearest')
            # ssim_inter=SSIM.ssim3D(hr_img,sr_inter)
            # ssim_inter=ssim_inter.cpu().detach().numpy()
            # inter_ssim.append(ssim_inter)


    column_ssim = ['test_ssim']
    # column_ssim_inter = ['inter_ssim']
    test_s = pd.DataFrame(columns=column_ssim, data=test_ssim)
    # test_s_i = pd.DataFrame(columns=column_ssim_inter,data=inter_ssim)
    test_s.to_csv('result/sand_csv/ssim_SAFMN_sand_NOSE.csv')
    # test_s_i.to_csv('result/carbon_csv/ssim_SRResNet3D.csv')

    print('用时  {:.3f} 秒'.format(time.time() - start))
