# -*- encoding: utf-8 -*-
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from MBHASR.model import HAMSR3D,SRCNN3D,SRResNet3D,RDN3D,SAFMN3D
from torch.optim.lr_scheduler import MultiStepLR
import datetime

from MBHASR.util import PSNR
from TorchTools.TorchNet.tools import calculate_parameters
from dataset import SRDataset
import torch.backends.cudnn as cudnn
import torch
from loss import *

batch_size = 1
epochs = 50 # lr= 1e-5: 30epoch时即将收敛，建议50个
workers = 1
start_epoch = 1
lr = 5e-5 # 1.SGD：0.01 可收敛；前10个epoch用0.1 后面每隔10epoch减半 2.Adam: 0.0001 收敛，建议1e-5
# beta = 1e-3         # 判别损失乘子
dataloader = 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/json_data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # 对卷积进行加速
# writer = SummaryWriter("train_HAMSR3D_logs")
# writer = SummaryWriter("train_SRResNet3D_logs")
# writer = SummaryWriter("train_RDN3D_logs")
writer = SummaryWriter("train_SAFMN_logs")
# tensorboard --logdir "C:\Users\Wang Jinye\PycharmProjects\pythonProject\MBHASR\train_SRResNet3D_logs"
# tensorboard --logdir "C:\Users\Wang Jinye\PycharmProjects\pythonProject\MBHASR\train_HAMSR3D_logs"
# tensorboard --logdir "C:\Users\Wang Jinye\PycharmProjects\pythonProject\MBHASR\train_SRCNN3D_logs"
# tensorboard --logdir "C:\Users\Wang Jinye\PycharmProjects\pythonProject\MBHASR\train_RDN3D_logs"
# tensorboard --logdir "C:\Users\Wang Jinye\PycharmProjects\pythonProject\MBHASR\train_SAFMN3D_logs"

def main():
    global start_epoch, writer

    train_dataset = SRDataset(data_folder=dataloader, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False,num_workers=workers, pin_memory=False)

    val_dataset = SRDataset(data_folder=dataloader, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=False)

    # model = SRCNN3D.Net(layers_num=22)
    # model = HAMSR3D.HAMSR3D()
    # model = SRResNet3D.SRResNet()
    # model = RDN3D.RDN3D()
    model = SAFMN3D.SAFMN(dim=64,n_blocks=12)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    # optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr)
    # 4.学习率动态调整
    schedular = MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.5, last_epoch=-1)
    # schedular_d= MultiStepLR(optimizer_d, milestones=[20, 30, 40], gamma=0.5, last_epoch=-1)
    model = model.to(device)
    # discriminator=discriminator.to(device)
    print('%s Created, Parameters: %d' % (model.__class__.__name__, calculate_parameters(model)))

    L1Loss=nn.L1Loss().to(device)
    FFT=FFTLoss().to(device)
    # criterion=nn.MSELoss().to(device)

    # adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    maxpsnr = 0
    maxepoch = 0
    test_interval = 1
    start = datetime.datetime.now()

    for epoch in range(start_epoch, epochs + 1):
        if epoch % test_interval == 0:

            psnr_val=0
            loss_val=0

            model.eval()
            with torch.no_grad():
                for i, (lr_imgs, hr_imgs) in enumerate(val_loader):

                    lr_imgs = lr_imgs.unsqueeze(1).to(device)/255.
                    hr_imgs = hr_imgs.unsqueeze(1).to(device)/255.
                    lr_imgs = lr_imgs.float()
                    sr_imgs = model(lr_imgs)

                    loss = L1Loss(sr_imgs, hr_imgs)+0.05*FFT(sr_imgs, hr_imgs)
                    # loss = criterion(sr_imgs,hr_imgs)
                    loss_val += loss
                    psnr = PSNR(hr_imgs.cpu().detach().numpy(), sr_imgs.cpu().detach().numpy())
                    psnr_val+=psnr

                loss = loss_val / len(val_dataset)
                psnr = psnr_val / len(val_dataset) # 平均psnr
                print(('epoch: %d , loss: %.4f, psnr: %.4f' % (epoch, loss, psnr)))

            writer.add_scalar('SAFMN3D/val_Loss', loss, epoch)
            writer.add_scalar('SAFMN3D/val_psnr', psnr, epoch)

            if psnr > maxpsnr:
                maxpsnr = psnr
                maxepoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/carbonate_3DSAFMN.pth')
                  # 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/carbonate_SRCNN3D.pth')
            # save_checkpoint(model, epoch, 'C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/checkpoint/')

        model.train()
        # discriminator.train()
        psnr_train = 0
        loss_train = 0
        # loss_adv = 0
        # loss_x = 0

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            lr_imgs = lr_imgs.unsqueeze(1).to(device)/255.
            hr_imgs = hr_imgs.unsqueeze(1).to(device)/255.
            lr_imgs = lr_imgs.float()
            sr_imgs = model(lr_imgs)

            # l1损失+FFT损失
            loss = L1Loss(sr_imgs, hr_imgs) + 0.05 * FFT(sr_imgs, hr_imgs)
            # loss = criterion(sr_imgs, hr_imgs)
            # sr_discriminated = discriminator(sr_imgs)
            # 对抗损失
            # adversarial_loss = adversarial_loss_criterion(
            #      sr_discriminated, torch.ones_like(sr_discriminated))  # 生成器希望生成的图像能够完全迷惑判别器，因此它的预期所有图片真值为1

            # 计算总的感知损失
            # perceptual_loss = loss + beta * adversarial_loss
            psnr = PSNR(sr_imgs.cpu().detach().numpy(),hr_imgs.cpu().detach().numpy())
            loss_train += loss
            # loss_adv += adversarial_loss
            psnr_train += psnr

            optimizer.zero_grad()
            loss.backward()
            # perceptual_loss.backward()
            # 5.梯度裁剪策略
            nn.utils.clip_grad_norm_(model.parameters(),0.5)
            optimizer.step()
            # ----------------------2.判别器更新 - ---------------------------
            # 判别器判断
            # hr_discriminated = discriminator(hr_imgs)
            # sr_discriminated = discriminator(sr_imgs.detach())

            # 二值交叉熵损失
            # adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
            #                    adversarial_loss_criterion(hr_discriminated, torch.ones_like(
            #                        hr_discriminated))  # 判别器希望能够准确的判断真假，因此凡是生成器生成的都设置为0，原始图像均设置为1
            # loss_x += adversarial_loss

            # 后向传播
            # optimizer_d.zero_grad()
            # adversarial_loss.backward()
            # #
            # # 更新判别器
            # optimizer_d.step()
            # 记录损失
            if i == 1599:  # 如果是最后一个batch
                print("第 " + str(epoch) + " 个epoch训练结束")

        # 监控损失值变化
        psnr=psnr_train / len(train_dataset)
        loss=loss_train / len(train_dataset)
        # loss_adv = loss_adv / len(train_dataset)
        # loss_x = loss_x / len(train_dataset)
        writer.add_scalar('SAFMN3D/train_psnr', psnr, epoch)  # 每一个epoch的最后一个batch的平均loss
        writer.add_scalar('SAFMN3D/train_Loss', loss, epoch)  # 每一个epoch的最后一个batch的平均loss
        # writer.add_scalar('HAMSR3D/train_Loss', loss_adv, epoch)
        # writer.add_scalar('HAMSR3D/train_Loss', loss_x, epoch)

        # 手动释放内存
        del lr_imgs, hr_imgs, sr_imgs
        schedular.step()
        # schedular_d.step()
        running_lr = schedular.get_last_lr()
        print('第 %d 个epcoh的lr: %.6f' % (epoch, running_lr[0]))

    end = datetime.datetime.now()
    print('time: %s' % (str(end - start)[:-4]))  # 总时间
    print(('Bestepoch: %d , Bestpsnr: %.4f' % (maxepoch , maxpsnr)))
# 训练结束关闭监控
    writer.close()

if __name__ == '__main__':
    main()
