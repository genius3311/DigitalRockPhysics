from torch.autograd import Variable
from PIL import Image
import os
import sys
import pickle
#import dominate
# from dominate.tags import *
from collections import OrderedDict
import torch
# import cv2
import datetime
from ..DataTools.Loaders import to_pil_image
from ..DataTools.Loaders import VAR2PIL
from shutil import copy

class _FileLogger(object):
    """Logger for losses files"""
    def __init__(self, logger, log_name, title_list):
        """
        Init a new log term
        :param log_name: The name of the log
        :param title_list: list, titles to store
        :return:
        """
        assert isinstance(logger, Logger), "logger should be instance of Logger"
        self.titles = title_list
        self.len = len(title_list)
        self.log_file_name = os.path.join(logger.log_dir, log_name + '.csv')
        with open(self.log_file_name, 'w') as f:
            f.write(','.join(title_list) + '\n')

    def add_log(self, value_list):
        assert len(value_list) == self.len, "Log Value doesn't match"
        for i in range(self.len):
            if not isinstance(value_list[i], str):
                value_list[i] = str(value_list[i])
        with open(self.log_file_name, 'a') as f:
            f.write(','.join(value_list) + '\n')


class HTML:
    def __init__(self, logger, reflesh=0):
        self.logger = logger
        self.title = logger.opt.exp_name
        self.web_dir = logger.web
        self.img_dir = logger.img
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=self.title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):  #TODO:image
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


class Logger(object):
    """Logger for easy log the training process."""
    def __init__(self, name, exp_dir, opt, commend='',
                 HTML_doc=False, log_dir='log', checkpoint_dir='/tmp/checkpoint',
                 sample='samples', web='web', test_dir = 'test'):
        """
        Init the exp dirs and generate readme file
        :param name: experiment name 实验名称
        :param exp_dir: dir name to store exp 保存实验的文件名称
        :param opt: argparse namespace # 解析器的关键字
        :param log_dir:# log文件
        :param checkpoint_dir:检查点文件，不仅保存模型的参数，优化器参数，还有loss，epoch等
        :param sample:样品图像
        """
        self.name = name# 实验名称
        # self.exp_dir = os.path.join('experiments', os.path.abspath(exp_dir))
        # os.path.abspath：取绝对路径,os.path.join:路径拼接函数
        self.exp_dir = os.path.join(os.path.abspath('experiments'), exp_dir)# 保存实验结果的文件夹的绝对路径，在调用文件的同级生成
        self.log_dir = os.path.join(self.exp_dir, log_dir)# 保存的log文件的绝对路径，在exp_dir的下一级生成
        self.sample = os.path.join(self.exp_dir, sample)# 迭代过程中生成的图像路径，在exp_dir的下一级生成
        self.web = os.path.join(self.exp_dir, web)# web文件夹的绝对路径，在exp_dir的下一级生成
        self.img = os.path.join(self.web, 'images')  # web文件夹下的图像，TODO:image
        self.checkpoint_dir=checkpoint_dir# 检查点文件夹路径
        # self.checkpoint_dir = os.path.join('C/', checkpoint_dir)# 检查点文件的绝对路径
        self.test_dir = os.path.join(self.exp_dir, test_dir)# 测试文件夹的绝对路径
        self.opt = opt# 解析器的关键字
        # self.result_dir = os.path.join(exp_dir, opt.result_dir)
        try:# 生成以上名称的文件夹及文件
            os.mkdir(self.exp_dir)
            os.mkdir(self.log_dir)
            os.mkdir(self.checkpoint_dir)
            os.mkdir(self.sample)
            os.mkdir(self.web)
            os.mkdir(self.img)
            os.mkdir(self.test_dir)
            print('Creating: %s\n          %s\n          %s\n          %s\n          %s'
                  % (self.exp_dir, self.log_dir, self.sample, self.checkpoint_dir, self.test_dir))# 创建以上文件夹及文件
        except NotImplementedError:
            raise Exception('Check your dir.')# 否则抛出异常
        except FileExistsError:
            pass
        # Saving training params
        with open(os.path.join(self.exp_dir, 'run_commend.txt'), 'w') as f:# 在self.exp_dir下创建文本
            f.write(commend)# 在文件中写入命令
        if opt.train_file is not '':# 如果解析器关键字train_file的值不为空
            copy(opt.train_file, self.exp_dir)# 将文件train_file复制给文件exp_dir
        copy(opt.config_file, self.exp_dir)# 否则将文件config_file复制给文件exp_dir

        self.html_tag = HTML_doc# False
        if HTML_doc:# 如果为真
            self.html = HTML(self)
            self.html.add_header(opt.exp_name)
            self.html.save()# 保存

        self._parse()

    def _parse(self):
        """
        print parameters and generate readme file
        :return:
        """
        attr_list = list()# 生成attr_list的绝对路径
        exp_readme = os.path.join(self.exp_dir, 'exp_params.txt')# exp_params.txt的绝对路径

        for attr in dir(self.opt):
            if not attr.startswith('_'):# 检测是否以'_'开头
                attr_list.append(attr)# 不以'_'开头就添加到attr_list列表中
        print('Init parameters...')# 输出参数单元
        with open(exp_readme, 'w') as readme:# 打开exp_params.txt文件
            readme.write(self.name + '\n')# 第一行写入实验名称
            for attr in attr_list:# 迭代循环取出解析器的每一个关键字
                line = '%s : %s' % (attr, self.opt.__getattribute__(attr))
                print(line)# 以行为单位输出关键字和其对应的值
                readme.write(line)# 将其写入文件
                readme.write('\n')# 写一个换一行

    def init_scala_log(self, log_name, title_list):
        """
        Init a new log term
        :param log_name: The name of the log
        :param title_list: list, titles to store
        :return:
        """
        return _FileLogger(self, log_name, title_list)

    def _parse_save_name(self, tag, epoch, step='_', type='.pth'):
        return str(epoch) + step + tag + type# 返回.pth文件

    def save_epoch(self,  epoch, name, state_dict):
        """
        Torch save
        :param name:
        :param state_dict:
        :return:
        """
        torch.save(state_dict, os.path.join(self.checkpoint_dir, self._parse_save_name(name, epoch)))# 保存文件

    def save(self, name, model, optim, dataparallel=1):# 保存文件函数
        """
        Torch save
        :param name:
        :param state_dict:
        :return:
        """
        save_path = os.path.join(self.checkpoint_dir, name)# 保存的pth文件路径
        epoch = {'Name': name,
                 'state_dict': model.state_dict(),
                 'optim': optim.state_dict()}# 建立字典，输入相关信息(文件名称，使用的模型，优化器)
        # torch.save(model.state_dict(), save_path)
        torch.save(epoch, save_path)# name:包含了.pth,保存文件
        print(' saving: %s......' % save_path)# 输出保存完成

    def load_epoch(self, name, epoch):
        return torch.load(os.path.join(self.checkpoint_dir, self._parse_save_name(name, epoch)),map_location='cpu')

    def print_log(self, string, with_time=True, same_line=False):# 输出信息
        if with_time:
            time_stamp = datetime.datetime.now()# 当前时刻
            time_stamp = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')# 开始时刻
            time_stamp += string
            string = time_stamp
        print(string, end='\r') if same_line else print(string)# 输出结束时刻
        sys.stdout.flush()
        with open(os.path.join(self.log_dir, self.name + '.log'), 'a') as f:# 打开log文件
            f.write(string.strip('\n') + '\n')# 写入信息

    def _parse_web_image_name(self, Nom, tag, step='_', type='.png'):
        return 'No.' + str(Nom) + step + tag + type

    def _save_web_images(self, pil, name):
        save_path = os.path.join(self.img, name)
        pil.save(save_path)

    def _add_image_table(self, img_list, tag_list):
        assert len(img_list) == len(tag_list), 'check input'
        self.html.add_images(img_list, tag_list, img_list)
        self.html.save()

    def save_image_record(self, epoch, image_dict):
        img_list = list()
        tag_list = list()
        for tag, image in image_dict.items():
            image_name = self._parse_web_image_name(epoch, tag)
            img_list.append(image_name)
            tag_list.append(tag)
            image.save(os.path.join(self.img, image_name))
        self.html.add_header('Epoch: %d' % epoch)
        self._add_image_table(img_list, tag_list)

    def save_logger(self):
        with open(os.path.join(self.exp_dir, 'logger.pkl'), 'w') as f:
            pickle.dump(self, f)

    def save_training_result(self, im_name, im, dir=False, epoch=0):# 定义训练结果保存函数
        """
        save test result during training procedure, [tensor, Variable]
        :param im_name:输入图像名称
        :param im:输入图像
        :param dir: /sample/epoch/i.png
        :param epoch:
        :return:
        """
        if dir:# 如果dir为真
            save_path = os.path.join(self.sample, str(epoch))# 添加结果保存文件路径
            if not os.path.exists(save_path):# 如果该文件路径不存在
                os.mkdir(save_path)# 在该路径生成文件
        else:# 如果dir为假
            save_path = self.sample# 添加结果保存文件路径
        if isinstance(im, Variable):# 如果im是Variable变量类型，可以反向传播
            im = im.cpu() if im.is_cuda else im# 如果is_cuda为真im.cpu()，否则im本身
            im = VAR2PIL(torch.clamp(im, min=0.0, max=1.0))# 将im转换到0-1范围内，再用VAR2PIL函数将im转换成图片
        else:# 如果im不是Variable类型,即tensor类型
            im = to_pil_image(torch.clamp(im, min=0.0, max=1.0))# 将im转换到0-1范围内，再用to_pil_image将im张量转换成图像保存
        im.save(os.path.join(save_path, im_name))# 在该路径下保存图像结果

    def save_test_result(self, epoch_idx, test_set, im_name, im):
        epoch_folder = os.path.join(self.test_dir, str(epoch_idx))
        if not os.path.exists(epoch_folder):
            os.mkdir(epoch_folder)
        set_folder = os.path.join(epoch_folder, test_set)
        if not os.path.exists(set_folder):
            os.mkdir(set_folder)
        cv2.imwrite(os.path.join(set_folder, im_name), im)


    # def save_testing_results(self, im_name, im):
    #     cv2.imwrite(os.path.join(self.))