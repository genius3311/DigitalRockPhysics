# -*- encoding: utf-8 -*-

from util import create_data_lists

if __name__ == '__main__':

    # # 碳酸盐岩路径
    # create_data_lists(train_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_train_HR_tiff'],
    #                   train_et_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_train_HR_expand_tiff'],
    #                   train_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_train_LR_x4_tiff'],
    #                   train_lr_et_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_train_LR_x4_expand_tiff'],
    #                   test_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_test_HR_tiff'],
    #                   test_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_test_LR_x4_tiff'],
    #                   val_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_valid_HR_tiff'],
    #                   val_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_valid_LR_x4_tiff'],
    #                   output_folder='C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/json_data')

    # # 砂岩路径
    create_data_lists(train_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_train_HR_tiff'],
                      train_et_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_train_HR_expand_tiff'],
                      train_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_train_LR_x4_tiff'],
                      train_lr_et_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_train_LR_x4_expand_tiff'],
                      test_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_test_HR_tiff'],
                      test_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_test_LR_x4_tiff'],
                      val_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_valid_HR_tiff'],
                      val_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_valid_LR_x4_tiff'],
                      output_folder='C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/json_data')

    # 混合路径
    # create_data_lists(train_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_train_HR_tiff'],
    #                   train_et_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_train_HR_tiff'],
    #                   train_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_train_LR_x4_tiff'],
    #                   train_lr_et_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/carbonate_train_LR_x4_tiff'],
    #                   test_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_test_HR_tiff'],
    #                   test_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_test_LR_x4_tiff'],
    #                   val_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_valid_HR_tiff'],
    #                   val_lr_folders=['C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/dataset/sandstone_valid_LR_x4_tiff'],
    #                   output_folder='C:/Users/Wang Jinye/PycharmProjects/pythonProject/MBHASR/json_data')

