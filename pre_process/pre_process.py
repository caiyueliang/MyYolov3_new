# encoding:utf-8
import os
import shutil
import random
import cv2

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


# 写数据 flag:'a+':追加 |'w+'
def write_data(file_name, data, flag):
    with open(file_name, flag) as f:
        f.write(data)


def data_pre_process(root_path, old_label_file, output_label_file):
    class_index = "0"
    files_list = list()

    old_label_path = os.path.join(root_path, old_label_file)
    output_label_path = os.path.join(root_path, output_label_file)

    with open(old_label_path, 'r') as file:
        label_list = file.readlines()

    for label in label_list:
        label = label.replace('\n', '').rstrip()
        str_list = label.split(' ')
        print str_list
        image_path = str_list[0]
        print image_path
        label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
        print label_path

        write_data(output_label_path, image_path + '\n', 'a+')      # 写图片路径到output_label_file
        image = cv2.imread(os.path.join(root_path, image_path))
        h, w, c = image.shape
        print('(h, w, c): (%d, %d, %d)' % (h, w, c))
        save_str = class_index + " "
        for i, temp in enumerate(str_list[2:]):
            print i, str
            if i % 5 == 0:
                save_str += str(float(temp) / w) + " "
            elif i % 5 == 1:
            elif i % 5 == 2:
            elif i % 5 == 3:
            elif i % 5 == 4:
                # 写标签到（output_label_path, label_path）
                write_data(os.path.join(output_label_path, label_path), save_str + '\n', 'a+')
                save_str = class_index + " "
    return


# # 提取文件数量大于min_size的类
# def data_pre_process_1(root_path, output_path, min_size):
#     mkdir_if_not_exist(output_path)
#     for root, dirs, files in os.walk(root_path):
#         for dir in dirs:
#             dir_path = os.path.join(root, dir)
#             print(dir_path)              # 文件名
#
#             for root_1, dirs_1, files_1 in os.walk(dir_path):
#                 if len(files_1) >= min_size:
#                     shutil.copytree(dir_path, os.path.join(output_path, dir))
#     return
#
#
# # 分为train和test文件
# def data_pre_process_2(root_path, output_path, count):
#     mkdir_if_not_exist(output_path)
#
#     mkdir_if_not_exist(output_path)
#     mkdir_if_not_exist(os.path.join(output_path, 'train'))
#     mkdir_if_not_exist(os.path.join(output_path, 'test'))
#
#     for root, dirs, files in os.walk(root_path):
#         random.shuffle(files)
#
#         for i, file in enumerate(files):
#             dir_path = root.split('/')[-1]
#             mkdir_if_not_exist(os.path.join(output_path, 'train', dir_path))
#             mkdir_if_not_exist(os.path.join(output_path, 'test', dir_path))
#
#             file_path = os.path.join(root, file)
#             print(file_path)
#             if i < count:
#                 shutil.copy(file_path, os.path.join(output_path, 'test', dir_path, file))
#             else:
#                 shutil.copy(file_path, os.path.join(output_path, 'train', dir_path, file))
#     return
#
#
# # 每个类别里提取n张图片放到一个文件夹里
# def data_pre_process_3(root_path, output_path, count):
#     mkdir_if_not_exist(output_path)
#
#     mkdir_if_not_exist(output_path)
#     mkdir_if_not_exist(os.path.join(output_path, 'test'))
#
#     for root, dirs, files in os.walk(root_path):
#         random.shuffle(files)
#
#         for i, file in enumerate(files):
#             file_path = os.path.join(root, file)
#             print(file_path)
#             if i < count:
#                 shutil.copy(file_path, os.path.join(output_path, 'test', file))
#     return
#
#
# # 每个类别里提取n张图片放到一个文件夹里
# def data_pre_process_4(root_path, output_path, count):
#     mkdir_if_not_exist(output_path)
#
#     mkdir_if_not_exist(output_path)
#     mkdir_if_not_exist(os.path.join(output_path))
#
#     for root, dirs, files in os.walk(root_path):
#         random.shuffle(files)
#
#         for i, file in enumerate(files):
#             file_path = os.path.join(root, file)
#             print(file_path)
#             if i < count:
#                 shutil.move(file_path, os.path.join(output_path, file))
#     return


if __name__ == '__main__':
    data_pre_process('../../Data/yolo_data/car_detect_train/', 'car_detect_train_label.txt', 'image_path.txt')

    # data_pre_process_1('../../Data/car_classifier_new/', '../../Data/car_classifier_min_50/', 50)
    # data_pre_process_2('../../Data/car_classifier_min_50/', '../../Data/car_classifier_train/', 10)
    # data_pre_process_3('../../Data/car_classifier_min_50/', '../../Data/car_classifier_1/', 50)

    # data_pre_process_4('../../Data/head_tail_classifier/train/head/', '../../Data/head_tail_classifier/test/head/', 700)
    # data_pre_process_4('../../Data/head_tail_classifier/train/tail/', '../../Data/head_tail_classifier/test/tail/', 700)

    # 拆分车牌头的图片
    # data_pre_process('../../Data/car_head_classifier/head', '../../Data/head_classifier')
    # data_pre_process_1('../../Data/car_classifier/head_classifier/', '../../Data/car_classifier/head_classifier_min_1/', 1)
    # data_pre_process_1('../../Data/car_classifier/head_classifier/', '../../Data/car_classifier/head_classifier_min_20/', 20)
    # data_pre_process_1('../../Data/car_classifier/head_classifier/', '../../Data/car_classifier/head_classifier_min_40/', 40)
    # data_pre_process_2('../../Data/car_classifier/clean_car/car_data_1/', '../../Data/car_classifier/classifier_train_best/', 30)

    # data_pre_process('../../Data/car_head_classifier/tail', '../../Data/tail_classifier')
