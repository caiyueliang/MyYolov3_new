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


def data_pre_process(root_path, old_label_file, output_path, output_label_file, class_index):
    files_list = list()

    mkdir_if_not_exist(output_path)

    old_label_path = os.path.join(root_path, old_label_file)
    output_label_path = os.path.join(output_path, output_label_file)

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

        target_image_path = os.path.join(output_path, image_path)
        mkdir_if_not_exist(os.path.split(target_image_path)[0])
        shutil.copy(os.path.join(root_path, image_path), target_image_path)

        write_data(output_label_path, image_path + '\n', 'a+')      # 写图片路径到output_label_file
        image = cv2.imread(os.path.join(root_path, image_path))
        h, w, c = image.shape
        print('(h, w, c): (%d, %d, %d)' % (h, w, c))
        save_str = class_index + " "
        for i, temp in enumerate(str_list[2:]):
            print i, temp
            if i % 5 == 0:
                save_str += str((float(temp) + float(str_list[2 + 2 + i]) / 2) / w) + " "
            elif i % 5 == 1:
                save_str += str((float(temp) + float(str_list[2 + 2 + i]) / 2) / h) + " "
            elif i % 5 == 2:
                save_str += str(float(temp) / w) + " "
            elif i % 5 == 3:
                save_str += str(float(temp) / h) + " "
            elif i % 5 == 4:
                # 写标签到（output_label_path, label_path）
                print save_str
                write_data(os.path.join(output_path, label_path), save_str + '\n', 'a+')
                save_str = class_index + " "
    return


if __name__ == '__main__':
    data_pre_process('../../Data/yolo_data/car_detect_train/', 'car_detect_train_label.txt',
                     '../../Data/yolo_data_new/car_detect_train/', 'image_path.txt', "0")
    data_pre_process('../../Data/yolo_data/car_detect_test/', 'car_detect_test_label.txt',
                     '../../Data/yolo_data_new/car_detect_test/', 'image_path.txt', "0")

