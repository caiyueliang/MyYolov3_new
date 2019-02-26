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


# 从源数据中拷贝出来
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
        label_path = image_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        print label_path

        target_image_path = os.path.join(output_path, image_path)
        mkdir_if_not_exist(os.path.split(target_image_path)[0])
        shutil.copy(os.path.join(root_path, image_path), target_image_path)

        write_data(output_label_path, image_path + '\n', 'a+')      # 写图片路径到output_label_file
        image = cv2.imread(os.path.join(root_path, image_path))
        h, w, c = image.shape

        print('(h, w, c): (%d, %d, %d)' % (h, w, c))
        save_str = class_index + " "
        save_flag = True
        for i, temp in enumerate(str_list[2:]):
            print i, temp
            if i % 5 == 0:
                save_str += str((float(temp) + float(str_list[2 + 2 + i]) / 2) / w) + " "
            elif i % 5 == 1:
                save_str += str((float(temp) + float(str_list[2 + 2 + i]) / 2) / h) + " "
            elif i % 5 == 2:
                if float(temp) == 0.0:
                    save_flag = False
                save_str += str(float(temp) / w) + " "
            elif i % 5 == 3:
                if float(temp) == 0.0:
                    save_flag = False
                save_str += str(float(temp) / h) + " "
            elif i % 5 == 4:
                if save_flag is True:
                    # 写标签到（output_label_path, label_path）
                    print save_str
                    write_data(os.path.join(output_path, label_path), save_str + '\n', 'a+')
                save_str = class_index + " "
                save_flag = True
    return


# 在源数据中处理
def data_pre_process_1(root_path, old_label_file, class_index):
    files_list = list()

    old_label_path = os.path.join(root_path, old_label_file)

    with open(old_label_path, 'r') as file:
        label_list = file.readlines()

    for label in label_list:
        label = label.replace('\n', '').rstrip()
        str_list = label.split(' ')
        print str_list
        image_path = str_list[0]
        print image_path
        label_path = image_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        print label_path

        # target_image_path = os.path.join(root_path, image_path)
        # mkdir_if_not_exist(os.path.split(target_image_path)[0])
        # shutil.copy(os.path.join(root_path, image_path), target_image_path)

        image = cv2.imread(os.path.join(root_path, image_path))
        h, w, c = image.shape

        print('(h, w, c): (%d, %d, %d)' % (h, w, c))
        save_str = class_index + " "
        save_flag = True
        for i, temp in enumerate(str_list[2:]):
            print i, temp
            if i % 5 == 0:
                save_str += str((float(temp) + float(str_list[2 + 2 + i]) / 2) / w) + " "
            elif i % 5 == 1:
                save_str += str((float(temp) + float(str_list[2 + 2 + i]) / 2) / h) + " "
            elif i % 5 == 2:
                if float(temp) == 0.0:
                    save_flag = False
                save_str += str(float(temp) / w) + " "
            elif i % 5 == 3:
                if float(temp) == 0.0:
                    save_flag = False
                save_str += str(float(temp) / h) + " "
            elif i % 5 == 4:
                if save_flag is True:
                    # 写标签到（output_label_path, label_path）
                    print save_str
                    write_data(os.path.join(root_path, label_path), save_str + '\n', 'a+')
                save_str = class_index + " "
                save_flag = True
    return


def update_image_path(root_path, label_file):
    write_data(os.path.join(root_path, label_file), "", "w+")

    total_dir = list()
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            total_dir.append(dir)

    for dir in total_dir:
        print(dir)
        for root, dirs, files in os.walk(os.path.join(root_path, dir)):
            for file in files:
                if '.jpg' in file or '.jpeg' in file or '.png' in file:
                    print(file)
                    write_data(os.path.join(root_path, label_file), dir + os.sep + file + '\n', "a+")


def yolo_to_facebox(root_path, old_label_file, new_label_file, filter_label):
    old_label_path = os.path.join(root_path, old_label_file)
    new_label_path = os.path.join(root_path, new_label_file)
    write_data(new_label_path, '', 'w+')

    with open(old_label_path, 'r') as file:
        label_list = file.readlines()

    for label in label_list:
        image_path = label.replace('\n', '').rstrip()
        print('image_path', image_path)
        label_path = image_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        print('label_path', label_path)

        image = cv2.imread(os.path.join(root_path, image_path))
        h, w, c = image.shape
        print('(h, w, c): (%d, %d, %d)' % (h, w, c))

        with open(os.path.join(root_path, label_path), 'a+') as file:
            label_all = file.readlines()

        save_str = ""
        useful_label_num = 0
        for loc_str in label_all:
            print('loc_str', loc_str)
            loc_list = loc_str.rstrip().split(' ')
            print('loc_list', loc_list)

            if (int(loc_list[0]) + 1) in filter_label:
                save_str += str((float(loc_list[1]) * w) - (float(loc_list[3]) * w) / 2) + " "
                save_str += str((float(loc_list[2]) * h) - (float(loc_list[4]) * h) / 2) + " "
                save_str += str(float(loc_list[3]) * w) + " "
                save_str += str(float(loc_list[4]) * h) + " "
                save_str += str(int(loc_list[0]) + 1) + " "
                useful_label_num += 1

        if save_str != "":
            save_str = image_path + " " + str(useful_label_num) + " " + save_str
            write_data(new_label_path, save_str + '\n', 'a+')


if __name__ == '__main__':
    # # 从Faceboxes数据中拷贝出来
    # data_pre_process('../../Data/yolo/yolo_data/car_detect_train/', 'car_detect_train_label.txt',
    #                  '../../Data/yolo/yolo_data_new/car_detect_train/', 'image_path.txt', "0")
    # data_pre_process('../../Data/yolo/yolo_data/car_detect_test/', 'car_detect_test_label.txt',
    #                  '../../Data/yolo/yolo_data_new/car_detect_test/', 'image_path.txt', "0")
    # # 在源数据中处理
    # data_pre_process_1('../../Data/yolo/yolo_data_new/car_detect_train/', 'car_loc_label.txt', "1")
    # data_pre_process_1('../../Data/yolo/yolo_data_new/car_detect_test/', 'car_loc_label.txt', "1")

    # 更新所有的image信息，保存在image_path.txt
    update_image_path('../../Data/yolo/yolo_data_new/car_detect_train/', 'image_path.txt')
    update_image_path('../../Data/yolo/yolo_data_new/car_detect_test/', 'image_path.txt')

    # yolo数据转成Faceboxes数据
    yolo_to_facebox('../../Data/yolo/yolo_data_new/car_detect_train/', 'image_path.txt', 'faceboxes_label.txt', [1, 2])
    yolo_to_facebox('../../Data/yolo/yolo_data_new/car_detect_test/', 'image_path.txt', 'faceboxes_label.txt', [1, 2])

    yolo_to_facebox('../../Data/yolo/yolo_data_new/car_detect_train/', 'image_path.txt', 'plate_label.txt', [1])
    yolo_to_facebox('../../Data/yolo/yolo_data_new/car_detect_test/', 'image_path.txt', 'plate_label.txt', [1])

    yolo_to_facebox('../../Data/yolo/yolo_data_new/car_detect_train/', 'image_path.txt', 'car_label.txt', [2])
    yolo_to_facebox('../../Data/yolo/yolo_data_new/car_detect_test/', 'image_path.txt', 'car_label.txt', [2])
