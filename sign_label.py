# coding=utf-8
import cv2
import os
import time
import shutil
import subprocess
import traceback

# ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',
# 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

# events = [i for i in dir(cv2) if 'EVENT' in i]
# img = np.zeros((512, 512, 3), np.uint8)


# mouse callback function
# def mouse_click_events(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
def exe_cmd(cmd):
    s = subprocess.Popen(str(cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    s.wait()
    print(s)


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def get_files(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))      # os.path.join 获取完整路径
    return L


def get_img_files(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jepg"):
                L.append(os.path.join(root, file))      # os.path.join 获取完整路径
    return L


# 写数据 flag:'a+':追加 |'w+'
def write_data(file_name, data, flag):
    with open(file_name, flag) as f:
        f.write(data)


# 读数据 flag:'r'
def read_data(file_name, flag):
    with open(file_name, flag) as f:
        return f.read()


class SignLabel:
    def __init__(self):
    # def __init__(self, root_dir, image_dir, label_file, index_file, copy_dir, copy_file):
        # self.img_files = get_files(os.path.join(root_dir, image_dir))
        # self.image_dir = image_dir
        # self.label_file = label_file
        self.car_points = []
        self.draw_image = None
        # self.index_file = index_file
        # self.copy_dir = copy_dir
        # self.copy_file = copy_file

        # print("[len] ", len(self.img_files))
        return

    def mouse_click_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.draw_image, (x, y), 3, (0, 0, 255), -1)
            print('click: [%d, %d]' % (x, y))
            self.car_points.append((x, y))

            if len(self.car_points) % 2 == 0:
                cv2.rectangle(self.draw_image, self.car_points[len(self.car_points)-2], self.car_points[len(self.car_points)-1],
                              (0, 0, 255), 2)

    # def sign_start(self, restart=False):
    #     times = 1
    #
    #     mkdir_if_not_exist(os.path.join(self.copy_dir, self.image_dir))
    #
    #     cv2.namedWindow('sign_image')
    #     cv2.setMouseCallback('sign_image', self.mouse_click_events)    # 鼠标事件绑定
    #
    #     if restart is False:
    #         try:
    #             start_i = int(read_data(self.index_file, 'r'))
    #             print('start_index: ' + str(start_i))
    #         except Exception, e:
    #             print e
    #             start_i = 0
    #     else:
    #         start_i = 0
    #
    #     # for img_file in self.img_files:
    #     while start_i < len(self.img_files):
    #         print('[total] %d; [index] %d; [name] %s' % (len(self.img_files), start_i, self.img_files[start_i]))
    #
    #         self.img = cv2.imread(self.img_files[start_i])
    #         print(self.img.shape)
    #         self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
    #         print(self.img.shape)
    #         cv2.imshow('sign_image', self.img)
    #
    #         while True:
    #             cv2.imshow('sign_image', self.img)
    #
    #             # 保存这张图片
    #             k = cv2.waitKey(1) & 0xFF
    #             if k == ord('s') or k == ord('1'):
    #                 print('save ...')
    #                 data = self.img_files[start_i] + " " + str(len(self.car_points) / 2)
    #                 for i in range(len(self.car_points) / 2):
    #                     data += ' ' + str(self.car_points[2*i][0]/float(times)) + \
    #                             ' ' + str(self.car_points[2*i][1]/float(times)) + \
    #                             ' ' + str((self.car_points[2*i+1][0]-self.car_points[2*i][0])/float(times)) + \
    #                             ' ' + str((self.car_points[2*i+1][1]-self.car_points[2*i][1])/float(times)) + \
    #                             ' 1'
    #
    #                 # for (x, y) in self.car_points:
    #                 #     data += ' ' + str(x/float(times)) + ' ' + str(y/float(times))
    #                 data += '\n'
    #
    #                 write_data(self.label_file, data, 'a+')
    #                 start_i += 1
    #                 write_data(self.index_file, str(start_i), 'w')
    #                 self.car_points = []
    #                 break
    #
    #             if k == ord('2'):
    #                 print('save and copy ...')
    #                 data = self.img_files[start_i] + " " + str(len(self.car_points) / 2)
    #                 for i in range(len(self.car_points) / 2):
    #                     data += ' ' + str(self.car_points[2 * i][0] / float(times)) + \
    #                             ' ' + str(self.car_points[2 * i][1] / float(times)) + \
    #                             ' ' + str((self.car_points[2 * i + 1][0] - self.car_points[2 * i][0]) / float(times)) + \
    #                             ' ' + str((self.car_points[2 * i + 1][1] - self.car_points[2 * i][1]) / float(times)) + \
    #                             ' 1'
    #
    #                 data += '\n'
    #
    #                 write_data(self.label_file, data, 'a+')
    #                 write_data(self.copy_file, data, 'a+')
    #                 shutil.copy(self.img_files[start_i], os.path.join(self.copy_dir, self.image_dir,
    #                                                                   self.img_files[start_i].split('/')[-1]))
    #
    #                 start_i += 1
    #                 write_data(self.index_file, str(start_i), 'w')
    #                 self.car_points = []
    #                 break
    #
    #             if k == ord('d'):
    #                 print('delete ...')
    #                 exe_cmd('rm -r ' + self.img_files[start_i])
    #                 self.img_files.pop(start_i)
    #
    #                 print('[total] %d; [index] %d; [name] %s' % (len(self.img_files), start_i, self.img_files[start_i]))
    #                 self.img = cv2.imread(self.img_files[start_i])
    #                 print(self.img.shape)
    #                 self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0] * times))
    #                 print(self.img.shape)
    #                 cv2.imshow('sign_image', self.img)
    #                 self.car_points = []
    #
    #             # 重新加载图片
    #             if k == ord('r'):
    #                 print('resign ...')
    #                 self.img = cv2.imread(self.img_files[start_i])
    #                 print(self.img.shape)
    #                 self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0]*times))
    #                 print(self.img.shape)
    #                 cv2.imshow('sign_image', self.img)
    #                 self.car_points = []
    #
    #             if k == ord('c'):
    #                 print('change size ...')
    #                 if times == 2:
    #                     times = 1
    #                 else:
    #                     times = 2
    #                 self.img = cv2.imread(self.img_files[start_i])
    #                 print(self.img.shape)
    #                 self.img = cv2.resize(self.img, (self.img.shape[1]*times, self.img.shape[0] * times))
    #                 print(self.img.shape)
    #                 cv2.imshow('sign_image', self.img)
    #                 self.car_points = []

    # def clean_start(self, root_dir, label_path, restart=False):
    #     with open(label_path, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             str_list = line.split(' ')
    #             image_path = os.path.join(root_dir, str_list[0])
    #             img = cv2.imread(image_path)
    #             while True:
    #                 cv2.imshow('clean_image', img)
    #
    #                 k = cv2.waitKey(1) & 0xFF
    #                 if k == ord('y'):
    #                     print('save ...')
    #                     write_data(os.path.join('.', label_path.split('/')[-1]), line, 'a+')
    #                     break
    #                 if k == ord('n'):
    #                     break
    #     return

    # ============================================================================================================
    def show_image(self, image_file):
        label_file = image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        print(image_file)
        print(label_file)

        image = cv2.imread(image_file)
        h, w, c = image.shape
        print('(h, w, c): (%d, %d, %d)' % (h, w, c))

        with open(label_file, 'r') as file:
            label_list = file.readlines()

        for label in label_list:
            temp_list = label.rstrip().split(' ')
            x1 = float(temp_list[1]) * w - (float(temp_list[3]) * w / 2)
            y1 = float(temp_list[2]) * h - (float(temp_list[4]) * h / 2)

            x2 = float(temp_list[1]) * w + (float(temp_list[3]) * w / 2)
            y2 = float(temp_list[2]) * h + (float(temp_list[4]) * h / 2)

            if int(temp_list[0]) == 0:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            elif int(temp_list[0]) == 1:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        cv2.imshow('show_image', image)
        cv2.waitKey(0)

    def show_images(self, root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if '.txt' not in file:
                    self.show_image(os.path.join(root, file))

    # ============================================================================================================
    def draw_rectangle(self, image, label_list):
        # print(label_list)
        for index, label in enumerate(label_list):
            print (index + 1), label
            if int(label["class"]) == 0:
                cv2.rectangle(image, (label["points"][0], label["points"][1]), (label["points"][2], label["points"][3]), (0, 255, 0), 2)
            elif int(label["class"]) == 1:
                cv2.rectangle(image, (label["points"][0], label["points"][1]), (label["points"][2], label["points"][3]), (0, 255, 255), 2)

        return image

    def write_label(self, label_file, label_list, w, h):
        write_data(label_file, "", "w+")        # 清空原来的数据

        for label in label_list:
            save_str = label["class"] + " "
            save_str += str((float(label["points"][0]) + float(label["points"][2])) / 2 / w) + " "
            save_str += str((float(label["points"][1]) + float(label["points"][3])) / 2 / h) + " "
            save_str += str((float(label["points"][2]) - float(label["points"][0])) / w) + " "
            save_str += str((float(label["points"][3]) - float(label["points"][1])) / h) + "\n"
            write_data(label_file, save_str, 'a+')

    def sign_image(self, image_file, label_file):
        print(image_file)
        print(label_file)

        cv2.namedWindow('sign_image')
        cv2.setMouseCallback('sign_image', self.mouse_click_events)    # 鼠标事件绑定

        image = cv2.imread(image_file)
        h, w, c = image.shape
        print('(h, w, c): (%d, %d, %d)' % (h, w, c))

        if os.path.exists(label_file):
            with open(label_file, 'r') as file:
                label_list_str = file.readlines()
        else:
            label_list_str = list()

        print(label_list_str)
        label_list = list()
        for label in label_list_str:
            temp_list = label.rstrip().split(' ')
            x1 = float(temp_list[1]) * w - (float(temp_list[3]) * w / 2)
            y1 = float(temp_list[2]) * h - (float(temp_list[4]) * h / 2)

            x2 = float(temp_list[1]) * w + (float(temp_list[3]) * w / 2)
            y2 = float(temp_list[2]) * h + (float(temp_list[4]) * h / 2)

            label_list.append(dict())
            label_list[-1]["class"] = temp_list[0]
            label_list[-1]["points"] = (int(x1), int(y1), int(x2), int(y2))

        self.draw_image = self.draw_rectangle(image.copy(), label_list)

        while True:
            try:
                cv2.imshow('sign_image', self.draw_image)

                # 保存车牌标记框
                k = cv2.waitKey(1) & 0xFF
                if k == ord('1'):
                    print('======================================================')
                    print('[append] class 0: car plate ...')
                    if len(self.car_points) == 2:
                        label_list.append(dict())
                        label_list[-1]["class"] = '0'
                        label_list[-1]["points"] = (self.car_points[0][0], self.car_points[0][1],
                                                    self.car_points[1][0], self.car_points[1][1])
                        self.car_points = []
                        self.draw_image = self.draw_rectangle(image.copy(), label_list)
                    else:
                        print('[append] fail: %s' % self.car_points)
                # 保存车辆标记框
                if k == ord('2'):
                    print('======================================================')
                    print('[append] class 1: car ...')
                    if len(self.car_points) == 2:
                        label_list.append(dict())
                        label_list[-1]["class"] = '1'
                        label_list[-1]["points"] = (self.car_points[0][0], self.car_points[0][1],
                                                    self.car_points[1][0], self.car_points[1][1])
                        self.car_points = []
                        self.draw_image = self.draw_rectangle(image.copy(), label_list)
                    else:
                        print('[append] fail: %s' % self.car_points)
                # 重新加载图片
                if k == ord('r'):
                    print('======================================================')
                    print('resign ...')
                    self.car_points = []
                    self.draw_image = self.draw_rectangle(image.copy(), label_list)
                # 保存，显示下一张
                if k == ord('s'):
                    print('======================================================')
                    print('[save] ...')
                    self.write_label(label_file, label_list, w, h)
                    break
                # 退出，不保存，显示下一张
                if k == ord('q'):
                    print('======================================================')
                    print('[next] ...')
                    break
                if k == ord('d'):
                    print('======================================================')
                    print('[delete] image and label ...')
                    if os.path.exists(image_file):
                        os.remove(image_file)
                    if os.path.exists(label_file):
                        os.remove(label_file)

                    break

                # 删除标记框
                if k == ord('!'):
                    print('======================================================')
                    object = label_list.pop(0)
                    print('[delete] ...index: 1; label: %s' % object)
                    self.draw_image = self.draw_rectangle(image.copy(), label_list)
                if k == ord('@'):
                    print('======================================================')
                    object = label_list.pop(1)
                    print('[delete] ...index: 2; label: %s' % object)
                    self.draw_image = self.draw_rectangle(image.copy(), label_list)
                if k == ord('#'):
                    print('======================================================')
                    object = label_list.pop(2)
                    print('[delete] ...index: 3; label: %s' % object)
                    self.draw_image = self.draw_rectangle(image.copy(), label_list)
                if k == ord('$'):
                    print('======================================================')
                    object = label_list.pop(3)
                    print('[delete] ...index: 4; label: %s' % object)
                    self.draw_image = self.draw_rectangle(image.copy(), label_list)
                if k == ord('%'):
                    print('======================================================')
                    object = label_list.pop(4)
                    print('[delete] ...index: 5; label: %s' % object)
                    self.draw_image = self.draw_rectangle(image.copy(), label_list)

            except Exception:
                msg = traceback.format_exc()
                print msg

        return

    def sign_images(self, root_path, process_all=True):
        print('process_all: %s' % process_all)

        for root, dirs, files in os.walk(root_path):
            for file in files:
                if '.txt' not in file:
                    process_flag = True

                    # 只处理未处理过的
                    label_name = file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                    if os.path.exists(os.path.join(root, label_name)) \
                            and process_all is False:
                        process_flag = False

                    if process_flag is True:
                        self.sign_image(os.path.join(root, file), os.path.join(root, label_name))


if __name__ == '__main__':
    # show_image("../Data/yolo/yolo_data_new/car_detect_train/daozha_1/480466_闽DF3N37.jpg")
    # show_images(root_path="../Data/yolo/yolo_data_new_1/car_detect_train/")

    sign_label = SignLabel()
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/daozha_1/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/daozha_2/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/daozha_3/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/failed_1/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/failed_2/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/failed_3/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/failed_4/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/failed_5/", process_all=False)
    sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/failed_9/", process_all=False)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/ketuo_1/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/other_1/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_train/szlg_1/", process_all=True)
    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new/car_detect_test/", process_all=True)

    # sign_label.sign_images(root_path="../Data/yolo/yolo_data_new_1/car_detect_train/", process_all=False)
