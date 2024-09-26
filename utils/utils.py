import cv2
import numpy as np
import torch
import xlwt
import os
import time
import random
import json
import matplotlib.pyplot as plt

from configs.configuration import Config
from utils.data_utils import *

from sklearn.utils.multiclass import unique_labels
# for comfusion matrix
from sklearn.metrics import confusion_matrix


import xml.etree.ElementTree as ET
from configs.configuration import Config
from utils.data_utils import *


def Average(lst):
    return sum(lst) / len(lst)

def calculate_errors(model, test_loader, gt_test_dir, label_type, save_path, point_list, device='cpu' ):

    points_num = len(point_list)
    loss = np.zeros(points_num)
    num_err_below_20 = np.zeros(points_num)
    num_err_below_25 = np.zeros(points_num)
    num_err_below_30 = np.zeros(points_num)
    num_err_below_40 = np.zeros(points_num)

    accuracy_bellow_20 = []
    accuracy_bellow_25 = []
    accuracy_bellow_30 = []
    accuracy_bellow_40 = []

    img_num = 0
    for img_num, (img, heatmaps, _, img_name, _, _, scal_ratio_w, scal_ratio_h) in enumerate(test_loader):
        # print('image: ', img_name[0])

        img = img.to(device)
        model.to(device)
        outputs, _ = model(img)
        outputs = outputs[0].cpu().detach().numpy()


        pred = get_predict_point_from_heatmap(outputs, scal_ratio_w, scal_ratio_h, points_num)


        if label_type == "txt":
            ########## Working with TXT files ##########
            gt_path = gt_test_dir + '/' + img_name[0].split('.')[0] + '.txt'
            points_name, gt_x, gt_y = get_points_from_txt(points_num, gt_path)
        elif label_type == "json":

            ########## Working with JSON files ##########
            gt_path = gt_test_dir + img_name[0].split('.jpeg')[0] + '.json'
            points_name, gt_x, gt_y = get_points_from_json(point_list, gt_path)
        elif label_type == "xml":
            ########## Working with xml files ##########
            gt_path = gt_test_dir + '/annotations.xml'

            doc = ET.parse(gt_path)
            root = doc.getroot()
            xml_imageEles = root.findall('image')

            points_name, gt_x, gt_y = get_points_from_CVAT_xml(img_name[0], point_list, xml_imageEles)
            # print(points_name, gt_x, gt_y)


        # note_gt_road = note_gt_dir + '/' + img_name[0].split('.')[0] + '.txt'
        # gt_x, gt_y = get_points_from_txt(points_num, note_gt_road)

        gt_x = np.trunc(np.reshape(gt_x, (points_num, 1)))
        gt_y = np.trunc(np.reshape(gt_y, (points_num, 1)))
        gt = np.concatenate((gt_x, gt_y), 1)
        for j in range(points_num):
            error = np.sqrt((gt[j][0] - pred[j][0]) ** 2 + (gt[j][1] - pred[j][1]) ** 2)
            loss[j] += error
            if error <= 20:
                num_err_below_20[j] += 1
            elif error <= 25:
                num_err_below_25[j] += 1
            elif error <= 30:
                num_err_below_30[j] += 1
            elif error <= 40:
                num_err_below_40[j] += 1

    loss = loss / (img_num + 1)
    num_err_below_25 = num_err_below_25 + num_err_below_20
    num_err_below_30 = num_err_below_30 + num_err_below_25
    num_err_below_40 = num_err_below_40 + num_err_below_30

    row0 = ['Point names', '<=20', '<=25', '<=30', '<=40', 'mean_err']
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    for i in range(0, points_num):
        # point_name = [k for k,v in point_list.items() if v == i+1]
        sheet1.write(i + 1, 0, point_list[i])
        sheet1.write(i + 1, 1, num_err_below_20[i] / (img_num + 1))
        sheet1.write(i + 1, 2, num_err_below_25[i] / (img_num + 1))
        sheet1.write(i + 1, 3, num_err_below_30[i] / (img_num + 1))
        sheet1.write(i + 1, 4, num_err_below_40[i] / (img_num + 1))
        sheet1.write(i + 1, 5, loss[i])
        accuracy_bellow_20.append(num_err_below_20[i] / (img_num + 1))
        accuracy_bellow_25.append(num_err_below_25[i] / (img_num + 1))
        accuracy_bellow_30.append(num_err_below_30[i] / (img_num + 1))
        accuracy_bellow_40.append(num_err_below_40[i] / (img_num + 1))

    sheet1.write(points_num + 1, 0, 'AVERAGE')
    sheet1.write(points_num + 1, 1, Average(accuracy_bellow_20))
    sheet1.write(points_num + 1, 2, Average(accuracy_bellow_25))
    sheet1.write(points_num + 1, 3, Average(accuracy_bellow_30))
    sheet1.write(points_num + 1, 4, Average(accuracy_bellow_40))

    f.save(save_path)

    return Average(accuracy_bellow_20)

def predict(model, input_dir, output_json_dir, point_list, device='gpu', visualization=False):

    points_num = len(point_list)
    list_img = os.listdir(input_dir + 'images/')
    count = 0

    model.to(device)
    model.eval()

    start_time = time.time()

    CVM_1 = []
    CVM_2 = []
    CVM_3 = []
    CVM_4 = []
    CVM_5 = []
    CVM_6 = []

    for img in list_img:
        print('image: ', img)
        img_path = input_dir + 'images/' + img
        img_name = img.split('.jpeg')[0]


        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        img_resize = cv2.resize(img, (Config.resize_w, Config.resize_h))
        output_image = img_resize.copy()
        img_data = np.transpose(img_resize, (2, 0, 1))
        img_data = np.reshape(img_data, (1, 3, Config.resize_h, Config.resize_w))
        img_data = torch.from_numpy(img_data).float()

        scal_ratio_w = img_w / Config.resize_w
        scal_ratio_h = img_h / Config.resize_h

        img_data = img_data.to(device)

        outputs, _ = model(img_data)
        outputs = outputs[0].cpu().detach().numpy()

        pred = get_predict_point_from_heatmap(outputs, scal_ratio_w, scal_ratio_h, points_num)

        json_data = {"markerPoints": None, "contrast":0, "brightness":0, "color": ""}
        json_data["markerPoints"] = {}

        x_value = []
        y_value = []

        for j in range(points_num):

            ###### Write to JSON ######
            x, y = int(pred[j][0] / scal_ratio_w), int(pred[j][1] / scal_ratio_h) #### x,y of 512x480 image

            x_org, y_org = (int(pred[j][0])), (int(pred[j][1]))
            x_value.append(x)
            y_value.append(y)
            point_name = point_list[j]

        C2a = [int(x_value[0]), int(y_value[0])]
        C2m = [int(x_value[1]), int(y_value[1])]
        C2p = [int(x_value[2]), int(y_value[2])]
        C3ua = [int(x_value[3]), int(y_value[3])]
        C3up = [int(x_value[4]), int(y_value[4])]
        C3la = [int(x_value[5]), int(y_value[5])]
        C3m = [int(x_value[6]), int(y_value[6])]
        C3lp = [int(x_value[7]), int(y_value[7])]
        C4ua = [int(x_value[8]), int(y_value[8])]
        C4up = [int(x_value[9]), int(y_value[9])]
        C4la = [int(x_value[10]), int(y_value[10])]
        C4m = [int(x_value[11]), int(y_value[11])]
        C4lp = [int(x_value[12]), int(y_value[12])]


        C2_output = cross_distance_calculation(C2a, C2p, C2m, ratio=True)
        C3_output = cross_distance_calculation(C3la, C3lp, C3m, ratio=True)
        C4_output = cross_distance_calculation(C4la, C4lp, C4m, ratio=True)

        json_data = {"CVM_classification": None}
        json_data["CVM_classification"] = {}

        point_dict = {}
        point_dict['C2_d'] = C2_output
        point_dict["C3_d"] = C3_output
        point_dict["C4_d"] = C4_output
        json_data["CVM_classification"].update(point_dict)

        with open(output_json_dir + img_name + '.json', 'w') as fp:
            json.dump(json_data, fp)
        fp.close()

        if visualization:
            CVM_class_txt_filepath = input_dir + 'cvm/ ' + img_name + '.txt'
            with open(CVM_class_txt_filepath) as note:
                for line in note:
                    cvm_class = int(line)
                    if cvm_class == 1:
                        CVM_1.append([C2_output, C3_output, C4_output])
                    elif cvm_class == 2:
                        CVM_2.append([C2_output, C3_output, C4_output])
                    elif cvm_class == 3:
                        CVM_3.append([C2_output, C3_output, C4_output])
                    elif cvm_class == 4:
                        CVM_4.append([C2_output, C3_output, C4_output])
                    elif cvm_class == 5:
                        CVM_5.append([C2_output, C3_output, C4_output])
                    elif cvm_class == 6:
                        CVM_6.append([C2_output, C3_output, C4_output])
                    else:
                        print("The image {} has not annotated!".format(img_name))

        count+=1

    if visualization:
        plt.plot(CVM_1, np.ones_like(CVM_1)* 1, ls='dotted', c='b', lw=2)
        plt.plot(CVM_2, np.ones_like(CVM_2)* 2, ls='dotted', c='g', lw=2)
        plt.plot(CVM_3, np.ones_like(CVM_3)* 3, ls='dotted', c='r', lw=2)
        plt.plot(CVM_4, np.ones_like(CVM_4)* 4, ls='dotted', c='c', lw=2)
        plt.plot(CVM_5, np.ones_like(CVM_5)* 5, ls='dotted', c='m', lw=2)
        plt.plot(CVM_6, np.ones_like(CVM_6)* 6, ls='dotted', c='y', lw=2)

        plt.axis([0, 4, 0, 8])
        plt.savefig('CVM_plot.png')
        plt.show()

    averageFPS = 1.0 / ((time.time() - start_time) / count)
    print("FPS single images: ", averageFPS)







def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


