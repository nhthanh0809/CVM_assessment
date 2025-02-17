
import os

import numpy as np
import torch
import time
import cv2
from configs.configuration import Config
from data.points_labels_v2 import *
from model.CephaLandmark_v2 import CephaLandmark_v2
from model.CephaLandmark_v3 import CephaLandmark_v3
from utils.utils import prepare_device
from utils.data_utils import *
from utils.utils import cross_distance_calculation, euclid_distance_calculation, angle_calculation
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.utils import plot_confusion_matrix
from catboost import CatBoostClassifier, Pool
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    device = torch.device("cpu")

    CVM_CLASSES = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']

    input_dir = '/media/prdcv193/Data1/ThanhNH/Ceph/data/CVM/public/v1.0/'
    input_image_dir = input_dir + 'test/images/'
    annotation_file = input_dir + 'annotations.xml'

    point_list = []
    if Config.data_type == 'private':
        point_list = CVM_private_list
    elif Config.data_type == 'public':
        point_list = CVM_public_list

    if Config.light_weight_model:
        landmark_model = CephaLandmark_v3(points_num=len(point_list))
        landmark_model_version = 'Ceph_v3'
    else:
        landmark_model = CephaLandmark_v2(points_num=len(point_list))
        landmark_model_version = 'Ceph_v2'

    data_version = Config.data_version
    data_dir = Config.CVM_data_dir + Config.data_type + '/' + data_version + '/'

    train_img_dir = data_dir + 'train/images/'
    valid_img_dir = data_dir + 'test/images/'
    test_img_dir = data_dir + 'test/images/'

    if Config.label_type == 'json':
        gt_train_dir = data_dir + 'train/points/'
        gt_valid_dir = data_dir + 'test/points/'
        gt_test_dir = data_dir + 'test/points/'
    elif Config.label_type == 'xml':
        gt_train_dir = data_dir
        gt_valid_dir = data_dir
        gt_test_dir = data_dir

    experiment_name = Config.data_type + '_' + Config.data_version + Config.CVM_landmarks_experiment_name + landmark_model_version

    landmark_checkpoint_folder = './output/' + experiment_name + '/CVM_landmark/trained_models/'
    landmark_evaluation_result_folder = './output/' + experiment_name + '/CVM_landmark/evaluation_results/'

    landmark_best_checkpoint_folder = './output/' + experiment_name + '/CVM_landmark/best_model/'
    landmark_best_checkpoint_evaluation = './output/' + experiment_name + '/CVM_landmark/best_model/evaluation_results/'
    landmark_best_checkpoint_inference_folder = './output/' + experiment_name + '/CVM_landmark/best_model/inference_results/'

    CVM_classification_checkpoint_folder = './output/' + experiment_name + '/CVM_classification/' + Config.CVM_classification_experiment_name + '/'
    final_inference_output_folder = './output/' + experiment_name + '/final_inference_output/'
    if Config.draw_output_image:
        os.makedirs(final_inference_output_folder + 'images/', exist_ok=True)
        os.makedirs(final_inference_output_folder + 'cropped/', exist_ok=True)

    model_cvm_landmark = CephaLandmark_v2(points_num=len(CVM_public_list))
    model_cvm_classification = CatBoostClassifier()

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            model_cvm_landmark = torch.nn.DataParallel(model_cvm_landmark, device_ids=device_ids)

    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    checkpoint_list = os.listdir(landmark_best_checkpoint_folder)

    pred_y = []
    GT_y = []

    for checkpoint in checkpoint_list:
        if checkpoint.endswith('.pt'):
            # device = 'cuda:0'
            model_cvm_landmark.load_state_dict(
                torch.load(landmark_best_checkpoint_folder + checkpoint, map_location=device))
            model_cvm_classification.load_model(
                CVM_classification_checkpoint_folder + Config.CVM_classification_experiment_name)
            model_cvm_landmark.to(device)
            model_cvm_landmark.eval()

            image_list = os.listdir(input_image_dir)
            points_num = len(point_list)



            best_accuracy = 0
            for imageName in image_list:
                print(imageName)
                fileName = imageName.split('.jpeg')[0]
                imagePath = input_image_dir + imageName

                # cvm_file_path = input_dir + 'cvm/' + fileName + '.txt'
                img = cv2.imread(imagePath)
                img_h, img_w, _ = img.shape
                img_resize = cv2.resize(img, (Config.resize_w, Config.resize_h))
                output_image = img_resize.copy()
                img_data = np.transpose(img_resize, (2, 0, 1))
                img_data = np.reshape(img_data, (1, 3, Config.resize_h, Config.resize_w))
                img_data = torch.from_numpy(img_data).float()

                scal_ratio_w = img_w / Config.resize_w
                scal_ratio_h = img_h / Config.resize_h

                img_data = img_data.to(device)

                outputs, _ = model_cvm_landmark(img_data)
                outputs = outputs[0].cpu().detach().numpy()
                pred_landmark = get_predict_point_from_heatmap(outputs, scal_ratio_w, scal_ratio_h, points_num)

                C2a = [(pred_landmark[0][0]), (pred_landmark[0][1])]
                C2m = [(pred_landmark[1][0]), (pred_landmark[1][1])]
                C2p = [(pred_landmark[2][0]), (pred_landmark[2][1])]
                C3ua = [(pred_landmark[3][0]), (pred_landmark[3][1])]
                C3up = [(pred_landmark[4][0]), (pred_landmark[4][1])]
                C3la = [(pred_landmark[5][0]), (pred_landmark[5][1])]
                C3m = [(pred_landmark[6][0]), (pred_landmark[6][1])]
                C3lp = [(pred_landmark[7][0]), (pred_landmark[7][1])]
                C4ua = [(pred_landmark[8][0]), (pred_landmark[8][1])]
                C4up = [(pred_landmark[9][0]), (pred_landmark[9][1])]
                C4la = [(pred_landmark[10][0]), (pred_landmark[10][1])]
                C4m = [(pred_landmark[11][0]), (pred_landmark[11][1])]
                C4lp = [(pred_landmark[12][0]), (pred_landmark[12][1])]

                ###### Calculate CVM features #####

                C2Angle = angle_calculation(C2a, C2p, C2m)
                C3Angle = angle_calculation(C3la, C3lp, C3m)
                C4Angle = angle_calculation(C4la, C4lp, C4m)

                C2Conc = cross_distance_calculation(C2a, C2p, C2m, ratio=False)
                C3Conc = cross_distance_calculation(C3la, C3lp, C3m, ratio=False)
                C4Conc = cross_distance_calculation(C4la, C4lp, C4m, ratio=False)

                C3PAR = euclid_distance_calculation(C3up, C3lp) / euclid_distance_calculation(C3ua, C3la)
                C3BAR = euclid_distance_calculation(C3lp, C3la) / euclid_distance_calculation(C3ua, C3la)
                C3BPR = euclid_distance_calculation(C3lp, C3la) / euclid_distance_calculation(C3up, C3lp)

                C4PAR = euclid_distance_calculation(C4up, C4lp) / euclid_distance_calculation(C4ua, C4la)
                C4BAR = euclid_distance_calculation(C4lp, C4la) / euclid_distance_calculation(C4ua, C4la)
                C4BPR = euclid_distance_calculation(C4lp, C4la) / euclid_distance_calculation(C4up, C4lp)

                input_data = []
                input_data.append([C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle])

                input_data = np.array(input_data)

                # input_data = torch.tensor(input_data).float()

                pred_classification = model_cvm_classification.predict_proba(input_data)
                # print(np.argmax(pred_classification))
                pred_y.append(np.argmax(pred_classification))

                if os.path.isfile(annotation_file):
                    doc = ET.parse(annotation_file)
                    root = doc.getroot()
                    xml_imageEles = root.findall('image')
                    for imageEle in xml_imageEles:
                        if imageName == imageEle.attrib['name']:
                            cvm_className = get_tagName_from_CVAT_xml(imageEle)
                            # print(cvm_className)
                            cvm_class = CVM_CLASSES.index(cvm_className)
                            GT_y.append(cvm_class)


    accuracy = accuracy_score(GT_y, pred_y)
    print("Accuracy: ", accuracy * 100)
    plot_confusion_matrix(np.array(GT_y), np.array(pred_y), classes=np.asarray(CVM_CLASSES), normalize=True,
                          title='Best model - Normalized confusion matrix | Accuracy: ' + str(accuracy* 100))

    plt.savefig(final_inference_output_folder + 'Best_model_Normalized_confusion_matrix.png')

    plot_confusion_matrix(np.array(GT_y), np.array(pred_y), classes=np.asarray(CVM_CLASSES), normalize=False,
                          title='Best model - confusion matrix | Accuracy: ' + str(accuracy* 100))
    plt.savefig(final_inference_output_folder + 'Best_model_confusion_matrix.png')











