
import os
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

if __name__ == '__main__':

    point_list = []
    if Config.data_type == 'private':
        point_list = CVM_private_list
    elif Config.data_type == 'public':
        point_list = CVM_public_list

    points_num = len(point_list)

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
    os.makedirs(landmark_best_checkpoint_folder, exist_ok=True)
    os.makedirs(landmark_best_checkpoint_evaluation, exist_ok=True)
    os.makedirs(landmark_best_checkpoint_inference_folder, exist_ok=True)
    os.makedirs(landmark_best_checkpoint_inference_folder + 'images/', exist_ok=True)
    os.makedirs(landmark_best_checkpoint_inference_folder + 'cropped/', exist_ok=True)

    CVM_classification_checkpoint_folder = './output/' + experiment_name + '/CVM_classification/' + Config.CVM_classification_experiment_name + '/'
    os.makedirs(CVM_classification_checkpoint_folder, exist_ok=True)

    final_inference_output_folder = './output/' + experiment_name + '/final_inference_output/'

    os.makedirs(final_inference_output_folder + 'images/', exist_ok=True)
    os.makedirs(final_inference_output_folder + 'cropped/', exist_ok=True)

    input_image_dir = data_dir + 'train/images/'

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            landmark_model = torch.nn.DataParallel(landmark_model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    for checkpointName in os.listdir(landmark_best_checkpoint_folder):
        if os.path.isfile(best_checkpoint_folder + checkpointName):
            landmark_model.load_state_dict(torch.load(landmark_best_checkpoint_folder + checkpointName, map_location=device))
            landmark_model.to(device)
            landmark_model.eval()

            count = 0
            pre_data = []
            gt_data = []

            for image in os.listdir(input_image_dir):
                print(image)
                fileName = image.split('.jpeg')[0]
                imagePath = input_image_dir + image

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

                outputs, _ = model(img_data)
                outputs = outputs[0].cpu().detach().numpy()
                pred_landmark = get_predict_point_from_heatmap(outputs, scal_ratio_w, scal_ratio_h, len(point_list))

                if Config.draw_output_image:
                    for j in range(points_num):
                        x, y = int(pred_landmark[j][0]), int(pred_landmark[j][1])
                        point_name = point_list[j]
                        ########### Draw prediction point ############
                        img = cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=3)
                        img = cv2.putText(img, str(point_name), (x - 18, y - 18), cv2.FONT_HERSHEY_COMPLEX, 1,
                                          (255, 0, 0), 1, cv2.LINE_AA)

                    cv2.imwrite(landmark_best_checkpoint_inference_folder + 'images/' + fileName + '.jpeg', img)
                    cv2.imwrite(landmark_best_checkpoint_inference_folder + 'cropped/' + fileName + '_cropped.jpeg', cropped_image)
