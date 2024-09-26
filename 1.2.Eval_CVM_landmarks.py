import torch.optim as optim
from torch.utils.data import DataLoader

from configs.configuration import Config
from data.dataloader import CephaDataset
import numpy as np
import torch
from model.CephaLandmark_v2 import CephaLandmark_v2
from model.CephaLandmark_v3 import CephaLandmark_v3
from utils.utils import calculate_errors, prepare_device
import os
from data.points_labels_v2 import *


if __name__ == '__main__':

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
    os.makedirs(landmark_best_checkpoint_folder, exist_ok=True)
    os.makedirs(landmark_best_checkpoint_evaluation, exist_ok=True)
    os.makedirs(landmark_best_checkpoint_inference_folder, exist_ok=True)

    CVM_classification_checkpoint_folder = './output/' + experiment_name + '/CVM_classification/' + Config.CVM_classification_experiment_name + '/'
    os.makedirs(CVM_classification_checkpoint_folder, exist_ok=True)

    final_inference_output_folder = './output/' + experiment_name + '/final_inference_output/'

    os.makedirs(final_inference_output_folder + 'images/', exist_ok=True)
    os.makedirs(final_inference_output_folder + 'cropped/', exist_ok=True)

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            landmark_model = torch.nn.DataParallel(landmark_model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    evaluation_results = []
    checkpoint_name = []
    checkpoint_list = os.listdir(landmark_checkpoint_folder)
    for checkpoint in checkpoint_list:
        if os.path.isfile(landmark_checkpoint_folder + checkpoint):
            fileName = checkpoint.split('.pt')[0]
            output_filePath = landmark_evaluation_result_folder + fileName + '.xls'

            # model = torch.load(Config.checkpoint_folder + checkpoint, map_location=device)
            landmark_model.load_state_dict(torch.load(landmark_checkpoint_folder + checkpoint, map_location=device))
            landmark_model.eval()
            test_set = CephaDataset(test_img_dir, gt_test_dir, Config.label_type, Config.resize_h, Config.resize_w,
                                    point_list, Config.sigma, visualization=False)

            test_loader = DataLoader(dataset=test_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4)

            print("Evaluating model of CVM: " + checkpoint)
            avg_acc = calculate_errors(landmark_model, test_loader, gt_test_dir, Config.label_type, output_filePath, point_list,
                                       device)
            print("Average accuracy: ", avg_acc)
            print("======" * 30)
            evaluation_results.append(avg_acc)
            checkpoint_name.append(checkpoint)
        else:
            print("There is no checkpoint file here!")

    max_value = max(evaluation_results)
    max_index = evaluation_results.index(max_value)
    print("Best model of CVM is: ", checkpoint_name[max_index])

    #### Move best checkpoint to best_checkpoint_folder #####

    import shutil

    src_ckp = landmark_checkpoint_folder + checkpoint_name[max_index]
    dst_ckp = landmark_best_checkpoint_folder + checkpoint_name[max_index]

    shutil.copy(src_ckp, dst_ckp)

    xls_fileName = checkpoint_name[max_index].split('.pt')[0] + '.xls'
    src_xls = landmark_evaluation_result_folder + xls_fileName
    dst_xls = landmark_best_checkpoint_evaluation + xls_fileName

    shutil.copy(src_xls, dst_xls)


    for checkpoint in checkpoint_list:
        if os.path.isfile(landmark_checkpoint_folder + checkpoint):
            os.remove(landmark_checkpoint_folder + checkpoint)







