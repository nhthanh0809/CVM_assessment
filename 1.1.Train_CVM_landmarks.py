
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import prepare_device
from configs.configuration import Config
from data.dataloader import CephaDataset
from model.CephaLandmark_v2 import CephaLandmark_v2
from model.CephaLandmark_v3 import CephaLandmark_v3
from data.points_labels_v2 import *
from utils.utils import *
from utils.data_utils import *
import numpy as np
import torch
import os
import torch.nn as nn
import time





train_iterations = 0
valid_iterations = 0



def export_training_config(txt_export_path):
    with open(txt_export_path, 'w') as f:
        for item in vars(Config).items():
            line = str(item[0]) + ': ' + str(item[1]) + '\n'
            f.write(line)


def train_model_single_epoch(model, criterion, optimizer, lr_scheduler, train_loader, epoch_num, checkpoint_folder, checkpoint_name, writer, debug_steps=10, device='cpu'):

    model.train()
    loss_temp = 0
    training_total_loss = 0.0
    training_loss = 0.0
    training_loss_refine = 0.0

    steps = 0
    for i, (img, heatmaps, heatmaps_refine, img_name, x_all, y_all, scal_ratio_w, scal_ratio_h) in enumerate(train_loader):

        img = img.to(device)
        heatmaps = heatmaps.to(device)
        heatmaps_refine = heatmaps_refine.to(device)
        outputs, outputs_refine = model(img)


        ### Visualize heatmap during training ###

        if Config.visualize_output_on_training:
            temp_outputs = outputs[0].cpu().detach().numpy()
            pred = get_predict_point_from_heatmap(temp_outputs, scal_ratio_w, scal_ratio_h, len(temp_outputs), visualization=True)

        loss = criterion(outputs, heatmaps)
        ratio = torch.pow(Config.base_number, heatmaps)
        loss = torch.mul(loss, ratio)
        loss = torch.mean(loss)
        loss_temp += loss

        loss_refine = criterion(outputs_refine, heatmaps_refine)
        ratio_refine = torch.pow(Config.base_number, heatmaps_refine)
        loss_refine = torch.mul(loss_refine, ratio_refine)
        loss_refine = torch.mean(loss_refine)

        total_loss = loss + loss_refine
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        training_total_loss += total_loss.item()
        training_loss += loss.item()
        training_loss_refine += loss_refine.item()

        global train_iterations
        train_iterations +=1

        writer.add_scalar('model/trạining_total_loss', total_loss.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('model/training_loss', loss.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('model/training_loss_refine', loss_refine.cpu().detach().numpy(), train_iterations)
        writer.add_scalar('model/learning rate', lr_scheduler.get_last_lr()[0], train_iterations)

        steps += 1
        if i and i % debug_steps == 0:

            avg_total_loss = training_total_loss / steps
            avg_loss = training_loss / steps
            avg_loss_refine = training_loss_refine / steps
            print(
                f"Epoch: {epoch_num}, Step: {train_iterations - 1}, " +
                f"Average training Loss: {avg_loss:.4f}, " +
                f"Average training Loss refine {avg_loss_refine:.4f}, " +
                f"Average training Total Loss: {avg_total_loss:.4f}, " +
                f"Learning rate: {lr_scheduler.get_last_lr()[0]:.8f}"
            )

    lr_scheduler.step()

    if epoch % Config.save_weight_every_epoch == 0 or epoch == Config.num_epochs - 1:
        model_path = os.path.join(checkpoint_folder, checkpoint_name + f"-{epoch}.pt")
        torch.save(model.state_dict(), model_path)


def valid_model_by_epoch(model,valid_loader, epoch_num, writer, device='cpu'):
    model.eval()
    steps = 0
    validation_loss_temp = 0
    validation_total_loss = 0.0
    validation_loss = 0.0
    validation_loss_refine = 0.0
    with torch.no_grad():
        for i, (img, heatmaps, heatmaps_refine, img_name, x_all, y_all, _, _) in enumerate(valid_loader):

            img = img.to(device)
            heatmaps = heatmaps.to(device)
            heatmaps_refine = heatmaps_refine.to(device)
            outputs, outputs_refine = model(img)

            loss = criterion(outputs, heatmaps)
            ratio = torch.pow(Config.base_number, heatmaps)
            loss = torch.mul(loss, ratio)
            loss = torch.mean(loss)
            validation_loss_temp += loss

            loss_refine = criterion(outputs_refine, heatmaps_refine)
            ratio_refine = torch.pow(Config.base_number, heatmaps_refine)
            loss_refine = torch.mul(loss_refine, ratio_refine)
            loss_refine = torch.mean(loss_refine)

            total_loss = loss + loss_refine

            validation_total_loss += total_loss.item()
            validation_loss += loss.item()
            validation_loss_refine += loss_refine.item()

            global valid_iterations
            valid_iterations += 1

            writer.add_scalar('model/validation_total_loss', total_loss.cpu().detach().numpy(), valid_iterations)
            writer.add_scalar('model/validation_loss', loss.cpu().detach().numpy(), valid_iterations)
            writer.add_scalar('model/validation_loss_refine', loss_refine.cpu().detach().numpy(), valid_iterations)

            steps += 1

        avg_total_loss = validation_total_loss / steps
        avg_loss = validation_loss / steps
        avg_loss_refine = validation_loss_refine / steps
        print(
            f"[Validation - Epoch: {epoch_num}], " +
            f"Average validation Loss: {avg_loss:.4f}, " +
            f"Average validation Loss refine {avg_loss_refine:.4f}, " +
            f"Average validation Total Loss: {avg_total_loss:.4f}, "
        )


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



    #### write-out training config ####
    txt_config_path = './output/' + experiment_name + '/training_config.txt'
    export_training_config(txt_config_path)

    writer = SummaryWriter(landmark_checkpoint_folder + 'logs/')

    train_set = CephaDataset(train_img_dir,
                             gt_train_dir,
                             Config.label_type,
                             Config.resize_h,
                             Config.resize_w,
                             point_list,
                             Config.sigma,
                             transform=Config.transform,
                             visualization=Config.visualization)
    valid_set = CephaDataset(valid_img_dir,
                             gt_valid_dir,
                             Config.label_type,
                             Config.resize_h,
                             Config.resize_w,
                             point_list,
                             Config.sigma,
                             transform=Config.transform,
                             visualization=Config.visualization)

    train_loader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_set, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(landmark_model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss(reduction='none')
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, Config.lr_step_milestones, gamma=0.1, last_epoch=-1)

    for epoch in range(0, Config.num_epochs):

        train_model_single_epoch(model,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 train_loader=train_loader,
                                 epoch_num=epoch,
                                 checkpoint_folder=landmark_checkpoint_folder,
                                 checkpoint_name=experiment_name,
                                 writer=writer,
                                 debug_steps=Config.debug_steps,
                                 device=device)

        if epoch % Config.validation_step == 0 or epoch == Config.num_epochs - 1:
            valid_model_by_epoch(model,
                                 valid_loader=valid_loader,
                                 epoch_num=epoch,
                                 writer=writer,
                                 device=device)










