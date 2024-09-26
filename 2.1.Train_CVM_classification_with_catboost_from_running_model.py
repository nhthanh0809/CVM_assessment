
import torch
import numpy as np
import csv
import os
from utils.utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from configs.configuration import Config
from data.points_labels_v2 import *
from model.CephaLandmark_v2 import CephaLandmark_v2
from model.CephaLandmark_v3 import CephaLandmark_v3

from utils.data_utils import *
from utils.utils import *
from utils.utils import cross_distance_calculation, euclid_distance_calculation, angle_calculation

import sklearn
import hyperopt
import catboost
from catboost import CatBoostClassifier


from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
import colorama

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'catboost-hyperopt-log.txt', 'w' )


CVM_CLASSES = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['border_count'] = space['border_count']
    #params['rsm'] = space['rsm']
    return params


def objective(space):
    global obj_call_count, cur_best_loss, log_writer

    obj_call_count += 1

    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count, cur_best_loss))

    params = get_catboost_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str))

    model = catboost.CatBoostClassifier(iterations=20000,
                                        learning_rate=params['learning_rate'],
                                        depth=int(params['depth']),
                                        loss_function='MultiClass',
                                        use_best_model=True,
                                        task_type="GPU",
                                        early_stopping_rounds=3000,
                                        border_count=int(params['border_count']),
                                        verbose=False
                                        )

    model.fit(D_train, eval_set=D_test, verbose=False)
    nb_trees = model.tree_count_

    print('nb_trees={}'.format(nb_trees))

    y_pred = model.predict_proba(D_test.get_features())

    print(y_pred)
    print(D_test.get_label())
    test_loss = sklearn.metrics.log_loss(D_test.get_label(), y_pred, labels=CVM_CLASSES)
    acc = sklearn.metrics.accuracy_score(D_test.get_label(), np.argmax(y_pred, axis=1))

    log_writer.write(
        'loss={:<7.5f} acc={} Params:{} nb_trees={}\n'.format(test_loss, acc, params_str, nb_trees))
    log_writer.flush()

    if test_loss < cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)

    return {'loss': test_loss, 'status': STATUS_OK}


if __name__ == '__main__':

    device = torch.device("cpu")

    CVM_CLASSES = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']

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
    if Config.draw_output_image:
        os.makedirs(final_inference_output_folder + 'images/', exist_ok=True)
        os.makedirs(final_inference_output_folder + 'cropped/', exist_ok=True)



    xml_annotation_path = gt_train_dir + 'annotations.xml'
    doc = ET.parse(xml_annotation_path)
    root = doc.getroot()
    imageEles = root.findall('image')

    if Config.multi_gpus:
        device, device_ids = prepare_device(Config.n_gpu)
        if len(device_ids) > 1:
            landmark_model = torch.nn.DataParallel(landmark_model, device_ids=device_ids)
    else:
        device_id = 'cuda:' + str(Config.gpu_id)
        device = torch.device(device_id if torch.cuda.is_available() else "cpu")


    checkpoint_list = os.listdir(landmark_best_checkpoint_folder)
    for checkpoint in checkpoint_list:
        if os.path.isfile(landmark_best_checkpoint_folder + checkpoint):
            # model = torch.load(Config.checkpoint_folder + checkpoint, map_location=device)
            ##### Load best checkpoint of landmark detection model #######
            print('==='*30)
            print('Loading best checkpoint of landmark detection model ........')
            print('Best checkpoint in: ' + landmark_best_checkpoint_folder + checkpoint)
            landmark_model.load_state_dict(torch.load(landmark_best_checkpoint_folder + checkpoint, map_location=device))
            print('Loaded best checkpoint successfully!')
            landmark_model.to(device)
            landmark_model.eval()

            train_x = []
            train_y = []

            val_x = []
            val_y = []

            """
            Get train_x, train_y data from train set
            """

            for imageName in os.listdir(train_img_dir):
                for imageEle in imageEles:
                    if imageName == imageEle.attrib['name']:
                        # print("Train image: ", imageName)

                        cvm_className = get_tagName_from_CVAT_xml(imageEle)
                        cvm_class = CVM_CLASSES.index(cvm_className)
                        imagePath = train_img_dir + imageName

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


                        outputs, _ = landmark_model(img_data)
                        outputs = outputs[0].cpu().detach().numpy()

                        pred = get_predict_point_from_heatmap(outputs, scal_ratio_w, scal_ratio_h, len(point_list))

                        C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle = calculate_CVM_metrics_from_inference_result(pred, point_list)

                        train_x.append([C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle])
                        # train_x.append([C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR])
                        train_y.append(cvm_class)


            """
            Get val_x, val_y data from validation set
            """
            for imageName in os.listdir(test_img_dir):
                for imageEle in imageEles:
                    if imageName == imageEle.attrib['name']:
                        # print("Val image: ", imageName)

                        cvm_className = get_tagName_from_CVAT_xml(imageEle)
                        cvm_class = CVM_CLASSES.index(cvm_className)

                        imagePath = test_img_dir + imageName

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

                        outputs, _ = landmark_model(img_data)
                        outputs = outputs[0].cpu().detach().numpy()
                        pred = get_predict_point_from_heatmap(outputs, scal_ratio_w, scal_ratio_h, len(point_list))

                        C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle = calculate_CVM_metrics_from_inference_result(pred, point_list)

                        val_x.append([C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR, C2Angle, C3Angle, C4Angle])
                        # train_x.append([C3PAR, C3BAR, C3BPR, C4PAR, C4BAR, C4BPR])
                        val_y.append(cvm_class)

            train_x = np.array(train_x)
            train_y = np.array(train_y)
            val_x = np.array(val_x)
            val_y = np.array(val_y)

            ################################## Create hyper parameters optimizer ############################

            N_HYPEROPT_PROBES = 100
            HYPEROPT_ALGO = tpe.suggest
            colorama.init()

            obj_call_count = 0
            cur_best_loss = np.inf

            D_train = catboost.Pool(train_x, train_y)
            D_test = catboost.Pool(val_x, val_y)

            space = {
                'depth': hp.quniform("depth", 4, 12, 1),
                'border_count': hp.uniform('border_count', 32, 255),
                'learning_rate': hp.loguniform('learning_rate', -5.0, -2),
                'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 16),
            }

            trials = Trials()
            best = hyperopt.fmin(fn=objective,
                                 space=space,
                                 algo=HYPEROPT_ALGO,
                                 max_evals=N_HYPEROPT_PROBES,
                                 trials=trials,
                                 verbose=True)

            print('===' * 30)
            print('The best params:')
            print(best)
            print('\n\n')

            best.update({'border_count': int(best['border_count'])})

            classification_model = catboost.CatBoostClassifier(iterations=40000,
                                                loss_function='MultiClass',
                                                use_best_model=True,
                                                task_type='CPU',
                                                early_stopping_rounds=500,
                                                od_type="Iter",
                                                verbose=2000,
                                                **best
                                                )

            classification_model.fit(D_train, eval_set=D_test, verbose=2000)

            train_pred = classification_model.predict(train_x)
            train_acc = accuracy_score(train_y, train_pred)
            print('Train Accuracy: ', train_acc)

            test_pred = classification_model.predict(val_x)
            test_acc = accuracy_score(val_y, test_pred)
            print('Test Accuracy:', test_acc)

            plot_confusion_matrix(val_y, test_pred, classes=np.array(CVM_CLASSES), normalize=True,
                                  title='Best model - Normalized confusion matrix | Accuracy: ' + str(test_acc))
            plt.savefig(CVM_classification_checkpoint_folder + 'Best_model_Normalized_confusion_matrix.png')

            plot_confusion_matrix(val_y, test_pred, classes=np.array(CVM_CLASSES), normalize=False,
                                  title='Best model - confusion matrix | Accuracy: ' + str(test_acc))
            plt.savefig(CVM_classification_checkpoint_folder + 'Best_model_confusion_matrix.png')


            model_name  = Config.CVM_classification_experiment_name
            classification_model.save_model(CVM_classification_checkpoint_folder + model_name)

            ## After you train the model using fit(), save like this -
            # model.save_model('model_name')  # extension not required.


            # # And then, later load -
            # from catboost import CatBoostClassifier
            #
            # model = CatBoostClassifier()  # parameters not required.
            # model.load_model('model_name')









