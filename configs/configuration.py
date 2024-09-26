class Config():

    CVM_data_dir = '/home/data/CVM/'

    data_type = 'public' #### Public, Private

    data_version = 'v1.0'
    light_weight_model = False
    label_type = 'xml'  ### txt, json, xml

    CVM_landmarks_experiment_name = '_IBSI-2015_'
    CVM_classification_experiment_name = 'Catboost'
    visualization = False
    visualize_output_on_training = False
    draw_output_image = True
    export_detect_point_to_json = False

    transform = False
    multi_gpus = False
    n_gpu = 2
    gpu_id = 0
    resume = False
    base_number = 40
    resize_h = 512
    resize_w = 480
    sigma = 10

    num_epochs = 110
    lr = 1e-4
    lr_step_milestones = [30, 60, 90]
    # lr_step_milestones = [0,1,2]
    debug_steps = 10
    validation_step = 5
    batch_size = 1
    save_weight_every_epoch = 5






