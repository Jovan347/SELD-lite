# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        

        quick_test=False,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights = r"models/8_1_dev_split0_multiaccdoa_foa_model.h5", 

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir = r"C:\Users\gogomk\projects\seld-lite\data",

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir=r"C:\Users\gogomk\projects\seld-lite\runs\seld_feat_label_2022",
 
        model_dir='models/',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=32,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=64,        # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=1,
        fnn_size=64,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-4,

        # METRIC
        average = 'macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20

        
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['dataset_dir'] = r"C:\Users\gogomk\projects\seld-lite\data"
        params['feat_label_dir'] = r"C:\Users\gogomk\projects\seld-lite\runs\seld_feat_label_2022"
        params['unique_classes'] = 12
        params['nb_epochs'] = 30
        params['batch_size'] = 8
        params['patience'] = 20
        params['do_prune'] = False
        params['distill'] = True
        params['teacher_model_path'] = r"C:\Users\gogomk\projects\seld-lite\repos\seld-dcase2022-teacher\models\3_1_dev_split0_multiaccdoa_foa_model.h5"
        params['kd_alpha'] = 0.5


    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
    
    elif argv == '8':
        print("FOA + multi ACCDOA (STUDENT MODEL)\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

        #  Paths (same as teacher)
        params['dataset_dir'] = r"C:\Users\gogomk\projects\seld-lite\data"
        params['feat_label_dir'] = r"C:\Users\gogomk\projects\seld-lite\runs\seld_feat_label_2022"

        # Dataset
        params['unique_classes'] = 12

        # STUDENT CAPACITY 
        params['nb_cnn2d_filt'] = 32
        params['nb_rnn_layers'] = 1
        params['rnn_size'] = 64
        params['nb_fnn_layers'] = 1
        params['fnn_size'] = 64

        # Training
        params['batch_size'] = 8
        params['nb_epochs'] = 100
        params['patience'] = 20

        # COMPRESSION
        params['apply_dynamic_quant'] = True
        params['quant_dtype'] = 'qint8'  # or 'float16' sometimes
        params['dynamic_quant'] = True
        params['prune_amount'] = 0.3
        params['do_prune'] = True

        params['distill'] = True
        params['teacher_model_path'] = r"models\3_1_dev_split0_multiaccdoa_foa_model.h5"   # example; use your best teacher file
        params['kd_alpha'] = 0.5   # 0.0 = only GT, 1.0 = only teacher



    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params.setdefault('patience', int(params['nb_epochs']))     # Stop training if patience is reached

    #if '2020' in params['dataset_dir']:
     #   params['unique_classes'] = 14 
    #elif '2021' in params['dataset_dir']:
     #   params['unique_classes'] = 12
    #elif '2022' in params['dataset_dir']:
     #   params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
