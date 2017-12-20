class DefaultConfig:
    imdb_csv_path = '/home/gdshen/datasets/face/processed/imdb.csv'
    imdb_csv_train = '/home/gdshen/datasets/face/processed/train.csv'
    imdb_csv_test = '/home/gdshen/datasets/face/processed/test.csv'
    wiki_csv_path = '/home/gdshen/datasets/face/processed/wiki.csv'
    # asian_csv_path = '/home/gdshen/datasets/face/asian/agegenderFilter_frontal.csv'
    # asian_csv_train = '/home/gdshen/datasets/face/asian/train.csv'
    # asian_csv_test = '/home/gdshen/datasets/face/asian/test.csv'
    # asian_csv_path = '/home/gdshen/datasets/face/asian/agegenderFilter_frontal.csv'
    asian_csv_train = '/home/gdshen/2017-03-01/train.csv'
    asian_csv_test = '/home/gdshen/2017-03-01/test.csv'
    asian_imgs_dir = '/home/gdshen/2017-03-01/154'

    whole_csv_train = '/home/gdshen/datasets/face/whole/train.csv'
    whole_csv_test = '/home/gdshen/datasets/face/whole/test.csv'
    whole_imgs_base_dir = '/home/gdshen/datasets/face/whole'

    batch_size = 10
    num_workers = 4
    log_interval = 100
    epoch = 30
    learning_rate = 0.0001
    momentum = 0.9
    checkpoint_interval = 10
    checkpoint_dir = '/home/gdshen/checkpoint/pytorch'
    logs_dir = '/home/gdshen/datasets/logs/'
    fc_learning_rate = 0.001
    weight_decay = 0.0005
    decay_epoches = 10
    decay_gamma = 0.1
    using_pretrain_model = True
    pretrain_model_path = '/home/gdshen/checkpoint/pytorch/checkpoint_whole-30.pth'
