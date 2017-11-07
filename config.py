class DefaultConfig:
    imdb_csv_path = '/home/gdshen/datasets/face/processed/imdb.csv'
    imdb_csv_train = '/home/gdshen/datasets/face/processed/train.csv'
    imdb_csv_test = '/home/gdshen/datasets/face/processed/test.csv'
    wiki_csv_path = '/home/gdshen/datasets/face/processed/wiki.csv'
    asian_csv_path = '/home/gdshen/datasets/face/asian/agegenderFilter_frontal.csv'
    asian_csv_train = '/home/gdshen/datasets/face/asian/train.csv'
    asian_csv_test = '/home/gdshen/datasets/face/asian/test.csv'
    asian_imgs_dir = '/home/gdshen/datasets/face/asian/images'
    batch_size = 10
    num_workers = 4
    log_interval = 100
    epoch = 50
    learning_rate = 0.0001
    momentum = 0.9
    checkpoint_interval = 10
    checkpoint_dir = '/home/gdshen/datasets/checkpoint'
    logs_dir = '/home/gdshen/datasets/logs/'
    fc_learning_rate = 0.001
    weight_decay = 0.0005
    decay_epoches = 10
    decay_gamma = 0.1
