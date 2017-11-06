class DefaultConfig:
    imdb_csv_path = '/home/gdshen/datasets/face/processed/imdb.csv'
    wiki_csv_path = '/home/gdshen/datasets/face/processed/wiki.csv'
    asian_csv_path = '/home/gdshen/datasets/face/asian/agegenderFilter_frontal.csv'
    asian_imgs_dir = '/home/gdshen/datasets/face/asian/images'
    batch_size = 10
    num_workers = 1
    log_interval = 10
    epoch = 500
    learning_rate = 0.0001
    momentum = 0.9
    checkpoint_interval = 100
    checkpoint_dir = '/home/gdshen/datasets/checkpoint'
