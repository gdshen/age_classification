import csv
import glob
import os
import shutil
from datetime import date
from math import ceil
from random import shuffle

import scipy.io


def extract_age_from_filename(filename: str) -> int:
    """filename example nm0000685_rm299994880_1938-12-29_2010.jpg
    文件名的问题：
    1. 照片的拍摄日期在人的出生日期之前
    2. 人的出生日期存在不合法的日期，比如day是0，还有这种670663_2015-02-16UTC08:04_1941形式， 32339345_1994-11-18

    imdb 7607
    wiki 1302 不合法
    """
    filename = filename.split('.')[0]  # remove file extension
    dob = filename.split('_')[-2]  # dob -> date of birth
    try:
        dob = date(*[int(t) for t in dob.split('-')])
    except TypeError as e:
        print(filename)
        return -1
    except ValueError as e:
        print(filename)
        return -1
    photo_taken = date(int(filename.split('_')[-1]), 7, 1)
    days = photo_taken.toordinal() - dob.toordinal()
    if days > 0:
        age = date.fromordinal(days).year
        return age
    else:
        return -1


def parsing_mat(mat, data_type: str, output_path: str):
    """
    parsing data information from mat file
    :param mat: matlab mat file
    :param data_type: 'imdb' | 'wiki'
    :param output_path
    :return: two lists
    """
    # dirty method to extract information from mat
    data = scipy.io.loadmat(mat)[data_type][0][0]
    basepath = os.path.dirname(mat)
    dob = data[0][0]
    photo_taken = data[1][0]
    full_path = data[2][0]  # get a single path full_path[i][0]

    # calculate age
    result = []
    error_list = []
    unknown_age_number = 0
    for i in range(dob.size):
        days = date(photo_taken[i], 7, 1).toordinal() - dob[i]
        if days > 0 and date.fromordinal(days).year <= 100:
            age = date.fromordinal(days).year
            result.append((os.path.join(basepath, full_path[i][0]), age))
        else:
            error_list.append((os.path.join(basepath, full_path[i][0]), days // 365))
            unknown_age_number += 1
    # imdb 560, wiki 1386 image <= 0
    # including large than 100, imdb 719, wiki 1879
    print('unknown age number {}'.format(unknown_age_number))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, data_type + '.csv'), 'w') as csvfile:
        path_age_writer = csv.writer(csvfile)
        path_age_writer.writerows(result)

    with open(os.path.join(output_path, data_type + '_error.csv'), 'w') as csvfile:
        path_age_error_writer = csv.writer(csvfile)
        path_age_error_writer.writerows(error_list)

    return result, error_list


def copy_according_to_csv(csv_path, input_dir, ouput_dir):
    with open(csv_path) as f:
        csvreader = csv.reader(f)
        csvreader = list(csvreader)[1:]
        filenames = [row[0].split('\\')[-1] for row in csvreader]

    for filename in filenames:
        shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))

    print(f'copy {len(filenames)} files')


def training_testing_split(csv_path, output_dir, portion):
    with open(csv_path) as f:
        csvreader = csv.reader(f)
        csvreader = list(csvreader)
        csvreader_header = csvreader[0]
        csvreader = csvreader[1:]
        shuffle(csvreader)
        portion = ceil(portion * len(csvreader))
        train_list = csvreader[:portion]
        test_list = csvreader[portion:]
        with open(os.path.join(output_dir, 'train.csv'), 'w') as train_writer:
            csvwriter = csv.writer(train_writer)
            csvwriter.writerows([csvreader_header] + train_list)

        with open(os.path.join(output_dir, 'test.csv'), 'w') as test_writer:
            csvwriter = csv.writer(test_writer)
            csvwriter.writerows([csvreader_header] + test_list)


def merge_and_random_split(csv_files_path, train_portion, train_csv_path, test_csv_path):
    # csv_files_path = '/home/gdshen/datasets/face/whole/split_age'
    csv_file_paths = glob.glob(csv_files_path + '/*.csv')
    # train_portion = 0.8
    train_files_paths = []
    test_files_paths = []
    for csv_file_path in csv_file_paths:
        with open(csv_file_path) as f:
            csv_file_reader = csv.reader(f)
            csv_file_reader = list(csv_file_reader)[1:]
            shuffle(csv_file_reader)

            boundary = int(train_portion * len(csv_file_reader))
            train_files_paths.extend(csv_file_reader[:boundary])
            test_files_paths.extend(csv_file_reader[boundary:])

    with open(train_csv_path, 'w') as f:
        train_csv_writer = csv.writer(f)
        train_csv_writer.writerows(train_files_paths)

    with open(test_csv_path, 'w') as f:
        test_csv_writer = csv.writer(f)
        test_csv_writer.writerows(test_files_paths)


if __name__ == '__main__':
    # Process IMDB-WIKI
    # imdb_mat = '/home/gdshen/datasets/face/imdb_crop/imdb.mat'
    # wiki_mat = '/home/gdshen/datasets/face/wiki_crop/wiki.mat'
    # parsing_mat(imdb_mat, 'imdb', output_path='/home/gdshen/datasets/face/processed')
    # parsing_mat(wiki_mat, 'wiki', output_path='/home/gdshen/datasets/face/processed')

    # Process Asian Data
    # csv_path = '/home/gdshen/datasets/face/asian/agegenderFilter_frontal.csv'
    # input_dir = '/home/gdshen/datasets/face/asian/yueda889'
    # output_dir = '/home/gdshen/datasets/face/asian/images'
    # copy_according_to_csv(csv_path, input_dir, output_dir)

    # split asian train test
    # csv_path = '/home/gdshen/datasets/face/asian/agegenderFilter_frontal.csv'
    # output_dir = '/home/gdshen/datasets/face/asian/'
    # portion = 0.7
    # training_testing_split(csv_path, output_dir, portion)

    # split imdb train test
    # csv_path = '/home/gdshen/datasets/face/processed/imdb.csv'
    # output_dir = '/home/gdshen/datasets/face/processed'
    # portion = 0.7
    # training_testing_split(csv_path, output_dir, portion)

    # merge split age and random shuffle age
    merge_and_random_split('/home/gdshen/datasets/face/whole/split_age', 0.8,
                           '/home/gdshen/datasets/face/whole/train.csv', '/home/gdshen/datasets/face/whole/test.csv')
