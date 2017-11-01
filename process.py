from datetime import date
import scipy.io
import os


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


def parsing_mat(mat, data_type: str):
    """
    parsing data information from mat file
    :param mat: matlab mat file
    :param data_type: 'imdb' | 'wiki'
    :return: file path with age
    """
    # dirty method to extract information from mat
    data = scipy.io.loadmat(mat)[data_type][0][0]
    basepath = os.path.basename(mat)
    dob = data[0][0]
    photo_taken = data[1][0]
    full_path = data[2][0]  # get a single path full_path[i][0]

    # calculate age
    result = []
    unknown_age_number = 0
    for i in range(dob.size):
        days = date(photo_taken[i], 7, 1).toordinal() - dob[i]
        if days > 0:
            age = date.fromordinal(days).year
            result.append((os.path.join(basepath, full_path[i][0]), age))
        else:
            unknown_age_number += 1
    # imdb 560, wiki 1386 image <= 0
    print('unknown age number {}'.format(unknown_age_number))
    return result


if __name__ == '__main__':
    # imdb_dir = '/home/gdshen/datasets/face/imdb_crop'
    # wiki_dir = '/home/gdshen/datasets/face/wiki_crop'
    #
    # print(imdb_dir)
    # print(wiki_dir)
    # i = 0
    # for dirpath, dirnames, filenames in os.walk(imdb_dir):
    #     for filename in filenames:
    #         if filename.endswith('.jpg'):
    #             if extract_age_from_filename(filename) == -1:
    #                 # print(filename)
    #                 i += 1
    #
    # print(i)
    matfile = '/home/gdshen/datasets/face/imdb_crop/imdb.mat'
    matfile = '/home/gdshen/datasets/face/wiki_crop/wiki.mat'
    parsing_mat(matfile, 'wiki')
    pass
