import glob
import os
import random
import cv2
import numpy as np


def load_keys(train_dir_path, test_dir_path, NUMBER_OF_TESTS):
    # trainデータの作成trainデータのフォルダを渡すとそれをデータセットにする。
    train_path_list = glob.glob(train_dir_path + "/*")
    X_train = []
    for train_img_path in train_path_list:
        train_img = cv2.imread(train_img_path)
        X_train.append(train_img)
    X_train = np.array(X_train)

    # testデータの読み込み
    random.seed(0)
    normal_imgs = random.sample(glob.glob(test_dir_path + "/NORMAL/*"), NUMBER_OF_TESTS)
    dent_imgs = random.sample(glob.glob(test_dir_path + "/DENTED/*"), NUMBER_OF_TESTS)
    scratch_imgs = random.sample(glob.glob(test_dir_path + "/SCRATCHED/*"), NUMBER_OF_TESTS)
    X_test = []
    y_test = []
    # normal_imgのラベルは0
    for normal_img_path in normal_imgs:
        normal_img = cv2.imread(normal_img_path)
        X_test.append(normal_img)
        y_test.append(0)

    # dentのらベルルは1
    for dent_img_path in dent_imgs:
        dent_img = cv2.imread(dent_img_path)
        X_test.append(dent_img)
        y_test.append(1)

    # scratchのラベルは2
    for scrtch_img_path in scratch_imgs:
        scratch_img = cv2.imread(scrtch_img_path)
        X_test.append(scratch_img)
        y_test.append(2)
    # testデータ, testラベルをnumpy配列に
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # datasetの記録
    save_dataset(train_path_list, normal_imgs, dent_imgs, scratch_imgs)

    return X_train, X_test, y_test


def save_dataset(train_path_list, normal_imgs, dent_imgs, scratch_imgs):
    """
    データセットのパスをcsvに保存する
    :return:
    """
    import datetime
    now = datetime.datetime.now()
    dataset_dir = os.path.join("./dataset_path", "{0:%Y%m%d-%H%M%S}".format(now))
    os.makedirs(dataset_dir, exist_ok=True)
    # 使用したtrainデータをcsvに書き込み
    train_csv = os.path.join(dataset_dir, "train.csv")
    with open(train_csv, 'w') as f:
        f.write("train_path" + '\n')
        for train_path in train_path_list:
            f.write(str(train_path) + '\n')

    # 使用したtestデータをcsvに書き込み
    test_csv = os.path.join(dataset_dir, "test.csv")
    with open(test_csv, 'w') as f:
        f.write("normal_path,dent_path,scrtch_path" + '\n')
        for normal_img_path, dent_img_path, scratch_img_path in zip(normal_imgs, dent_imgs, scratch_imgs):
            f.write(str(normal_img_path) + ',' + str(dent_img_path) + ',' + str(scratch_img_path) + '\n')


if __name__ == '__main__':
    train_path = "/home/kiyo/Pictures/key_pictures/Dec_18_2018/dataset_path/learn"
    test_dir_path = ""
    X_train = load_keys(train_path)
