from BIGAN.bigan import BIGAN
from BIGAN.data_loader import load_keys
import datetime
from keras.optimizers import Adam
import numpy as np
import os


def normalize(X):
    return (X - 127.5) / 127.5


def denormalize(X):
    return ((X + 1.0) / 2.0 * 255.0).astype(dtype=np.uint8)


def log_plotter(train_path, optimizer, epochs, ETA, save_dir):
    save_path = os.path.join(save_dir, "log.txt")
    with open(save_path, 'a') as f:
        f.write("train paht: {}".format(train_path) + "\n")
        f.write("epoch　は {}".format(epochs) + "\n")
        f.write("処理にかかった時間は {}".format(str(ETA)) + '\n')
        f.write(optimizer.get_config() + "\n")


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    train_path = "/home/kiyo/Pictures/key_pictures/Dec_18_2018/dataset/2019_0127/train"
    test_path = "/home/kiyo/Pictures/key_pictures/Dec_18_2018/dataset/2019_0127/test"
    batch_size = 16
    epochs = 40000
    input_dim = 64
    NUMBER_OF_TESTS = 120

    # optimizer
    optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)

    ### 0. prepare data
    X_train, X_test, y_test = load_keys(train_dir_path=train_path, test_dir_path=test_path, NUMBER_OF_TESTS=120)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    input_shape = X_train[0].shape



    ### 1. learn generator & discriminator
    dcgan = BIGAN(optimizer=optimizer, input_shape=input_shape)
    # ファイル名を時間にする
    now = datetime.datetime.now()
    # 実行した日付のついたフォルダをrootにする
    save_root = os.path.join("./learn", "{0:%Y%m%d-%H%M%S}".format(now))
    save_result_dir = os.path.join(save_root, "result")
    save_weight_dir = os.path.join(save_root, "weights")
    os.makedirs(save_result_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)
    d_losses_r, d_losses_f, g_losses = dcgan.train(X_train, epochs, batch_size, save_result_dir, save_weight_dir)
    # ファイル名を時間にする

    # lossをcsvに書き込む
    # result, weightsと同じようにファイル名を日付で指定する。
    loss_dir = os.path.join(save_root, "loss")
    os.makedirs(loss_dir, exist_ok=True)
    loss_path = os.path.join(loss_dir, "loss.csv")
    with open(loss_path, 'w') as f:
        f.write("g_loss,d_loss_r,d_loss_f" + '\n')
        for g_loss, d_loss_r, d_loss_f in zip(g_losses, d_losses_r, d_losses_f):
            f.write(str(g_loss) + ',' + str(d_loss_r) + str(d_loss_f) + '\n')

    # lossのプロット
    # loss_plotter(loss_csv_path=loss_path)
    ETA = datetime.datetime.now() - start_time
    print("処理にかかった時間は {} ".format(ETA))
    # 学習に関する記録を保存
    log_plotter(train_path=train_path, optimizer=optimizer, epochs=epochs, ETA=ETA, save_dir=save_root)
