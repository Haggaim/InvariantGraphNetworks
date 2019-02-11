import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np



BASE_DIR = os.path.abspath("../experiments")


def write_to_file_doc(train_acc, train_loss, test_acc, test_loss, epoch, config):
    """
    creates if not exist and update sumary csv file of the the training
    """
    columns = ['experiment_name', 'epoch', 'train_loss', 'train_accuracy', 'test_loss', "test_accuracy", 'timestamp']
    directory = config.summary_dir
    if os.path.exists(directory + '/exp_summary.csv'):
        f = pd.read_csv(directory + '/exp_summary.csv')
    else:
        f = pd.DataFrame(columns=columns)
    val = [config.exp_name, epoch, train_loss, train_acc, test_loss, test_acc, config.timestamp]
    f = f.append(pd.DataFrame([val], columns=columns))
    f.to_csv(directory + '/exp_summary.csv')


def create_experiment_results_plot(title, parameter, dir, log=False):
    """
    create plot of chosen parameter during training
    :param title: the first part of plot title
    :param parameter: the parameter you want to plot. loss accuracy
    :param dir: the directory in which the plot will be saves
    :param log: boolean to chose if you want semilog scale
    :return: name of the saved plot file
    """
    df = pd.read_csv(filepath_or_buffer=dir + '/exp_summary.csv')
    epochs = df["epoch"]
    train_param = df["train_{0}".format(parameter)]

    test_param = df["test_{0}".format(parameter)]

    if log:
        plt.semilogy(epochs, train_param, 'r', label='train')
        plt.semilogy(epochs, test_param, 'b', label='test')
        plt.ylabel(parameter + " semilog")
        axes = plt.gca()
        axes.set_ylim([10 ** -3, 10 ** 1])
    else:
        plt.plot(epochs, train_param, 'r', label='train')
        plt.plot(epochs, test_param, 'b', label='test')
        plt.ylabel(parameter)
        axes = plt.gca()
        axes.set_ylim([0.1, 0.95])
    plt.xlabel('Epochs')


    plt.title(title)
    plt.legend()
    file_name = (os.path.join(dir, (title + parameter + ".png")))
    plt.savefig(file_name)

    plt.close()
    return file_name


'''def summary_10fold_results(directory, epochs, chosen_epoch):
    accuracy_arr = np.empty((epochs+1, 0))
    folders = get_folders_list(directory)
    for folder in folders:
        df = pd.read_csv(filepath_or_buffer=folder + "/exp_summary.csv")
        curr_acc = np.expand_dims(df["test_accuracy"].values, axis=1)
        accuracy_arr = np.append(accuracy_arr, curr_acc, axis=1)

    print(accuracy_arr.shape)
    print("Results")
    print(np.mean(accuracy_arr, axis=1)[chosen_epoch])
    print(accuracy_arr.std(axis=1)[chosen_epoch])'''



def get_folders_list(directory):
    r = []
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            r.append(os.path.join(directory, name))
    return r


def doc_results(acc, loss, exp, fold, summary_dir):
    columns = ['experiment_num', 'fold_num', "acc", "loss"]
    if os.path.exists(summary_dir + '/exp_summary.csv'):
        f = pd.read_csv(summary_dir + '/exp_summary.csv')
    else:
        f = pd.DataFrame(columns=columns)
    values = [exp, fold, acc, loss]
    f = f.append(pd.DataFrame([values], columns=columns))
    f.to_csv(summary_dir + '/exp_summary.csv')


def summary_10fold_results(summary_dir):
    df = pd.read_csv(summary_dir+"/exp_summary.csv")
    acc = np.array(df["acc"])
    print("Results")
    print("Mean Accuracy = {0}".format(np.mean(acc)))
    print("Mean std = {0}".format(np.std(acc)))

if __name__ =="__main__":
    print(BASE_DIR)


