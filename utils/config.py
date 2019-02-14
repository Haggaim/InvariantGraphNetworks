import json
from easydict import EasyDict
import os
import datetime

NUM_LABELS = {'COLLAB':0, 'IMDBBINARY':0, 'IMDBMULTI':0, 'MUTAG':7, 'NCI1':37, 'NCI109':38, 'PROTEINS':3, 'PTC':22, 'DD':89}
NUM_CLASSES = {'COLLAB':3, 'IMDBBINARY':2, 'IMDBMULTI':3, 'MUTAG':2, 'NCI1':2, 'NCI109':2, 'PROTEINS':2, 'PTC':2}
LEARNING_RATES = {'COLLAB': 0.00256, 'IMDBBINARY': 0.00064, 'IMDBMULTI': 0.00064, 'MUTAG': 0.00512, 'NCI1':0.00256, 'NCI109':0.00256, 'PROTEINS': 0.00064, 'PTC': 0.00128}
DECAY_RATES = {'COLLAB': 0.7, 'IMDBBINARY': 0.4, 'IMDBMULTI': 0.7, 'MUTAG': 0.7, 'NCI1':0.7, 'NCI109':0.7, 'PROTEINS': 0.7, 'PTC': 0.6}
CHOSEN_EPOCH = {'COLLAB': 100, 'IMDBBINARY': 40, 'IMDBMULTI': 150, 'MUTAG': 130, 'NCI1': 99, 'NCI109': 99, 'PROTEINS': 20, 'PTC': 9}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.num_classes = NUM_CLASSES[config.dataset_name]
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join("../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.parent_dir, "checkpoint/")
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.learning_rate = LEARNING_RATES[config.dataset_name]
        config.decay_rate = DECAY_RATES[config.dataset_name]
    config.n_gpus = len(config.gpu.split(','))
    config.gpus_list = ",".join(['{}'.format(i) for i in range(config.n_gpus)])
    config.devices = ['/gpu:{}'.format(i) for i in range(config.n_gpus)]
    return config

if __name__ == '__main__':
    config = process_config('../configs/example.json')
    print(config.values())
