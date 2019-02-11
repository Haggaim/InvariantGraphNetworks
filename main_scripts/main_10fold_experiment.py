import os
from data_loader.data_generator import DataGenerator
from models.invariant_basic import invariant_basic
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments %s" % e)
        exit(0)

    os.environ["CUDA_VISIBLE_devices"] = config.gpu
    import tensorflow as tf
    import numpy as np
    tf.set_random_seed(100)
    np.random.seed(100)
    print("lr = {0}".format(config.learning_rate))
    print("decay = {0}".format(config.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    for exp in range(1, config.num_exp+1):
        for fold in range(1, 11):
            print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
            # create your data generator
            config.num_fold = fold
            data = DataGenerator(config)
            gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            gpuconfig.gpu_options.visible_device_list = config.gpu
            sess = tf.Session(config=gpuconfig)
            # create an instance of the model you want
            model = invariant_basic(config, data)
            # create trainer and pass all the previous components to it
            trainer = Trainer(sess, model, data, config)
            # here you train your model
            acc, loss = trainer.train()
            doc_utils.doc_results(acc, loss, exp, fold, config.summary_dir)
            sess.close()
            tf.reset_default_graph()

    doc_utils.summary_10fold_results(config.summary_dir)

if __name__ == '__main__':
    main()
