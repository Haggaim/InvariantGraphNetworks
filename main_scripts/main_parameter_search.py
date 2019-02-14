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

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    import tensorflow as tf
    import numpy as np
    tf.set_random_seed(100)
    np.random.seed(100)
    base_summary_folder = config.summary_dir
    base_exp_name = config.exp_name
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    for lr in [0.00008*(2**i) for i in range(8)]:
        for decay in [0.6, 0.7, 0.8, 0.9]:
            config.learning_rate = lr
            config.decay_rate = decay
            config.exp_name = base_exp_name + " lr={0}_decay={1}".format(lr, decay)
            curr_dir = os.path.join(base_summary_folder, "lr={0}_decay={1}".format(lr, decay))
            config.summary_dir = curr_dir
            create_dirs([curr_dir])
            # create your data generator
            data = DataGenerator(config)
            gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            gpuconfig.gpu_options.visible_device_list = config.gpus_list
            gpuconfig.gpu_options.allow_growth = True
            sess = tf.Session(config=gpuconfig)
            # create an instance of the model you want
            model = invariant_basic(config, data)
            # create trainer and pass all the previous components to it
            trainer = Trainer(sess, model, data, config)
            # here you train your model
            acc, loss = trainer.train()
            sess.close()
            tf.reset_default_graph()

    doc_utils.summary_10fold_results(config.summary_dir)

if __name__ == '__main__':
    main()
