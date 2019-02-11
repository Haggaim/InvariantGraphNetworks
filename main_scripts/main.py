import os
from data_loader.data_generator import DataGenerator
from models.invariant_basic import invariant_basic
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs

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

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpuconfig.gpu_options.visible_device_list = config.gpu
    sess = tf.Session(config=gpuconfig)
    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    model = invariant_basic(config, data)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
