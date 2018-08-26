import os
import time
import logging
from argparse import ArgumentParser
import tensorflow as tf
import yaml

from evaluate import Evaluator
from models import *
from utils import DataReader, AttrDict, available_variables, expand_feed_dict


class BreakLoopException(Exception):
    pass


def train(config):
    """Train a model with a config file."""
    logger = logging.getLogger('')
    data_reader = DataReader(config=config)
    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
    model.build_train_model(test=config.train.eval_on_dev)

    train_op, loss_op = model.get_train_op(name=None)
    global_saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    summary_writer = tf.summary.FileWriter(config.model_dir)

    with tf.Session(config=sess_config) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        # Reload variables from disk.
        if tf.train.latest_checkpoint(config.model_dir):
            available_vars = available_variables(config.model_dir)
            if available_vars:
                saver = tf.train.Saver(var_list=available_vars)
                saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
                for v in available_vars:
                    logger.info('Reload {} from disk.'.format(v.name))
            else:
                logger.info('Nothing to be reload from disk.')
        else:
            logger.info('Nothing to be reload from disk.')

        evaluator = Evaluator()
        evaluator.init_from_existed(model, sess, data_reader)

        global dev_bleu, toleration
        dev_bleu = evaluator.evaluate(**config.dev) if config.train.eval_on_dev else 0
        toleration = config.train.toleration

        def train_one_step(batch, loss_op, train_op):
            feed_dict = expand_feed_dict({model.src_pls: batch[0], model.dst_pls: batch[1]})
            step, lr, loss, _ = sess.run(
                [model.global_step, model.learning_rate,
                 loss_op, train_op],
                feed_dict=feed_dict)
            if step % config.train.summary_freq == 0:
                summary = sess.run(model.summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step)
            return step, lr, loss

        def maybe_save_model():
            global dev_bleu, toleration

            def save():
                mp = config.model_dir + '/model_step_{}'.format(step)
                global_saver.save(sess, mp)
                logger.info('Save model in %s.' % mp)

            if config.train.eval_on_dev:
                new_dev_bleu = evaluator.evaluate(**config.dev)
                if config.train.toleration is None:
                    save()
                else:
                    if new_dev_bleu >= dev_bleu:
                        save()
                        toleration = config.train.toleration
                        dev_bleu = new_dev_bleu
                    else:
                        toleration -= 1
            else:
                save()

        try:
            step = 0
            for epoch in range(1, config.train.num_epochs+1):
                for batch in data_reader.get_training_batches(epoches=1):

                    # Train normal instances.
                    start_time = time.time()
                    step, lr, loss = train_one_step(batch, loss_op, train_op)
                    logger.info(
                        'epoch: {0}\tstep: {1}\tlr: {2:.6f}\tloss: {3:.4f}\ttime: {4:.4f}'.
                        format(epoch, step, lr, loss, time.time() - start_time))
                    # Save model
                    if config.train.save_freq > 0 \
                       and step > 0 \
                       and step % config.train.save_freq == 0:
                        maybe_save_model()

                    if config.train.num_steps is not None and step >= config.train.num_steps:
                        raise BreakLoopException("BreakLoop")

                    if toleration is not None and toleration <= 0:
                        raise BreakLoopException("BreakLoop")

                # Save model per epoch if config.train.save_freq is less or equal than zero
                if config.train.save_freq <= 0:
                    maybe_save_model()
        except BreakLoopException as e:
            logger.info(e)

        logger.info("Finish training.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    logging.basicConfig(filename=config.model_dir + '/train.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # Train
    train(config)
