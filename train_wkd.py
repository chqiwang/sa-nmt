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


def wrap_scope(input_ckpt_path, output_ckpt_path, scope):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.variable_scope(scope):
                var_list = tf.contrib.framework.list_variables(input_ckpt_path)
                var_names, var_shapes = zip(*var_list)
                reader = tf.contrib.framework.load_checkpoint(input_ckpt_path)
                var_values = [reader.get_tensor(name) for name in var_names]
                new_var_list = [tf.get_variable(name, initializer=value)
                                for name, value in zip(var_names, var_values)]
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(new_var_list)
                saver.save(sess, output_ckpt_path)


def train(config, teacher_config):
    """Train a model with a config file."""
    logger = logging.getLogger('')
    data_reader = DataReader(config=config)
    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
    with tf.variable_scope('teacher'):
        teacher_model = eval(teacher_config.model)(config=teacher_config, num_gpus=0)
    model.build_train_model(test=config.train.eval_on_dev, teacher_model=teacher_model)

    train_op, loss_op = model.get_train_op(name=None)
    global_saver = tf.train.Saver([v for v in tf.global_variables() if not v.name.startswith('teacher')])

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    summary_writer = tf.summary.FileWriter(config.model_dir)

    with tf.Session(config=sess_config) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())

        # Reload teacher variables from disk.
        logger.info('Load teacher model parameters...')
        teacher_vars = tf.global_variables('teacher')
        teacher_saver = tf.train.Saver(var_list=teacher_vars)
        tmp_ckpt = '/tmp/teacher-{}.ckpt'.format(os.getpid())
        wrap_scope(tf.train.latest_checkpoint(teacher_config.model_dir), tmp_ckpt, 'teacher')
        teacher_saver.restore(sess, tmp_ckpt)
        for v in teacher_vars:
            logger.info('Reload {} from disk.'.format(v.name))

        # Reload student variables from disk.
        logger.info('Load student model parameters...')
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
    parser.add_argument('-t', '--teacher_config', dest='teacher_config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    teacher_config = AttrDict(yaml.load(open(args.teacher_config)))
    # Logger
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    logging.basicConfig(filename=config.model_dir + '/train.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # Train
    train(config, teacher_config)
