from __future__ import print_function

import codecs
import commands
import os
import time
import logging
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from tempfile import mkstemp

import yaml

from models import *
from utils import DataReader, AttrDict, expand_feed_dict


def roll_back_to_previous_version(config):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            var_list = tf.contrib.framework.list_variables(config.model_dir)
            var_names, var_shapes = zip(*var_list)
            reader = tf.contrib.framework.load_checkpoint(config.model_dir)
            var_values = [reader.get_tensor(name) for name in var_names]
            new_var_list = []
            for name, value in zip(var_names, var_values):
                if name == 'encoder/src_embedding/kernel':
                    name = 'src_embedding'
                elif name == 'decoder/dst_embedding/kernel':
                    name = 'dst_embedding'
                elif name == 'decoder/softmax/kernel':
                    name = 'dst_softmax'
                new_var_list.append(tf.get_variable(name, initializer=value))
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(new_var_list)
            saver.save(sess, os.path.join(config.model_dir, 'new_version'))
    config.num_shards = 1


class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self):
        pass

    def init_from_config(self, config):
        self.model = eval(config.model)(config, config.test.num_gpus)
        self.model.build_test_model()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config)

        # Restore model.
        try:
            tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint(config.model_dir))
        except tf.errors.NotFoundError:
            roll_back_to_previous_version(config)
            tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint(config.model_dir))

        self.data_reader = DataReader(config)

    def init_from_frozen_graphdef(self, config):
        frozen_graph_path = os.path.join(config.model_dir, 'frozen_graph.pb')
        # If the file doesn't existed, create it.
        if not os.path.exists(frozen_graph_path):
            logging.warning('The frozen graph does not existed, use \'init_from_config\' instead'
                            'and create a frozen graph for next use.')
            self.init_from_config(config)
            saver = tf.train.Saver()
            save_dir = '/tmp/graph-{}'.format(os.getpid())
            os.mkdir(save_dir)
            save_path = '{}/ckpt'.format(save_dir)
            saver.save(sess=self.sess, save_path=save_path)

            with tf.Session(graph=tf.Graph()) as sess:
                clear_devices = True
                output_node_names = ['loss_sum', 'predictions']
                # We import the meta graph in the current default Graph
                saver = tf.train.import_meta_graph(save_path + '.meta', clear_devices=clear_devices)

                # We restore the weights
                saver.restore(sess, save_path)

                # We use a built-in TF helper to export variables to constants
                output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,  # The session is used to retrieve the weights
                    tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                    output_node_names  # The output node names are used to select the useful nodes
                )

                # Finally we serialize and dump the output graph to the filesystem
                with tf.gfile.GFile(frozen_graph_path, "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                    logging.info("%d ops in the final graph." % len(output_graph_def.node))

                # Remove temp files.
                os.system('rm -rf ' + save_dir)
        else:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.allow_soft_placement = True
            self.sess = tf.Session(config=sess_config)
            self.data_reader = DataReader(config)

            # We load the protobuf file from the disk and parse it to retrieve the
            # unserialized graph_def
            with tf.gfile.GFile(frozen_graph_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Import the graph_def into current the default graph.
            tf.import_graph_def(graph_def)
            graph = tf.get_default_graph()
            self.model = AttrDict()

            def collect_placeholders(prefix):
                ret = []
                idx = 0
                while True:
                    try:
                        ret.append(graph.get_tensor_by_name('import/{}_{}:0'.format(prefix, idx)))
                        idx += 1
                    except KeyError:
                        return tuple(ret)

            self.model['src_pls'] = collect_placeholders('src_pl')
            self.model['dst_pls'] = collect_placeholders('dst_pl')
            self.model['predictions'] = graph.get_tensor_by_name('import/predictions:0')

    def init_from_existed(self, model, sess, data_reader):
        self.sess = sess
        self.model = model
        self.data_reader = data_reader

    def beam_search(self, X):
        return self.sess.run(self.model.predictions, feed_dict=expand_feed_dict({self.model.src_pls: X}))

    def loss(self, X, Y):
        return self.sess.run(self.model.loss_sum, feed_dict=expand_feed_dict({self.model.src_pls: X, self.model.dst_pls: Y}))

    def translate(self, src_path, output_path, batch_size, keep_bpe_flag):
        logging.info('Translate %s.' % src_path)
        _, tmp = mkstemp()
        fd = codecs.open(tmp, 'w', 'utf8')
        count = 0
        token_count = 0
        epsilon = 1e-6
        start = time.time()
        for X in self.data_reader.get_test_batches(src_path, batch_size):
            Y = self.beam_search(X)
            Y = Y[:len(X)]
            sents = self.data_reader.indices_to_words(Y)
            assert len(X) == len(sents)
            for sent in sents:
                print(sent, file=fd)
            count += len(X)
            token_count += np.sum(np.not_equal(Y, 3))  # 3: </s>
            time_span = time.time() - start
            logging.info('{0} sentences ({1} tokens) processed in {2:.2f} minutes (speed: {3:.4f} sec/token).'.
                         format(count, token_count, time_span / 60, time_span / (token_count + epsilon)))
        fd.close()
        if not keep_bpe_flag:
            # Remove BPE flag, if have.
            os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, output_path))
            os.remove(tmp)
        else:
            os.system("mv {} {}" % (tmp, output_path))
        logging.info('The result file was saved in %s.' % output_path)

    def ppl(self, src_path, dst_path, batch_size):
        logging.info('Calculate PPL for %s and %s.' % (src_path, dst_path))
        token_count = 0
        loss_sum = 0
        for batch in self.data_reader.get_test_batches_with_target(src_path, dst_path, batch_size):
            X, Y = batch
            loss_sum += self.loss(X, Y)
            token_count += np.sum(np.greater(Y, 0))
        # Compute PPL
        ppl = np.exp(loss_sum / token_count)
        logging.info('PPL: %.4f' % ppl)
        return ppl

    def evaluate(self, batch_size, **kargs):
        """Evaluate the model on dev set."""
        src_path = kargs['src_path']
        output_path = kargs['output_path']
        cmd = kargs['cmd'] if 'cmd' in kargs else\
            "perl multi-bleu.perl {ref} < {output} 2>/dev/null | awk '{{print($3)}}' | awk -F, '{{print $1}}'"
        cmd = cmd.strip()
        logging.info('Evaluation command: ' + cmd)
        keep_bpe_flag = False
        if 'keep_bpe_flag' in kargs:
            keep_bpe_flag = kargs['keep_bpe_flag']
        self.translate(src_path, output_path, batch_size, keep_bpe_flag)
        bleu = None
        if 'ref_path' in kargs:
            ref_path = kargs['ref_path']
            try:
                bleu = commands.getoutput(cmd.format(**{'ref': ref_path, 'output': output_path}))
                bleu = float(bleu)
            except ValueError, e:
                logging.warning('An error raised when calculate BLEU: {}'.format(e))
                bleu = 0
            logging.info('BLEU: {}'.format(bleu))
        if 'dst_path' in kargs:
            self.ppl(src_path, kargs['dst_path'], batch_size)
        return bleu


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    logging.basicConfig(level=logging.INFO)
    evaluator = Evaluator()
    if config.test.frozen:
        evaluator.init_from_frozen_graphdef(config)
    else:
        evaluator.init_from_config(config)
    for attr in config.test:
        if attr.startswith('set'):
            evaluator.evaluate(config.test.batch_size, **config.test[attr])
    logging.info("Done")
