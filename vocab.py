import codecs
import logging
import os
from argparse import ArgumentParser
from collections import Counter

import yaml

from utils import AttrDict


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        words = l.split()
        word2cnt.update(Counter(words))
    word2cnt.update({"<PAD>":   10000000000000,
                     "<UNK>":   1000000000000,
                     "<S>":     100000000000,
                     "</S>":    10000000000})
    with codecs.open(fname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    logging.basicConfig(level=logging.INFO)
    if os.path.exists(config.src_vocab):
        logging.info('Source vocab already exists at {}'.format(config.src_vocab))
    else:
        make_vocab(config.train.src_path, config.src_vocab)
    if os.path.exists(config.dst_vocab):
        logging.info('Destination vocab already exists at {}'.format(config.dst_vocab))
    else:
        make_vocab(config.train.dst_path, config.dst_vocab)
    logging.info("Done")
