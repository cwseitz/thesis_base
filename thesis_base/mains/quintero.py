import argparse
import collections
import numpy as np
import cellquant as cq
from thesis_base.utils import ConfigParser

def main(config):

    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Simple Config')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []
    config = ConfigParser.from_args(args, options)
    main(config)
