#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SYNOPSIS

	python SCRIPT.py [-h,--help] [-v,--verbose]


DESCRIPTION

	Concisely describe the purpose this script serves.


ARGUMENTS

	-h, --help		show this help message and exit
	-v, --verbose		verbose output


AUTHOR

	Doug McGeehan


LICENSE

	Copyright 2017 Doug McGeehan - GNU GPLv3

"""

__appname__ = "cnn"
__author__ = "Doug McGeehan"
__version__ = "0.0pre0"
__license__ = "GNU GPLv3"
__dev__ = True # used for debug messages in logs


import argparse
from datetime import datetime
import sys
import os
import logging

logger = logging.getLogger(__appname__)


IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']


def main(args):
    archive_files = set()
    for db in args.datasets:
        if os.path.isdir(db):
            archive_files.update(check_for_dataset(indir=db))
        elif os.path.isfile(db) and is_archive(db):
            archive_files.add(db)
        else:
            raise 'Unable to handle file type: {}'.format(db)
        
        logger.debug('Dataset file: {}'.format(db))


def check_for_dataset(indir):
    files = [f for f in os.listdir(indir) \
             if os.path.isfile(os.path.join(indir, f)) and \
             is_archive(f)]
    if not files:
        raise 'No archive files found within {}'.format(indir)
    logger.debug('Archive files: {}'.format(files))
    return files


def is_archive(filename):
    extension = filename.split('.')[-1]
    if extension.lower() in IMPLEMENTED_ARCHIVE_EXTENSIONS:
        return True
    else:
        return False

def setup_logger(args):
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', __appname__ + '.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    ch.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Description printed to command-line if -h is called."
    )
    # during development, I set default to False so I don't have to keep
    # calling this with -v
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=__dev__, help='verbose output')

    def path(*args):
        abspath = os.path.dirname(os.path.abspath(__file__))
        if len(args) == 0: # assume it was called for cwd
            pass # leave as cwd

        else:
            initial_path = os.path.join(*args)
            if not os.path.isabs(initial_path):
                abspath = os.path.join(abspath, initial_path)

        assert os.path.exists(abspath), "Path doesn't exist: {}".format(
            abspath
        )
        return abspath

    parser.add_argument('datasets', metavar='DATASET', nargs='+', type=path,
                        help='verbose output')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    try:
        start_time = datetime.now()

        args = get_arguments()
        setup_logger(args)
        logger.debug('Command-line arguments:')
        for arg in vars(args):
            value = getattr(args, arg)
            logger.debug('\t{argument_key}:\t{value}'.format(argument_key=arg,
                                                           value=value))

        logger.debug(start_time)

        main(args)

        finish_time = datetime.now()
        logger.debug(finish_time)
        logger.debug('Execution time: {time}'.format(
            time=(finish_time - start_time)
        ))
        logger.debug("#" * 20 + " END EXECUTION " + "#" * 20)

        sys.exit(0)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Something happened and I don't know what to do D:")
        sys.exit(1)
