import os
import logging
import subprocess
import shutil

try:  # this is my own package, but it might not be present
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30

IMPLEMENTED_ARCHIVE_EXTENSIONS = ['zip', 'tgz']
EXTRACTOR_SCRIPT_SOURCE = 'http://askubuntu.com/a/338759'
EXTRACTOR_SCRIPT = 'extract.sh'
logger = logging.getLogger(__name__)


def get_dataset_dir(dataset_dir):
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), dataset_dir)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    return dataset_dir


def get_datasets(indir):
    files = set()
    for dirpath, _, filenames in os.walk(indir):
        files.update([os.path.join(dirpath, f)
                      for f in filenames
                      if is_archive(f)])
    if not files:
        raise Exception(
            'Error loading datasets. Please download from this url:\n'
            'https://sites.google.com'
            '/site/qianmingjie/home/datasets/cnn-and-fox-news')

    logger.debug('Archive files: {}'.format(files))
    return files


def is_archive(filename):
    extension = filename.split('.')[-1]
    if extension.lower() in IMPLEMENTED_ARCHIVE_EXTENSIONS:
        return True
    else:
        return False


def decompress(file, to, dataset_dir):
    if os.path.exists(to):
        logger.debug('Already existing file/dir at {}.'.format(to))
        logger.debug('No extraction will be done for {}.'.format(
            os.path.basename(file)))
        return

    extractor = os.path.join(dataset_dir, EXTRACTOR_SCRIPT)
    if not os.path.isfile(extractor):
        raise Exception('No archive extractor script found at {path}.\n'
                        'Create it from this post: {url}'.format(
            path=extractor,
            url=EXTRACTOR_SCRIPT_SOURCE))

    # take snapshot of directory so that the extracted directory can be spotted
    current_files = set(os.listdir(os.getcwd()))

    logger.info('Extracting dataset. This might take a while.')
    subprocess.run(['bash', extractor, file])
    logger.debug('Extraction complete.')

    logger.debug('Moving to {}'.format(to))
    new_files = list(set(os.listdir(os.getcwd())) - current_files)
    relocate(new_files, to)

    logger.info('Extraction complete. Uncompressed files'
                ' are within {}'.format(to))
    for f in os.listdir(to):
        dirname = os.path.basename(to)
        path = os.path.join('...', dirname, f)
        logger.debug('\t{}'.format(path))


def relocate(new_files, to):
    logger.debug('New files after extraction: {}'.format(new_files))
    if len(new_files) > 1:
        # move all files
        os.makedirs(to, exist_ok=True)
        for f in new_files:
            shutil.move(os.path.join(os.getcwd(), f), to)
        logger.debug('{} files moved.'.format(len(new_files)))

    elif len(new_files) == 1:
        new_path = os.path.join(os.getcwd(), new_files[0])
        if os.path.isfile(new_path):
            logger.debug('Extracted only one file.')
            os.makedirs(to, exist_ok=True)
            shutil.move(new_path, to)

        else:
            logger.debug('Extracted a whole directory.')
            os.rename(new_path, to)