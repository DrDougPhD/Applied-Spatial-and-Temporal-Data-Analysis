import logging
import os
import shutil
import pickle
logger = logging.getLogger(__name__)

try:  # this is my own package, but it might not be present
    from lib.lineheaderpadded import hr
except:
    hr = lambda title, line_char='-': line_char * 30 + title + line_char * 30


def these(data, n, file_relocation, files, pickle_to):
    if file_relocation:
        logger.debug(hr('Relocating Created Files'))
        logger.debug('Storing files in {}'.format(file_relocation))
        for f in files:
            logger.debug(f)
            filename = os.path.basename(f)
            dst = os.path.join(file_relocation, filename)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            # shutil.copy(f, args.relocate_files_to)
            os.rename(f, os.path.join(file_relocation, filename))

    # pickle the data
    pickle.dump(data, open(pickle_to.format(num_items=n), 'wb'))