
import logging
logger = logging.getLogger('cnn.'+__name__)
class LoggingObject(object):
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)

    def debug(self, msg):
        logger.debug(msg)

    def info(self, msg):
        logger.info(msg)