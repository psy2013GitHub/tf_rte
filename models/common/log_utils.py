
import sys
from pathlib import Path
import tensorflow as tf
import logging

def init_log_path(path):
    Path(path).mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('{}/main.log'.format(path)),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers