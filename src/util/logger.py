import os
import sys

from tensorboardX import SummaryWriter
from datetime import datetime


class Logger(object):
    def __init__(self, val=True, filename="print.log"):
        self.Time = datetime.now().strftime('%Y-%m-%d_%H%M')
        self.path = 'output/' + self.Time
        self.log_filename = filename
        os.makedirs(self.path) if os.path.exists(self.path) is False else None
        self.run_path = '{}/{}'.format(self.path, 'tb')

        # common log
        self.terminal = sys.stdout
        self.terminal.write(self.path)

        # init tensorboardX
        self.train_writer = None
        self.val_writer = None
        self.tensorboard_init(val)

    def printlog(self, message):
        message = str(message)
        self.terminal.write(message + '\n')

        log = open(os.path.join(self.path, self.log_filename), "a", encoding='utf8', )
        log.write(message + '\n')
        log.close()

    def tensorboard_init(self, val=True):
        if val:
            self.train_writer = SummaryWriter(self.run_path+'/train')
            self.val_writer = SummaryWriter(self.run_path+'/val')
        else:
            self.train_writer = SummaryWriter(self.run_path)

    def get_path(self):
        return self.path
