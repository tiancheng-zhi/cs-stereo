import visdom
import numpy as np

class Visualizer():
    def __init__(self, server='http://localhost', port=8097, env='main'):
        self.vis = visdom.Visdom(server=server, port=port, env=env, use_incoming_socket=False)
        self.iteration = []
        self.dpn_nlogloss = []
        self.stn_nlogloss = []
        self.epoch = []
        self.rmse = []

    def state_dict(self):
        return {'iteration': self.iteration, 'dpn_nlogloss': self.dpn_nlogloss, 'stn_nlogloss': self.stn_nlogloss, 'epoch': self.epoch, 'rmse': self.rmse}

    def load_state_dict(self, state_dict):
        self.iteration = state_dict['iteration']
        self.dpn_nlogloss = state_dict['dpn_nlogloss']
        self.stn_nlogloss = state_dict['stn_nlogloss']
        self.epoch = state_dict['epoch']
        self.rmse = state_dict['rmse']

    def plot_loss(self):
        self.vis.line(
            X=np.array(self.iteration),
            Y=np.array([self.stn_nlogloss, self.dpn_nlogloss]).T,
            opts={
                'title': '-LogLoss',
                'legend': ['STN', 'DPN'],
                'xlabel': 'epoch',
                'ylabel': '-logloss'},
            win=0)

    def plot_rmse(self):
        self.vis.line(
            X=np.array(self.epoch),
            Y=np.array(self.rmse),
            opts={
                'title': 'RMSE',
                'legend': ['rmse'],
                'xlabel': 'epoch',
                'ylabel': 'rmse'},
            win=1)

    def image(self, im, idx):
        self.vis.image(im, opts={'title':'Level ' + str(idx)}, win=idx + 2)
