import os
import glob

def make_savepath(folder='results/data', savepath=None):
    if savepath is None:
        paths = glob.glob('{}_*'.format(folder))
        if len(paths):
            n = max([int(p.split('_')[-1]) for p in paths])
        else:
            n = 0
        savepath = '{}_{}'.format(folder, n + 1)
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    return savepath
