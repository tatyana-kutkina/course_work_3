from sklearn import metrics
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

def find_tau(data):
    data = data.reshape((len(data), 1))
    kbins = KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')
    data_trans = kbins.fit_transform(data)

    # find usable time delay via mutual information
    tau_max = 50
    mis = []
    data = data_trans.reshape(len(data_trans, ))

    for tau in range(1, tau_max):
        unlagged = data[:-tau]
        lagged = np.roll(data, -tau)[:-tau]
        mis.append(metrics.mutual_info_score(lagged, unlagged))

        if len(mis) > 1 and mis[-2] < mis[-1]:  # return first local minima
            tau -= 1
            return tau


