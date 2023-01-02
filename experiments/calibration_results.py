# import torch
# import numpy as np
#
# DATASET = "FMNIST"
# # DATASET = "CIFAR10"
# MODEL_NAME = "CNN"
# # MODEL_NAME = "AllCNN"
# SEED = 30
#
# PATH = '/home/mjazbec/laplace/BNN-predictions/experiments/'
# fname = PATH + 'models/500_epochs/' + '_'.join([DATASET, MODEL_NAME, str(SEED)]) + '_{delta:.1e}.pt'
#
# deltas = np.logspace(-2.0, 3.0, 16)
# # deltas = np.insert(deltas, 0, 0)  # add unregularized network
#
# LA_TYPE = 'lap_kron'
# # LA_TYPE = 'map'
#
# best_acc, best_nll, best_ece = 0., 1., 1.
# for delta in deltas:
#     print("DELTA ", delta)
#     params = torch.load(fname.format(delta=delta), map_location=torch.device('cpu'))
#     # print(params)
#     print(params.keys())
#     print(params[LA_TYPE].keys())
#     acc = params[LA_TYPE]['acc_te']
#     nll = params[LA_TYPE]['nll_te']
#     ece = params[LA_TYPE]['ece_te']
#     print(f"acc: {acc}",
#           f"nll: {nll}",
#           f"ece: {ece}")
#     if acc >= best_acc:
#         best_acc = acc
#     if nll <= best_nll:
#         best_nll = nll
#     if ece <= best_ece:
#         best_ece = ece
# print(best_acc, best_nll, best_ece)

import torch
import numpy as np
import pandas as pd
from typing import Dict, List

DATASET = "FMNIST"
# DATASET = "CIFAR10"
MODEL_NAME = "CNN"
# MODEL_NAME = "AllCNN"
SEED = 30

PATH = '/home/mjazbec/laplace/BNN-predictions/experiments/'
fname = PATH + 'models/500_epochs/' + '_'.join([DATASET, MODEL_NAME]) + '_{seed}' + '_{delta:.1e}.pt'

LA_TYPE = 'lap_kron'
# LA_TYPE = 'map'

def bnn_preds_replication(eval_metric: str, la_type: str,
                          seeds: List[int], path: str,
                          deltas: np.array = np.logspace(-2.0, 3.0, 16)) -> pd.Series:
    res = {"ACC": [], "NLL": [], "ECE": []}
    for seed in seeds:
        best_acc, best_nll, best_ece = 0., 1., 1.
        best_delta_acc, best_delta_nll, best_delta_ece = None, None, None
        for delta in deltas:
            params = torch.load(path.format(delta=delta, seed=seed), map_location=torch.device('cpu'))

            acc = params[la_type]['acc_va']
            nll = params[la_type]['nll_va']
            ece = params[la_type]['ece_va']
            if acc >= best_acc:
                best_acc, best_delta_acc = acc, delta
            if nll <= best_nll:
                best_nll, best_delta_nll = nll, delta
            if ece <= best_ece:
                best_ece, best_delta_ece = ece, delta

        for metric, delta in zip(["NLL", "ECE", "ACC"], [best_delta_nll, best_delta_ece, best_delta_acc]):
            params = torch.load(path.format(delta=delta, seed=seed), map_location=torch.device('cpu'))
            res[metric].append(params[la_type][f'{metric.lower()}_te'])

    return pd.Series({metric: f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}" for metric, vals in res.items()},
                     name=la_type)


res_df = pd.DataFrame()
for la_type in ['map', 'lap_diag_nn', 'lap_kron_nn', 'lap_kron_dampnn' , 'lap_diag', 'lap_kron']:
# for la_type in ['map', 'lap_kron']:
    print(f"computing metrics for {la_type}")
    res_df = res_df.append(bnn_preds_replication(None, la_type=la_type, seeds=[10, 20, 30, 50], path=fname), ignore_index=False)

print(res_df)
