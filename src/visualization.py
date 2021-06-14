import matplotlib.pyplot as plt
import numpy as np
import collections


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def visu(pytrec_dict, file_loc, filename):
    metrics = []
    metrics_averaged_values = {}
    for k, v in pytrec_dict.items():
        for kk, vv in v.items():
            metrics.append(kk)
            # average
            if kk not in metrics_averaged_values.keys():
                metrics_averaged_values[kk] = []
            metrics_averaged_values[kk].append(vv)
    metric_value = metrics_averaged_values.copy()
    for metric in metrics:
        metrics_averaged_values[metric] = np.mean(metrics_averaged_values[metric])
    x = list(metrics_averaged_values.keys())
    y = list(metrics_averaged_values.values())
    print(f'metrics value:{metric_value}')
    print(f'metrics averaged value:{metrics_averaged_values}')
    plt.bar(x, y)
    plt.xticks(rotation=45)
    plt.savefig(f'{file_loc}{filename}.jpg')


def vis(pytrec_dict, file_loc, filename):
    test_dict = {}
    metrics = []
    for k, v in pytrec_dict.items():
        for kk, vv in v.items():
            for kkk, vvv in vv.items():
                metrics.append(kkk)
    metrics = list(set(metrics))
    for met in metrics:
        print(met)
        for k, v in pytrec_dict.items():
            for kk, vv in v.items():
                for kkk, vvv in vv.items():
                    if kkk in met:
                        test_dict[f'{k}_{kkk}'] = vvv
                        break
    sorted_dict = {k: v for k, v in sorted(test_dict.items(), key=lambda item: item[1], reverse=True)}
    for met in metrics:  # ndcg, map
        demo_dict = {}
        x, y = [], []
        for k, v in sorted_dict.items():
            if k.__contains__(met):  # ndcg
                demo_dict[k] = v
        for k, v in demo_dict.items():
            x.append(k)
            y.append(v)
            if len(x) > 20:
                break
        plt.figure()
        figure = plt.gcf()
        figure.set_size_inches(10, 14)

        plt.bar(x, y)
        plt.xticks(rotation=90, fontsize='small')
        plt.savefig(f'{file_loc}{filename}_{met}.jpg', dpi=100)
