import matplotlib.pyplot as plt
import os


def visualize(pytrec_dict, evaluation_charts, filename):
    os.chdir(evaluation_charts)
    new_dict = {}
    for key, value in pytrec_dict.items():
        for k, v in value.items():
            for k2, v2 in v.items():
                if v2 > 0.05:
                    new_dict[k] = v
        new_dictionary = {str((k, k1)): v1 for k, v in new_dict.items() for k1, v1 in v.items()}
        plt.figure(figsize=(20, 10))
        plt.bar(new_dictionary.keys(), new_dictionary.values(), width=.5, color='b')
        plt.xticks(rotation=90, fontsize='small')
        plt.savefig(f'{filename}.jpg')
