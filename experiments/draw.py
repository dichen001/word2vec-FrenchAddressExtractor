__author__ = 'amrit'

import matplotlib.pyplot as plt
import os, pickle
import operator
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors

if __name__ == '__main__':

    fileB = ['SE0' ,'SE6','SE1','SE3', 'SE8', 'cs', 'diy','photo','rpg','scifi']
    F_final1={}

    # path = '/home/amrit/GITHUB/e-disc/Amrit/2016-08-30/dump/'
    # for root, dirs, files in os.walk(path, topdown=False):
    #     for name in files:
    #         a = os.path.join(root, name)
    #         with open(a, 'rb') as handle:
    #             F_final = pickle.load(handle)
    #             F_final1 = dict(F_final1.items() + F_final.items())
    with open('dump/' + 'test_positive_on_4_kernels.pickle', 'rb') as handle:
        F_final = pickle.load(handle)
    print(F_final)
    #print(F_final1)
    # learners = ['dual', 'primal']
    smotes = ['S', 'noS']
    targets = ['pos', 'neg']
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # measures = ['precision', 'recall', 'f1', 'f2']
    measures = ['f1', 'f2']
    font = {'size'   : 60}
    plt.rc('font', **font)
    paras={'lines.linewidth': 10,'legend.fontsize': 35, 'axes.labelsize': 60, 'legend.frameon': False,'figure.autolayout': True}
    plt.rcParams.update(paras)

    for i, target in enumerate(targets):
        plt.figure(num=i, figsize=(25, 15))

        for s in smotes:
            for measure in measures:
                median=[]
                iqr=[]
                #print(x)
                for kernel in kernels:
                    median.append(np.median(F_final[target][s][kernel][measure]))
                    iqr.append(np.percentile(F_final[target][s][kernel][measure], 75) - np.percentile(F_final[target][s][kernel][measure], 25))
                X = range(len(kernels))
                line, = plt.plot(X, median, marker='o', markersize=20, label=s + '-' + measure + ' median')
                plt.plot(X, iqr, linestyle="-.", color=line.get_color(), marker='*', markersize=20, label=measure + ' iqr')
        #plt.ylim(-0.1,1.1, )
        #plt.ytext(0.04, 0.5, va='center', rotation='vertical', fontsize=11)
        #plt.text(0.04, 0.5,"Rn (Raw Score)", labelpad=100)
        plt.xticks(X, kernels)
        plt.ylabel("Performance", labelpad=20)
        plt.xlabel("SVM Kernels",labelpad=25)
        plt.legend(bbox_to_anchor=(1.07, 1.13), loc=1, ncol = 4, borderaxespad=0.1)
        plt.savefig(target + "_performance.png")
