import pandas as pd
import numpy as np
import os
import re
import random
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import pairwise_distances

def normalize(df):
    df_min = df.min().min()
    df_max = df.max().max()
    df_normalized = (df - df_min) / (df_max - df_min)
    df_normalized = df_normalized * 2 - 1
    return df_normalized

def somers_d(x, y):
    concordant = 0
    discordant = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)

def get_llm(epoch):
    llm_file = '/root/LLM/backup/tmpfinal.txt'
    output_file = '/root/LLM/backup/mse.txt'
    CelebA_Attr_file = "/root/DEAR-main/celeba/list_attr_celeba.txt"
    ans = []
    num_idx = [16, 10, 2, 8, 12, 4]
    with open(llm_file, "r") as ffile:
        info = ffile.readlines()
        for line in info:
            line = re.sub('[^\d]', ' ', line)
            line = line.split()
            if len(line) < 17:
                for _ in range(18):
                    line.append(0)
            line = np.array(list(map(int, line)))
            ans.append(line[num_idx])
        ans = np.array(ans)
    label_idx = [5, 4, 20, 24, 9, 18]
    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_file.readline()
        att = np.array(Attr_file.readline().split())
        att = att[label_idx]
        att[2] = 'Gender'
        att[3] = 'Beard'
        att[4] = 'Blond'
        att[5] = 'Makeup'
        coef = pd.DataFrame(np.zeros((len(att), len(att))), columns=att, index=att)
        df = pd.DataFrame(ans[:, :], columns=att)
        for column in df.columns:
            for ycolumn in df.columns:
                column_index = df.columns.get_loc(column)
                ycolumn_index = df.columns.get_loc(ycolumn)
                mask = (df.iloc[:, column_index] != 0) | (df.iloc[:, ycolumn_index] != 0)

                xcolumn_data = df.loc[mask, column]
                ycolumn_data = df.loc[mask, ycolumn]
                somers_d_value = somers_d(xcolumn_data, ycolumn_data)
                coef.loc[column, ycolumn] = somers_d_value
        coef = normalize(coef)
        print(coef)
        np.savetxt('/root/LLM/backup/g2coef.csv', coef.to_numpy(), delimiter=',')
        mask = np.eye(coef.shape[0], dtype=bool)
        with open(output_file, "a") as ofile:
            ofile.write('coef' + str(coef) + '\n\n')
            fig_name = '/root/LLM/backup/coefg1' + '_' + epoch + '.png'
            plt.clf()
            fig = sns.heatmap(coef, annot=True, fmt='.2f', cmap="coolwarm", linewidths=0.3, linecolor="grey", mask=mask, vmin=-1, vmax=1)
            plt.tight_layout()
            fig.get_figure().savefig(fig_name)

get_llm('')
