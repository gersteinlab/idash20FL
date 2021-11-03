import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
sns.set()

eps = [1]

for i in eps:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Results/IID_Equal/Epsilon_' + str(i)))
    
    df_list = []

    for j in range(0, 10):
        with open(data_dir + '/precision_' + str(j) + '.pkl', 'rb') as f:
            precision = pkl.load(f)
        with open(data_dir + '/recall_' + str(j) + '.pkl', 'rb') as f:
            recall = pkl.load(f)

        for k in range(0, len(precision)):
            df_list.append([precision[k], recall[k]])

    df = pd.DataFrame(df_list, columns = ['Precision', 'Recall'])
    table = df.groupby(['Recall']).describe()
    print(df.groupby(['Recall']).describe())