import matplotlib.pyplot as plt
import os
import numpy as np
import pickle as pkl
import seaborn as sns
sns.set()

eps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
acc = []
loss = []

for i in eps:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Results/IID_Equal/Epsilon_' + str(i)))
    sample_acc = []
    sample_loss = []
    
    for j in range(0, 20):
        with open(data_dir + '/acc_' + str(j) + '.pkl', 'rb') as f:
            sample_acc.append(pkl.load(f)[-1])

    for j in range(0, 20):
        with open(data_dir + '/loss_' + str(j) + '.pkl', 'rb') as f:
            sample_loss.append(pkl.load(f)[-1])

    acc.append(sample_acc)
    loss.append(sample_loss)

plt.figure(figsize=(30, 10))
plt.boxplot(acc, labels=eps)
plt.title('Test Set Accuracy vs Epsilon Value: IID and Equally Distributed Data', fontsize=20)
plt.xlabel('Epsilon', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.savefig('IID_Equal_Accuracy.svg', dpi=500, bbox_inches='tight')

plt.figure(figsize=(30, 10))
plt.yscale('log', base=10)
plt.boxplot(loss, labels=eps)
plt.title('Test Set Loss vs Epsilon Value: IID and Equally Distributed Data', fontsize=20)
plt.xlabel('Epsilon', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.savefig('IID_Equal_Loss.svg', dpi=500, bbox_inches='tight')