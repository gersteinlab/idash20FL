import matplotlib.pyplot as plt
import os
import numpy as np
import pickle as pkl
import seaborn as sns
sns.set()

eps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

iterations = []

for i in eps:
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'Results/IID_Equal/Epsilon_' + str(i)))
    sample_iterations = []
    
    for j in range(0, 20):
        with open(data_dir + '/acc_' + str(j) + '.pkl', 'rb') as f:
            sample_iterations.append(len(pkl.load(f)))

    iterations.append(sample_iterations)

plt.figure(figsize=(30, 10))
plt.boxplot(iterations, labels=eps)
plt.title('Iterations till Convergence vs Epsilon Value: IID and Equally Distributed Data', fontsize=20)
plt.xlabel('Epsilon', fontsize=16)
plt.ylabel('Iterations', fontsize=16)
plt.savefig('IID_Equal_Iterations.svg', dpi=500, bbox_inches='tight')