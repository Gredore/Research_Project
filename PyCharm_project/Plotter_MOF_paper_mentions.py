import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 8))

sns.set_theme(style="whitegrid")

csv_ingest = np.loadtxt("MOF_paper_mentions.csv", dtype=int, delimiter=',')

ax = sns.barplot(x=csv_ingest[:,0], y=csv_ingest[:,1], palette="Blues_d")

plt.xlabel('Year of publication', fontsize=20)
plt.ylabel('Web of Science Results', fontsize=20)

plt.xticks(range(1, 24, 2))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig('../MOF_paper_mentions.png', format='png', dpi=300)
plt.show()

