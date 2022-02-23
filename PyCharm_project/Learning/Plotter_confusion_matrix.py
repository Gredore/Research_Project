from matplotlib import pyplot as plt
import seaborn as sns

sns.set(font_scale=2)

fig, ax = plt.subplots(figsize=(10, 8))

cf_matrix = [[0.3, 0.24],[0.26, 1]]
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', annot_kws = {'fontsize': 20})


ax.set_title('SVM Confusion Matrix [weighting=electronegativity]\n\n', fontsize = 20);
ax.set_xlabel('Predicted Values', fontsize = 20)
ax.set_ylabel('Actual Values ', fontsize = 20);

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Unstable','Stable'], fontsize = 20)
ax.yaxis.set_ticklabels(['Unstable','Stable'], fontsize = 20)

#fig.axes[1].xaxis.set_ticklabels(['Unstable','Stable'], fontsize = 20)
## Display the visualization of the Confusion Matrix.
plt.show()