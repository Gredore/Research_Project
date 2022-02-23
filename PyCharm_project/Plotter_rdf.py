import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 8))


RDF_path = "../RDF_Outputs_electroneg/s40.csv"

csv_ingest = np.loadtxt(RDF_path,dtype=np.float64, delimiter=',')
rdf_values = csv_ingest[:,1]
r_values = np.arange(0, 20, 0.1)
padded_rdf2 = np.pad(rdf_values, (0, 200 - rdf_values.size), 'constant')

ax.plot(r_values,padded_rdf2, linewidth=2, color='b', label='Electronegativity')


RDF_path = "../RDF_Outputs_vdW/s40.csv"

csv_ingest = np.loadtxt(RDF_path,dtype=np.float64, delimiter=',')
rdf_values = csv_ingest[:,1]
r_values = np.arange(0, 20, 0.1)
padded_rdf3 = np.pad(rdf_values, (0, 200 - rdf_values.size), 'constant')

ax.plot(r_values,padded_rdf3, linewidth=2, color='g', label='VdWaals volume')


RDF_path = "../RDF_Outputs_unit/s40.csv"

csv_ingest = np.loadtxt(RDF_path,dtype=np.float64, delimiter=',')
rdf_values = csv_ingest[:,1]
r_values = np.arange(0, 20, 0.1)
padded_rdf1 = np.pad(rdf_values, (0, 200 - rdf_values.size), 'constant')


plt.rcParams.update({'font.size': 25})
ax.plot(r_values,padded_rdf1, linewidth=2, color='k', label='Unit $(P_i = P_j = 1)$')
plt.xlabel('R / $\AA$', fontsize=30)
plt.ylabel('RDF score', fontsize=30)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


ax.fill_between(
    r_values, padded_rdf2, padded_rdf1,
    interpolate=True, color="blue", alpha=0.25
)

ax.fill_between(
    r_values, padded_rdf3, padded_rdf1,
    interpolate=True, color="green", alpha=0.25
)

ax.legend(title="Weighting", title_fontproperties={'weight': 'bold'})
#plt.savefig('../s40_RDFs.png', format='png', dpi=200)
plt.show()