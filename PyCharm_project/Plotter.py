import numpy as np
import matplotlib.pyplot as plt

RDF_path = "../RDF_Outputs_electroneg/s15.csv"

csv_ingest = np.loadtxt(RDF_path,dtype=np.float64, delimiter=',')
rdf_values = csv_ingest[:,1]
r_values = np.arange(0, 20, 0.1)
padded_rdf = np.pad(rdf_values, (0, 200 - rdf_values.size), 'constant')


plt.rcParams.update({'font.size': 16})
plt.plot(r_values,padded_rdf, linewidth=1, color='k')
plt.xlabel('R / $\AA$')
plt.ylabel('RDF score')
plt.show()