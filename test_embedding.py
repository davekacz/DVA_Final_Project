# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from util import Utility, kernel_pca, graph_embedding
plt.close('all')
# Inputs
ut = Utility(taxi = 'yellow', day_night = 'day', 
             day_of_week = 'weekday')
year = 2019
month = 6
top = 10
embedding = 'kernelpca' # or 'laplacian'
#%% Read in the data
edges, trips = ut.load_data(month, year)
zones = pd.read_csv('data/taxi_zones.csv')
id_to_zone = dict(zip(zones['locationid'], zones['zone']))
#%% Get the state transition matrix
edges = ut.state_transition(edges)
n = max(edges['source'].max(), edges['target'].max())
node_ids = np.arange(n)
P = ut.df_to_coo(edges, n, data_col='weight')
P = P.toarray()
# Remmove isolated nodes
mask = np.isclose(np.sum(P, axis =1),1)
node_ids = node_ids[mask]
P = P[node_ids[:, np.newaxis], node_ids[np.newaxis,:]]
w, v = np.linalg.eig(P.T)
#%%
v = v[:, np.isclose(w.real, 1)]
v= v[:,0]
v = v.real/v.real.sum()
# Get the most profitable pickup zones
ranked_pu_zones = ut.weighted_profit(trips, v.ravel(), node_ids+1)
assert np.all((ranked_pu_zones['pickup_zone'].values - 1) == node_ids), 'something wrong'
pickupscore = ranked_pu_zones['weighted_profit'].values
pickupscore = pickupscore/pickupscore.sum()
# Get the undirected adjacency matrix
edges = ut.symmetrize_df(edges, self_edges=True)
A = ut.df_to_coo(edges, n)
A = A.toarray()
A = A[node_ids[:, np.newaxis], node_ids[np.newaxis,:]]
# fig, ax = plt.subplots(figsize=(8,6))
# ax.spy(A, markersize = 0.5)
# ax.set_title('Sparsity of the Adjacency Matrix\n')
# Graph model
if embedding == 'kernelpca':
    model = kernel_pca(k=2)
else:
    model = graph_embedding(k=2)
#%% Vizualize the graph
proj = model.fit_transform(A)
# if rank == 'degree':
#     degree = A.sum(axis=1).ravel()
#     degree = degree/degree.sum()
#     idx = np.argsort(degree)[-top:]
# elif rank =='pagerank':
degree = v.ravel()
do_idx = np.argsort(-degree)[:top]
    

pu_idx = np.argsort(-pickupscore)[:top]

idx = pd.DataFrame({'id':pu_idx, 'label':0, 'rank': pickupscore[pu_idx]})
idx = pd.concat([idx, pd.DataFrame({'id':do_idx, 'label':1, 'rank': degree[do_idx]})])
fig, ax = ut.plot_graph(proj, adj_mx = A, rank = pickupscore,
                        node_ids = node_ids+1, 
                     id_to_name=id_to_zone, idx = idx, thumb_frac= 0.03)
ax.grid(False)
# ax.set_title('Top {0} Pickup zones by profitability for Taxi type = {1}, Month = {2}, Year={3}' \
#              .format(top, taxi.upper(), month, year))
# fig.savefig('PU_{4}_{0}_{1}_{2}_{3}.png'.format(taxi, month, year, rank, embedding),
#             bbox_inches = 'tight')
    
        