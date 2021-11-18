# -*- coding: utf-8 -*-
# Utility functions and class definitions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
from scipy import sparse
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from datetime import date
from dateutil.relativedelta import relativedelta
#%%
class Utility:
    def __init__(self, taxi = 'yellow', day_night = 'day', 
                 day_of_week = 'weekday', duration = 8,
                 exclude_ids = None, nexp = 500):
        assert taxi in ['yellow', 'green']
        assert day_night in ['day', 'night']
        assert day_of_week in ['weekday', 'weekend']
        assert duration > 1
        assert nexp > 10
        self.taxi = taxi
        self.day_night = day_night
        self.dow = day_of_week
        self.duration = duration
        self.exclude_ids = exclude_ids
        self.nexp = nexp
        
    def load_data(self, month, year=2019):
        # Check for input data consistency
        assert year in range(2016,2020), 'year should be from 2015 to 2020'
        assert month in range(1,13) ,'month should be an integer from 1 to 12'
        # Set the filepath to data
        filepath = 'TAXI_trips_{0}.csv'.format(year)
        if self.taxi == 'yellow':
            cab_type = 1
        if self.taxi == 'green':
            cab_type = 2
        # Read in the data
        df = pd.read_csv('data/' + filepath)
        # df = df[(df['avg_trip_time']>=0.08) & 
        #         (df['avg_trip_time'] <= 3)]
        # Subset the data for a month and cab_type
        edges = df[(df['month'] == month) & 
                   (df['cab_type_id']==cab_type) &
                   (df['day_night']==self.day_night) & 
                   (df['day_of_week']==self.dow)]
        trips = edges[['pickup_zone', 'dropoff_zone','avg_trip_time',
                       'avg_total_amount', 'num_trips']].copy()
        # Define the appropriate edge weight
        edges = edges[['pickup_zone', 'dropoff_zone', 'num_trips']]
        edges.rename(columns = {'pickup_zone': 'source', 'dropoff_zone': 'target',
                                'num_trips':'value'}, inplace = True) 
        return edges, trips
#%% Function for drawing the graph
# This function is borrowed and modified from 
# https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    def plot_graph(self, proj, adj_mx, rank, node_ids=None, id_to_name = None,
                   idx = None, thumb_frac=0.05):
        fig, ax = plt.subplots(figsize=(10, 10))
        if idx is not None:
            if isinstance(idx, pd.DataFrame):
                show = idx['id'].values
                label = idx['label'].values
                rank = idx['rank'].values
            else:
                show = idx
                rank = rank[show]
            proj = proj[show,:]
            node_ids = node_ids[show]
            adj_mx = adj_mx[show[:, np.newaxis], show[np.newaxis,:]]
        # Normalized edge weights 
        adj_mx = adj_mx/adj_mx.max()
        # First add the nodes
        if isinstance(idx, pd.DataFrame):
            ax.scatter(proj[:, 0], proj[:, 1], s = rank*20000, c = label,
                       cmap = ListedColormap(['red', 'blue']), marker='.')
        else:
            ax.scatter(proj[:, 0], proj[:, 1], s = rank*10000, c = rank,
                       cmap = 'YlOrRd', marker='.')
        # Add the edges
        adj_mx = sparse.coo_matrix(adj_mx)
        segments = [[proj[i, :], proj[j, :]]
                for i, j in zip(adj_mx.row, adj_mx.col) if i>j]
        linewidths = [k*8 for i, j, k in zip(adj_mx.row, adj_mx.col, adj_mx.data) if i>j]
        lc = LineCollection(segments, linewidths = linewidths,
                            alpha =0.5, zorder=0)
        ax.add_collection(lc)
        # Add the node labels
        if node_ids is not None:
            min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
            shown_ids = np.array([2 * proj.max(0)])
            for i in range(proj.shape[0]):
                dist = np.sum((proj[i] - shown_ids) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_ids = np.vstack([shown_ids, proj[i]])
                locationid = node_ids[i]
                if id_to_name is not None:
                    if locationid in id_to_name:
                        name = id_to_name[locationid]
                    else:
                        name = locationid
                else:
                    name = locationid
                ax.annotate(name,
                            xy = proj[i],
                            xytext = (-10,10*rank[i]),
                            textcoords = 'offset pixels',
                            size = 9, color = 'black')
        return fig, ax
            
#%% function for symmetrizing an edges df
    def symmetrize_df(self, edges, self_edges = True):
        # Remove self edges and makes the graph undirected
        assert 'source' in edges.columns
        assert 'target' in edges.columns
        assert 'value' in edges.columns
    
        edges_transpose = edges.rename(columns={'source': 'target', 'target': 'source'})
        edges_all = pd.concat([edges, edges_transpose], sort=False) \
                    .reset_index(drop=True) \
                    .groupby(['source', 'target'], as_index = False) \
                    .sum()
                    
        if self_edges:
            edges_all['value'][edges_all['source'] == edges_all['target']] /= 2
        else:
            edges_all = edges_all[edges_all['source'] != edges_all['target']]
        return edges_all
#%% function for calculating the probability of going from source to target
    def state_transition(self, edges):
        assert 'source' in edges.columns
        assert 'target' in edges.columns
        assert 'value' in edges.columns
        
        if self.exclude_ids is not None:
            mask = (edges['source'].isin(self.exclude_ids) |
                    edges['target'].isin(self.exclude_ids))
            edges = edges[~mask]
        
        print('Initial number of rows in edge list -', len(edges))
        # only keep nodes which have greater than 1 outgoing edge
        for step in range(2):
            temp = edges[['source', 'value']].groupby('source', as_index= False).sum()
            temp = temp['source'][temp['value'] > 1]
            mask = edges['source'].isin(set(temp.values))
            edges = edges[mask]
            # temp = edges[['target', 'value']].groupby('target', as_index= False).sum()
            # temp = temp['target'][temp['value'] > 1]
            # mask = edges['target'].isin(set(temp.values))
            # edges = edges[mask]
            # Remove nodes which only have self-edges
            temp = edges['source'][edges['source'] != edges['target']]
            mask = edges['source'].isin(set(temp.values))
            edges = edges[mask] 
            # Only keep zones which have both incoming and outgoing edges
            mask = (edges['source'].isin(set(edges['target'].values)) &
                    edges['target'].isin(set(edges['source'].values)))
            edges = edges[mask]
        
        print('Final number of rows in edge list - ', len(edges))
        pickups = edges[['source', 'value']].groupby('source', as_index= False).sum()
        pickups.rename(columns = {'value':'total_value'}, inplace = True)
        # pickups = pickups[pickups['total_value']>0]
        edges = pd.merge(edges, pickups, on = 'source')
        edges['weight'] = edges['value']/edges['total_value']
        return edges
    
 # function for converting a edges df to cco sparse matrix   
    def df_to_coo(self, edges, n, data_col = 'value'):
        """Returns a Scipy coordinate sparse matrix, given an input graph in a data frame representation."""
        assert 'source' in edges.columns
        assert 'target' in edges.columns
        assert data_col  in edges.columns
    
        values = edges[data_col]
        rows = edges['source'].values - 1
        cols = edges['target'].values - 1
        return sparse.coo_matrix((values, (rows, cols)), shape=(n, n))

#%% Function to calculate weighted profit
    def weighted_profit(self, trips, score, node_ids):
        df = pd.DataFrame({'dropoff_zone': node_ids, 'score': score})
        df = pd.merge(trips, df, on = 'dropoff_zone')
        df['weighted_profit'] = (df['avg_total_amount'] *
                                 df['num_trips'] * df['score'])
        df = df[['pickup_zone', 'num_trips', 'weighted_profit']] \
             .groupby('pickup_zone', as_index= False).sum()
        df.rename(columns = {'num_trips':'total_trips'}, inplace = True)
        # df = df[df['total_trips']>0]
        df['weighted_profit'] = df['weighted_profit']/df['total_trips']
        temp = pd.DataFrame({'pickup_zone': node_ids})
        df = pd.merge(temp, df, how = 'left', on= 'pickup_zone')
        df.fillna(0, inplace = True)
        return df[['pickup_zone', 'weighted_profit']]
    #%% function to compute the transition proabilities and the pagerank 
    def pagerank(self, edges, trips, top =10):
        edges = self.state_transition(edges)
        n = max(edges['source'].max(), edges['target'].max())
        node_ids = np.arange(n)
        P = self.df_to_coo(edges, n, data_col='weight')
        P = P.toarray()
        # Remmove isolated nodes
        mask = np.isclose(np.sum(P, axis =1),1)
        node_ids = node_ids[mask]
        P = P[node_ids[:, np.newaxis], node_ids[np.newaxis,:]]
        node_ids = node_ids + 1 # actual location ids start from 1 not 0
        w, v = np.linalg.eig(P.T)
        # v = v[:, np.isclose(w.real, 1)]
        v = v[:,0]
        v = v.real/v.real.sum()
        assert len(v.ravel()) == len(node_ids), 'something wrong with P'
        self.dropoffscore_ = v.ravel()
        # Get the most profitable pickup zones
        ranked_pu_zones = self.weighted_profit(trips, v.ravel(), node_ids)
        assert np.all(ranked_pu_zones['pickup_zone'].values == node_ids), \
            'something wrong with weighted profit calculation'
        pickupscore = ranked_pu_zones['weighted_profit'].values
        pickupscore = pickupscore/pickupscore.max()
        self.pickupscore_ = pickupscore
        # Get the undirected adjacency matrix
        idx = np.argsort(-pickupscore)[:top]
        print('Feasible number of nodes in the transition matrix - ', len(node_ids))
        return P, node_ids, idx
#%% Function to simulate taxi path
    def random_walk(self, P, node_ids, trip_data, start = None):
        assert np.all(np.isclose(np.sum(P, axis =1),1)), \
            'something wrong in the state transition matrix'
        if start is None:
            start = np.random.choice(node_ids)
        total_amount = 0
        trip_time = 0
        # for step in range(10):
        while trip_time <= self.duration:
            startid = np.nonzero(node_ids == start)[0][0]
            end = np.random.choice(node_ids, p = P[startid,:])
            total_amount += trip_data.loc[(start, end), 'avg_total_amount']
            trip_time = (trip_time + 
                         trip_data.loc[(start,end), 'avg_trip_time'] + 0.1)
            start = end
        return total_amount
#%% Function to simulate randomwalk over many experiments
    def simulation(self, P, node_ids, trips, idx = None):
        # Simulate a random walk 100 times
        total_amount = []
        if idx is None:
            idx = range(len(node_ids))
        trip_data = trips.set_index(['pickup_zone','dropoff_zone'])
        for experiment in range(self.nexp):
            start = np.random.choice(node_ids[idx])
            total_amount.append(self.random_walk(P, node_ids, trip_data, start))
        return np.array(total_amount)
#%% Function to run the simulation over a specified time range
    def monthly_sim(self, start, end, top = 10):
        current  = start
        dates = []
        random_start = []
        pr_start = []
        while current <= end:
            print('Running simulation for month {0} of year {1}'.format(current.month,
                                                                        current.year))
            edges, trips = self.load_data(current.month, current.year)
            P, node_ids, idx = self.pagerank(edges, trips, top)
            try:
                temp1 = np.mean(self.simulation(P, node_ids, trips, idx))
                temp2 = np.mean(self.simulation(P, node_ids, trips))
                random_start.append(temp2)
                pr_start.append(temp1)
                dates.append(current)
            except:
                print('something wrong for month of {0} in year {1}'.format(current.month,
                                                                            current.year))
                pass
            current = current + relativedelta(months=1)
        df = pd.DataFrame({'random': random_start, 'pagerank': pr_start},
                          index=pd.to_datetime(dates))
        return df
    #%% Function to run the simulation over a specified time range
    def monthly_pred(self, start, end, top = 10):
        current  = start
        dates = []
        random_start = []
        pr_start = []
        while current < end:
            next_mon = current + relativedelta(months=1)
            print('Running simulation for month {0} of year {1}'.format(next_mon.month,
                                                                        next_mon.year))
            edges, trips = self.load_data(current.month, current.year)
            _, ids, idx = self.pagerank(edges, trips, top)
            edges, trips = self.load_data(next_mon.month, next_mon.year)
            P, node_ids, _ = self.pagerank(edges, trips, top)
            idx = [np.nonzero(node_ids == i)[0][0] 
                   for i in ids[idx] if np.isin(i, node_ids)]
            try:
                temp1 = np.mean(self.simulation(P, node_ids, trips, idx))
                temp2 = np.mean(self.simulation(P, node_ids, trips))
                pr_start.append(temp1)
                random_start.append(temp2)
                dates.append(next_mon)
            except:
                print('something wrong for month of {0} in year {1}'.format(next_mon.month,
                                                                            next_mon.year))
                pass
            current = next_mon
        df = pd.DataFrame({'random': random_start, 'pagerank': pr_start},
                          index=pd.to_datetime(dates))
        return df
    #%% Find the best zone among neigbors
    def top_neighbor(self, current_zone, month, year,
                     time_remaining, transition_time = 0.2):
        temp = self.duration
        edges, trips = self.load_data(month, year)
        P, node_ids, _ = self.pagerank(edges, trips)
        trips.set_index(['pickup_zone','dropoff_zone'], inplace = True)
        neighbors = trips.loc[current_zone].copy()
        mask = neighbors['avg_trip_time'] <= transition_time
        neighbors = neighbors[mask]
        neighbors.index.rename('zone_id', inplace = True)
        neighbors.loc[current_zone, 'avg_trip_time'] = 0
        neighbors['expected_total_amount'] = np.nan
        for zone in neighbors.index:
            if zone in node_ids:
                self.duration = time_remaining - neighbors.loc[zone, 'avg_trip_time']
                total_amount = []
                for experiment in range(50):
                    try:
                        total_amount.append(self.random_walk(P, node_ids, trips, zone))
                    except:
                        break
                neighbors.loc[zone, 'expected_total_amount'] = np.mean(total_amount)
        self.duration = temp        
        return neighbors        
        
#%% Plot the distributions from randonwalk simulations
    def dist_plot(self, data):
        '''
        Parameters
        ----------
        data : df containing the columns to be plotted 
    
        Returns
        -------
        fig, ax
    
        '''
        fig, ax = plt.subplots()
        for col in data.columns:
            sns.kdeplot(data[col], fill= True, ax = ax, label = col, legend =True)
        ax.set_xlabel('total amount')
        return fig, ax
#%% eigendecomposition of the normalized graph Laplacian
class graph_embedding:
    def __init__(self, k = 2):
        self.k = k
        
    def fit_transform(self, A):
        """
        A: symmetrical adjacency matrix
        """
        D = np.diag(1/np.sqrt(np.sum(A, axis=1)).ravel())
        # D = sparse.spdiags(1/np.sqrt(np.sum(A, axis=1)).ravel(), diags=0, m=A.shape[0], n=A.shape[1])
        L = D @ A @ D
        w, v = sparse.linalg.eigsh(L, k =self.k, which = 'LM')
        return v #* np.sqrt(w[np.newaxis,:])
#%% Kernel PCA (eigen decomposition of the Gram matrix)
class kernel_pca:
    def __init__(self, k = 2):
        self.k = k
    
    def fit_transform(self, A):
        """
        A: symmetrical adjacency matrix
        """
        n = A.shape[0]
        # D = np.diag(np.sum(A, axis=1).ravel())
        H = np.eye(n) - np.ones((n,n))/n
        C = H @ A @ H
        C = (C + C.T)/(2*n)
        w, v = sparse.linalg.eigsh(C, k =self.k, which = 'LM')
        return v# * np.sqrt(w[np.newaxis,:])