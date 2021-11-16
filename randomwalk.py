# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from util import plot_graph, symmetrize_df, df_to_coo, state_transition
from util import weighted_profit, kernel_pca, graph_embedding, random_walk
nbexp = 1000
plt.close('all')
# Inputs
year = 2016
month = 12
taxi = 'yellow' # 'uber', 'lyft', 'yellow' or 'green'
rank = 'pagerank' # pagerank or degree
top = 5
weight = 'trips' #'trips'
embedding = 'kernelpca' # or 'laplacian'
#%% Check for input data consistency
assert year in range(2015,2021), 'year should be from 2015 to 2020'
assert month in range(1,13) ,'month should be an integer from 1 to 12'
assert taxi in ['yellow', 'green', 'uber', 'lyft']
assert rank in ['pagerank', 'degree']
assert top > 1
assert weight in ['trips', 'fare']
assert embedding in ['kernelpca', 'laplacian']
#%% Constant
zones = pd.read_csv('data/taxi_zones.csv')
id_to_zone = dict(zip(zones['locationid'], zones['zone']))

def computedate(taxi, year,month, weight):
    # Set the filepath to data
    if taxi in ['uber', 'lyft']:
        filepath = '{0}_trips_{1}.csv'.format(taxi.upper(), year)
        weight = 'trips'
        print('Uber and Lyft have no fare data. Hence setting edge weight to number of trips')
    else:
        filepath = 'TAXI_trips_{0}.csv'.format(year)
        if taxi == 'yellow':
            cab_type = 1
        if taxi == 'green':
            cab_type = 2
    #%% Read in the data
    df = pd.read_csv('data/' + filepath)
    
    #%% Subset the data for a month and cab_type
    if taxi in ['uber', 'lyft']: 
        edges = df[df['month'] == month]
    else:
        edges = df[(df['month'] == month) & (df['cab_type_id']==cab_type)]
        trips = edges[['pickup_zone', 'dropoff_zone', 'avg_total_amount', 'num_trips']].copy()
    # Define the appropriate edge weight
    if weight == 'fare':   
        edges = edges[['pickup_zone', 'dropoff_zone', 'avg_total_amount', 'num_trips']]
        edges['value'] = edges['avg_total_amount'] * edges['num_trips']
        edges = edges[edges['value'] > 0]
    else:
        edges = edges[['pickup_zone', 'dropoff_zone', 'num_trips']]
        edges.rename(columns = {'num_trips':'value'}, inplace = True)
    edges = edges[['pickup_zone', 'dropoff_zone', 'value']]    
    edges.rename(columns={'pickup_zone': 'source', 'dropoff_zone': 'target'},
                 inplace = True)
    #%% Get the state transition matrix
    exclude_ids = [264,265]
    edges = state_transition(edges, exclude_ids)
    n = max(edges['source'].max(), edges['target'].max())
    node_ids = np.arange(n)
    P = df_to_coo(edges, n, data_col='weight')
    P = P.toarray()
    edges = symmetrize_df(edges, self_edges=True)
    
    # Remmove isolated nodes
    temp = np.sum(P, axis=1)
    node_ids = node_ids[temp>0]
    P = P[node_ids[:, np.newaxis], node_ids[np.newaxis,:]]
    return node_ids, P, trips, edges, n

def piclocation(edges,rank, top,n,node_ids,embedding,id_to_zone,taxi,month,year,P,trips):
    w , v = np.linalg.eig(P.T)
    # v = v[:, np.isclose(w.real, 1, atol=1e-2)]
    v= v[:,0]
    v = v.real/v.real.sum()
    # Get the most profitable pickup zones
    if 'trips' in locals():
        ranked_pu_zones = weighted_profit(trips, v.ravel(), node_ids)
        assert np.all((ranked_pu_zones['pickup_zone'].values - 1) == node_ids), 'something wrong'
        pickupscore = ranked_pu_zones['weighted_profit'].values
        pickupscore = pickupscore/pickupscore.sum()
    # Get the undirected adjacency matrix

    A = df_to_coo(edges, n)
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
    if rank == 'degree':
        degree = A.sum(axis=1).ravel()
        degree = degree/degree.sum()
        idx = np.argsort(degree)[-top:]
    elif rank =='pagerank':
        degree = v.ravel()
        idx = np.argsort(degree)[-top:]
        
    fig, ax = plot_graph(proj, adj_mx = A, rank = degree, node_ids = node_ids, 
                         id_to_name=id_to_zone, idx = idx, thumb_frac= 0.03)
    ax.grid(False)
    ax.set_title('Top {0} zones by {1} for Taxi type = {2}, Month = {3}, Year={4}' \
                 .format(top, rank, taxi.upper(), month, year))
    fig.savefig('DO_{4}_{0}_{1}_{2}_{3}.png'.format(taxi, month, year, rank, embedding),
                bbox_inches = 'tight')
    #%% Plot the best pickup zones
    if 'trips' in locals():
        idx = np.argsort(pickupscore)[-top:]
        
        fig, ax = plot_graph(proj, adj_mx = A, rank = pickupscore, node_ids = node_ids, 
                             id_to_name=id_to_zone, idx = idx, thumb_frac= 0.03)
        ax.grid(False)
        ax.set_title('Top {0} Pickup zones by profitability for Taxi type = {1}, Month = {2}, Year={3}' \
                     .format(top, taxi.upper(), month, year))
        fig.savefig('PU_{4}_{0}_{1}_{2}_{3}.png'.format(taxi, month, year, rank, embedding),
                    bbox_inches = 'tight')
    return idx


def simulation(P,node_ids,trips, count = 10, dataset=dict(), year =2016, month =1 ):
    if (year, month) in list(dataset):
        return dataset[year, month]
    total_amount = dict()
    for nodeid in node_ids:
        total_amount[nodeid] = []
        for experiment in range(count):
            total_amount[nodeid].append(random_walk(P, node_ids, trips, start=nodeid))
    return total_amount



def chechalgos(year,month, dataset=dict()):

    node_ids, P, trips, edges, n = computedate(taxi, year,month, weight)
    
    
    
    # Simulate a random walk 10 times on each zones
    total_amount = simulation(P,node_ids,trips, count = nbexp, dataset=dataset, month=month,year=year)
    pathchoosen = []
    print("phase 1")
    for nodeid in list(total_amount):
        pathchoosen += total_amount[nodeid]
    print('average total_amount made over a trip when starting at a random recommandation',
      np.mean(pathchoosen))
    
    algochoices= dict()
    total_amountexperiment = []
    """
    for experiment in range(nbexp*len(node_ids)):
        total_amountexperiment.append(random_walk(P, node_ids, trips))
    
    print('average total_amount made over 10 trips when starting at a random zone',
          np.mean(total_amountexperiment))
    """
    realylazy = [0]*top
    algochoices = dict()
    algochoices["Naive human selection"] = [1, 8,261,132, 87,232,216,184,218,219]
    algochoices["realylazy"]= realylazy
    print('Results for {0} {1}'.format(month, year))
    month2 = (month-2)%12 +1
    if month2 == 12 :
        year0 = year-1
    else:
        year0 = year
    print("phase 2")
    node_ids, P, trips, edges, n = computedate(taxi, year0,month2, weight)
    algochoices['pagerank {0}'.format('one month before')]= piclocation(edges,'pagerank', top,n,node_ids,embedding,id_to_zone,taxi,month,year,P,trips)
    total_amountsimulated = simulation(P,node_ids,trips, count = nbexp, dataset=dataset, month=month2,year=year0)
    total_amountsimulated = {k: v for k, v in sorted(total_amountsimulated.items(), key=lambda item: np.mean(item[1]))}
    algochoices['best based on a simulation of  {0}'.format('one month before')] = list(total_amountsimulated)[-10::]
    total_amountexperiment = []
    startchoosen = []
    """
    for experiment in range(top*nbexp):
        start = np.random.choice(node_ids[algochoices['pagerank one month before']])
        startchoosen += [start]
        total_amountexperiment.append(random_walk(P, node_ids, trips, start = start))
    
    print('average total_amount made over 10 trips when starting at a profit zone',
          np.mean(total_amountexperiment))
    """
    year2 = year-1
    print("phase 3")
    node_ids, P, trips, edges, n = computedate(taxi, year2,month, weight)
    algochoices['pagerank of  {0}'.format('one year before')]= piclocation(edges,'pagerank', top,n,node_ids,embedding,id_to_zone,taxi,month,year,P,trips)
    total_amountsimulated = simulation(P,node_ids,trips, count = 10, dataset=dataset, month=month,year=year2)
    total_amountsimulated = {k: v for k, v in sorted(total_amountsimulated.items(), key=lambda item: np.mean(item[1]))}
    algochoices['best based on a simulation of  {0}'.format('one year before')] = list(total_amountsimulated)[-10::]
    bestcurent = list({k: v for k, v in sorted(total_amount.items(), key=lambda item: np.mean(item[1]))})[-10::]
    algochoices['Curent top 10']= bestcurent
    '''
    plt.close('all')
    results = dict()
    for algos in algochoices :
        pathchoosen = []
        for nodeid in algochoices[algos]:
            if nodeid in total_amount:
                pathchoosen += total_amount[nodeid]
            else:
                print("unable to count {0} in algo".format(nodeid))
        print('average total_amount made over a trip when starting with {0} recommandation'.format(algos),
          np.mean(pathchoosen))
        results[algos] = pathchoosen
    results['Histogram with {0}'.format('pagerank one month before with individual simulation')]  = total_amountexperiment
    '''
    algochoices['random'] = list(total_amount)
    results= dict()
    for algos in algochoices :
        total_amountrand = []
        candidats = []
        for i in algochoices[algos] :
            if i in algochoices['random']:
                candidats += [i]
        possiblestart = candidats
        for i in range(top*nbexp):
            total_amountrand += [np.random.choice(total_amount[np.random.choice(possiblestart)])]
        results[algos]=total_amountrand
    # create with hue but without legend
    for i in list(results):
        ax = sns.histplot(results[i],kde=True, legend=True,label=('{0} : (mu = {1} sd = {2})'.format(i, int(np.mean(results[i])),int(np.std(results[i])))))
    print("phase 4")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Number of paths')
    plt.xlabel('Profit')
    plt.title('Histogram with {0}'.format('pagerank one month before with individual simulation'))
    
    plt.show()
    pd.DataFrame(results).boxplot(rot=20, fontsize=7)
    return results
#chechalgos(year,month)
import pickle

def dumppkl (filename, data):
    with open(filename, 'wb') as pklfile:
        pickle.dump( data, pklfile )
def loadpkl (filename):
    with open(filename, 'rb') as pklfile:
        data = pickle.load( pklfile )
    return data



def computebestdays():
    pathsmonths = dict()
    
    pathsmonths = loadpkl ('pathsmonths2015.plk')
    for yearbis in [2015]:
        for monthbis in range(1,13):
            try:
                if(yearbis, monthbis) not in list( pathsmonths):
                    print ('start : {0} {1}'.format(monthbis, yearbis))
                    node_ids, P2, trips2, edges2, n = computedate(taxi, yearbis,monthbis, weight)
                    total_amountsimulated = simulation(P2,node_ids,trips2, count = nbexp)
                    #total_amountsimulated = {k: v for k, v in sorted(total_amountsimulated.items(), key=lambda item: np.mean(item[1]))}
                    #bestdays[yearbis, monthbis] = list(total_amountsimulated)[-10::]
                    pathsmonths[yearbis, monthbis]= total_amountsimulated
                    print ('end : {0} {1}'.format(monthbis, yearbis))
                    dumppkl ('pathsmonths2015.plk', pathsmonths)
            except:
                pass
#computebestdays()
"""
dataset = dict()
datasetsimu = dict()
yearsanalysis = range(2018,2021)
for yearanalysis in yearsanalysis:
    pathsmonths = loadpkl ('pathsmonths{0}.plk'.format(yearanalysis))
    datasetsimu.update(pathsmonths)
    for months in list(pathsmonths):
         valeursdataset = dict()
         for zone in list(pathsmonths[months[0],months[1]]):
             if zone in list(id_to_zone):
                 name = id_to_zone[zone]
             else:
                 name = zone
             valeursdataset[name] = np.mean(pathsmonths[months[0],months[1]][zone])
         dataset[yearanalysis,months[1]] = valeursdataset

algosresults = dict()

for onedate in list(datasetsimu):
    print(onedate)
    res = chechalgos(onedate[0],onedate[1],datasetsimu )
    meanres = dict()
    for algo in list(res):
        meanres[algo]=np.mean(res[algo])
    algosresults[onedate] = meanres


#pd.DataFrame(dataset).to_csv("datasettaximax.csv")
pd.DataFrame(algosresults).to_csv("algosresultstaximax.csv")
"""
#https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/

'''

plt.close('all')
plt.hist(total_amountexperiment )
plt.ylabel('Number of paths')
plt.xlabel('Profit')
plt.title('Histogram with {0}'.format('pagerank one month before with individual simulation'))
plt.show()
'''
'''

'''