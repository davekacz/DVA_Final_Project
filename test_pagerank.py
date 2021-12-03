#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns; sns.set()
from util import Utility
from datetime import date
plt.close('all')
taxi = 'yellow'
day_night = 'night'
dow = 'weekend'
ut = Utility(taxi = taxi, day_night = day_night, 
             day_of_week = dow, duration = 8,
             exclude_ids = None, nexp = 1000)
#%% Check for input data consistency
start  = date(2016,1,1)
end = date(2019,12,1)
top = 10

zones = pd.read_csv('data/taxi_zones.csv')
id_to_zone = dict(zip(zones['locationid'], zones['zone']))
#%% run the monthly simulation
df = ut.monthly_sim(start, end, top)

fig, ax =  plt.subplots(figsize=(10, 10))
for col in df.columns:
    ax.plot(df.index, df[col], label = col)
# Major ticks every 6 months.
fmt_half_year = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(fmt_half_year)  

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)     
# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()
ax.set_xlabel('month')
ax.set_ylabel('Average total amount made over {0} experiments'.format(ut.nexp))
ax.set_title('''Random vs. Top {0} starting nodes from the current month 
             {1} hr randomwalks during {2} of {3}s'''.format(top, ut.duration, 
             ut.day_night, ut.dow))
ax.legend()
fig.savefig('random_vs_pagerank_current_month_{0}_{1}.png'.format(day_night,
                                                                  dow), bbox_inches = 'tight')
df.to_csv('random_vs_pagerank_current_month_{0}_{1}.csv'.format(day_night,
                                                                  dow))
#%% run the monthly predictions for the next month
df = ut.monthly_pred(start, end, top)

fig, ax =  plt.subplots(figsize=(10, 10))
for col in df.columns:
    ax.plot(df.index, df[col], label = col)
# Major ticks every 6 months.
fmt_half_year = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(fmt_half_year)  

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)     
# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()
ax.set_xlabel('month')
ax.set_ylabel('Average total amount made over {0} experiments'.format(ut.nexp))
ax.set_title('''Random vs. Top {0} starting nodes from the previous month
             {1} hr randomwalks during {2} of {3}s'''.format(top, ut.duration, 
             ut.day_night, ut.dow))
ax.legend()
fig.savefig('random_vs_pagerank_next_month_{0}_{1}.png'.format(day_night,
                                                               dow), bbox_inches = 'tight')
df.to_csv('random_vs_pagerank_next_month_{0}_{1}.csv'.format(day_night,
                                                               dow))
#%% Compare the distributions of total amounts of random vs. pagerank starting nodes
year = 2017
month = 7
edges, trips = ut.load_data(month, year)
P, node_ids, idx = ut.pagerank(edges, trips, top)
try:
    random_start = ut.simulation(P, node_ids, trips)
    pr_start = ut.simulation(P, node_ids, trips, idx)
except:
    print('something wrong')
df = pd.DataFrame({'random': random_start, 'pagerank': pr_start})
fig, ax = ut.dist_plot(df)
ax.set_ylabel('probability density')
ax.set_xlabel('total amount')
ax.set_title('''Probability distribution of total amount made from a random 
             starting point vs. a top {0} node for {1}-{2} \
(during {3} of {4}s)'''.format(top, year, month, ut.day_night, ut.dow))
ax.legend()
fig.savefig('kde_{0}_{1}_{2}_{3}.png'.format(month, year,
                                             day_night, dow), bbox_inches = 'tight')