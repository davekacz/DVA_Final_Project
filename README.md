
## DESCRIPTION

This package contains all the relevant code for our NYC Taxi Pickup Zone recommendation project.  We use both a PageRank algorithm and a RandomWalk algorithm to show NYC Taxi drivers the best place to pick up fares.

The PageRank algorithm provides a score for each of the zones which are then used in our visualization as an overview of where a driver could head in NYC to start their day if they could start anywhere with the goal of providing the most income to the driver.

The Random Walk algorithm is used to compare zones that are nearby the user's current zone to show where they might head to pick up a fare close by. It takes the current zone, the hours left in the driver's day, and how far they'd be willing to drive to a new zone as input.  It then finds all the neighboring zones within that willing to drive time and simulates 100 random walks for the remaining time (after accounting for the unproductive time spent traveling to the neighboring zone). The transition probability matrix used in the RandomWalk model is the same as that used for PageRank. It then calculates the average of those walks and returns that information to the user.

**visualization.py** - Contains all code relevant to our interactive visualization implemented using Dash/Plotly.

**util.py** - Contains code to run our PageRank algorithm and Random Walk algorithm for the visualization. The core functionality is implemented in a class named Utility. This class contains methods for loading historical data into memory, building a graph and transition probability matrix, Pagerank, RandomWalk, and finding the best neighboring zones. All other code files reference this base class.

**test_embedding.py** - Script to generate 2D graph embeddings of the data using a kernel PCA approach. This is still experimental and not implemented in the visualization

**test_pagerank.py** - Script to evaluate the PageRank model versus a random selection of pickup zones. Generates plots comparing the average total amount a driver can make in a 8 hr day based on 1000 RandoWalks when picking a zone at random vs. picking a top 10 zone as recommended by the PageRank model

**SourceDataPull.ipynb** - Contains the code and SQL query we used to pull and aggregate the data from our PostgreSQL database.

### Data Folder:
**TAXI_trips_XXXX.csv** - Contains our aggregated NYC Taxi trip data for each pickup and drop-off zone.  As well as month, day or night field, the day of the week, average total fare, average trip time, and average time to make the trip.

**centers.csv** - Contains the longitude and latitude center for each zone to be used in the visualization for markers.

**taxi_geo_small.json** - Contains the geo JSON for all the taxi zones in NYC to be used in the visualization.

**TaxiZone_Name_Borough.csv** - Contains each zones id, name, and the borough they're located in for the hover data in our visualization.

## INSTALLATION

If you would like to skip the installation, we've hosted the project in two places.  Simply head to: https://dva-warriors.azurewebsites.net/

If for some reason that page is not working or down, our backup host is located here, albeit running a bit slower: https://dva-warriors-backup.azurewebsites.net/

To set up the environment locally:

1. Create a new python 3.8 environment in Anaconda.
2. Install the CMD.exe prompt in the Home Tab and open it.  
3. Navigate to the root directory for this project.
4. In the command prompt type: conda install --file requirements.txt
5. If prompted to proceed, enter y (YES).

The following libraries and versions will be installed to your enviorment.  
pandas
numpy
seaborn
scipy
dash==1.19.0
plotly==5.4.0
plotly_express==0.4.1
matplotlib
Brotli==1.0.9
click==8.0.3
dash-core-components==1.15.0
dash-html-components==1.1.2
dash-renderer==1.9.0
dash-table==4.11.2
Flask==2.0.2
Flask-Compress==1.10.1
importlib-metadata==4.8.2
itsdangerous==2.0.1
Jinja2==3.0.3
MarkupSafe==2.0.1
six==1.16.0
tenacity==8.0.1
typing_extensions==4.0.0
Werkzeug==2.0.2
zipp==3.6.0

## EXECUTION

Once again, we suggest you head to: https://dva-warriors.azurewebsites.net/ 
or our backup site: https://dva-warriors-backup.azurewebsites.net/

If installing locally,

1. Enter the CMD.exe prompt in Anaconda and navigate to the root directory of this project
2. In the command prompt type: PYTHON visualization.py
3. Then simply head to http://127.0.0.1:8050/ in your favorite browser (Chrome Preferred) to view our visualization.  

To interact with the visualization simply choose which dataset you'd like to work with from the first set of dropdowns.  
The PageRank algorithm will update automaticaly.  
If you'd like to enter the RandomWalk algorithm, simply click on a zone, which reprsents where the cab driver is.  
The reccomendations can be customized by the two time sliders above the map.  
To return to pagerank mode, simply double click on another zone.  


We have also included some other work we built while in the exploratory phase of this project.    

If you run in the command prompt: python test_embedding.py, plots will be created in the root directory showing a 2d graph embedding of the top 10 drop off and pickup zones.  

If you run in the command prompt: python test_pagerank.py, plots will be generated evaluateing our PageRank model with random selection of pickup zones.  
