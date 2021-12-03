---------------------------------------------------------------------------------------
DESCRIPTION
---------------------------------------------------------------------------------------
This package contains all the relevant code for our NYC Taxi Pickup Zone recommendation project.  We use both a PageRank algorithm
and a RandomWalk algorithm to show NYC Taxi drivers the best place to pick up fares.

The PageRank algorithm provides a score for each of the zones which are then used in our visualization as an overview of where a driver
could head in NYC to start their day if they could start anywhere.

The Random Walk algorithm is used to compare zones that are nearby the user's current zone to show where they might head to pick up a fare close by.
It takes the current zone, the hours left in the driver's day, and how far they'd be willing to drive to a new zone as input.  It then finds all the
neighboring zones within that willing to drive time and simulates 50 random walks within the time left in the drivers days from those zones using a
steady-state probability distribution.  It then calculates the average of those walks and returns that information to the user.

visualization.py - Contains all code relevant to our interactive visualization.

util.py - Contains code to run our PageRank algorithm and Random Walk algorithm for the visualization.  It also contains...

test_embedding.py - Contains...

test_pagerank.py - Contains...

SourceDataPull.ipynb - Contains the code and SQL query we used to pull and aggregate the data from our PostgreSQL database.

Data Folder:
TAXI_trips_XXXX.csv - Contains our aggregated NYC Taxi trip data for each pickup and drop-off zone.  As well as
month, day or night field, the day of the week, average total fare, average trip time, and average time to make the trip.

centers.csv - Contains the longitude and latitude center for each zone to be used in the visualization for markers.

taxi_geo_small.json - Contains the geo JSON for all the taxi zones in NYC to be used in the visualization.

TaxiZone_Name_Borough.csv - Contains each zones id, name, and the borough they're located in for the hover data in our visualization.

---------------------------------------------------------------------------------------
INSTALLATION
---------------------------------------------------------------------------------------
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

---------------------------------------------------------------------------------------
EXECUTION
---------------------------------------------------------------------------------------
Once again, we suggest you head to: https://dva-warriors.azurewebsites.net/ 
or our backup site: https://dva-warriors-backup.azurewebsites.net/

If installing locally,

1. Enter the CMD.exe prompt in Anaconda and navigate to the root directory of this project
2. In the command prompt type: PYTHON visualization.py
3. Then simply head to http://127.0.0.1:8050/ in your favorite browser (Chrome Preferred) to view our visualization.  


**********Notes for myself*********

Piazza Post on scope of Readme:
https://piazza.com/class/kqihppjwbhk3vf?cid=1865

README.txt - a concise, short README.txt file, corresponding to the "user guide". This file should contain:

DESCRIPTION - Describe the package in a few paragraphs
INSTALLATION - How to install and setup your code
EXECUTION - How to run a demo on your code


[Optional, but recommended] DEMO VIDEO - Include the URL of a 1-minute *unlisted* YouTube video in this txt file. 
The video would show how to install and execute your system/tool/approach 
(e.g, from typing the first command to compile, to system launching, and running some examples). 
Feel free to speed up the video if needed (e.g., remove less relevant video segments). 
This video is optional (i.e., submitting a video does not increase scores; not submitting one does not decrease scores). 
However, we recommend teams to try and create such a video, 
because making the video helps teams better think through what they may want to write in the README.txt, 
and generally how they want to "sell" their work.

