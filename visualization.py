# Download the libraries mentioned below and
# Run this app with `python visualization.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
from dash.html.Select import Select
import plotly.express as px
import pandas as pd
import json
import numpy as np 
import ipdb
#Page rank and Random Walk Functions
from util import Utility

#set access token for mapbox
px.set_mapbox_access_token('pk.eyJ1IjoiZGF2ZWthY3oiLCJhIjoiY2t2ZnEyNTVnNDNvNDJvcXBvdGpkd2V6OCJ9.KZLuPalXbe5r40WG13fcUg')
# ---------------------------------
#               TO DO
# 1 - Need a function from Karns work - 
# Input: Zone Selected, 3 Values from Dropdowns, time left slider
# Output: pd.DataFrame with all zone id's - ranks for choropleth - 
# any other data we want to show - avg fare, avg time to travel
# borough, etc...  
#
# 2 - Need centers of each zone csv
#
# 3 - Need to highlight the selection in some way.  Outline - or maybe just red dot in center?  
#
#-------------------------------------

app = dash.Dash(__name__)

#read data on zone information - this will be removed, final version
# zone_df = pd.read_csv('data/zone_info.csv')

#initialize our utility algorithms
util = Utility(taxi = 'yellow', day_night = 'night', 
                 day_of_week = 'weekday', duration = 8)

#pull in zone names and borough location - other data we want to show could be added here 
zone_names = pd.read_csv('data/TaxiZone_Name_Borough.csv')

#read in geojson
taxi_geo = json.load(open('data/taxi_geo_small.json'))
centers = pd.read_csv('data/centers.csv')
centers.set_index('zone_id', inplace = True)
#create selection dataframe this was going to be used to show selection, all 0's, 1 for the selection
#Don't think it's going to work like this though
'''location_id = [x for x in range(264)]
select_df = pd.DataFrame(location_id, columns=['location_id'])
select_df['score'] = 0'''

#The original figure - Leftover from original code - think this should go and be updated using callback
#when the dropdowns initialize and cause the callback function to be triggered.   
'''fig = px.choropleth_mapbox(zone_df, geojson=taxi_geo, color='first',
                        locations="zone", featureidkey="properties.locationid",
                        center={"lat": 40.7128, "lon": -74.0060},
                        color_continuous_scale="greens",
                        opacity = .35,
                        mapbox_style="open-street-map", 
                        zoom=10,
                        height = 800,
                        hover_name = 'name',
                        hover_data=['borough']
                        )

fig.update_layout(clickmode='event+select')'''

#Create the website layout
app.layout = html.Div(children=[
    html.H1(children='NYC Cabbie Director'), 
    html.H4(children='''Welcome to the NYC Cabbie Director!  Our app hopes to help NYC Yellow cab drivers
                        find a zone near their current zone that will lead them to having a more profitable day!'''),
    html.H4(children='''Select the month, weekday or weekend, and the time, and the year to query:'''),
    #First Dropdown
    html.Div([
        dcc.Dropdown(
        id='month_select',
        options=[
            {'label': 'January', 'value': 1},
            {'label': 'February', 'value': 2},
            {'label': 'March', 'value': 3},
            {'label': 'April', 'value': 4},
            {'label': 'May', 'value': 5},
            {'label': 'June', 'value': 6},
            {'label': 'July', 'value': 7},
            {'label': 'August', 'value': 8},
            {'label': 'September', 'value': 9},
            {'label': 'October', 'value': 10},
            {'label': 'Novembor', 'value': 11},
            {'label': 'December', 'value': 12},
        ], 
        value=1 #defaults to the first option
        ),
    ], style={'width': '15%', 'display': 'inline-block'}),

    #Second Dropdown
    html.Div([
        dcc.Dropdown(
        id='day_type_select',
        options=[
            {'label': 'Weekday', 'value': 'weekday'},
            {'label': 'Weekend', 'value': 'weekend'},
        ], 
        value='weekday' #defaults to the first option
        ),
    ], style={'width': '15%', 'display': 'inline-block'}), 

    #Third Dropdown
    html.Div([
        dcc.Dropdown(
        id='time_select',
        options=[
            {'label': 'Day (6:00 AM - 6:00 PM)', 'value': 'day'},
            {'label': 'Night (6:00 PM - 6:00 AM)', 'value': 'night'},
        ], 
        value='day' #defaults to the first option
        ),
    ], style={'width': '15%', 'display': 'inline-block'}),

    #Fourth Dropdown
    html.Div([
        dcc.Dropdown(
        id='year_select',
        options=[
            {'label': '2019', 'value': 2019},
            {'label': '2018', 'value': 2018},
            {'label': '2017', 'value': 2017},
            {'label': '2016', 'value': 2016},
        ], 
        value=2019 #defaults to the first option
        ),
    ], style={'width': '15%', 'display': 'inline-block'}),


    #Text above slider bar
    html.H4(children='''Select how many hours are left in your day:'''),

    #slider bar to select how many hours are left in the day
    html.Div([
        dcc.Slider(
        id='time_slider',
        min = 1,
        max = 8,
        step = 1,
        marks={
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',

    },
        value= 8 #defaults to the first option
        ),

    ], style={'width': '60%'}),
    
    #Text above slider bar
    html.H4(children='''Select the zone you are currently in, double click another zone to reset your selection:'''),

    #Inset Chloropeth Graph
    dcc.Graph(
        id='choropleth',
        #figure = fig,
        style={'width': '75%', 'display': 'inline-block'},
        config = {'doubleClick': 'reset+autosize'} 
    ),

    #Show what's currently clicked
    html.Div([
            dcc.Markdown("""
                **Click Data**

                Selected Point:
            """),
            #The portion that actually shows the selection
            html.Div(id='selected-data')
        ]),
    html.H5(children='Hours left in your day: ', style={'display': 'inline-block'}),
    html.H5(id='slider-output-container', style={'display': 'inline-block'})
])

#Old callback - can't have two callbacks updating the same figure - 
#in our case our choropleth mapbox 
'''@app.callback(
    Output('choropleth', 'figure'),
    Input('month_select', 'value')
)

#takes arguments in order from the callback function, in this case the month_select value

def update_choropleth_dropdowns(score_value):
    #returns an updated figure
    fig = px.choropleth_mapbox(zone_df, geojson=taxi_geo, color=score_value,
                           locations="zone", featureidkey="properties.locationid",
                           center={"lat": 40.7128, "lon": -74.0060},
                           color_continuous_scale="greens",
                           opacity = .35,
                           mapbox_style="open-street-map", 
                           zoom=10,
                           height = 800,
                           hover_name = 'name',
                           hover_data=['borough']
                           )

    fig.update_layout(clickmode='event+select')

    return fig'''

#Callback function - Updates if zone selected in choropleth, or one of the drodowns
#month_select, day_type_select, time_select

@app.callback(
    Output('selected-data', 'children'),
    Output('choropleth', 'figure'),
    Output('slider-output-container', 'children'),
    Input('choropleth', 'selectedData'),    
    Input('month_select', 'value'),
    Input('day_type_select', 'value'),
    Input('time_select', 'value'),
    Input('time_slider', 'value'),
    Input('year_select', 'value'))

def display_selected_data(selectedpoints, month_selection, day_selection,
                          time_selection, time_slider, year_selection):

    #supposed to show how and why the callback was called - can't figure out
    #can maybe just use if statement below instead?
    ipdb.set_trace()
    ctx = dash.callback_context
    changed_id = ctx.triggered[0]['prop_id'].split('.')[0]

    print(changed_id)
    # ------------
    #Should call Karns algorith here, pass it all the inputs here,
    #output our zone_df 
    # ------------

    #update utility with relevant information: 
    util.day_of_week = day_selection
    util.day_night = time_selection
    util.duration = time_slider

    #Checks if there are any selected points, returns zoomed out map if None
    if not selectedpoints:
        #If no zones are selected, return unzoomed overall map

        #run pagerank algorithm:
        year = year_selection
        month = month_selection
        edges, trips = util.load_data(month, year)
        P, node_ids, idx = util.pagerank(edges, trips, 10)

        #create score dictionary
        score_dict = dict(zip(node_ids, util.pickupscore_))

        #create dataframe for choropleth map
        #Code to create choro_df for visualization

        zones = [x + 1 for x in range(265)]

        choro_df = pd.DataFrame(zones, columns = ['Zone_ID'])
        choro_df['pickup_score'] = choro_df['Zone_ID'].map(score_dict)
        choro_df.replace(np.nan, 0, inplace=True)
 
        choro_df = pd.merge(choro_df, zone_names, left_on='Zone_ID', right_on='LocationID')
        choro_df.drop(columns='LocationID', inplace=True)

        #fig = {} #attempt to reset figure - trying to remove current/best points

        fig = px.choropleth_mapbox(choro_df, geojson=taxi_geo, color='pickup_score',
                        locations="Zone_ID", featureidkey="properties.locationid",
                        center={"lat": 40.6908, "lon": -74.0060},
                        color_continuous_scale="greens",
                        opacity = .5,
                        mapbox_style="streets", 
                        zoom=10,
                        height = 800,
                        hover_name = 'zone',
                        hover_data=['borough'],
                        )

        fig.update_mapboxes(pitch=45)
        fig.update_layout(clickmode='event+select', title = 'NYC Cabbie Director', coloraxis_showscale=False,
        margin={"r":0,"t":0,"l":0,"b":0})

        print('No Zone Selected')

        #Return NA for no zone selected, and the mapbox, and the time selected
        return 'NA', fig, time_slider

    #If a point is selected... returns zoomed in - need table of zone centers to zoom to
    else:
        location = int(json.dumps(selectedpoints['points'][0]['location']))

        #update the selection DF for selection visualization - don't think this is goign to work
        '''select_df = pd.DataFrame(location_id, columns=['location_id'])
        select_df['score'] = 0
        select_df['score'].iloc[int(location)] = 1'''
        neighbors = util.top_neighbor(location, month_selection, 
                                      year_selection, time_slider)
        base_amount = neighbors.loc[location, 'expected_total_amount']
        
        neighbors['pct_extra'] = (100 * (neighbors['expected_total_amount'] 
                                         - base_amount)/base_amount)
        neighbors = pd.merge(zone_names, neighbors, how = 'left',
                             left_on = 'LocationID', right_index = True)
        # neighbors.fillna(0, inplace=  True)
        #if a zone is selected, zoom in, show zone dependent mapbox
        fig = px.choropleth_mapbox(neighbors, geojson=taxi_geo, color='pct_extra',
                locations="LocationID", featureidkey="properties.locationid",
                center={"lat": centers.loc[location, 'avg_lat'], 
                        "lon": centers.loc[location, 'avg_long']},
                color_continuous_scale="rainbow",
                opacity = .5,
                mapbox_style="streets", 
                zoom=11.5,
                height = 800,
                hover_name = 'zone',
                hover_data=['borough', 'avg_trip_time', 'expected_total_amount'],
                )

        #Adds pitch to camera instead of top down view
        fig.update_mapboxes(pitch=45)
        # ipdb.set_trace()
        #Adds title and removes the choropleth scale on the right.
        #fig.update_layout(title = 'NYC Cabbie Director', coloraxis_showscale=False)

        #Add Scatter Plot to render the Best Location to pickup
        #symbols would be cool, but work differently, don't render if close unless you zoom in.
        if neighbors['pct_extra'].idxmax():
            best_zone = neighbors.loc[neighbors['pct_extra'].idxmax(), 'LocationID']
            fig.add_scattermapbox(lat = [centers.loc[best_zone, 'avg_lat']],
                    lon = [centers.loc[best_zone, 'avg_long']],
                    mode = 'markers+text',
                    text = ['Best Location'],  #a list of strings, one  for each geographical position  (lon, lat)              
                    below='', 
                    marker_size=15, marker_color='rgb(0,0,255)', 
                    textposition = "bottom center", textfont=dict(size=16, color='black'),
                    name = 'Best Location')

        #Add Scatter Plot to render the Current location 
        fig.add_scattermapbox(lat = [centers.loc[location, 'avg_lat']],
                lon = [centers.loc[location, 'avg_long']],
                mode = 'markers+text',
                text = ['Current Location'],  #a list of strings, one  for each geographical position  (lon, lat)              
                below='',                 
                marker_size=15, marker_color='rgb(235, 0, 100)',
                textposition = "bottom center", textfont=dict(size=16, color='black'),
                name = 'Current Location')

        #Remove legend on the right - selections do weird things
        #fig.update(layout_showlegend=False)

        #Disabled allowing anymore selections after the first one.  Behaviour is a little odd and unexpected.  
        #fig.update_layout(clickmode='event+select', margin={"r":0,"t":0,"l":0,"b":0})

        #Return zone selected, and the mapbox
        return location, fig, time_slider





if __name__ == '__main__':
    #Use this line if running locally
    app.run_server(debug=True)

    #Uncomment this line if hosting on the web, and comment out above line
    #app.run_server(debug=True, host='0.0.0.0', port='80')