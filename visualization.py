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
# import ipdb
# import plotly.io as pio
#Page rank and Random Walk Functions
from util import Utility

#set access token for mapbox
px.set_mapbox_access_token('pk.eyJ1IjoiZGF2ZWthY3oiLCJhIjoiY2t2ZnEyNTVnNDNvNDJvcXBvdGpkd2V6OCJ9.KZLuPalXbe5r40WG13fcUg')


app = dash.Dash(__name__)

#initialize our utility algorithms
util = Utility(taxi = 'yellow', day_night = 'night', 
                 day_of_week = 'weekday', duration = 8)

#pull in zone names and borough location - other data we want to show could be added here 
zone_names = pd.read_csv('data/TaxiZone_Name_Borough.csv')

#read in geojson and centers
taxi_geo = json.load(open('data/taxi_geo_small.json'))
centers = pd.read_csv('data/centers.csv')
centers.set_index('zone_id', inplace = True)
home_position = {"lat": 40.6908, "lon": -74.0060}
home_zoom = 10
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
    
    #Text above slider bar2
    html.H4(children='''Select how many minutes you want to drive to your pickup zone:'''),
    
    #slider bar to select how many hours are left in the day
    html.Div([
        dcc.Slider(
            id='time_slider_driving_duration',
            min = 0,
            max = 4,
            step = None,            
            marks={
            1: '15',
            2: '30',
            3: '45',
            4: '60'
            },
            value= 1 #defaults to the first option
        ),
        ], 
        style={'width': '15%'}
    ),    
    
    
    #Text above slider bar
    html.H4(children='''Select the zone you are currently in, double click another zone to reset your selection:'''),

    #Inset Chloropeth Graph
    dcc.Graph(
        id='choropleth',
        figure = {},
        style={'width': '75%', 'display': 'inline-block'},
        config = {'doubleClick': 'reset+autosize',
                  'doubleClickDelay': 600} 
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
    html.H5(id='slider-output-container', style={'display': 'inline-block'}),
    dcc.Store(id = 'location')
])

@app.callback(
    Output('selected-data', 'children'),
    Output('choropleth', 'figure'),
    Output('slider-output-container', 'children'),    
    Output('choropleth', 'selectedData'),
    Output('location', 'data'),
    Input('choropleth', 'selectedData'),
    Input('choropleth', 'relayoutData'), 
    Input('month_select', 'value'),
    Input('day_type_select', 'value'),
    Input('time_select', 'value'),
    Input('time_slider', 'value'),
    Input('year_select', 'value'),
    Input('time_slider_driving_duration', 'value'),
    Input('choropleth', 'figure'),
    Input('location', 'data'))

def display_selected_data(selectedpoints, relaydata,
                          month_selection, day_selection,
                          time_selection, time_slider, year_selection,
                          time_slider_driving_duration, figure,
                          last_clicked):
        # print('clickedpoint', clickedpoint)
        print ('selectedpoints:', selectedpoints)
        print ('relaypoints:', relaydata)
        fig = figure
        if last_clicked is None:
            location = 'null'
        else:
            location  = json.loads(last_clicked)
        # ipdb.set_trace()
        if relaydata is not None:
            if ('mapbox.center' in relaydata):
                if ((relaydata['mapbox.center'] == home_position) and 
                    (relaydata['mapbox.zoom'] == home_zoom)):
                    run_callback = True
                    selectedpoints = None
                    last_clicked = json.dumps(None)
                    location = 'null'
                elif selectedpoints is not None:
                    temp = selectedpoints['points'][0]['location']
                    if temp == location:
                        run_callback = False
                    else:
                        run_callback = True
                        location = selectedpoints['points'][0]['location']
                        last_clicked = json.dumps(location)
                else:
                    if location == 'null':
                        run_callback = True
                    else:
                        run_callback = False
            elif selectedpoints is not None:
                temp = selectedpoints['points'][0]['location']
                if temp == location:
                    run_callback = False
                else:
                    run_callback = True
                    location = selectedpoints['points'][0]['location']
                    last_clicked = json.dumps(location)
            else:
                if location == 'null':
                    run_callback = True
                else:
                    run_callback = False
        elif selectedpoints is not None:
            temp = selectedpoints['points'][0]['location']
            if temp == location:
                run_callback = False
            else:
                run_callback = True
                location = selectedpoints['points'][0]['location']
                last_clicked = json.dumps(location)
        else:
            if location == 'null':
                run_callback = True
            else:
                run_callback = False
        # ctx = dash.callback_context
        # changed_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # print(changed_id)

        #update utility with relevant information: 
        if run_callback:
            util.day_of_week = day_selection
            util.day_night = time_selection
            util.duration = time_slider
            year = year_selection
            month = month_selection
            transition_time = (int(time_slider_driving_duration)*15)/60
        #Checks if there are any selected points, returns zoomed out map if None
            if location == 'null':
                print('No Zone Selected')
                #If no zones are selected, return unzoomed overall map
                #run pagerank algorithm:
                edges, trips = util.load_data(month, year)
                P, node_ids, idx = util.pagerank(edges, trips, 10)
    
                #create score dictionary
                score_dict = dict(zip(node_ids, util.pickupscore_))
    
                #create dataframe for choropleth map
                #Code to create choro_df for visualization
                zones = [x + 1 for x in range(263)]
    
                choro_df = pd.DataFrame(zones, columns = ['Zone_ID'])
                choro_df['pickup_score'] = choro_df['Zone_ID'].map(score_dict)
                choro_df.fillna(0, inplace=True)
        
                choro_df = pd.merge(choro_df, zone_names, left_on='Zone_ID', right_on='LocationID')
                choro_df.drop(columns='LocationID', inplace=True)
                
                df = choro_df
                color_data = 'pickup_score'
                location_ID = "Zone_ID"
                fkey = "properties.locationid"
                center_data = home_position
                color_scale = "RdBu"
                zoom_level = home_zoom
                hover_dataset = ['borough']
                color_midpoint = np.mean(choro_df['pickup_score'].values)
                range_color = [choro_df['pickup_score'].min(), 
                               choro_df['pickup_score'].max()]
    
            else:
                print('Calculating top neigbors...')
                neighbors = util.top_neighbor(location, month, year,
                                              time_slider, transition_time)
                print('done!')
                base_amount = neighbors.loc[location, 'expected_total_amount']
                
                neighbors['pct_extra'] = (100 * (neighbors['expected_total_amount'] 
                                                - base_amount)/base_amount)         
                neighbors = pd.merge(zone_names,neighbors,how = 'left',
                                    left_on='LocationID', right_index = True)
                
                df = neighbors
                color_data = 'pct_extra'
                location_ID = "LocationID"
                fkey = "properties.locationid"
                center_data = {"lat": centers.loc[location, 'avg_lat'],
                               "lon": centers.loc[location, 'avg_long']}
                color_scale = "RdBu"
                zoom_level = 11.5
                color_midpoint = 0
                range_color = [neighbors['pct_extra'].min(), 
                               neighbors['pct_extra'].max()]
                hover_dataset = ['borough', 'avg_trip_time', 'expected_total_amount']
                # neighbors['pct_extra'].fillna(-100, inplace=True)
            fig = px.choropleth_mapbox(
                            df, 
                            geojson=taxi_geo, 
                            color=color_data,
                            locations=location_ID, 
                            featureidkey=fkey,
                            center=center_data,
                            color_continuous_scale=color_scale,
                            range_color = range_color,
                            color_continuous_midpoint  = color_midpoint,
                            opacity = .5,
                            mapbox_style="streets", 
                            zoom=zoom_level,
                            height = 800,
                            hover_name = 'zone',
                            hover_data=hover_dataset,
                            )
    
            fig.update_mapboxes(pitch=45)
            fig.update_layout(clickmode='event+select', 
                              title = 'NYC Cabbie Director', 
                              coloraxis_showscale=True,
                              margin={"r":0,"t":0,"l":0,"b":0})

        #Add Scatter Plot to render the Best Location to pickup
        #symbols would be cool, but work differently, don't render if close unless you zoom in.
        # if (location != 'NULL') and neighbors['pct_extra'].idxmax():
        #     best_zone = neighbors.loc[neighbors['pct_extra'].idxmax(), 'LocationID']
        #     fig.add_scattermapbox(lat = [centers.loc[best_zone, 'avg_lat']],
        #             lon = [centers.loc[best_zone, 'avg_long']],
        #             mode = 'markers+text',
        #             text = ['Best Location'],  #a list of strings, one  for each geographical position  (lon, lat)              
        #             below='', 
        #             marker_size=15, marker_color='rgb(0,0,255)', 
        #             textposition = "bottom center", textfont=dict(size=16, color='black'),
        #             name = 'Best Location')

        # #Add Scatter Plot to render the Current location 
        # if location != 'NULL':
        #     fig.add_scattermapbox(lat = [centers.loc[location, 'avg_lat']],
        #             lon = [centers.loc[location, 'avg_long']],
        #             mode = 'markers+text',
        #             text = ['Current Location'],  #a list of strings, one  for each geographical position  (lon, lat)              
        #             below='',                 
        #             marker_size=15, marker_color='rgb(235, 0, 100)',
        #             textposition = "bottom center", textfont=dict(size=16, color='black'),
        #             name = 'Current Location')
        # ipdb.set_trace()
        return location, fig, time_slider, selectedpoints, last_clicked

if __name__ == '__main__':
    #Use this line if running locally
    app.run_server(debug=True)

    #Uncomment this line if hosting on the web, and comment out above line
    #app.run_server(debug=True, host='0.0.0.0', port='80')