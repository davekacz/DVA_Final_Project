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
home_view = {'center': {"lat": 40.6908, "lon": -74.0060},
             'zoom': 10}
#Create the website layout
app.layout = html.Div(children=[
    html.H1(children='NYC Cabbie Director', className="title"), 
    html.H4(children='''Welcome to the NYC Cabbie Director!  Our app hopes to help NYC Yellow cab drivers
                        find a zone near their current zone that will lead them to having a more profitable day!''', className="subtitle"),
    html.H4(children='''Select the month, weekday or weekend, and the time, and the year to query:''', className="subtitle"),
    html.Br(),
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
    ], style={'width': '25%', 'display': 'inline-block', 'align-items': 'center', 'justify-content': 'center'}),
    
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
    ], style={'width': '25%', 'display': 'inline-block', 'align-items': 'center', 'justify-content': 'center'}),
    
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
    ], style={'width': '25%', 'display': 'inline-block', 'align-items': 'center', 'justify-content': 'center'}),


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
    ], style={'width': '25%', 'display': 'inline-block', 'align-items': 'center', 'justify-content': 'center'}),


    #Text above slider bar
    html.Br(),			  
    html.Br(), 	
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
        value= 4 #defaults to the first option
        ),

    ], style={
            "width": "100%",
            "height": "100%"
        }		 
    ),
    
    #Text above slider bar2
    html.H4(children='''Select how many minutes you want to drive to your pickup zone:'''),
    
    #slider bar to select how many hours are left in the day
    html.Div([
        dcc.Slider(
            id='time_slider_driving_duration',
            min = 1,
            max = 3,
            step = None,            
            marks={
            1: '10',
            2: '20',
            3: '30'
            },
            value= 2 #defaults to the first option
        ),
        ], 
        style={
            "width": "100%",
            "height": "100%"
        }	 
    ),       
    
    
    #Text above slider bar
    html.H4(children='''Select the zone you are currently in, double click anywhere zone to reset your selection:'''),

    #Inset Chloropeth Graph
    dcc.Graph(
        id='choropleth',
        figure = {},
        style={
            "width": "100%",
            "height": "100%"
        },
        config = {'doubleClick': 'reset+autosize'} 
    ),

    #Show what's currently clicked
    html.Div([
            dcc.Markdown("""Selected Zone:"""),
            #The portion that actually shows the selection
            html.Div(id='selected-data')
        ]),
    
    # html.H5(children='Hours left in your day: ', style={'display': 'inline-block'}),
    # html.H5(id='slider-output-container', style={'display': 'inline-block'}),
    dcc.Store(id = 'lastselection'),
    dcc.Store(id = 'lastzoom')
])

@app.callback(
    Output('selected-data', 'children'),
    Output('choropleth', 'figure'),
    # Output('slider-output-container', 'children'),    
    # Output('choropleth', 'selectedData'),
    Output('lastselection', 'data'),
    Output('lastzoom', 'data'),
    Input('choropleth', 'selectedData'),
    Input('choropleth', 'relayoutData'), 
    Input('month_select', 'value'),
    Input('day_type_select', 'value'),
    Input('time_select', 'value'),
    Input('time_slider', 'value'),
    Input('year_select', 'value'),
    Input('time_slider_driving_duration', 'value'),
    Input('choropleth', 'figure'),
    Input('lastselection', 'data'),
    Input('lastzoom', 'data'))

def display_selected_data(selectedpoints, relaydata,
                          month_selection, day_selection,
                          time_selection, time_slider, year_selection,
                          time_slider_driving_duration, figure,
                          last_selection, last_zoom):
        # print('clickedpoint', clickedpoint)
        # print ('selectedpoints:', selectedpoints)
        # print ('relaypoints:', relaydata)
        util.day_of_week = day_selection
        util.day_night = time_selection
        util.duration = time_slider
        year = year_selection
        month = month_selection
        transition_time = (int(time_slider_driving_duration)*10)/60
        fig = figure
        if selectedpoints is not None:
            if ('location' in selectedpoints['points'][0]):
                location = selectedpoints['points'][0]['location'] 
            else:
                location = 0
        else:
            location = 0
        current_selection = {'dow':day_selection,
                             'day_night': time_selection,
                             'duration':time_slider,
                             'transition': transition_time,
                             'year': year_selection,
                             'month':month_selection,
                             'location': location}
        if last_selection is not None:
            last_selection  = json.loads(last_selection)
            
        if current_selection == last_selection:
            run_callback = False
        else:
            run_callback = True
            last_selection = current_selection
        
        # ipdb.set_trace()
        if last_zoom is not None:
            last_zoom  = json.loads(last_zoom)
            
        if relaydata is not None:
            if ('mapbox.center' in relaydata):
                current_zoom = {'center': relaydata['mapbox.center'],
                                'zoom': relaydata['mapbox.zoom']}
                if (current_zoom == home_view) and (last_zoom != home_view):
                    run_callback = True
                    selectedpoints = None
                    location = 0
                    current_selection['location'] = 0
                    last_selection = current_selection
                last_zoom = current_zoom
                    
               
        # ctx = dash.callback_context
        # changed_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # print(changed_id)

        #update utility with relevant information: 
        if run_callback:
        #Checks if there are any selected points, returns zoomed out map if None
            if location == 0:
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
                
                choro_df["Text"] = choro_df.apply(lambda row: "At " + row["borough"] + " borough, " + row["zone"] + " zone," +  " the pickup score is:" + str(round(row["pickup_score"],2)), axis=1)
                
                df = choro_df
                color_data = 'pickup_score'
                location_ID = "Zone_ID"
                fkey = "properties.locationid"
                center_data = home_view['center']
                color_scale = "RdBu"
                zoom_level = home_view['zoom']
                hover_dataset = {'Text':True, 'Zone_ID':False, 'borough':False,'zone':False, 'pickup_score':False}
                color_midpoint = np.mean(choro_df['pickup_score'].values)
                range_color = [choro_df['pickup_score'].min(), 
                               choro_df['pickup_score'].max()]
                last_zoom = home_view
                #This is used later in add_scattermapbox, as best_zone is only relevant after a click, we put zero here to avoid getting
                # an error in add_scattermapbox. I added a 0,0,0 row in centers.csv for this purpose
                best_zone = 0
            else:
                print('Calculating top neigbors...')
                neighbors = util.top_neighbor(location, month, year,
                                              time_slider, transition_time)
                print('done!')
                base_amount = neighbors.loc[location, 'expected_total_amount']
                
                neighbors['pct_extra'] = (100 * (neighbors['expected_total_amount'] 
                                                - base_amount)/base_amount)    
                neighbors[neighbors['pct_extra']<0] = np.nan
                neighbors = pd.merge(zone_names,neighbors,how = 'left',
                                    left_on='LocationID', right_index = True)
                #This is constructing a meaningful mouse-hover/over text, it creates a new column in neighbors to store the hover text
                neighbors["Text"] = neighbors.apply(lambda row: 
                    "There is no financial benefit in driving to " + row["borough"] + " borough, " + row["zone"] + " zone," +  " compared to your current zone."
                    if (np.isnan(row["pct_extra"]) or (row["pct_extra"]==0))                   
                    else
                        "If you drive to " + row["borough"] + " borough, " + row["zone"] + " zone," +  " the avg trip time for a pickup is: " + str(round(row["avg_trip_time"]*60,2)) 
                        + " minutes. <br> You will make on avg $" + str(round(row["avg_total_amount"],2)) + " in the time left of your day. This amount is $" + str(round(row["avg_total_amount"]*(row["pct_extra"]/100),2)) + ' more than you current zone.', axis=1)
 
                df = neighbors
                color_data = 'pct_extra'
                location_ID = "LocationID"
                fkey = "properties.locationid"
                center_data = {"lat": centers.loc[location, 'avg_lat'],
                               "lon": centers.loc[location, 'avg_long']}
                color_scale = "RdBu"
                zoom_level = 11.5
                color_midpoint = neighbors['pct_extra'].max()/2
                range_color = [neighbors['pct_extra'].min(), 
                               neighbors['pct_extra'].max()]
                #This selects which columns you want to show on hover
                hover_dataset = {'Text':True, 'LocationID':False, 'borough':False,'zone':False, 'avg_trip_time':False, 'avg_total_amount' :False, 'num_trips':False, 'expected_total_amount':False, 'pct_extra':False}
                last_zoom = {'center': center_data,
                             'zoom': zoom_level}
                if neighbors['pct_extra'].idxmax():
                    best_zone = df.loc[df['pct_extra'].idxmax(), 'LocationID']
                else:
                    best_zone = 0
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
            
            fig.add_scattermapbox(lat = [centers.loc[best_zone, 'avg_lat']],
                    lon = [centers.loc[best_zone, 'avg_long']],
                    mode = 'markers+text',
                    text = ['Best Location'],  #a list of strings, one  for each geographical position  (lon, lat)
                    below='',
                    marker_size=20, 
                    marker_color='rgb(0,0,255)',                                                        
                    textposition = "bottom center", 
                    textfont=dict(size=20, color='black'),                                                                    
                    name = 'Best Location',
                    hoverinfo="none",
                    # hide from the legend
                    showlegend=False)
            
            fig.add_scattermapbox(lat=[centers.loc[location, 'avg_lat']],
                                       lon = [centers.loc[location, 'avg_long']],
                                       mode = 'markers+text',
                                       text = ['Current Location'],  #a list of strings, one  for each geographical position  (lon, lat)
                                       below='',
                                       marker_size=20, 
                                       marker_color='rgb(235, 0, 100)',
                                       textposition = "bottom center", 
                                       textfont=dict(size=20, color='black'),
                                       name = 'Current Location',
                                       hoverinfo="none",
                                       # hide from the legend
                                       showlegend=False )
    
            # fig.update_mapboxes(pitch=45)
            fig.update_layout(clickmode='event+select', 
                              title = 'NYC Cabbie Director', 
                              coloraxis_showscale=True,
                              margin={"r":0,"t":0,"l":0,"b":0})

            #Add Scatter Plot to render the Best Location to pickup
            #symbols would be cool, but work differently, don't render if close unless you zoom in.
            # if (location != 'null') and neighbors['pct_extra'].idxmax():
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
            # if location != 'null':
            #     fig.add_scattermapbox(lat = [centers.loc[location, 'avg_lat']],
            #             lon = [centers.loc[location, 'avg_long']],
            #             mode = 'markers+text',
            #             text = ['Current Location'],  #a list of strings, one  for each geographical position  (lon, lat)              
            #             below='',                 
            #             marker_size=15, marker_color='rgb(235, 0, 100)',
            #             textposition = "bottom center", textfont=dict(size=16, color='black'),
            #             name = 'Current Location')
        # ipdb.set_trace()
        return location, fig, json.dumps(last_selection), json.dumps(last_zoom)

if __name__ == '__main__':
    #Use this line if running locally
    app.run_server(debug=True)

    #Uncomment this line if hosting on the web, and comment out above line
    #app.run_server(debug=True, host='0.0.0.0', port='80')