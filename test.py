            from util import Utility    
            import numpy as np 
            import pandas as pd
            
            util = Utility(taxi = 'yellow', day_night = 'night', 
                 day_of_week = 'weekday', duration = 8)
            
            zone_names = pd.read_csv('data/TaxiZone_Name_Borough.csv')
            
            location = 'NULL'
            
            #If no zones are selected, return unzoomed overall map

            #run pagerank algorithm:
            year = 2019
            month = 10
            edges, trips = util.load_data(month, year)
            P, node_ids, idx = util.pagerank(edges, trips, 10)

            #create score dictionary
            score_dict = dict(zip(node_ids, util.pickupscore_))
            #print (score_dict)
            zones = [x + 1 for x in range(265)]

            choro_df = pd.DataFrame(zones, columns = ['Zone_ID'])
            choro_df['pickup_score'] = choro_df['Zone_ID'].map(score_dict)
            choro_df.replace(np.nan, 0, inplace=True)
    
            choro_df = pd.merge(choro_df, zone_names, left_on='Zone_ID', right_on='LocationID')
            choro_df.drop(columns='LocationID', inplace=True)
            
            choro_df.head(20)
            
            
            
            location = 12
            
            neighbors = util.top_neighbor(location, month, 
                                        year, 5)
            neighbors.head(5)
            
            base_amount = neighbors.loc[location, 'expected_total_amount']
            
            neighbors['pct_extra'] = (100 * (neighbors['expected_total_amount'] 
                                            - base_amount)/base_amount)         
            

            neighbors = pd.merge(zone_names,neighbors,how = 'left',
                               left_on='LocationID', right_on='Zone_ID', right_index = True)
            
            neighbors.head(20)
