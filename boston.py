""" ***************************************************************************
# * File Description:                                                         *
# * Using data downloaded from Yahoo Finace, we construct visual tools        *
# * to confirm stock market trends.                                           *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries                                                    *
# * 2. Helper Functions: Use to read data                                     *
# * 3. Read data                                                              *
# * 4a. Visualize Data: Line Plot                                             * 
# * 4b. Visualize Data: Prepare data for Candlestick Chart                    *
# * 4b. Visualize Data: Make Candlestick Chart                                *
# * 5a. Simple Moving Average                                                 *
# * 5b. Exponential Moving Average                                            *
# * 5c. Popular SMA and EMA                                                   *
# * 5d. Candlesticks with Moving Averages                                     *
# * 6a. Candlestick charts, Moving Averages, and Volume: Crunching the numbers*
# * 6b. Candlestick charts, Moving Averages, and Volume: Figure               *
# *                                                                           *
# * --------------------------------------------------------------------------*
# * AUTHORS(S): Frank Ceballos                                                *
# * --------------------------------------------------------------------------*
# * DATE CREATED: Sept 2, 2019                                                *
# * --------------------------------------------------------------------------*
# * NOTES:                                                                    *
# * ************************************************************************"""


###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import pandas as pd
import numpy as np
import folium
import matplotlib.cm
import seaborn as sns
from folium.plugins import HeatMap, HeatMapWithTime

###############################################################################
#                    2. Helper Functions to Set Map Colors                    #
###############################################################################
values = hour
number_of_colors = 24
cmap_name = "YlOrRd"
# Define color vector function
def ColorVector(values, number_of_colors, cmap_name):
    buckets = pd.qcut(values, number_of_colors,  duplicates='drop').codes
    cmap = sns.color_palette(cmap_name, number_of_colors)

    colors = []
    for i in range(0, number_of_colors):
        r = int(cmap[i][0]*255)
        g = int(cmap[i][1]*255)
        b = int(cmap[i][2]*255)
        color = "#{0:02x}{1:02x}{2:02x}".format(r, g, b)  
        colors.append(color)
    values = []
    for j in range(0, len(buckets)):
        value = colors[buckets[j]]
        values.append(value)
    return values
  
# Define color categories function
def ColorCategories(variable, colors):
    properties['Colors'] = colors
    counts = properties.groupby(['Colors', variable]).size().reset_index().rename(columns={0:'count'})
    color_list = counts.Colors.unique()
    for color in color_list:
        temp = counts[(counts.Colors == color)]
        print(color, min(temp[variable]), max(temp[variable]), len(temp))
     
        
###############################################################################
#                             3. Read in Data                                 #
###############################################################################
# Read in the data
data = pd.read_csv('crime.csv',encoding='latin-1')

# Remove NaN from dataframe
data = data.dropna(subset = ["Lat", "Long"]).iloc[0:10000, :]

# Define columns of interest
columns = ["OFFENSE_CODE_GROUP", "HOUR", "OFFENSE_DESCRIPTION", "OCCURRED_ON_DATE",
           "STREET", "Lat", "Long"]

# Get columns of interest
data = data[columns]


###############################################################################
#                          4. Prepare Map Components                          #
###############################################################################
# prepare map components
lats = data["Lat"].values
lngs = data["Long"].values
description = data["OFFENSE_DESCRIPTION"].values
date = data["OCCURRED_ON_DATE"].values
hour = data ["HOUR"].values

# Colors
numColors = 24
colors = ColorVector(hour, numColors, "YlOrRd")

###############################################################################
#                               5. Make Map                                   #
###############################################################################
# create base map
m = folium.Map(location=[np.mean(lats), np.mean(lngs)], 
               tiles = 'Stamen Toner', zoom_start = 10)

# loop through properties to add map layers
for i in range(len(lats)):
    folium.Circle(location=[lats[i], lngs[i]], 
                  popup = ("""<b>{}</b><br>Description: {}""".format(date[i], description[i])),
                  radius = 15, 
                  color = colors[i],
                  fill = True,
                  fill_opacity=.30).add_to(m)
    
# save map as html file    
m.save('view-1.html')


###############################################################################
#                               6. Heat Map                                   #
###############################################################################
# Make list of list
heat_data = [[lats[ii], lngs[ii]] for ii in range(len(lats))]

# Create base map
m = folium.Map(location=[np.mean(lats), np.mean(lngs)], 
               tiles = 'Stamen Toner', zoom_start = 10)

# Create heat map
HeatMap(heat_data,
        radius = 10).add_to(m)
 
# Save map as html file    
m.save('Heat Map.html')



###############################################################################
#                          7. Heat Map with Time                              #
###############################################################################
# Read in the data
data = pd.read_csv('crime.csv',encoding='latin-1')
police_stations = pd.read_csv("Boston_Police_Stations.csv")

# Remove NaN from dataframe
data = data.dropna(subset = ["Lat", "Long"])

# Make a list of list
heat_data = []

for hour in range(24):
    temp_data = data.loc[data["HOUR"] == hour]
    temp_lat = temp_data.Lat.values
    temp_long = temp_data.Long.values
    
    temp_heat = [[temp_lat[ii], temp_long[ii]] for ii in range(len(temp_long))]
    
    # Append
    heat_data.append(temp_heat)


# Create base map
m = folium.Map(location=[np.mean(lats), np.mean(lngs)], 
               tiles = 'Stamen Toner', zoom_start = 10)


# Create heat map
HeatMapWithTime(heat_data, name = "taco",radius = 20, auto_play=True, max_opacity=0.8).add_to(m)


# Add police stations
for ii in range(len(police_stations)):
    folium.Marker(
        location=[police_stations.Y[ii], police_stations.X[ii]],
        popup='Police Stations',
        icon=folium.Icon(icon='hdd')
        ).add_to(m)


# Save map as html file    
m.save('Heat Map with Time.html')



