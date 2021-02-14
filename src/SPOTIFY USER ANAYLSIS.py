#Importing necessarries libraries

import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import io
import requests
from collections import Counter
from itertools import product
import warnings
import re
warnings.filterwarnings('ignore')

#Reading data from .json format, converting into a pandas object
request_zero = requests.get('https://raw.githubusercontent.com/jmbarrios27/SPOTIFY-USER-ANALYSIS/main/data/StreamingHistory0.json')
request_zero = pd.io.json.json_normalize(request_zero.json())

request_one = requests.get('https://raw.githubusercontent.com/jmbarrios27/SPOTIFY-USER-ANALYSIS/main/data/StreamingHistory1.json')
request_one = pd.io.json.json_normalize(request_one.json())

request_two = requests.get('https://raw.githubusercontent.com/jmbarrios27/SPOTIFY-USER-ANALYSIS/main/data/StreamingHistory2.json')
request_two = pd.io.json.json_normalize(request_two.json())

#Concat three files
music_search = pd.concat([request_zero , request_one , request_two])
music_search.head()

#Let´s inspect the dataframe
print(f'Seacrh Dataframe shape is: {music_search.shape}')
print()
print(f'Seacrh Dataframe information is: {music_search.info()}')
print()
print(f'Seacrh Dataframe description is: {music_search.describe()}')
print()
print(f'Looking for NaN values in the dataframe: {music_search.isna().sum()}')
print()

search_sorted = music_search.sort_values(by='endTime')
print(f"The first Day of the dataframe is: {search_sorted['endTime'].head(1)}")
print("The last Day of the dataframe is: ",search_sorted['endTime'].tail(1))

#Converting endTime column into Datetime type date
music_search['endTime'] = pd.to_datetime(music_search['endTime'])

#Extracting month, date, and hour separately
#Creating new columns based on the time info
music_search['Month'] = music_search.endTime.dt.month
music_search['Day'] = music_search.endTime.dt.day
music_search['Hour'] = music_search.endTime.dt.hour

#Creating Seconds Column1
music_search['SecondsListened'] = music_search['msPlayed'] * 0.001
music_search['HoursListened'] =  music_search['SecondsListened'] * 0.000277778

#Dropping the msPlayed column 
music_search = music_search.drop(columns=['msPlayed'])

print('Final Result of the Dataframe')
music_search.head()

#Grouping by artist
most_listened = music_search.groupby(by='artistName').sum()
most_listened = most_listened.sort_values(by='HoursListened', ascending=False)
top_ten = most_listened.head(10)


plt.figure(figsize=(6,4))
sns.barplot(data=top_ten, x=top_ten.index, y='HoursListened',palette='Set2')
plt.xticks(rotation=90)
plt.title('TOP 10 ARTISTS')
plt.xlabel('Artists')
plt.ylabel('Hours Played')
plt.grid()
plt.show()

#Lets drop all the observations with 0.0 hours listened, because it means the song wasn´t listened at all
least_searched = music_search[music_search['HoursListened']!= 0]
least_searched = least_searched.groupby(by='artistName').sum()
least_searched = least_searched.sort_values(by='HoursListened',ascending=False)
least_searched = least_searched.tail(10)

plt.figure(figsize=(6,4))
sns.barplot(data=least_searched, x=least_searched.index, y='HoursListened',palette='Set3')
plt.xticks(rotation=90)
plt.title('LAST 10 ARTISTS')
plt.xlabel('Artists')
plt.ylabel('Hours Played')
plt.grid()
plt.show()


# #### **10 most listened songs and the 10 least**

# In[9]:


most_track = music_search.groupby(by='trackName').sum()
most_track = most_track.sort_values(by='HoursListened', ascending=False)
most_track = most_track.head(10)



plt.figure(figsize=(6,4))
sns.barplot(data=most_track, x=most_track.index, y='HoursListened',palette='Set2')
plt.xticks(rotation=90)
plt.title('TOP LISTENED TRACKS')
plt.xlabel('TRACK')
plt.ylabel('Hours Played')
plt.grid()
plt.show()

#Lets drop all the observations with 0.0 hours listened, because it means the song wasn´t listened at all
last_tracks = music_search[music_search['HoursListened']!= 0]
last_tracks = last_tracks.groupby(by='trackName').sum()
last_tracks = last_tracks.sort_values(by='HoursListened',ascending=False)
last_tracks = last_tracks.tail(10)

print()
plt.figure(figsize=(6,4))
sns.barplot(data=last_tracks, x=last_tracks.index, y='HoursListened',palette='tab10')
plt.xticks(rotation=90)
plt.title('LAST 10 TRACKS')
plt.xlabel('Tracks')
plt.ylabel('Hours Played')
plt.grid()
plt.show()


# #### **how many hours of music were listened to per month?**

# In[10]:


#Grouping by Month
month = music_search.groupby(by='Month').sum()
month = month.drop(columns=['Day','Hour','SecondsListened'])
plt.figure(figsize=(8,8))
month.plot(color='skyblue')
plt.xticks(month.index)
plt.title('HOURS LISTENED PER MONTH')
plt.ylabel('Hours Listened')
plt.show()


# #### **how many hours of music were listened to per day of the month?**

# In[104]:


#Grouping Data by Day of the month
day = music_search.groupby(by='Day').sum()
day = day.drop(columns=['Month','Hour','SecondsListened'])

day.plot(color='darkblue')
plt.xticks(day.index, rotation=90)
plt.title('HOURS LISTENED PER DAY OF THE MONTH')
plt.ylabel('Hours Listened')
plt.show()


# #### **how many hours of music were listened to per Hour of the Day?**

# In[12]:


hour = music_search.groupby(by='Hour').sum()
hour = hour.drop(columns=['Month','Day','SecondsListened'])

hour.plot(color='darkgreen')
plt.xticks(hour.index, rotation=90)
plt.title('MUSIC LISTENED BY HOURS OF THE DAY')
plt.ylabel('Hours Listened')
plt.show()


# ## **SEARCH QUERIES**
# 
# In this section we are going to analyze the queries made.
# The information is from the 8 of november 2020 
# 
# to february the first of  2021

# In[13]:


search_url = "https://raw.githubusercontent.com/jmbarrios27/SPOTIFY-USER-ANALYSIS/main/data/SearchQueries.json"
search_url = requests.get(search_url)
search = pd.io.json.json_normalize(search_url.json())
search.head()


# **Data check**

# In[14]:


print()
search.describe()
print()
search.info()
print()
search.isna().sum()
print()
search.shape


# **No NaN values**

# ### **Data Augmentation and Data cleaning**

# In[15]:


#Function to clean Text
def iphoneClean(text):
    text = re.sub(r'ARM64',' ',text)
    text = re.sub(r'_', ' ',text)
    return text


def timeclean(timetext):
    timetext = re.sub(r'UTC', ' ',timetext)
    timetext = re.sub(r'Z', ' ',timetext)
    timetext = re.sub(r'T', ' ', timetext)
    timetext = re.sub(r' [ [ ] ]', ' ', timetext)
    return timetext.replace('[', '')


# In[16]:


#Apllying functions
search['platform'] = search['platform'].apply(iphoneClean)
search['searchTime'] = search['searchTime'].apply(timeclean)

#converting searchTime into datetime data type
#Converting endTime column into Datetime type date
search['searchTime'] = pd.to_datetime(search['searchTime'])



#Extracting month, date, and hour separately
#Creating new columns based on the time info
search['Month'] = search.searchTime.dt.month
search['Day'] = search.searchTime.dt.day
search['Hour'] = search.searchTime.dt.hour

search.head()


# In[17]:


#Let´s see the most performed queries
queries = search['searchQuery'].value_counts()
queries = queries.head(10)
queries = pd.DataFrame(queries)

sns.barplot(data=queries, y=queries.index, x='searchQuery')
plt.title('TOP 10 QUERIES')
plt.ylabel('Term')
plt.xlabel('Count')
plt.grid()


# ##### **Platforms Usage**

# In[18]:


sns.countplot(data=search, x='platform')
plt.title('PLATFORMS SOURCE QUERIES')
plt.xlabel('Platform')
plt.ylabel('Count of quieres per platform')
plt.show()


# ##### **QUERIES MADE BY MONTH**

# In[19]:


query_month = search
query_month['sum'] = 1
query_month = query_month.groupby(by='Month').sum()
sns.barplot(data=query_month,x=query_month.index, y='sum')
plt.title('Queries Per Month')
plt.ylabel('Count of Queries')


# ##### **QUERIES MADE BY DAY  OF THE MONTH**

# In[20]:


query_day = search
query_day['sum'] = 1
query_day = query_day.groupby(by='Day').sum()
query_day = query_day.drop(columns=['Month','Hour'])
plt.figure(figsize=(8,9))
sns.barplot(data=query_day,x=query_day.index, y='sum')
plt.title('Queries Per Day')
plt.ylabel('Count of Queries')


# #### **QUERIES MADE BY HOUR  OF THE DAY**

# In[21]:


query_hour = search
query_hour['sum'] = 1
query_hour = query_hour.groupby(by='Hour').sum()
plt.figure(figsize=(8,9))
sns.barplot(data=query_hour,x=query_hour.index, y='sum',palette='Set2')
plt.title('Queries Per Hour of the Day')
plt.ylabel('Count of Queries')


# ## **MY LIBRARY**
# 
# Here is the information of the music libray of the account

# In[22]:


#Read columns
library_url = 'https://raw.githubusercontent.com/jmbarrios27/SPOTIFY-USER-ANALYSIS/main/data/YourLibrary.json'
library = requests.get(library_url)
library = library.json()

for columns in library:
    print(columns)


# In[23]:


#Read Json file locally
path = 'C:\\Users\\Asus\\Desktop\\data\\YourLibrary.json'

with open(path) as f:
    tracks = pd.DataFrame(json.load(f)['tracks'])

tracks.head()


# ##### **Dataframe info check**

# In[103]:


print()
tracks.info()
print()
print(f'Dataframe shape {tracks.shape}')
print()
track_count = 1
for column in tracks.columns:
    print('column',track_count, 'is',column)
    track_count = track_count + 1


# ### **FAVORTITE ALBUMS, ARTISTS AND TRACKS ON MY LIBRARY**

# In[71]:


#Splitting by category
albums = tracks['album'].value_counts()
artists = tracks['artist'].value_counts()
track = tracks['track'].value_counts()

#Transforming into Pandas object
albums = pd.DataFrame(albums)
artists = pd.DataFrame(artists)
track = pd.DataFrame(track)

#renaming Columns
albums.columns = [['Album Played']]
artists.columns = [['Artists Played']]
track.columns = [['Track Played']]

#Filtering for the first 10 observations
albums = albums.head(10)
artists = artists.head(10)
track = track.head(10)


# #### **Favorite Album**

# In[75]:


ax = albums.plot.bar(color='green')
ax.get_legend().remove()
plt.xticks(rotation=90)
plt.title('FAVORITE ALBUMS')
plt.xlabel('Album Name')
plt.ylabel('Album Played')
plt.show()


# #### **Favorite  Artist or Band**

# In[79]:


ax = artists.plot.bar(color='purple')
ax.get_legend().remove()
plt.xticks(rotation=90)
plt.title('FAVORITE ARTISTS')
plt.xlabel('Artist or Band Name')
plt.ylabel('How many Times Artist was played')
plt.show()


# #### **Favorite Tracks**

# In[80]:


ax = track.plot.bar(color='orange')
ax.get_legend().remove()
plt.xticks(rotation=90)
plt.title('FAVORITE TRACKS')
plt.xlabel('Track Name')
plt.ylabel('Nº of times track was played')
plt.show()

