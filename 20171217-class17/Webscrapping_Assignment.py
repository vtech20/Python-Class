# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:16:06 2017

@author: Vignesh
"""
from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
 
###### Retrieving tags from html web pages ############################
r = requests.get('http://www.mationalytics.com/sample.html')
soup = BeautifulSoup(r.content)
 
for i in soup.find_all('li'): # list element
    print("\nText in List item:",i.text)
     
for i in soup.find_all('a'): 
    print("Text in List item:",i['href'])
     
for i in soup.find_all('img'): # images
    print("\nText in List item:",i['src'])
    
for i in soup.find_all('a'):
    print(i['href'])    
 
######## Retrieving tables in html pages ################################
 
links = ["https://www.icc-cricket.com/rankings/mens/team-rankings/test",
         "https://www.icc-cricket.com/rankings/mens/team-rankings/odi",
         "https://www.icc-cricket.com/rankings/mens/team-rankings/t20i"]
 
tables = list()
 
for url in links:
     
    html = requests.get(url).content
 
    soup = BeautifulSoup(html,"lxml")
 
    temp_df = pd.read_html(str(soup))[0]
     
    tables.append(temp_df)
 
test_df = tables[0].iloc[:,[1,3]]
odi_df = tables[1].iloc[:,[1,3]]
t20_df = tables[2].iloc[:,[1,3]]
 
## Assignment 1
# inner merge all the three tables and find sum of points across
 
odi_test_merge = pd.merge(odi_df,test_df,on='Team',how='inner') # MERGING ODI MATCHES BY TEAM NAMES
odi_test_merge.columns = ['Team','ODI_Points','Test_Points'] # MERGING COLUMNS
odi_test_t20_merge = pd.merge(odi_test_merge,t20_df,on='Team',how='inner') # # MERGING T20 MATCHES BY TEAM NAME
odi_test_t20_merge = odi_test_t20_merge.rename(columns={'Points':'T20_Points'}) ## RENAMING COLUMN NAMES FOR DATA ANALYSIS
 
# which team has the highest sum of points across all 3 cricket team formats ?
 
odi_test_t20_merge['Total_Points'] = odi_test_t20_merge.iloc[:,1:4].apply(np.sum,axis=1) 
 
odi_test_t20_merge.sort_values(by='Total_Points',ascending=False,inplace=True)
 
print (odi_test_t20_merge.head(n=1),'Maximum points in all segments')
 
## Assignment 2
 
# url : http://www.espncricinfo.com/wi/engine/series/1078425.html?view=pointstable
 
# Sum the points of each team in the points by match table and 
# create a dataframe with columns 'Team' and "Total points" and compare it with Points Table
 
url = "http://www.espncricinfo.com/wi/engine/series/1078425.html?view=pointstable"
 
r = requests.get(url) ### fetching data
     
html = r.content ### filtering content from fetched data
     
### pd.read_html  is used to search for tables in a html
read_tables = pd.read_html(html) 
 
pt = read_tables[0] ### taking the first dataframe
 
pt.columns = pt.iloc[0,:] ## retrieving column names
 
pt=pt.iloc[1:,:] ### dropping first row
 
## slicing needed columns
pt_slice=pt.loc[:,['Teams','Pts']]
 
## taking second dataframe
pbm = read_tables[2]
pbm = pbm.iloc[1:,:]
 
### separating as two dataframes 
pbm_slice_1 = pbm.iloc[:,[1,2]]
pbm_slice_2 = pbm.iloc[:,[3,4]]
 
## changing column names
pbm_slice_1.columns=['Teams','Pts']
pbm_slice_2.columns=['Teams','Pts']
 
### concating both dataframes one after other
pbm_merge = pd.concat([pbm_slice_1,pbm_slice_2])
 
### converting column data type to integer
pbm_merge['Pts'] = pbm_merge['Pts'].astype(int)
 
### grouping points based on Ream names
pbm_tally = pbm_merge.groupby('Teams',as_index=False).sum()
 
 
## sorting highest points
pbm_tally = pbm_tally.sort_values('Pts',ascending=False)
 
###printing both tables to find comparison
print ("Table1: \n",pt_slice,"\n Table2 : \n",pbm_tally)