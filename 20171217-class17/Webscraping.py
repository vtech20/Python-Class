# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:13:51 2017

@author: Vignesh
"""

import requests
from bs4 import BeautifulSoup
from lxml import html
import numpy as np
r = requests.get("http://www.mationalytics.com/sample.html")

soup = BeautifulSoup(r.content,"lxml")

###To extract he values of list element ################
for i in soup.find_all('li'):
    print("\n text:",i.text)

with open('E:\\Python Class\\20171217-class17\\sample.html', "r") as f:    
    page = f.read()
    
import pandas as pd

#For Single link

link = "https://www.icc-cricket.com/rankings/mens/team-rankings/test"

html = requests.get(link).content

soup = BeautifulSoup(html,"lxml")

temp_df = pd.read_html(str(soup))[0]

#For multiple links

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
temp_df.dtypes
merged_team_inner = pd.merge(test_df,odi_df,how="inner",
                               left_on="Team",right_on="Team")
merged_team_inner1 = pd.merge(merged_team_inner,t20_df,how="inner",
                              left_on="Team",right_on="Team")

merged_team_inner1["Total"] = merged_team_inner1["Points_x"] + merged_team_inner1["Points_y"] + merged_team_inner1["Points"]

### Activity ####

# url : http://www.espncricinfo.com/wi/engine/series/1078425.html?view=pointstable

# Sum the points of each team in the points by match table and 
# create a dataframe with columns 'Team' and "points", check with Points Table



link1 = "http://www.espncricinfo.com/wi/engine/series/1078425.html?view=pointstable"
html_1 = requests.get(link1).content
soup_1 = BeautifulSoup(html_1,"lxml")
temp_df_1 = pd.read_html(str(soup_1))[0]
temp_df_2 = pd.read_html(str(soup_1))[2]
temp_df_1.dtypes
scp_1 = temp_df_1.iloc[1:9,:]
scp_1.columns = ["Teams","Mat","Won","Lost","Tied","N/R","Pts","Net RR","For","Against"]
scp_1["Pts"]=pd.to_numeric(scp_1['Pts'],errors =  'coerce')
scp_1.dtypes
scp_2 = temp_df_2.iloc[1:57,:]
scp_2.columns = ["Result Date","Team","Pts","Team1","Pts1","tb"]
scp_2.dtypes
scp_2["Pts"]=pd.to_numeric(scp_2['Pts'],errors =  'coerce')
scp_2["Pts1"]=pd.to_numeric(scp_2['Pts1'],errors =  'coerce')
scp_2.dtypes
div_1 = scp_2.iloc[:,1:3]
div_2 = scp_2.iloc[:,3:5]
div_1_grp_sum = div_1.groupby("Team",as_index=False).sum()
div_2_grp_sum = div_2.groupby("Team1",as_index=False).sum()
merged_tbl_inner = pd.merge(div_1_grp_sum,div_2_grp_sum,how="inner",left_on="Team",right_on="Team")
merged_tbl_inner["Total_pts"] = merged_tbl_inner["Pts"] + merged_tbl_inner["Pts1"]

match_table_sum = merged_tbl_inner.iloc[:,[1,4]]
Points_table_sum = scp_1.iloc[:,[0,6]]



  