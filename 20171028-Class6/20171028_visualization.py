import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
wgdata = pd.read_csv("data/wg.csv")

############### EXPLORATORY ANALYSIS AND DATA QUALITY CHECKS #############
# number of nans in wg column
wgdata["wg"].isnull().sum()

# number of nans in metmin
sum(wgdata["metmin"].isnull())

# how many observations had a null either in wg or in metmin column?
sum(wgdata["wg"].isnull() | wgdata["metmin"].isnull())

# extract the observations which don't have null in "wg" or "metmin" columns
condn1 = wgdata["wg"].notnull()
condn2 = wgdata["metmin"].notnull()
wgclean = wgdata.loc[condn1 & condn2,:].reset_index(drop=True)

wgclean = wgdata.dropna() # simpler way of cleaning data

# how many people have gained weight above average weight gain of the data
avg_wt = wgclean["wg"].mean()
sum(wgclean["wg"] > avg_wt)

##############3 VISUALIZATION #############################################

################ HISTOGRAM #####################################################
# visualizes the distribution of data
# Bins the values and generates a bar plot of count of observations in each bin

# Using Pandas plotting function
hist_plt = wgclean["wg"].plot.hist()
hist_plt.set_xlabel("Weight Gain")
hist_plt.set_ylabel("Number of observations")
hist_plt.set_title("Distribution of Weight Gain")
hist_plt.set_xlim([5,45])

# Using matplotlib
plt.hist(wgclean["wg"],color="red")
plt.xlabel("Weight Gain")
plt.ylabel("Number of observations")
plt.title("Distribution of Weight Gain")
plt.xlim([0,50])

# custom binning
wgclean["wg"].plot.hist(bins=[0,20,40,60]) # Pandas
plt.hist(wgclean["wg"],bins=[0,10,20,30,40,50,60,70]) #Matplotlib

sum((wgclean["wg"]>= 0) & (wgclean["wg"]<10)) # 69 observations with wg between 0 and 10
sum((wgclean["wg"]>= 10) & (wgclean["wg"]<20)) #90 observations with wg between 10 and 20

plt.hist(wgclean["wg"],bins=[0,20,40,60])

######################## BOXPLOT #############################################
## Tukey's Boxplot 
## An alternative to six sigma kind of analysis
## Mean is replaced by median
## +3sigma replaced by Upper Whisker Line
## -3sigma replaced by lower whisker line
## Upper Whisker Line = Q3 + 1.5*IQR
## Lower Whisker L:ne = Q1 - 1.5*IQR
## Values outside thw Whisker lines can be treated as OUTLIERS

# randomly samply 100 observations from a normal distribution with mean 10 and 
   # standard deviation 1
a1 = np.random.normal(10,1,100) 
plt.hist(a1)
np.mean(a1)
np.median(a1)
# mean and median would be closer 

a2 = np.append(a1,100000) # addong an outlier data point
plt.hist(a2)
np.mean(a2) # mean gets significantly affected
np.median(a2) # median is robust to outliers

# IQR = Inter Quartile Ramge
# IQR = 75th percentile - 25th percentile
scores_20 = np.random.randint(0,100,21)
scores_20_sorted = np.sort(scores_20)
np.percentile(scores_20,50) # 50th percentile  or 2nd quartile or median
np.median(scores_20)
np.percentile(scores_20,100) # maximum value
np.percentile(scores_20,0) #minimum value
np.percentile(scores_20,25) #25th percentile or 1st quartile
np.percentile(scores_20,75) #75th percentile or 3rd quartile

wgclean["wg"].plot.box()
wgclean["wg"].describe()
Q3 = np.percentile(wgclean["wg"],75) 
Q1 = np.percentile(wgclean["wg"],25) 
IQR = Q3 - Q1
UWL = Q3 + 1.5*IQR # Upper Whisket line is capped with maximum value
LWL = Q1 - 1.5*IQR # Lower whisker line is capped with minimum value
np.min(wgclean["wg"]) # Lower whisker line is capped at 2 doesn't go below that
np.max(wgclean["wg"]) # maximum value is above UWL

# Comparison of distribution 
wgclean.boxplot(column="wg",by="Gender")
wgclean.boxplot(column="wg",by="Shift")

# Matplotlib version of boxplot
plt.boxplot(wgclean["wg"])

# Q: Can the Boxplot display negative values? Yes
pos_values = np.random.rand(50)
neg_values = -1*np.random.rand(30)
all_values = np.append(pos_values,neg_values)
plt.boxplot(all_values)


################# SCATTER PLOT ##############################################
######## X - Y Plot 
# Visualize relationship between variables/attributess

# Pandas
sct_plt = wgclean.plot.scatter("metmin","wg")
sct_plt.set_xlabel("Activities")
sct_plt.set_ylabel("Weight Gain")

# Matplotlib
# Plotting 2 independent series/array/list
plt.scatter(wgclean["metmin"],wgclean['wg'])
# plotting columns of a data frame
plt.scatter("metmin",'wg',data = wgclean)
plt.scatter("metmin",'wg',data = wgclean,c = "red")
plt.scatter("metmin",'wg',data = wgclean,c = "blue")

# Different color for male and female observations
wgclean["custom_color"] = "red" # creating a color column and filling all values with red
# changing the color of Male observations to blue
wgclean.loc[wgclean["Gender"] == "M","custom_color"] = "blue"
plt.scatter("metmin",'wg',data = wgclean,c = "custom_color")

################### Line Plot ###################################################
stockdata = pd.read_csv("data/Stock_Price.csv")

stck_plot = stockdata["DELL"].plot.line()
stck_plot.set_xlabel("Time Instance")
stck_plot.set_ylabel("Closing Stock Price of Dell")

stck_plot = stockdata["Intel"].plot.line()
stck_plot.set_xlabel("Time Instance")
stck_plot.set_ylabel("Closing Stock Price of Intel")

# Plotting multiple columns of a data frame
stck_plot = stockdata.plot.line()
stck_plot.set_xlabel("Time Instance")
stck_plot.set_ylabel("Comparison of stockl prices of Intel vs Dell")


#plt.scatter("DELL","Intel",data=stockdata)

# if data time is provided in index, it becomes a time series data
pd.date_range(start = "2017-01-01",periods=76,freq="D")
pd.date_range(start = "2017-01-01",periods=76,freq="M")

stockdata.index = pd.date_range(start = "2017-01-01",periods=76,freq="D")

stck_plot = stockdata.plot.line()
stck_plot.set_xlabel("Time Instance")
stck_plot.set_ylabel("Comparison of stockl prices of Intel vs Dell")

# xtick labels will be adapted based on the duration and frequency of data

########################### Bar Plot #############################################

stockdata.loc[:,"DELL"].plot.bar()
# Bar plot of first 10 days stock prices
stockdata.iloc[0:10,0].plot.bar() # Dell
stockdata.iloc[0:10,1].plot.bar() # Intel
stockdata.iloc[0:10,:].plot.bar() # Dell vs Intel

# Mix of integer and name indexing could be done with .ix
# NOTE: It is deprecated
#stockdata.ix[0:10,"DELL"].plot.bar() # Dell vs Intel

























































