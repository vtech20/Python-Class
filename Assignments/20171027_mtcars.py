# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

mtcars = pd.read_csv("data/mtcars.csv")

#3. How many rows in the data?
mtcars.shape[0]
#4. How many columns in the data?
mtcars.shape[1]
#5. What are the column names?
mtcars.columns
#6. Use describe command to understand the statistical summary.
mtcars.describe()

#7. Average miles per gallon (mpg) of all cars
mtcars["mpg"].mean()

#8. Average mpg of automatic transmission cars
condn = mtcars["am"]== 0
mtcars.loc[condn,"mpg"].mean()
mtcars.loc[mtcars["am"]==0,"mpg"].mean() # last 2 lines in 1 line

#9. Average mpg of manual transmission cars
mtcars.loc[mtcars["am"]==1,"mpg"].mean()

#10. Average Displacement of cars with 4 gears
mtcars.loc[mtcars["gear"] == 4, "disp"].mean()

#11. Average Horse power of cars with 3 carb
mtcars.loc[mtcars["carb"] == 3, "hp"].mean()

#12. Average mpg of automatic cars with 4 gears
condn1 = mtcars["am"] == 0
condn2 = mtcars["gear"] == 4
mtcars.loc[condn1 & condn2,"mpg"].mean()
# last 3 lines in 1 line
mtcars.loc[(mtcars["am"] == 0) & (mtcars["gear"] == 4),"mpg"].mean()

#13. Average qsec of cars with mpg above 
# average mpg and weight below average weight
mtcars.loc[(mtcars["mpg"] > mtcars["mpg"].mean()) &
(mtcars["wt"] < mtcars["wt"].mean()),"qsec"].mean()

#14. Entire row of the vehicle which has the highest miles per gallon
mtcars.loc[mtcars["mpg"] == mtcars["mpg"].max(),:] # implementation1
mtcars[mtcars["mpg"] == mtcars["mpg"].max()] #implementation2
# implementation3
mtcars_sorted_mpg = mtcars.sort_values("mpg",ascending = False)
mtcars_sorted_mpg.iloc[0,:]

#15. Entire row of vehicle with the highest horsepower
mtcars.loc[mtcars["hp"] == mtcars["hp"].max(),:]

#16. Mileage and hp of car with highest weight
mtcars.loc[mtcars["wt"] == mtcars["wt"].max(),["mpg","hp"]]

#17. Calculate ratio of mpg to carb for each car and 
# calculate the average of ratio
np.mean(mtcars["mpg"]/mtcars["carb"])

# Q: Can I add the ratio as another column and take a mean of that column?
mtcars["mpg_carb_ratio"] = mtcars["mpg"]/mtcars["carb"]
np.mean(mtcars["mpg_carb_ratio"])

#18. Weight of the car with the minimum displacement
mtcars.loc[mtcars["disp"] == mtcars["disp"].min(),"wt"]

#19. Slice all columns of 3 gear cars
mtcars.loc[mtcars["gear"]==3,:]

#20. Slice mpg, displacement and hp columns of 
#manual transmission cars
mtcars.loc[mtcars["am"] == 1, ["mpg","disp","hp"]]

#21. What is average mpg of 3, 4 and 5 gear cars. Save output as a list/array/series 

#22. What is average hp, average wt, average qsec, average vs for 3, 4 and 5 gear cars. Save output as a matrix or data frame


