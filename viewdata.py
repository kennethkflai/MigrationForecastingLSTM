# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

import pandas as pd

df = pd.read_csv("demographics.csv")

df2 = df.groupby("Country of origin").sum()
df2 = df2[df2.Total >= 1000000]

countryLabel = list(df2.index)
ageLabel = ["0 - 4",
            "5 - 11",
            "12 - 17",
            "18 - 59",
            "60",
            "other"]
genderLabel = ["Female", "Male"]

for index in range(len(countryLabel)):
    if countryLabel[index] == "Unknown ":
        del countryLabel[index]
        break;
        
temp = [df2.loc[lbl, "Total"] for lbl in countryLabel]

temp_df = df[df["Country of origin"]=="South Sudan"]
cLabel = "South Sudan"

for gLabel in genderLabel:
    for aLabel in ageLabel:
        temp_dictionary = {"Age":aLabel, "Gender":gLabel,"Origin":cLabel}
        temp =  temp_df.loc[:, ["Year", gLabel+" " +aLabel]]
        if len(temp) <22:
            for year in range(2001,2023):
                if (temp["Year"]==year).any() == False:
                    temp = temp.concat({'Year': year, gLabel+" " +aLabel: 0}, ignore_index=True)

            temp = temp.sort_values("Year")
#            temp = temp.iloc[:,1]
            print(temp)
                
#            temp = np.array(temp)