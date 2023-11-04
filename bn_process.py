#process data to convert from xlsx to csv for reading to a Bayesian Network

import pandas as pd
import numpy as np

            
#save the extracted data in .csv files
def saveData(provinceData):
    
    root = r"data_csv2//"
    for p in provinceData.keys():
        df = pd.DataFrame(provinceData[p])
        
        df.to_csv(root + p + ".csv")
        
#convert numerical data containing "," to numbers without
def convertData(tempData):
    for i in range(len(tempData)):
        tempData[i] = tempData[i].replace(',','')
            
    tempData[tempData=='--'] = 0
    
    return np.int32(tempData)
        
if __name__ == "__main__":
    
    #provinceData = extractData(r"EN_ODP-PR-Citz.xlsx")
    
    data = np.array(pd.read_excel("EN_ODP-PR-Citz.xlsx"))
    temp = data[4:-7,:]
    
    
    rows = []

    for i, d in enumerate(temp):
        values = list(d)
        country = values.pop(0)
        values = convertData(np.array(values))
        
        store=[]
        for index in range(len(values)):
            if ((index+1)%4==0) or ((index+1)%17 == 0):
                continue
            else:
                store.append(values[index])
                
        rows.append([country] + list(store[:-2]))

month = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
colName = list([month[(i)%12+1]+str(2015+i//12) for i in range(len(rows[0])-1)])

df = pd.DataFrame(rows, columns=["Country"] + colName)

df.to_csv("originCountry.csv")
#    saveData(provinceData)
#     f = pd.read_csv(root+"Alberta.csv")
    
df2 = pd.read_csv("originCountry.csv")

