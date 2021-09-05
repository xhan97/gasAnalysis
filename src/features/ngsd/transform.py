import pandas as pd
import numpy as np


data = pd.read_excel("ng sd data.xlsx",sheet_name="daily sd",skiprows=[0,1])
data["date"] = data["date"].astype(str)  + " 06:20:00"
data.rename ({"date":"datetime"},axis=1,inplace=True)

data.to_csv("dailySd.csv",index=False,float_format='%.2f')
data = pd.read_excel("ng sd data.xlsx",sheet_name="weekly storage survey actual",skiprows=[0])

data["release"] = data["release"].astype(str)  + "09:30:00"

data.rename ({"release":"datetime"},axis=1,inplace=True)

data.to_csv("weeklyStorageSurvey.csv",index=False,float_format='%.2f')
