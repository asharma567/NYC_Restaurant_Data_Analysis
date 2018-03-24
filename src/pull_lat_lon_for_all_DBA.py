from helpers import *
import pandas as pd
import joblib

df = pd.read_csv('~/Downloads/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')
df = df.drop_duplicates(["DBA"])

lat_lons = {}
idx = []
dba =[]
addresses = []
coords=[]

lat_lons = joblib.load('lat_lons.pkl')

for i, row in enumerate(df[["DBA",'BUILDING', 'STREET', 'BORO','ZIPCODE']].to_records()):
    try:
        addr = \
            str(row[1]) + ',' + \
            str(row[2]) + ' ' + \
            str(row[3]) + ',' +  \
            str(row[4]) + ',' + \
            str(row[5])[:-2]


        addresses.append(addr)
        dba.append(row[1])
        idx.append(row[0])
        

        if lat_lons.get(addr): 
            continue

        lat_lons[addr] = get_lat_lon(addr)
        coords.append(lat_lons[addr])
        joblib.dump(lat_lons, 'lat_lons.pkl')

        print (i)
    
    except:
        print ('CRASHED!')
        joblib.dump(lat_lons, 'lat_lons.pkl')
    



joblib.dump(idx, 'idx.pkl')
joblib.dump(coords, 'coords.pkl')
joblib.dump(addresses, 'addresses.pkl')