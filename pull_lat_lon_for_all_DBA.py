from helpers import *
import pandas as pd
import joblib

df = pd.read_csv('/Users/ajaysharma/Downloads/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')
df = df.drop_duplicates(["DBA"])

lat_lons = {}
idx = []
dba =[]
addresses = []
coords=[]

for i, row in enumerate(df[["DBA",'BUILDING', 'STREET', 'BORO','ZIPCODE']].to_records()[::-1]):
    addr = \
        str(row[1]) + ',' + \
        str(row[2]) + ' ' + \
        str(row[3]) + ',' +  \
        str(row[4]) + ',' + \
        str(row[5])[:-2]

    addresses.append(addr)
    dba.append(row[1])
    idx.append(row[0])
    lat_lons[addr] = get_lat_lon(addr)

    coords.append(lat_lons[addr])
    joblib.dump(lat_lons, 'lat_lons.pkl')
    





joblib.dump(idx, 'idx.pkl')
joblib.dump(coords, 'coords.pkl')
joblib.dump(addresses, 'addresses.pkl')