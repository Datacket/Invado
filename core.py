import json 
import pandas as pd
import numpy as np
import pymysql
import pymysql.cursors as pycurse
from datetime import datetime
from model_animal_tracking import *
import tensorflow as tf
from io import StringIO
from datetime import timedelta
from flask import Flask,jsonify,request
from sklearn.preprocessing import OneHotEncoder
app=Flask(__name__)
@app.route("/save",methods=["POST"])
def reply():
    #lat,long,city,date of sighting, time of sighting, species, month of sighting
    #float,float,string,date,str(M,A,E,N),
    lat=request.args.get('lat',None)
    lon=request.args.get('lon',None)
    tos=request.args.get('tos',None)
    dos=request.args.get('dos')
    print(dos)
    dt1=datetime.strptime(dos,'%Y-%m-%d %H:%M:%S')
    dos=str(dos).split(' ')[0]
    mos=int(dos.split('-')[1])
    spec=request.args.get('spec',None)
    dt2=datetime.now()
    try:
        conn=pymysql.connect(host="127.0.0.1",user="root",db='details',password="891998",cursorclass=pycurse.DictCursor)
        with conn.cursor() as cur:
                sql="INSERT INTO DETAILS (date,lat,lon,tos,spec,mos) VALUES(\'{}\',{},{},\'{}\',\'{}\',{})".format(*list(map(str,[dos,lat,lon,tos,spec,mos])))
                cur.execute(sql)
                conn.commit()
        return jsonify({"Status":200})
    except Exception as e:
        return jsonify({"Status":str(e)})
    var=model.fit(list(map(str,[lat,lon,tos,spec,mos])))
def lat_long(tup, list_long_lang, radius):
    fres = []
    for l in list_long_lang:
        dis_for_l = edis(tup, l)
        if is_short_enough(dis_for_l, radius):
            fres.append(l)
            if len(fres) == 15:
                break
    return fres
    #return sorted(fres)[:15]

def edis(X, Y):
    x1, y1, x2, y2 = X[0], X[1], Y[0], Y[1]
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

def is_short_enough(deg_dist, radius):
    dist_in_km = np.cos(deg_dist) * 110
    return True if dist_in_km < radius else False
from tqdm import tqdm
@app.route("/",methods=["GET"])
def get():
    centre=list(map(float,[request.args.get('lat',None),request.args.get('lon',None)]))
    date=request.args.get('dos',None)
    mos=int(date.split('-')[1])
    print("Hello world!")
    if True:
        conn=pymysql.connect(host="127.0.0.1",user="root",db='details',password="891998",cursorclass=pycurse.DictCursor)
        with conn.cursor() as curr:
            sql="SELECT * FROM DETAILS"
            curr.execute(sql)
            result=curr.fetchall()
        latitude=[]
        longitude=[]
        print("Hello world!")
        for i in tqdm(result):
            latitude.append(i['lat'])
            longitude.append(i['lon'])
            l=list(zip(latitude,longitude))
            lt_ln=lat_long(centre,l,5)
        df=pd.DataFrame(result)
        df["spec"] = df["spec"].apply(lambda x : x.lower())
        df["spec"] = df["spec"].apply(lambda x : "snake" if x == "cobra" else x)
        spec_copy = df["spec"].copy()
        df["spec"]=df["spec"].apply(str.lower).astype("category").cat.codes
        df["tos"]=df["tos"].astype("category").cat.codes

        oh1=OneHotEncoder().fit(np.array(df["spec"]).reshape(-1,1))
        l=oh1.transform(np.array(df["spec"]).reshape(-1,1)).toarray()
        #l=l[:,1:]
        oh2=OneHotEncoder().fit(np.array(df["tos"]).reshape(-1,1))
        l2=oh2.transform(np.array(df["tos"]).reshape(-1,1)).toarray()
        #l2=l2[:,1:]
        s2=np.concatenate([np.array(df["lat"]).reshape(-1,1),np.array(df["lon"]).reshape(-1,1),np.array(df["mos"]).reshape(-1,1),l2],axis=1)
        wlc=WildlifeCraziness(s2.shape[1],l.shape[1])
        wlc.load_dataset(s2,l)
        print("Hello World!!")
        wlc.fit()
        print("World")
        dat=[np.array(centre[0]).reshape(-1,1),np.array(centre[1]).reshape(-1,1),np.array(mos).reshape(-1,1)]
        test={}
        for i in "MEAN":
            #dat.append(np.array(l2.transform(i)).reshape(-1,1))
            if i == 'A':
                arr = [1, 0, 0, 0]
            elif i == 'E':
                arr = [0, 1, 0, 0]
            elif i == 'M':
                arr = [0, 0, 1, 0]
            else:
                arr = [0, 0, 0, 1]
            l=sorted(set(spec_copy))
            #print (l)
            #print(np.concatenate([np.array(dat).reshape(-1,1),np.array(arr).reshape(-1,1)]).shape)
            
            prediction=wlc.predict(np.concatenate([np.array(dat).reshape(-1,1),np.array(arr).reshape(-1,1)]).T, l)
            test[i]=prediction
        test["lat_ln"]=lt_ln
        return jsonify(test)
            # l2 as JSON 
    #except Exception as e:
     #  return jsonify({"Status":str(e)})
    
app.run(host="0.0.0.0",port=10400,debug=True)
