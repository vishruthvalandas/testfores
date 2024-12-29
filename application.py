from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as np
import pickle
from sklearn.preprocessing import StandardScaler



application=Flask('__name__')
app=application

#import ridge regressor and standard scler pickle

ridge_model=pickle.load(open('model/ridge.pkl','rb'))
standard_scaler=pickle.load(open('model/scalar.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temp=int(request.form.get('Temperature'))
        rh=int(request.form.get('RH'))
        ws=int(request.form.get('Ws'))
        rain=float(request.form.get('Rain'))
        ffmc=float(request.form.get('FFMC'))
        dmc=float(request.form.get('DMC'))
        isi=float(request.form.get('ISI'))
        classes=int(request.form.get('Classes'))
        region=int(request.form.get('Region'))
        new_data_scaled=standard_scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host='0.0.0.0')