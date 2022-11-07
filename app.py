from flask import Flask, render_template, request
import pickle


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sb


from collections import OrderedDict
from cbfv.composition import generate_features

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

#import io
#import base64

app = Flask(__name__)
model = pickle.load(open('model-TE2.pkl', 'rb'))


def merge_data(formula, sin_temp, measure_temp):
    
    table = pd.DataFrame(data = {'formula':[formula], 'sinter temp Celcius':[sin_temp], 'measurement temperature' : [measure_temp], 'target':[1]})
    
    data_and_features, y_train, formulae_train, skipped_train = generate_features(table, elem_prop='Magpie', drop_duplicates=False, extend_features=True, sum_feat=True)
    
    data_and_selectFeatures = data_and_features[[
    'measurement temperature', 
    'mode_Number', 
    'avg_GSvolume_pa',
    'mode_GSvolume_pa', 
    'mode_Row', 
    #'range_NpValence', 
    #'range_NValence',
    'range_SpaceGroupNumber', 
    'sum_GSbandgap', 
    'dev_CovalentRadius',
    'sinter temp Celcius', 
    'max_NpUnfilled', 
    #'dev_Number',
    'avg_SpaceGroupNumber', 
    #'pressure MPa', 
    'range_NpUnfilled',
    'sum_CovalentRadius', 
    'max_MendeleevNumber', 
    'mode_Electronegativity',
    'mode_NUnfilled', 
    'dev_NdValence', 
    'dev_NpValence', 
    #'dev_NValence',
    'avg_NUnfilled', 
    'sum_Electronegativity',
    'sum_NValence',
    'sum_NsValence', 
    'mode_Column', 
    'dev_GSmagmom', 
    'sum_GSmagmom']]
    
    
    
    return data_and_selectFeatures
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER    

@app.route("/")
@app.route('/index')
def hello():
   
    return render_template('index.html' )

def generate_feature():
    formula = str(request.form['formula'])
    sin_temp = int(request.form['sin_temp'])
    measure_temp = int(request.form['measure_temp'])
    featureTotal = merge_data(formula, sin_temp, measure_temp)
    #scaler = StandardScaler()
   # fit_data = scaler.fit_transform(featureTotal)
   # feature = normalize(featureTotal)
    
    
    return featureTotal

@app.route("/cal_zt", methods=['POST'])
def cal_zt():
    set_of_zt = []
    material = str(request.form['formula'])
    sinter_temp = int(request.form['sin_temp'])
    set_of_zt = []
    temperatures = [500,550,600,650,700]
    set_of_zt = []
    set_of_zt = []
    for i in temperatures:
        zt = model.predict(merge_data(material,sinter_temp,i ))[0]
        set_of_zt.append(zt)
    
    return set_of_zt, temperatures, material,sinter_temp

@app.route("/plot_zt", methods=['POST'])
def plot_zt():
    zt_list = []

    set_of_zt, temperatures, material,sinter_temp = cal_zt()
    
    zt_list = set_of_zt
    plt = matplotlib.pyplot
    plt.plot(temperatures, zt_list, 'o', ms=9, mec='k', mfc='red', alpha=0.4)
    plt.xlabel(f' Temperature')
    plt.ylabel(f'ZT')
    plt.title(f' ZT dependence temperature of {material} sintered at {sinter_temp} C')
    plt.savefig(f'./static/IMG/prediction_{material}_{sinter_temp}.jpg')
    plt.close()
    plt.cla()
    plt.clf()
    
    return f'show picture below'



@app.route("/predict", methods=['POST'])
def predict():
    #formula = str(request.form['formula'])
    #sin_temp = int(request.form['sin_temp'])
    #measure_temp = int(request.form['measure_temp'])
    #feature = merge_data(formula, sin_temp, measure_temp)
    try:
        pic = plot_zt()
        error =""
    except:
        error = "You put the wrong fomula form, try again!"
        pic = ""
    formula = str(request.form['formula'])
    sin_temp = int(request.form['sin_temp'])
    #measure_temp = int(request.form['measure_temp'])
    

    #x= generate_feature()
    #prediction = model.predict(x)
    #output = prediction[0]
    Flask_Logo = os.path.join(app.config['UPLOAD_FOLDER'], f'prediction_{formula}_{sin_temp}.jpg')
    
    return render_template('index.html', image=Flask_Logo, pic=pic, error=error)


if __name__ == "__main__":
    app.run()
    
    
    
    
    
    
    