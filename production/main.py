import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras import datasets, layers, models
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import keras.backend as K
from keras.models import load_model
from keras.utils import custom_object_scope
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
np.random.seed(0)
import seaborn as sns
from datetime import datetime
import seaborn as sb
import math
from pathlib import Path
import cgi
from time import sleep
from flask import Flask, render_template, request

#init flask app
app = Flask(__name__)

#Helper functions
def rmse(y_true, y_pred): #defining the Root Mean Squared Error function
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def yeartodate_scaled():
    day_of_year = datetime.now().timetuple().tm_yday
    return day_of_year / 365

def data_setup(): #CSV and H5 file imports and
    airports_df = pd.read_csv('airports.csv')
    airlines_df = pd.read_csv('airlines.csv')
    ontime_10423 = pd.read_csv('ontime_10423.csv')


    with custom_object_scope({'rmse': rmse}):
        print("About to load y1")
        modely1 = load_model('modely1.h5')
        print("Load done y1")
        modely2 = load_model('modely2.h5')


    #Setting up the input matrix

    X_data = ontime_10423.iloc[:,:-64]
    X_data.drop(['ORIGIN_AIRPORT_ID','DEP_DELAY','CANCELLED'], axis=1, inplace=True)
    collist = X_data.columns.tolist()
    #input_df = pd.DataFrame({'feature': collist, 'val': 0* len(collist)})

    #first_row = input_df.iloc[0]
    #input_df = input_df.iloc[1:]
    #input_df = input_df.append(first_row, ignore_index=True)


    #creating a list of airlines for users to pick from
    airline_list = []
    for col in collist:
        if col.startswith('OP_UNIQUE_CARRIER_'):
            airline_list.append(col.replace('OP_UNIQUE_CARRIER_', ''))

    #creating a dictionary to map OP_UNIQUE_CARRIER to CARRIER_NAME
    carrier_dict = airlines_df.set_index('OP_UNIQUE_CARRIER')['CARRIER_NAME'].to_dict()

    #using the map function to replace the values in airline_list
    airline_list = [carrier_dict.get(airline, airline) for airline in airline_list]

    #creating a list of destination airports for users to pick from
    airport_list = []
    for col in collist:
        if col.startswith('DEST_AIRPORT_ID_'):
            airport_list.append(col.replace('DEST_AIRPORT_ID_', ''))
    airport_list = pd.Series(airport_list).astype('int64').tolist()

    #creating a dictionary to map AIRPORT_ID to DISPLAY_AIRPORT_NAME
    airport_dict = airports_df.set_index('AIRPORT_ID')['DISPLAY_AIRPORT_NAME'].to_dict()

    #using the map function to replace the values in airport_list
    airport_list = [airport_dict.get(airport, airport) for airport in airport_list]
    print("Data loading is done!")
    return airport_list, airline_list, collist

#user input prediction function
def user_pred(numpy_array_input):  #input is shape (43,), all OHE except the last 

    #make delay prediction with the model
    raw_delay_prediction = modely1.predict(numpy_array_input)
    transformed_delay_prediction = np.exp(raw_delay_prediction) -30

    #make cancellation prediction with the model
    cancellation_prediction = modely2.predict(numpy_array_input)

    return transformed_delay_prediction, cancellation_prediction

def run_pred(input_dest, input_airline):

    new_list = [0]*len(collist) #Resetting input

    #SAMPLE INPUT
    
    #input_airline = 9      #Remember, this starts at 0  #drop down menu appears as user starts typing, index is stored
    #input_dest = 2         #Remember, this starts at 0  #drop down menu appears as user starts typing, index is stored

    #EXECUTION

    collist[input_airline] = 1          #Executes the addition of airline to input
    collist[input_dest+17] = 1          #Executes the addition of airport to input
    collist[-1] = round(yeartodate_scaled(),3)  #Executes the addition of scaled YTD

    X_input = np.array(collist.reshape(-1,43))

    #test prediction
    prediction = user_pred(X_input)
    delay_pred = str(prediction[0])
    delay_pred = delay_pred[2:-8]

    #output to be sent to user
    cancellation_pred = str(prediction[1])
    cancellation_pred = cancellation_pred[2:-7]

    return delay_pred,cancellation_pred

global airport_list, airline_list, collist
airport_list, airline_list, collist = data_setup()

#Flask components
@app.route('/') #routes to html page at ('/')
def index():
    return render_template('template.html')

@app.route('/predict', methods=['GET','POST'])


def predict(): # Make prediction based on selected values
    airport_index = 2 # request.get_data()
    airline_index = 3

    #airport_index = int(index chosen from airport_list)
    #airline_index = int(index chosen from airline_list)

    delay, cancellation = run_pred(airport_index, airline_index)
    print("Your delay is " + str(delay))
    print("Your cancellation time is " + str(cancellation))



if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)
    





