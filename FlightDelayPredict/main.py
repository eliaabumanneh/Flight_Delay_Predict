import streamlit as st
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


#Streamlit Componenets

#initialising containers
header = st.beta_container()
input = st.beta_container()

current_dir = os.getcwd()
with header: 
    st.title('Flight Delay Predict')
    st.write('Data Science Bootcamp Capstone Project')
    st.write('Elia Abu-Manneh')
    st.write('April 12 2023')

    
    st.write(current_dir)

airline_list = ['Delta', 'United', 'American', 'Southwest']


with input:
    selected_index = st.selectbox('Select an airline:', airline_list)


def rmse(y_true, y_pred): #defining the Root Mean Squared Error function
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

airports_df = Path(__file__) / 'airports.csv'
airlines_df = Path(__file__) / 'airlines.csv'
ontime_10423 = Path(__file__) / 'ontime_10423.csv'

#airports_df = pd.read_csv('airports/csv')
#airlines_df = pd.read_csv('airlines.csv')
#ontime_10423 = pd.read_csv('ontime_10423.csv')

with custom_object_scope({'rmse': rmse}):
    modely1 = keras.models.load_model('modely1.h5')
    modely2 = keras.models.load_model('modely2.h5')


#user input prediction function
def user_pred(numpy_array_input):  #input is shape (43,), all OHE except the last 

    #make delay prediction with the model
    raw_delay_prediction = modely1.predict(numpy_array_input)
    tranformed_delay_prediction = np.exp(raw_delay_prediction) -30

    #make cancellation prediction with the model
    cancellation_prediction = modely2.predict(numpy_array_input)

    return tranformed_delay_prediction, cancellation_prediction

#Setting up the input matrix

X_data = ontime_10423.iloc[:,:-64]
X_data.drop(['ORIGIN_AIRPORT_ID','DEP_DELAY','CANCELLED'], axis=1, inplace=True)
collist = X_data.columns.tolist()
input_df = pd.DataFrame({'feature': collist, 'val': 0* len(collist)})

first_row = input_df.iloc[0]
input_df = input_df.iloc[1:]
input_df = input_df.append(first_row, ignore_index=True)


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

def yeartodate_scaled():
    day_of_year = datetime.now().timetuple().tm_yday
    return day_of_year / 365





#SAMPLE INPUT
input_df['val'] = 0      #Resetting input
input_airline = 14        #Remember, this starts at 0  #drop down menu appears as user starts typing, index is stored
input_dest = 8         #Remember, this starts at 0  #drop down menu appears as user starts typing, index is stored

#EXECUTION

input_df.iloc[input_airline,1] = 1          #Executes the addition of airline to input
input_df.iloc[input_dest+17,1] = 1          #Executes the addition of airport to input
input_df.iloc[-1,-1] = round(yeartodate_scaled(),3)  #Executes the addition of scaled YTD

X_input = np.array(input_df.iloc[:,1]).reshape(-1,43)

#test prediction
prediction = user_pred(X_input)
delay_pred = str(prediction[0])
delay_pred = delay_pred[2:-8]

#output to be sent to user
cancellation_pred = str(prediction[1])
cancellation_pred = cancellation_pred[2:-7]


print("Expected Delay for this flight is: " + str(delay_pred) + " Minutes")
print("Expected Probability of Cancellation for this flight is: " + str(cancellation_pred))
