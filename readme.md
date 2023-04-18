# Flight Delay Predict

#### Data Science Capstone Project

# TLDR;

User inputs their origin airport, destination and airline. The tool predicts the expected flight delay and likelihood of cancellation.


# Project Description
A tool that allows users to get a prediction on the expected delay of their flight and the liklihood of cancellation using historical data. 

In [Part 1](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_1_Data_Processing.ipynb), Data was collection and pre-processed.

In [Part 2.1](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_2.1_Modelling_Linear.ipynb), the data was modelling using a linear model and XGBoost. The models were evaluated and compared based on a number of metrics. Despite it's interpretability, the linear model was not performant. Therefore, we attempt to model the data using a neural network was

In [Part 2.2](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_2.2_Keras_Modelling-No-Clipping-No-Weather.ipynb), a Keras model was created, trained and used to make predictions. The predictions were evaluated and compared with the linear models. 

In [Part 3](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_3_User_interface.ipynb), a user interface (in juypter noteobok) was created to allow for custom predictions. Future porting of the model would only require viewing this notebook. 


# Project Limitations
* Data Limitation: The data used in this tool is limited to the contingous United States and limited to the airport data provided by the Buereau of Transportation Statistics. 

* Modelling Limitation: Due to the large size of the data, the models have been subdivided by originairport to reduces modelling complexity by 2 orders of Magnitude. 

# Data Sources
* [Buereau of Transportation Statistics - Airline On-Time Performance Data](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD&QO_anzr=Nv4yv0r%FDb0-gvzr%FDcr4s14zn0pr%FDQn6n&QO_fu146_anzr=b0-gvzr) 
* [Kaggle - 2019 Airline Delays w/Weather and Airport Detail by Jen Wadkins -  Raw Data](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations)

* [NOAA National Centers for Environmental Information - Global Historical Climatology Network daily (GHCNd) ](https://noaa-ghcn-pds.s3.amazonaws.com/index.html#csv/by_year/)

# Languages Used
* Python
* HTML

# Softwares Used
* Visual Code Studio
* Google Chrome
* GitHub Desktop
* GitHub
* CUDNN

# Python Packages Used
* Pandas
* Numpy
* Scikit-learn
* Tensorflow 
* Keras
* Matplotlib
* Xgboost
* Statsmodels

# Legal

This project was created and is intended for educational purposes only. The creator(s) nor any user(s) nor distributor(s) claim legal responsibility for any information, data, claim or prediction infered or supplied by this program whether implicitly or explicitly. The program is not to be sold or resold. The information provided by the program is not to be sold or resold. The project may be used by others for educational and non-commerical purposes. 
