# Flight Delay Predict

![image](https://user-images.githubusercontent.com/59853149/232960546-57ccabb8-4da3-4524-8f29-260d28002c1d.png)

# TLDR;

User inputs their origin airport, destination and airline. The tool predicts the expected flight delay and likelihood of cancellation.

# Project Walkthrough
A tool that allows users to get a prediction on the expected delay of their flight and the likelihood of cancellation using historical data. 

In [Part 1](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_1_Data_Processing.ipynb) Data Exploration and Organisation, historical data was collected, explored and compiled. Pre-processing of the data consisted of removing rows and columns with significant levels of Nan and 0 values. Feature engineering was carried out by removing features that are not releveant to our model, as well as creating dummy variables to features such as airline name. 


In [Part 2.1](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_2.1_Modelling_Linear.ipynb), processed data was modelled using an OLS linear model and XGBoost. Data was transformed logarithmically to fit a normal distribution before modelling.
![1](https://user-images.githubusercontent.com/59853149/233203925-5fe57cbd-a5c2-47e7-b5db-53be3c7f4745.png)
![2](https://user-images.githubusercontent.com/59853149/233204204-368b7edb-264c-4461-9be8-7c4ac2b5dc90.png)

Note: Blue indicates statistically significant results. Red indicates otherwise. 

The models were evaluated and compared based on their MSE, RMSE, MAE and R^2 metrics. Despite their interpretability, the linear models were not performant. Therefore, we attempt to model the data using a neural network.

In [Part 2.2](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_2.2_Keras_Modelling-No-Clipping-No-Weather.ipynb), a Keras model was created, trained and used to make predictions. The Keras model performed better than the linear models. The predictions were evaluated and compared with the linear models. The model's architecture and weights were saved for future use. 
![image](https://user-images.githubusercontent.com/59853149/233199863-4e9303d9-bd8e-4ced-ae39-2c69e8b2d9e7.png)
![image](https://user-images.githubusercontent.com/59853149/233201100-9bcf824a-8a22-4d4a-b371-f594c9d8fa87.png)





In [Part 3](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_3_User_interface.ipynb), a user interface (in juypter noteobok) was created to allow for custom predictions. Future porting of the model (eg: creating a Flask/Django deployment) would only require viewing this notebook. 

A [Flask app](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/production/app.py) was created using Python that uses an HTML page for user interaction. It was deployed locally. Below is a sample use case: 
![image](https://user-images.githubusercontent.com/59853149/232960546-57ccabb8-4da3-4524-8f29-260d28002c1d.png)




# Project Limitations
* Data Limitation: The data used in this tool is limited to the contingous United States and limited to the airport data provided by the Buereau of Transportation Statistics. 

* Modelling Limitation: Due to the large size of the data, the models have been subdivided by origin airport to reduces modelling complexity by 2 orders of Magnitude. 

# Data Sources
* [Buereau of Transportation Statistics - Airline On-Time Performance Data](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD&QO_anzr=Nv4yv0r%FDb0-gvzr%FDcr4s14zn0pr%FDQn6n&QO_fu146_anzr=b0-gvzr) 
* [Kaggle - 2019 Airline Delays w/Weather and Airport Detail by Jen Wadkins -  Raw Data](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations)

* [NOAA National Centers for Environmental Information - Global Historical Climatology Network daily (GHCNd) ](https://noaa-ghcn-pds.s3.amazonaws.com/index.html#csv/by_year/)

# Languages Used
* Python
* HTML

# Software Used
* Visual Code Studio
* Google Chrome
* GitHub Desktop
* GitHub
* cuDNN

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
