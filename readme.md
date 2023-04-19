# Flight Delay Predict

![image](https://user-images.githubusercontent.com/59853149/232960546-57ccabb8-4da3-4524-8f29-260d28002c1d.png)

# TLDR;

User inputs their origin airport, destination and airline. The tool predicts the expected flight delay and likelihood of cancellation.

# Project Walkthrough
A tool that allows users to get a prediction on the expected delay of their flight and the liklihood of cancellation using historical data. 

In [Part 1](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_1_Data_Processing.ipynb) Data Exploration and Organisation, historical data was collected, explored and compiled. Pre-processing of the data consisted of removing rows and columns with significant levels of Nan and 0 values. Feature engineering was carried out by removing features that are not releveant to our model, as well as creating dummy variables to features such as airline name. 


In [Part 2.1](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_2.1_Modelling_Linear.ipynb), processed data was modelled using an OLS linear model and XGBoost. Data was transformed logarithmically to fit a normal distribution before modelling.
![image](https://user-images.githubusercontent.com/59853149/233200473-6287324a-a4f6-46f6-97db-dea9b0d68f5d.png)
![image](https://user-images.githubusercontent.com/59853149/233200580-72ae10f0-59de-4d8d-9119-1bfe30f1af95.png)


The models were evaluated and compared based on their MSE, RMSE, MAE and R^2 metrics. Despite their interpretability, the linear models were not performant. Therefore, we attempt to model the data using a neural network.

In [Part 2.2](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_2.2_Keras_Modelling-No-Clipping-No-Weather.ipynb), a Keras model was created, trained and used to make predictions. The Keras model performed better than the linear models. The predictions were evaluated and compared with the linear models. The model's architecture and weights were saved for future use. 
![image](https://user-images.githubusercontent.com/59853149/233199863-4e9303d9-bd8e-4ced-ae39-2c69e8b2d9e7.png)





In [Part 3](https://github.com/eliaabumanneh/Flight_Delay_Predict/blob/main/bin/Part_3_User_interface.ipynb), a user interface (in juypter noteobok) was created to allow for custom predictions. Future porting of the model (eg: creating a Flask/Django deployment) would only require viewing this notebook. 





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
* Keras

# Legal

This project was created and is intended for educational purposes only. The creator(s) nor any user(s) nor distributor(s) claim legal responsibility for any information, data, claim or prediction infered or supplied by this program whether implicitly or explicitly. The program is not to be sold or resold. The information provided by the program is not to be sold or resold. The project may be used by others for educational and non-commerical purposes. 
