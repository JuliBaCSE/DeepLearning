# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#__________________________

# Importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# now we have to select the right column that we need and create a numpy array 
# since we only need the first column we take 1:2 since 2 is excluded
# as result we have a numpy array of column 1
# .values will create the numpy array
training_set = dataset_train.iloc[:, 1:2].values

#__________________________

# Feature Scaling

# here it is recommended to use normalisation
# whenever sigmoid function in the output layer => recom. using normalisation
# therefore we use the minmaxscalar class from the sklearn library
# all our scaled stockprizes will be between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
# since we want to keep our original stock values we create a new on which we apply
# fit => get min max
# transform => apply on each scaling
training_set_scaled = sc.fit_transform(training_set)

#__________________________

# Creating a data structure with 60 timesteps and 1 output

# that will be what the rnn has to remember
# !!! timesteps has to be selected right otherwise overfitting
# 60 timestepts => at each time t rnn will look at 60 previous timesteps
# => past information on which it learn and understand
# 60 is gained by experiments !!

# x_train is the input and y_train the output <+> both inti with empty list
# since we need 60 previous stock prices we have to start with 60 
# (0-59 => 60; start one after => 60)
# last value is at 1257 since in range last is excluded => take 1258
# @ each iteration we want to append the 60 previous stockprizes to x_train
# y_train will has the stock prize at time t+1 => i 
# f.e. i = 60 => x_train has all from 0 to 59 and y_train has value at index 60 which will get predicted

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# since x_ and y_train are list we have to convert them into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

#___________________________

# Reshaping

# with that extra dimension we can add some more indicator to predict the stockprize
# currently only the 60 previous days work as indicator
# therefore we use the reshape function 
# this is also important to have the right input format for the RNN 
# (check keras docu recurrent layers) => input 3D array (tensor) 

# => we need batch_size (1198 or .shape[0] = lines), timesteps (60 or .shape[1] = columns) 
# and input_dim = 1 which equals the number of indicators (currently 1 = google stock prize)

# f.e. apply stock prize depend on samsung stock prize => add this dimension
# currently our array has 2 dimensions (1198 lines and 60 columns)
# here we add a new dimension 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#:::::::::::::::::::::::::::::::::::::::::::

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#___________________________

# Initialising the RNN

# we init regressor as a sequence of layers
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# at first we add the LSTM layer

# units => number of LSTM cells (simply neurons)/memory untis in our LSTM
# since capturing the trends of a stockprice is really complex we want a high dimensionality 
# => we take 50 to better capture the up and downwards trends

# return_sequences => since we have a stacked LSTM we have to set it to true if after this layer another follows

# input_shape => the shape that we created 
# we only have to input the last two dimensions (timesteps and indicators) since we first one will automatically be taking into account

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# we also need some dropout regularisation because we want to avoid overfitting
# the argument is the dropout rate => rate of neurons that gets dropped (here 20 %)
# => 20% of 50 = 10 => 10 neurons will be dropped at each iteration
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# since this is our last layer we can use the default of return_.. (false)
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# to have fully connected we use the dense class
# since we have one output => stockprize at t+1 => units = 1
regressor.add(Dense(units = 1))

#___________________________

# Compiling the RNN

# here we use mean_squared_error since we calculate regression
# for optimizer check keras documentation
# RMSprop could also be used but here adam is better
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#___________________________

# Fitting the RNN to the Training set

# fit will connect NN to the training set and execute over the number of epochs
# increase epochs until convergence
# batch size is the number of stock prizes going into the NN
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# be aware if loss is too small it could be overfitting

#:::::::::::::::::::::::::::::::::::::::::::

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#___________________________

# Getting the predicted stock price of 2017

# since to predict the stockprices in january we need the previous 60 stock prizes from last year (training set) and partly from the test set
# => we have to concat both 
# since we dont want to scale the test values we first create a new set with the original training set and test set and scale afterwards
# scaling is importating since our rnn is trained on scaled values
# we use concat from pandas input the two data frames since we only want the open stock prizes we want only the column 'Open' 
# and axis is what we want to concat (lines or columns) 
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# at each day we need to 60 previous => lower bound first day we want to predict - 60 and upper bound is the last stockprize
# again we want a numpy array => .values to shape it to a numpy array we use .reshape (otherwise waring format will pop up)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
# to scale our inputs for the NN we use our scaling object and the transform 
# since our sc is already fitted to the training set and we want the same scaling we only need the transform (not fit transform)
inputs = sc.transform(inputs)

# the upper bound is now different since we only look 20 (4 x 5 days) financial days -> 60 + 20 = 80
# since we do prediction we dont need y_train (the true values)
# again we have to get the 3D fromat as previous => .reshape
# since we want the actual stockprize we have to inverse our scaling
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#___________________________

# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()