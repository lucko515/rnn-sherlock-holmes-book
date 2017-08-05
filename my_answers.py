import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    #defining counter to the zero
    i = 0
    #entering splitting loop
    while (i+window_size) != len(series):
        #each input is the sequence of vaues
        X.append(series[i:i+window_size])
        #each output is the single value
        y.append(series[i+window_size])
        i += 1

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #Defining the model of the network
    model = Sequential()
    #adding the LSTM layer with 16 units and tanh activation function
    model.add(LSTM(5, activation='tanh', input_shape=(window_size, 1)))
    #output of the each time step should be single number (predicted value of the stocks)
    model.add(Dense(units=1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    english_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                  'u', 'v', 'w', 'x', 'y', 'z', ' ']

    #This loop looks char at the time
    for i in range(len(text)):
        #if the current char is in eng letters or it is in punctuation we do nothing
        if (text[i] in english_letters) or (text[i] in punctuation):
            continue
        #if the char is some other than english letters, space or punctuation we replace it with the space
        else:
            text = text.replace(text[i], ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    #Define counter to 0 (zero)
    i = 0
    #Entering the loop to split data
    while i < (len(text)-window_size):
        #each input is the sequence of characters
        inputs.append(text[i:i+window_size])
        #each output is the single character
        outputs.append(text[i+window_size])
        #increasing the counter by the value of step_size
        i += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential() #Defining the model type 
    #Using LSTM hidden layer with 128 units, activation tanh
    model.add(LSTM(200, activation='tanh', input_shape=(window_size, num_chars)))
    #Our output layer has the number of neurons equal to number of characters in our vocab
    #Because we have many classes (for each char one class) we are using softmax activation function to create probs from logits outputs 
    model.add(Dense(num_chars, activation='softmax'))
    return model
