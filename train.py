import os
from typing import Type
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import sigmoid
from tensorflow.keras.losses import Loss
from tensorflow.python.ops.numpy_ops import np_config

def loadDatasets(folder="./games"):
    for file in os.listdir(folder): #iterates through all game files (includes both .pgn and .hdf5 files)
        if not file.endswith(".hdf5"): #ignore any .pgn files
            continue

        file = os.path.join(folder, file) #defines full path to hdf5 file
        try:
            yield h5py.File(file, "r") #returns opened files using generator, so they don't have to all be processed at once
        except:
            print(f"Failed to read: {file}")

def getData(dataSetGroups=["xp","xq","xr"]): 
    data = [] #stores the grouped boards for each game 
    for file in loadDatasets(): #iterates through returned the generator for the datasets 
        try:
            data.append([file[dataSetGroup] for dataSetGroup in dataSetGroups]) #appends the current files dataset groups     
        except:
            raise
    print(f"Reading {len(data)} dataset files...\n")
    #Stacks all datasets, with data from the same game on the same row (e.g. columns [0:64] represent Xp)
    data =np.vstack([np.hstack(dataset) for dataset in data]) #vstack stacks the hstacked datasets vertically (row wise)
                                                              #hstack stacks each datset's groups horizontally (column wise)                                                 
    print("*********************************************")
    print("Splitting dataset into train/dev/test sets...")
    print("*********************************************\n")
    trainData, testData = partitionDataset(data, data.shape[0]) #split dataset into train/dev/test partitions
    print(f"Successfully partitioned dataset into:\n<trainData of length {len(trainData)}>\n<testData of length {len(testData)}>\n")
     #slices datasets to access the different boards, each of which have a 64 elements long
    XpTrain, XqTrain, XrTrain = trainData[:, :64], trainData[:, 64:128], trainData[:, 128:192]
    XpTest, XqTest, XrTest = testData[:, :64], testData[:, 64:128], testData[:, 128:192]

    return XpTrain, XqTrain, XrTrain, XpTest, XqTest, XrTest


def partitionDataset(dataset, datasetSize, trainSplit=0.9, valSplit=0, testSplit=0.1, shuffle=True):
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        rng = np.random.default_rng(1)
        rng.shuffle(dataset, axis=0) #shuffling the dataset yields more realistic validation and test metrics
    
    trainSize = int(trainSplit * datasetSize) #number of elements in train partition
    #valSize = int(valSplit * datasetSize) #number of elements in validation partition
    
    trainDataset = dataset[:trainSize] #slices dataset to include elements up to index trainSize
    #valDataset = dataset[trainSize : trainSize+valSize] #slices dataset to include elements from index trainSize up to trainSize + valSize
    testDataset = dataset[trainSize:] #slices dataset to include remaining elements (index above trainSize+valSize)

    return [trainDataset, testDataset]


def train():
    XpTrain, XqTrain, XrTrain, XpTest, XqTest, XrTest = getData(['xp', 'xq', 'xr'])
    model = MyModel()
    #print(model.call(XpTrain[0]))
    #print(model.call(XqTrain[0]))
    #print(model.call(XpTrain[0]))
    #print(model.call(XrTrain[0]))
    for i in range(5): #stochastic gradient descent (batches of size 1)
      loss =  model.call(XqTrain[i], XpTrain[i], XrTrain[i], training=True)
      print(f"Iteration {i} Loss: {loss}")
    print((model.call(XpTest[0], training=False) - model.call(XqTest[0], training=False))*1000000000000000)
    print((model.call(XqTest[0], training=False) % 0.1)*10000000000000)
    print((model.call(XpTest[0], training=False) % 0.1)*10000000000000)
    #tf.print(model.dotWs)
    #print(tf.reduce_sum(model.call(XqTrain[0])))
    #print(tf.reduce_sum(model.call(XrTrain[0])))
    
    #b = model.call(XqTrain[0]) == model.call(XpTrain[0])
    #print([i for i in range(len((model.call(XqTrain[0]) == model.call(XpTrain[0])))) if b[i]==False])
    
    #print(model.call(XpTrain[0]))
    #print(model.call(XrTrain[0]))
    #print(model.call(XqTrain[1]))
    
    #Fetches all the weights/biases connected to the layers in trained network, and passes them to saveVariables() to be saved to an external file (for later use in the actual game)
    #allVariables = [model.dotWs, model.dotBs, model.dense1.get_weights()[0], model.dense1.get_weights()[1], model.dense2.get_weights()[0], model.dense2.get_weights()[1], model.dense3.get_weights()[0], model.dense3.get_weights()[1]]
    #saveVariables(allVariables) 
    #tf.io.write_file("model1", "\n".join(list(map(str, allVariables))))
    
    #model.save(".//SavedModels//model1.tf", save_format="tf")
    #xqOutput = model.call(XqTrain[0], training =True) 
    #xpOutput = model.call(XpTrain[0], training=True)
    #xrOutput = model.call(XrTrain[0], training=True)
    
    #print(f"Xp Output: {xpOutput}\nXr Output: {xrOutput}\nXq Output: {xqOutput}")
    #lossObject = LogLikelihood(xqOutput, XpTrain[0], XrTrain[0], model)

    #print(lossObject.call())
    #model.compile(optimizer="adam", loss=LogLikelihood(xqOutput,XpTrain[0], XrTrain[0], model))
def saveVariables(variables):
  with open("model1.txt", "w") as file: #file for the trained model to be saved to
    for variable in variables: #itertes through each of the passed variables
      contents = [] #list for tensor's elements to be appended to
      if len(variable.shape) == 2: #for dense layer variables (shape=(768,2048))
        for i in range(variable.shape[0]): #iterates through rows in tensor
          for j in range(variable.shape[1]): #iterates elements in said rows (columns)
            contents.append(float(variable[i][j])) #copies each element by value to the content list
      else: #for dot layer variables
        for i in range(variable.shape[0]): #iterates through each element in the tensor
          contents.append(float(variable[i])) #copies each element by value to the content list
      file.write(str(contents)+"\n") #appends the current variables contents to the file, followed by a new line for readability
    
      
#Defines the structure and functionality of the neural network 
class MyModel(tf.keras.Model): #Inherits from the tensorflow Model class, allowing it to make use of tf's inbuilt methods for efficient usage of the network

  def __init__(self, inputUnits=768, hiddenUnits=2048):
    super().__init__() #initialises tensorflow's Model class variables 
    #self.initialise = tf.keras.initializers.GlorotUniform()
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True) #optimizes network's params using SGD with momentum (with learning rate 0.030), beta of 0.9 averages over last 10 games (1/1-0.9)
    self.dotWs = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(2048,)), name="dotWs", dtype=tf.float32) #initialises the weights for the final dot layer, uses Glorot normal to produce non-zero random values -
                                                                                                                  #ensuring that the output won'b be zero for the first few iterations
    self.dotBs = tf.Variable(tf.keras.initializers.Zeros()(shape=(768,)), name="dotBs", dtype=tf.float32) #initialises the biases which are added to the vector outputted from the dot product
    self.dotOnes = tf.keras.initializers.Ones()(shape=(768,)) #vector of 1s, used outisde of training - where it's dotted with the outputted vector to produce a single value (sum)
    self.dropout = tf.keras.layers.Dropout(0.2) #to be used on each layer (on the output neurons), randomly turns neurons off with probability 0.2
    self.inputLayer = tf.keras.layers.Flatten(input_shape=(inputUnits,)) #Flattens the inputted board into a feature vector (input must be of length 768) 
    self.dense1 = tf.keras.layers.Dense(hiddenUnits, activation=tf.nn.relu, bias_initializer=tf.keras.initializers.Constant(0.1)) 
    self.dense2 = tf.keras.layers.Dense(hiddenUnits, activation=tf.nn.relu, bias_initializer=tf.keras.initializers.Constant(0.1)) #The three hidden layers, each contains 2048 densely connected neurons, uses the RelU actiavtion function   
    self.dense3 = tf.keras.layers.Dense(hiddenUnits, activation=tf.nn.relu, bias_initializer=tf.keras.initializers.Constant(0.1)) #and initialises it's weights using glorot normal and biases to a small +ve constant (0.1) 
    #self.dotLayer = tf.keras.layers.Dot(axes=1)#dot product with output of final hidden layer, produces single value
    #self.dotLayer.add_weight(name="dotW",shape=(768,2048), initializer=self.initialise, trainable=True) #add initialised weights for final 'dot' layer
    
    
  def call(self, input, xp=None, xr=None, training=False):
    #with tf.GradientTape() as tape:
    if training: #dropout and updates only performed during training
      with tf.GradientTape() as tape: #Tracks how the network's variables change (the gradients) as forward prop is carried out- to be used in updates later on
        #Each board is fed through the network using the same process (input is xq)
        input = self.getInputVector(input) #convert board to sequence of 768 bits 
        input = self.inputLayer(input) #flattens input 
        input = self.dense1(input) #feeds input through first layer, outputs matrix of shape (768, 2048) 
        input = self.dropout(input, training=training) #randomly deactivates elements from the output (neurons) with probablity 0.2 (this sets them to 0.)
        input = self.dense2(input) #feeds output from first layer into second layer, again outputs matrix of shape (768,2048)
        input = self.dropout(input, training=training) 
        input = self.dense3(input) #feeds output from second layer into third layer
        input = self.dropout(input, training=training)
        #print(input)    
        #input = tf.math.add(tf.tensordot(input, self.dotWs, axes=1), self.dotBs) #outputs a vector of shape (768,)
        input = tf.tensordot(input, self.dotWs, axes=1)
        #print(input)      
        xp = self.getInputVector(xp)
        xp  = self.inputLayer(xp)
        xp = self.dense1(xp)
        xp = self.dropout(xp, training=training)     
        xp = self.dense2(xp)  
        xp = self.dropout(xp, training=training)    
        xp = self.dense3(xp)
        xp = self.dropout(xp, training=training)    
        #xp = tf.math.add(tf.tensordot(xp, self.dotWs, axes=1), self.dotBs)
        #print(xp)
        xp = tf.tensordot(xp, self.dotWs, axes=1)

        xr = self.getInputVector(xr)
        xr = self.inputLayer(xr)
        xr = self.dense1(xr)
        xr = self.dropout(xr, training=training)     
        xr = self.dense2(xr)  
        xr = self.dropout(xr, training=training)    
        xr = self.dense3(xr)
        xr = self.dropout(xr, training=training)    
        #xr = tf.math.add(tf.tensordot(xr, self.dotWs, axes=1), self.dotBs)
        #print(xr)
        xr = tf.tensordot(xr, self.dotWs, axes=1)
        
        #First part tends to 0 as the difference between input (x) and xr increases, therefore an adapated sigmoid function is used (1/1+e^x)
        #Second part tends to 0 as the difference between input (x) and xp decreases, therefore sigmoid is used (1/1+e^-x)
        cost = -tf.math.reduce_mean(tf.math.log(sigmoid(-(tf.math.abs(input-xr))))) - 10*tf.math.reduce_mean(tf.math.log(sigmoid(tf.math.abs(input- xp)))) #produces single value
      #"w:0", "dense/kernel:0", "dense/bias:0", "dense_1/kernel:0", "dense_1/bias:0", "dense_2/kernel:0", "dense_2/bias:0"
      trainableVariables = [var for var in tape.watched_variables()] #list of the variables which GradientTape found to be interacted with during the above process
      grads = tape.gradient(cost, trainableVariables) #list of tensors denoting the gradients of the watched variables with respect to the final cost function (how much they impact the cost) 
      self.optimizer.apply_gradients(zip(grads,trainableVariables)) #takes the watched variables and their respective gradients,                                                  
      return cost                                                   #and using the given optimizer (SGD), updates their values in an attempt to optimize the network (boost it's accuracy)
    
    else: #during test/game time 
      input = self.getInputVector(input) #convert board to sequence of 768 bits 
      input = self.inputLayer(input) #flattens input
      input = self.dense1(input) #feeds input through first layer, outputs matrix of shape (768, 2048) 
      input = self.dense2(input) #feeds output from first layer into second layer, again outputs matrix of shape (768,2048)    
      input = self.dense3(input) #feeds output from second layer into third layer   
      input = tf.tensordot(input, self.dotWs, axes=1) #+ self.dotBs #outputs a vector of shape (768,)
      input = tf.tensordot(input, self.dotOnes, axes=1) #turns output vector into single value
      return tf.math.abs(input) #returns the absolute value of the output (sometimes it's negative)

  # Convert input into a 12 * 64 list
  def getInputVector(self, inputBoard):
    bitBoards = [] #stores the different piece boards
    for piece in [1,2,3,4,5,6, 8,9,10,11,12,13]: #the different piece 'types'
      bitBoards.append((((inputBoard == piece)).astype(float))*10) #appends 'board' with '100' on squares where the was piece present, '0' where it wasn't (on the current board)
                                                                    #results in 12 differnt 'boards' - one for each piece
    return tf.concat(bitBoards, axis=0) #concatenates the 12 different piece/bit boards into a single list of 768 '100s' and '0s'



#Defines custom loss function - calculated over the three boards 
#class LogLikelihood(Loss): 
#    def __init__(self, xqOut, xpIn, xrIn, model):
#        super().__init__()
#        self.xq = xqOut
#        self.xp = model.call(xpIn, training=True)
#        self.xr = model.call(xrIn, training=True)
    #compute loss
#    def call(self):
#        randomDiff = self.xq - self.xr
#        parentDiff =  self.xq + self.xp
#        lossA = -np.log(sigmoid(randomDiff)).mean()
#        lossB = -np.log(sigmoid(parentDiff)).mean()
#        lossC = -np.log(sigmoid(-parentDiff)).mean()
#        return lossA + lossB + lossC  








if __name__ == "__main__":  
    train()