# Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


data = train[['trestbps' , 'chol' , 'thalach' , 'oldpeak', 'target']]
x = data[['trestbps' , 'chol' , 'thalach' , 'oldpeak']]
y = data['target'].tolist()
x.insert(0, "xNode", np.ones(x.shape[0]), True)

testData = test[['trestbps' , 'chol' , 'thalach' , 'oldpeak', 'target']]
xTest = testData[['trestbps' , 'chol' , 'thalach' , 'oldpeak']]
yTest = testData['target'].tolist()
xTest.insert(0, "xNode", np.ones(xTest.shape[0]), True)

featureNames = ["xNode", 'trestbps' , 'chol' , 'thalach' , 'oldpeak', 'target']


# Data Normalization for testing data
temp4 = (xTest["trestbps"] - np.mean(x["trestbps"])) / (np.amax(x["trestbps"]) - np.amin(x["trestbps"]))
temp5 = (xTest["chol"] - np.mean(x["chol"])) / (np.amax(x["chol"]) - np.amin(x["chol"]))
temp6 = (xTest["thalach"] - np.mean(x["thalach"])) / (np.amax(x["thalach"]) - np.amin(x["thalach"]))
del xTest["trestbps"]
del xTest["chol"]
del xTest["thalach"]
xTest.insert(1, "trestbps", temp4, True)
xTest.insert(2, "chol", temp5, True)
xTest.insert(3, "thalach", temp6, True)

xTest.reset_index(inplace=True)
del xTest["index"]


# Data Normalization for training data
temp1 = (x["trestbps"] - np.mean(x["trestbps"])) / (np.amax(x["trestbps"]) - np.amin(x["trestbps"]))
temp2 = (x["chol"] - np.mean(x["chol"])) / (np.amax(x["chol"]) - np.amin(x["chol"]))
temp3 = (x["thalach"] - np.mean(x["thalach"])) / (np.amax(x["thalach"]) - np.amin(x["thalach"]))
del x["trestbps"]
del x["chol"]
del x["thalach"]
x.insert(1, "trestbps", temp1, True)
x.insert(2, "chol", temp2, True)
x.insert(3, "thalach", temp3, True)

x.reset_index(inplace=True)
del x["index"]



m = x.shape[0]
n = x.shape[1]

def hypothesis(x, theta):
    temp = np.dot(x, theta)
    return 1/(1+np.exp(-temp))

# Maximum likelihood estimation
def cost(theta, x, y):
    h = hypothesis(x, theta)
    result = 0.0
    for i in range (m): 
        result +=  ((y[i] * np.log(h[i])) + ((1 - y[i]) * np.log(1-h[i]) ) )
    return ((-1 / (m)) * result)[0]


# Get correct guesses 
def cost2(theta, x, y):
    h = hypothesis(x, theta)
    correct = 0
    incorrect = 0
    for i in range (len(x)) :
        if (h[i] >= 0.5) :
            if (y[i] == 1):
                correct += 1
            else:
                incorrect += 1
        else:
            if (y[i] == 0):
                correct += 1
            else:
                incorrect += 1
    return correct
    

# Cost Function in Logistic regression  
def J(theta, x, y, feature):
    result = 0
    h = hypothesis(x, theta)    
    for i in range(m):
        result += (h[i] - y[i]) * x[featureNames[feature]][i]
    return result

costIterations = []
def gradientDescent(theta, x, y, iterations, alpha):
    for iteration in range (iterations):
        tempTheta = []
        for i in range (n):
            temp = theta[i] - ((alpha / m) * J(theta, x, y, i))
            tempTheta.append(temp)
        costIterations.append(cost(theta, x, y))
        # Simultaneously change theta        
        theta = tempTheta
        
   
    
    #print("Cost: ", costIterations)
    
    plt.figure(figsize=(10,6))
    plt.title("Cost")
    
    Niterations = []
    for i in range (iterations):
        Niterations.append(i)
    plt.plot(Niterations, costIterations)
    plt.grid(True) #Always plot.grid true!
    plt.show()
    return theta
theta = np.zeros((x.shape[1], 1))

#if no normalization
#ans = gradientDescent(theta, x, y, 1000, 0.000075)

ans = gradientDescent(theta, x, y, 300, 0.09)

print(ans)
# Predict
yPredicted = cost2(ans, xTest, yTest)
print(yTest)
print(hypothesis(xTest, ans))
print("Correct Guesses: ", yPredicted)
print("Incorrect Guesses: ", len(yTest) - yPredicted)
print("Accuracy:", yPredicted/ len(yTest))

