#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #python library used mainly for computing
import pandas as pd #python library 
import matplotlib.pyplot as plt
def ComputeCost(X,y,theta): # defining function
    n=len(y)                  # length 
    h=X.dot(theta)
    square_err=(h-y)**2       #formulas
    return 1/(2*m)*np.sum(square_err)   #returns the value
def gradientDescent (X,y,theta,alpha,num_iters):  # defining function 
    n=len(y)  #length 
    J_history=[]
    for i in  range(num_iters):    # using for loop for iterations
        h=X.dot(theta)
        error=np.dot(X.transpose(),(h-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(ComputeCost(X,y,theta))
    return theta,J_history  # returns the value back to function
data=pd.read_csv('ex1data1.txt',header=None)
data_n=data.values
m=len(data_n[:,0]) # getting length and storing it in m 
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
ComputeCost(X,y,theta)
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
theta[0,0]
plt.scatter(data[0],data[1]) #to view plot
plt.plot(X,theta[1,0]*X+theta[0,0]) # to view the plot
plt.show # to view the plot 


# In[2]:


import numpy as np   #importing python library
import pandas as pd  #importing python library
import matplotlib.pyplot as plt
def ComputeCost(X,y,theta):   # defining a function 
    n=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) #returns the value back to function
def gradientDescent (X,y,theta,alpha,num_iters):# defining gradient function
    n=len(y) #lenth of y is stored in n
    J_history=[]
    for i in  range(num_iters):# defined for loop for performing iterations required
        h=X.dot(theta)
        error=np.dot(X.transpose(),(h-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(ComputeCost(X,y,theta))
    return theta,J_history #returns the value back to the function 
data=pd.read_csv('ex1data1.txt',header=None)
plt.scatter(data[0],data[1]) # used for plotting
data_n=data.values
m=len(data_n[:,0]) #lenght is stored in variable m
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
ComputeCost(X,y,theta)
theta,J_history=gradientDescent(X,y,theta,0.01,1500)
theta[0,0]
plt.scatter(data[0],data[1]) # used for plotting 
plt.plot(X,theta[1,0]*X+theta[0,0]) # used to display plot
plt.show # used to display plot


# In[ ]:




