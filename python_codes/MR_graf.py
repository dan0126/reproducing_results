import os.path
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import sys
import random
import math
import pandas as pd
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

#__author__     = 'Dan Ni Lin'
#__email__      = 'dan_lin@usp.br'


"This code is about the multiple regression exercise which can be found in this article:"

' Machine Learning for Fluid Property Correlations: Classroom Examples with MATLAB '

"Lisa Joss and Erich A. Müller " 


def v(b): #This part returns the data made available by Professor Muller to carry out the activity.

    basepath = '/Users/dannilin/Desktop/IC/dados/'

    l = []

    dat = open(basepath + b).readlines()
    
    for i in range(len(dat)):

         l.append(float((dat[i].rstrip('\n')).replace(',' , '.')))

    return l


def alea():

    l = []

    for i in range(100):

        l.append(random.randint(0,5999))

    return l

def multi():

#############################_Genrates Data_##################################

    tb = v('boiling.txt')
    
    tc = v('critical.txt')

    mw = v('mol.txt')

    ac = v('acentric.txt')

    num = alea()

    l_mw = []

    l_ac = []

    l_t = [] #Observed values

    for i in num:

        l_mw.append(mw[i])

        l_ac.append(ac[i])

        l_t.append(tb[i]/tc[i])

###############################_Solves the math system_#############################

    
    X = np.matrix([np.repeat(1,100),l_mw,l_ac]) #transp
    
    y = np.transpose(np.matrix(l_t))
        
    x_t = np.transpose(X) #normal
    
    #Solves the system
    #Predicts Tb / Tc and turns the matrix into a list

    l_solução = np.squeeze(np.asarray(((np.linalg.inv(X.dot(x_t))).dot(X)).dot(y)))
    #Return the values of the coeficient of our plane equation

###############################_Predicted Values_##############################

    
    l_previsto = []
    
    for i1 in range(len(l_mw)):

        l_previsto.append(l_solução[0]+l_solução[1]*l_mw[i1]+l_solução[2]*l_ac[i1])
        
    

#############################_Calculates R_squared_##############################

    a = 0

    b = 0

    c = 0

    d = 0

    e = 0

    for i2 in range(len(l_previsto)):

        a += l_t[i2]*l_previsto[i2]

        b += l_previsto[i2] 

        c += l_t[i2] 

        d += l_previsto[i2]*l_previsto[i2]

        e += l_t[i2]*l_t[i2]

    r2 = np.round(abs((len(l_t)*a-(b*c)))/np.sqrt(((len(l_t)*(d))-(b*b))*((len(l_t)*e)-(c*c)))*100,decimals = 2)


    print('Rˆ2',r2,'%')
                              

##########################_Calculates AAD/%_##################################

    a1 = 0 #contagem

    for i3 in range(len(l_t)):

        a1 += (np.abs(np.round(l_previsto[i3],decimals = 2) - l_t[i3])/l_t[i3])

    aad = np.round(a1/len(l_t)*100,decimals = 2)

    print('AAD',aad,'%')


########################_Graphic_###########################################

    X = np.transpose(np.matrix([mw, ac])) #mw, ac
    
    Y = l_t #t

    l1 = np.array(np.transpose(X[:, 0])[0])
    l2 = np.array(np.transpose(X[:, 1])[0])
    
    z = Y

    x = []
    y = []

    for i4 in range(len(l1)): #transforms l1 and l2 into list

        for j in range(len(l1[i4])):

            x.append(l1[i4][j])

            y.append(l2[i4][j])

    l_r_t = [] #list with all the values ​​of the ratio of tb and tc
    Z1 = []

    for i5 in range(len(tb)):

        l_r_t.append(tb[i5]/tc[i5])
        Z1.append(l_solução[0]+l_solução[1]*mw[i5]+l_solução[2]*ac[i5])



 

    f, ax = plt.subplots(figsize=(8,8))
    ax.set(xlim=(0.55, 0.85), ylim=(0.55, 0.85)) ## se quiser limitar os eixos
    plt.scatter(Z1, l_r_t, c='orange', alpha=0.4, label='Test')
    plt.scatter(l_previsto, l_t, c='blue', alpha=0.2, label='Train') #100 to train
    
    plt.xlabel('Predicted values', fontsize=20)
    plt.ylabel('Database values', fontsize=20)
    #plt.title('(a)', fontsize=22)
    plt.legend()

    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.2')

    plt.show()

multi()

































