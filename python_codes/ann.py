#!/usr/bin/env python

'''ann.py: Script para (...)'''

__author__     = 'Dan Ni Lin'
__email__      = 'dan_lin@usp.br'

## 2021 - é bom colocar um cabeçalho pra lembrar o que o código faz, etc.

# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense

## definir uma random seed
## isso garante reprodutibilidade dos resultados
RSEED = 42 

def loadtext(b): #retorna a lista dos valores
    ## Tente usar nomes mais explícitos para as funções/métodos
    ## Essa poderia chamar load_text(file) ou loadText(file)
    ## É sempre bom indicar o parâmetro passado e o que o método retorna

    basepath = '/Users/dannilin/Desktop/IC/dados/'
    l = []
    dat = open(basepath + b).readlines()    
    for i in range(len(dat)):
    	l.append(float((dat[i].rstrip('\n')).replace(',' , '.')))
    return l


def dados(): #gera dados a serem previsto
    ## Pelo que entendi esse método calcula tb/tc para cada ponto
    ## Bons nomes: compute_boc, boiling_over_critical
    
    tb = loadtext('boiling.txt')    
    tc = loadtext('critical.txt')
    l_t = [] #valores observados

    for i in range(len(tb)):
        l_t.append(tb[i]/tc[i])

    return l_t


def ann(): #creating the training and test datasets
    ## Esse método está estranho porque ele faz tudo - cria, treina e testa a NN
    ## Outra coisa estranha é que você não está passando nenhuma variável
    ## Sugestão: train_ann(data); então você poderia chamar ann(dados())
    ## ou usar a mesma rede pra treinar com uma outra base de dados

    target_data = dados()

    target = (np.array(target_data)).reshape(len(target_data),1) # dados a ser previsto no formato vetorial

    predictors = np.transpose([loadtext('mol.txt'),loadtext('acentric.txt')])  # dados para prever o target no formato de matriz, os dados estao dispostos em colunas

    predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size = 0.2, random_state=RSEED)
    ## a linha acima está muito longa, não dá pra ler na tela sem usar o scroll - você pode quebrar em 2 train_test_split
    ## mudei a proporção da base de testes para 20%; você sempre deve usar a maiorida dos dados pra treinar

    #Define model
    model = Sequential()
    model.add(Dense(4, input_dim= 2, activation= 'relu'))
    model.add(Dense(8, activation= 'relu'))
    model.add(Dense(16, activation= 'relu')) ## aumentei um pouco a rede
    model.add(Dense(1))

    #Define an optimizer and the loss measure for trainig
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    model.summary()
    
    # model.fit(predictors_train, target_train, epochs=20) ##
                                
    # Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada
    ## Tem vários jeitos de usar o callbacks - esse aqui está muito estranho
    '''
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        # print('.', end='') ## Comentei essa linha pra não ficar imprimindo
    '''

    EPOCHS = 200 # é mais do que suficiente, nesse caso
    print('Start training for {} epochs ...'.format(EPOCHS))
    history = model.fit(predictors_train, target_train, epochs=EPOCHS, 
    	validation_split = 0.2, verbose=0) #, callbacks=[PrintDot()]) 
    ## pode colocar verbose=1 pra acompanhar no terminal, se quiser.
    
    #Visualize the progress of the training model
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('\n History head: {}'.format(hist.head()))
    print('\n History tail: {}'.format(hist.tail())) ## Evite imprimir
    ## ... mas se precisar, tente imprimir algo antes do número pra identificar no log.

    ## Você entende o que o programa está fazendo aqui? 
    ## Ele está comparando o model.predict com o valor real do dataset, pra estimar o erro RMSE do modelo.

    #Predict on the test data and compute evaluation metrics
    pred_train = model.predict(predictors_train)
    print('\n Train RMSE: {}'.format(np.sqrt(mean_squared_error(target_train, pred_train))))
    ## a mesma coisa que comentei acima

    pred = model.predict(predictors_test) ## mudei aqui para test
    print('\n Test RMSE: {}'.format(np.sqrt(mean_squared_error(target_test, pred))))
    ## O erro na base de testes é normalmente maior
    
    print('Predicted values:', pred_train) ## são muitos dados pra imprimir
    return model

## Adicionei algumas coisas pra tornar o código mais funcional

model = ann() ## com o método ann retornando o seu modelo treinado, você pode salvar ele
model.save('nn_trained.h5') ## assim você não precisa ficar treinando sempre 

from keras.models import load_model
#model = load_model('nn_trained.h5') # already trained? ## você pode usar essa linha pra carregar o modelo

'''
Adicionei essa parte para plotar alguns resultados
Block 2 - Plotting section
'''

## Como você carregou os dados dentro do método ann, é preciso carregar de novo aqui
## (as variáveis são locais)

target_data = dados()
target = (np.array(target_data)).reshape(len(target_data),1) # dados a ser previsto no formato vetorial
predictors = np.transpose([loadtext('mol.txt'),loadtext('acentric.txt')])  # dados para prever o target no formato de matriz, os dados estao dispostos em colunas
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=RSEED)



r2_test = sklearn.metrics.r2_score(y_test, model.predict(X_test))
print(' ')
print('Rˆ2 ', np.round(r2_test,decimals = 2)*100,'%')
print(' ')


print ('Making graphics...')

import matplotlib as mpl
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D

f, ax = plt.subplots(figsize=(8,8))
ax.set(xlim=(0.55, 0.85), ylim=(0.55, 0.85)) ## se quiser limitar os eixos

plt.scatter(model.predict(X_train), y_train, c='blue', alpha=0.2, label='Train')
plt.scatter(model.predict(X_test), y_test, c='orange', alpha=0.4, label='Test')
plt.xlabel('Predicted values', fontsize=20)
plt.ylabel('Database values', fontsize=20)
#plt.title('(a)', fontsize=22)
plt.legend()

diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.2')

# plt.subplots_adjust(left=0.16, bottom=0.16, right=0.94) ## ajustar a posição da caixa
plt.savefig('plot.png', dpi=300)
plt.show()



