# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:20:44 2022

Copyright TheAmirHK
"""
# In[] libraries
import numpy as np 
import time 
import matplotlib.pyplot as plt
import math
from collections import Counter
from matplotlib import pyplot
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE 
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SVMSMOTE 
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.under_sampling import NearMiss 

plt.rcParams.update({'font.size': 12})

# In[] split data to train_set and test_set
data = np.load('address to Train_data_1e6_sim.npy')

raw_data = data [np.logical_and(np.logical_not
                                (data[:,15] < 0), np.logical_and(np.logical_not
                                (data[:,16] < 0),np.logical_not
                                (data[:,17] < 0)))]
for i in range (3):
    raw_data[:,i + 15] = (1 - raw_data[:,i + 15])*1e6
    raw_data[:,i + 15] = [ math.ceil( k ) for k in raw_data[:,i + 15] ]
raw_data_y = raw_data[:,15]

## the bound reprsents industrial target which illustrates the number of defectives per million 
## 0.9995 means number of defectives per million are limited to 500 pairs
bound = 0.9995 
data = data [np.logical_not(data[:,15] < bound)]

for i in range (3):
    data[:,i + 15] = (1 - data[:,i + 15])*1e6
    data[:,i + 15] = [ math.ceil( k ) for k in data[:,i + 15] ]
    
# In[] split data to train_set and test_set
def get_data (which_gear):
    imbalanced_data = data[:,:]
    if which_gear == "spur_gear":
        imbalanced_data_X , imbalanced_data_y = np.concatenate( (imbalanced_data[:,0:3],imbalanced_data[:,7:10] ), axis=1) , imbalanced_data[:,16]
    if which_gear == "pinion_gear":
        imbalanced_data_X , imbalanced_data_y = np.concatenate( (imbalanced_data[:,3:6],imbalanced_data[:,10:13] ), axis=1) , imbalanced_data[:,17]        
    if which_gear == "paired":
        imbalanced_data_X , imbalanced_data_y = imbalanced_data[:,:15] , imbalanced_data[:,15]
    if which_gear =="KTE":
        imbalanced_data_X,imbalanced_data_y = imbalanced_data[:,:6] , imbalanced_data[:,-1]
    
        
    imbalanced_data_set = np.column_stack((imbalanced_data_X , imbalanced_data_y))
    return imbalanced_data_X , imbalanced_data_y, imbalanced_data_set
        

imbalanced_data_X , imbalanced_data_y, imbalanced_data_set = get_data("paired")

# In[] data balancing techniques
def data_balancing (imbalanced_data_X,imbalanced_data_y_labeled , technique, method):
    
    if technique == 'none':
        if method == None:        
            balanced_data_x, balanced_data_y = imbalanced_data_X, imbalanced_data_y_labeled 
                       
    if technique =='over_sampling':
        if method == 'random':
            balancing_model = RandomOverSampler (random_state = 42 )
            
        if method == 'smote':
            balancing_model = SMOTE (random_state = 42 )
            
        if method == 'b_smote':
            balancing_model = BorderlineSMOTE (random_state = 42 )
            
        if method == 'smotenc':
            balancing_model = SMOTENC (random_state = 42 )

        if method == 'smoten':
            balancing_model = SMOTEN (random_state = 42 )
                                            
        if method == 'svmsmote':
            balancing_model = SVMSMOTE (random_state = 42 )
            
        if method == 'adasyn':
            balancing_model = ADASYN (random_state = 42 )                                

        if method == 'kmeansmote':
            balancing_model = KMeansSMOTE (random_state = 42 )
            
    if technique =='under_sampling':
        if method == 'random':
            balancing_model = RandomUnderSampler (random_state=42 )
            
        if method == 'nearmiss_1':
            balancing_model = NearMiss (version = 1, n_neighbors=3)
            
        if method == 'nearmiss_2':
            balancing_model = NearMiss (version = 2, n_neighbors=3 )
            
        if method == 'nearmiss_3':
            balancing_model = NearMiss (version = 3, n_neighbors=1 )
      
    balanced_data_x, balanced_data_y = balancing_model.fit_resample(imbalanced_data_X, imbalanced_data_y_labeled)     
    balanced_data = np.column_stack((balanced_data_x, balanced_data_y))
    return balanced_data_x, balanced_data_y, balanced_data 

# In[] return poly fit
def poly_fit (inputs,outputs, deg):
    x = np.linspace(min (inputs) , max(inputs) , len(outputs))
    y = outputs
    z = np.polyfit(x, y, deg)
    f = np.poly1d(z)
    x_new = np.linspace(x[0], x[-1], len(outputs))
    y_new = f(x_new)
    return x_new,y_new

# In[] target output balanced and imbalanced data visualization
def plot_target(imbalanced_data_y, balanced_data_y):
    
    counter_imbalanced_data = Counter(imbalanced_data_y)
    counter_balanced_data = Counter(balanced_data_y)    

    fig, axs = plt.subplots(1,3)
            
    axs[0].hist(raw_data_y, bins=200, color = [(51/255,161/255,201/255)])
    axs[0].set_ylabel('Occurrence')
    axs[0].set_xlabel('dppm')    
    
    axs[1].bar(counter_imbalanced_data.keys(), counter_imbalanced_data.values(), width = 1, color = [(51/255,161/255,201/255)])

    axs[1].set_ylabel('Occurrence')
    axs[1].set_xlabel('dppm')
    
    axs[2].bar(counter_balanced_data.keys(), counter_balanced_data.values(), width = 1, color = [(51/255,161/255,201/255)])
    axs[2].set_ylabel('Occurrence')
    axs[2].set_xlabel('dppm ')
     
    pyplot.show()
    
# In[] target input balanced and imbalanced data visualization
def plot_input(data, title):
    
    feature_names = ['Tpitcherror_spur_gear', 'Trunout_spur_gear', 'Tformdefect_spur_gear','Tpitcherror_crown_wheel' , 'Trunout__crown_wheel', 'Tformdefect_crown_wheel','T_k_assembly',\
                 'Kpitcherror_spur_gear', 'Krunout_spur_gear', 'Kformdefect_spur_gear','Kpitcherror_crown_wheel', 'Krunout_crown_wheel','Kformdefect_crown_wheel', 'K_k_assembly', 'KTE',\
                 'Lambda','Conf. Gear #1','Conf. Gear #2' ]
                 
    x_label = ['µm', 'µm', 'µm','µm' , 'µm', 'µm','µm',\
                 '-', '-', '-','-', '-','-', '-']
                 
    csfont = {'fontname':'Comic Sans MS'}
    
    fig, axs = plt.subplots(2, 7)
    plt.suptitle(title, fontsize = 24, **csfont)
    
    try:
        for i in range (2):
            for j in range (7):
                inf = axs[i,j].hist(data[:,7*i+j], bins=200,color = [(51/255,161/255,201/255)])
                x, y = poly_fit(data[:,7*i+j],inf[0],5)
                axs[i,j].plot(x, y,"--",color = 'k', linewidth=1, label = "trendline")
                axs[i,j].set_title(feature_names[7*i+j], fontsize = 10, **csfont)
                axs[i,j].set_xlabel("{}".format(x_label[7*i+j]))
                axs[i,j].legend(loc='upper left', fontsize = 'x-small')
        fig.tight_layout()                
        plt.show()
    except:
        pass
   
# In[] plot       
balanced_x, balanced_y, balanced_data = data_balancing(imbalanced_data_X, imbalanced_data_y, 'over_sampling', 'random')
plot_target(imbalanced_data_y, balanced_y)
plot_input(balanced_data, "Balanced target input")
plot_input(raw_data, "Raw Input")

