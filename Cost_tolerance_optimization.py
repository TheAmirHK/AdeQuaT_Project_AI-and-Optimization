# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:08:46 2022

Copyright TheAmirHK
"""
import numpy as np
import pickle
from scipy.stats import truncnorm, norm
from mealpy.utils.visualize import *
import tensorflow as tf
from numpy import argmax
from mealpy.evolutionary_based.DE import SADE


## Note: Cost values in this model are classified as condifientioal data and the following cost parameters do not represent the industry case !
## However, the pickle file representes and evaluates the conformance of real case gears.

save_path = './trained_surrogate.h5'
loaded_model = tf.keras.models.load_model(save_path)
# In[] SADE optimisation run
def gear_conformity_rate (Tolerances, K_values, distribution_type):
    Tolerances = np.array(Tolerances)
    Tpitcherror_spur_gear = Tolerances[0]
    Trunout_spur_gear = Tolerances[1]
    Tformdefect_spur_gear = Tolerances[2]
    
    Kpitcherror_spur_gear = K_values[0]
    Krunout_spur_gear = K_values[1]
    Kformdefect_spur_gear = K_values[2]
    
    lower_bound = 5
    upper_bound = float('inf')
           
    if distribution_type == 'truncated': 
        Spur_gear_pitcherror_failure = truncnorm.cdf(Tpitcherror_spur_gear, lower_bound, upper_bound,0, np.sqrt(Tpitcherror_spur_gear/(3*Kpitcherror_spur_gear)))
        Spur_gear_runout_failure = truncnorm.cdf(Trunout_spur_gear, lower_bound, upper_bound,0, np.sqrt(Trunout_spur_gear/(3*Krunout_spur_gear)))
        Spur_gear_formdefect_failure = truncnorm.cdf(Tformdefect_spur_gear, lower_bound, upper_bound,0, np.sqrt(Tformdefect_spur_gear/(3*Kformdefect_spur_gear)))

    if distribution_type == 'normal':
        
        Spur_gear_pitcherror_failure = norm.cdf(Tpitcherror_spur_gear, 0, Tpitcherror_spur_gear/(3*Kpitcherror_spur_gear))
        Spur_gear_runout_failure = norm.cdf(Trunout_spur_gear, 0, Trunout_spur_gear/(3*Krunout_spur_gear))
        Spur_gear_formdefect_failure = norm.cdf(Tformdefect_spur_gear, 0, Tformdefect_spur_gear/(3*Kformdefect_spur_gear))
        
    Spur_gear_conformity_rate = Spur_gear_pitcherror_failure * Spur_gear_runout_failure * Spur_gear_formdefect_failure
    
    return Spur_gear_conformity_rate

# In[] SADE optimisation run
def Cost_function(solution):
    
    al=0.0027 #Error type I
    be=0.00005 #Error type II

    Spur_gear_conformity_rate, Pinion_gear_conformity_rate = gear_conformity_rate (solution [0:3],solution [7:10],'truncated'), gear_conformity_rate (solution [3:6],solution [10:13],'truncated')
    Spur_gear_conformity_rate = Spur_gear_conformity_rate*(1-al)+(1-Spur_gear_conformity_rate)*be
    Pinion_gear_conformity_rate = Pinion_gear_conformity_rate*(1-al)+(1-Pinion_gear_conformity_rate)*be
    arrays = np.array(solution)
    arrays = arrays.reshape(1,-1)
    dppm = argmax(loaded_model.predict (arrays))
    conformity = (1-dppm/(1e6))
    conformity = conformity*(1-al)+(1-conformity)*be
    
    # cost parameters 120, 367, 24, 95, 20, 30, 43
    Manufacturing_cost = 120/Spur_gear_conformity_rate + 367/Pinion_gear_conformity_rate
    Scrap_cost =  24*(1-Spur_gear_conformity_rate)/Spur_gear_conformity_rate + 95*(1-Pinion_gear_conformity_rate)/Pinion_gear_conformity_rate
    Inspection_cost = 20/Spur_gear_conformity_rate + 30/Pinion_gear_conformity_rate
    Assembly_cost = 43/conformity
    
    
    # Industrial requirement and constraint
    pen1 = 0
#   this constriant limits the number of non-conformed pairs to 50 
    if (argmax(loaded_model.predict (arrays)) > 50):
        
#    this constriant limits the ratio of non-conformed spur gears to 0.01%
#    or (Spur_gear_conformity_rate < 0.999)\ 
    
#    this constriant limits the ratio of non-conformed crown wheel to 0.01%
#    or (Pinion_gear_conformity_rate < 0.999):
    
        pen1= 1e4
        
    return Manufacturing_cost + Scrap_cost + Inspection_cost + Assembly_cost + pen1

def amend_position(solution, lowerbound, upperbound):
    pos = np.clip(solution, lowerbound, upperbound)
    return pos.astype(int)

KTE  = 26
K_value = 1.2
#misalignment_value = 5
problem_dict1 = {
    "fit_func": Cost_function,
    "lb": [10, 10, 5, 10, 10, 5, 5, K_value, K_value, K_value, K_value, K_value, K_value,K_value, KTE],
    "ub": [20, 20, 10, 25, 25, 20, 10, K_value, K_value, K_value, K_value, K_value, K_value, K_value, KTE],
    "minmax": "min",
    "verbose": False,
}


# In[] SADE optimisation run
epoch = 20
pop_size = 10
model = SADE(problem_dict1, epoch, pop_size)


# In[] Print solution
best_position, best_fitness = model.solve()
model.history.save_global_objectives_chart(filename="hello/goc")
model.history.save_local_objectives_chart(filename="hello/loc")

sol = np.array(best_position)
sol = sol.reshape(1,-1)
print (argmax(loaded_model.predict (sol)))
test_x, test_y = gear_conformity_rate (best_position [0:3],best_position [7:10],'normal'), gear_conformity_rate (best_position [3:6],best_position [10:13],'normal')
print (test_x, test_y)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
