# -*- coding: utf-8 -*-
"""
@author: amirh
"""
import numpy as np
import tensorflow as tf
from numpy import argmax

save_path = "address to the \traiend_surrogate.h5' "
loaded_model = tf.keras.models.load_model(save_path)

def predict_conformity(gdeviations):
    
    """Please do not change the CP_value value. This value is an expert value which is 
    exported from the shopfloor for the specific case study"""
        
    """gdeviations represents the array of geometric devitions on the two gears and is tructured as follow:
    gdeviations = [Tpitcherror_spur_gear, Trunout_spur_gear, Tformdefect_spur_gear, Tpitcherror_spur_gear, Trunout_spur_gear, Tformdefect_spur_gear, Tmisalignment, KTE_value].
    Note that the KTE_value is the admissible KTE value to the designer and the conformity is predicted due to the admissible value,consequently.
    Moreover, the clinical and defined domain for the surrogate model is specifed as follows:
    deviations_var_bound = [(10,30), (10,30), (5,20), (10,30), (10,30), (5,20), (15,30)]
    KTE_var_bound = [15,30] ## KTE admissible interval 
    It should be noted, out of trained boundry, the prediction is not promising"""    
    CP_value = np.array([1.5]*7)
    
   
    arrays = np.array(gdeviations)
    arrays = np.insert (arrays, 7, CP_value, axis=0) 
    arrays = arrays.reshape(1,-1)
        
    dppm = argmax(loaded_model.predict (arrays))
    conformity = (1-dppm/(1e6))
    
    return conformity


gdeviations = [Tpitcherror_spur_gear, Trunout_spur_gear, Tformdefect_spur_gear, Tpitcherror_spur_gear, Trunout_spur_gear, Tformdefect_spur_gear, Tmisalignment, KTE_value]
print(predict_conformity(gdeviations))
