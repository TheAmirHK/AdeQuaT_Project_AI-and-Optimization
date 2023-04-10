# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:33:52 2022

@author: amirh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
import time 
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
start_time = time.time()
     
### define gears specifications: Tolerance (Pitch error, Runout, Form_defect, Numebr of Teeth, Assembly) and k and Pressure angle
def Optimization_interface (Tolerances, K_values, KTE, distribution_type):
    
    Tolerances = np.array(Tolerances)
    Tpitcherror_spur_gear = Tolerances[0]
    Trunout_spur_gear = Tolerances[1]
    Tformdefect_spur_gear = Tolerances[2]
    Tpitcherror_pinion_gear = Tolerances[3]
    Trunout_pinion_gear = Tolerances[4]
    Tformdefect_pinion_gear = Tolerances[5]
    T_k_assembly = Tolerances[6]
    
    Kpitcherror_spur_gear = K_values[0]
    Krunout_spur_gear = K_values[1]
    Kformdefect_spur_gear = K_values[2]
    Kpitcherror_pinion_gear = K_values[3]
    Krunout_pinion_gear = K_values[4]
    Kformdefect_pinion_gear = K_values[5]
    K_k_assembly = K_values[6]
     
        
    spur_gear_char = [Tpitcherror_spur_gear, Kpitcherror_spur_gear, Trunout_spur_gear, Krunout_spur_gear, Tformdefect_spur_gear, Kformdefect_spur_gear, 13] 
    pinion_gear_char = [Tpitcherror_pinion_gear, Kpitcherror_pinion_gear, Trunout_pinion_gear, Krunout_pinion_gear, Tformdefect_pinion_gear,Kformdefect_pinion_gear, 19]
    T_k_assembly = [T_k_assembly, K_k_assembly]
    Pressure_angle = 23     
    Number_of_Monte_Carlo = 1000000 ### Note that the number of simulation plays a crucial role in the accuracy of the experiments and industry partner data
    Adimissible_KTE_boundary = KTE
    lower_bound = 0
    upper_bound = float('inf')
    distribution_type = 'truncated' ### Define the distribution to be 'normal' or 'truncated' 
  
    ### A function to generate set of random deviation on a gear
    def generate_random_deviation( Tpitcherror, Kpitcherror, Trunout, Krunout, Tformdefect, Kformdefect, Numebr_of_teeth):

        distance_k_k1 = np.zeros(Numebr_of_teeth)        
        form_defect_k_k1 = np.zeros(Numebr_of_teeth)
        
        
        if distribution_type == 'truncated':
            
            d_Pitcherror = truncnorm.rvs(lower_bound, upper_bound,0,Tformdefect/(3*Kformdefect), size=Numebr_of_teeth) ### Truncated normal random
            d_Formdefect = truncnorm.rvs(lower_bound, upper_bound,0,Tformdefect/(3*Kformdefect), size=Numebr_of_teeth) ### Truncated normal random
            d_Runout = truncnorm.rvs(lower_bound, upper_bound,0, Trunout/(3*Krunout), size=1) ### Truncated normal random

        elif distribution_type == 'normal':
            
            d_Pitcherror = np.random.normal(0, Tpitcherror/(3*Kpitcherror), (Numebr_of_teeth))   ### Normal random                        
            d_Formdefect = np.random.normal(0, Tformdefect/(3*Kformdefect), (Numebr_of_teeth))   ### Normal random        
            d_Runout = np.random.normal(0, Trunout/(3*Krunout))      ### Normal random  

        else:
            print('Please check your distribution type, either normal or truncated')
            
            
        for i in range (Numebr_of_teeth):
            distance_k_k1[i] = np.abs(d_Pitcherror[i-1] - d_Pitcherror[i] )               
            form_defect_k_k1[i] = np.abs(d_Formdefect[i-1] - d_Formdefect[i] )
            
                
        return  distance_k_k1, d_Runout, form_defect_k_k1
    
    
    ### Generate set of random deviations on a specific pair of gears
    def gerate_deviations_set_on_Spur_and_Pinion ():
        
        generate_random_deviation_on_spur= []
        generate_random_deviation_on_pinion = []
            
        generate_random_deviation_on_spur = generate_random_deviation( spur_gear_char[0], spur_gear_char[1],\
                                                                      spur_gear_char[2], spur_gear_char[3], spur_gear_char[4], spur_gear_char[5], spur_gear_char[6])
        generate_random_deviation_on_spur = list(generate_random_deviation_on_spur)
        
        generate_random_deviation_on_pinion = generate_random_deviation( pinion_gear_char[0], pinion_gear_char[1],\
                                                                        pinion_gear_char[2], pinion_gear_char[3], pinion_gear_char[4], pinion_gear_char[5], pinion_gear_char[6])
        generate_random_deviation_on_pinion = list(generate_random_deviation_on_pinion)
        
        d_assembly = truncnorm.rvs(lower_bound, upper_bound, 0, T_k_assembly[0]/(3*T_k_assembly[1]), size=1) ### Truncated normal random
            
        return generate_random_deviation_on_spur, generate_random_deviation_on_pinion, d_assembly
    
    
    def gear_conformity_rate ():
        
        if distribution_type == 'truncated':        
        
            Spur_gear_pitcherror_failure = truncnorm.cdf(Tpitcherror_spur_gear, lower_bound, upper_bound,0, Trunout_spur_gear/(3*1))
            Spur_gear_runout_failure = truncnorm.cdf(Trunout_spur_gear, lower_bound, upper_bound,0, Trunout_spur_gear/(3*1))
            Spur_gear_formdefect_failure = truncnorm.cdf(Tformdefect_spur_gear, lower_bound, upper_bound,0, Tformdefect_spur_gear/(3*1))
            Spur_gear_conformity_rate = Spur_gear_pitcherror_failure * Spur_gear_runout_failure * Spur_gear_formdefect_failure
            
            Pinion_gear_pitcherror_failure = truncnorm.cdf(Tpitcherror_pinion_gear, lower_bound, upper_bound,0, Trunout_pinion_gear/(3*1))
            Pinion_gear_runout_failure = truncnorm.cdf(Trunout_pinion_gear, lower_bound, upper_bound,0, Trunout_pinion_gear/(3*1))
            Pinion_gear_formdefect_failure = truncnorm.cdf(Tformdefect_pinion_gear, lower_bound, upper_bound,0, Tformdefect_pinion_gear/(3*1))
            Pinion_gear_conformity_rate = Pinion_gear_pitcherror_failure * Pinion_gear_runout_failure * Pinion_gear_formdefect_failure   
        
        if distribution_type == 'normal':
            
            Spur_gear_pitcherror_failure = norm.cdf(Tpitcherror_spur_gear, lower_bound, upper_bound,0, Trunout_spur_gear/(3*1))
            Spur_gear_runout_failure = norm.cdf(Trunout_spur_gear, lower_bound, upper_bound,0, Trunout_spur_gear/(3*1))
            Spur_gear_formdefect_failure = norm.cdf(Tformdefect_spur_gear, lower_bound, upper_bound,0, Tformdefect_spur_gear/(3*1))
            Spur_gear_conformity_rate = Spur_gear_pitcherror_failure * Spur_gear_runout_failure * Spur_gear_formdefect_failure
            
            Pinion_gear_pitcherror_failure = norm.cdf(Tpitcherror_pinion_gear, lower_bound, upper_bound,0, Trunout_pinion_gear/(3*1))
            Pinion_gear_runout_failure = norm.cdf(Trunout_pinion_gear, lower_bound, upper_bound,0, Trunout_pinion_gear/(3*1))
            Pinion_gear_formdefect_failure = norm.cdf(Tformdefect_pinion_gear, lower_bound, upper_bound,0, Tformdefect_pinion_gear/(3*1))
            Pinion_gear_conformity_rate = Pinion_gear_pitcherror_failure * Pinion_gear_runout_failure * Pinion_gear_formdefect_failure   
                  
        return Spur_gear_conformity_rate, Pinion_gear_conformity_rate 
    
    ### A function to run Monte-Carlo simulation 
    def Monte_carlo ( Number_of_iterations):
        
        x,y,z = gerate_deviations_set_on_Spur_and_Pinion ()
            
        alpha = Pressure_angle*np.pi/180   
        KTE_list = list()
        
        Pitch_error_distance_on_spur = list()
        Pitch_error_distance_on_pinion = list()
        
        Runout_error_on_spur = list()
        Runout_error_on_pinion = list()
        
        Form_defect_on_spur = list()
        Form_defect_on_pinion = list()
        
        Assembly_error = list() 
            
        for iter in tqdm(range (Number_of_iterations)):
          x,y,z = gerate_deviations_set_on_Spur_and_Pinion ()
          
          Pitch_error_distance_on_spur.append(x[0])
          Pitch_error_distance_on_pinion.append(y[0])
          
          Runout_error_on_spur.append(x[1])
          Runout_error_on_pinion.append(y[1])
          
          Form_defect_on_spur.append(x[2])
          Form_defect_on_pinion.append(y[2])
          
          Assembly_error.append(z)
          
          KTE = max(x[0]) + max (y[0]) + np.sin(alpha * (x[1]+y[1])) + 2*np.sin( alpha * z ) + max(x[2]) + max (y[2])
          KTE_list.append(KTE)
        
        design_space = list()
        design_space = [Pitch_error_distance_on_spur, Pitch_error_distance_on_pinion, Runout_error_on_spur, Runout_error_on_pinion, \
                        Form_defect_on_spur, Form_defect_on_pinion, Assembly_error]
        ### Design space (Runout, Pitch error, Form defect, and Assembly error) and the resoponse return here
        return design_space, KTE_list 

        
    ### Monte-Carlo iteration analysis
    Figure = plt.plot
    def Plot_function (Number_of_Monte_Carlo):
        Number_of_Monte_Carlo = Number_of_Monte_Carlo
        design_space, KTE_list = Monte_carlo(Number_of_Monte_Carlo)
        
        mean_total = np.array([np.mean(KTE_list) ] *Number_of_Monte_Carlo)
        max_total = np.array([max(KTE_list) ] *Number_of_Monte_Carlo)
#        min_total = np.array([min(KTE_list) ] *Number_of_Monte_Carlo)
        adimissible_KTE_boundary = np.array([Adimissible_KTE_boundary] *Number_of_Monte_Carlo)
        
            ### Plot KTE for each iteration
        plt.plot([i+1 for i in range (Number_of_Monte_Carlo)], KTE_list, 'o')
        
        plt.plot([i+1 for i in range (Number_of_Monte_Carlo)],mean_total,'k-',\
                  [i+1 for i in range (Number_of_Monte_Carlo)],max_total,'k--',[i+1 for i in range (Number_of_Monte_Carlo)],adimissible_KTE_boundary,'k-.' )
        
        plt.legend(["Simulated","Mean value", "Max value", "Admissible KTE value"],fontsize=14,loc="upper right")
        plt.xlabel("Index of Monte Carlo iteration",fontsize=15)
        plt.ylabel("KTE",fontsize=15)
        plt.grid()
        plt.show()
        
        return Figure
    """ the following lines activate the plotting which are not recommneded during the design of experiment process
    plt.figure()
    Plot_function (Number_of_Monte_Carlo)
    """      
    design_space, KTE_list = Monte_carlo ( Number_of_Monte_Carlo)    
    Adimissible_KTE = list ()
    Adimissible_KTE = [i for i in KTE_list if i <Adimissible_KTE_boundary]
    
    return  np.size (Adimissible_KTE) /Number_of_Monte_Carlo, gear_conformity_rate ()[0], gear_conformity_rate ()[1]

