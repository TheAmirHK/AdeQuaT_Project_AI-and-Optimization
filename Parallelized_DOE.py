# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:41:42 2022

@author: amirh
"""
import os
import numpy as np
import Gear_simulation as Gd
from skopt.sampler import Lhs
from skopt.sampler import Hammersly
from skopt.space import Space
import time 
from func_timeout import func_timeout, FunctionTimedOut
import multiprocess
import matplotlib.pyplot as plt  
import skopt



try:
    os.mkdir("/home/BeeGFS/Laboratories/LCFC/akhezri/Run")    
except OSError as error:
#    print(error)
    pass


""" Within this function, we can generate the design space entering the population size and the sampling method.
    The sampling method includes Random (random), Ratio optimized hypercube sampling (LHS), and Hammersly sampling (hammersly).
    The output is a .npy format file where the first 7 coloumns return random tolerances, the second 7 coulumns return k values,
    the 14th coloumn indicates KTE value, and the last ones are the responses (assembly rate, spur conformity, pinion conformity)  """
    
def design_of_experience (population_size , sampling_model):
    
    ## Initialization
    pop_s = population_size
    
    Tolerances_var_bound = [(1 , 30.001)]*7 ## The admissible tolerance interval
    K_values_var_bound = [(0.7 , 3.001)]*7   ## The practical deviaiton coefficient
    KTE_var_bound = [(3.5 , 30.001)]*1 ## KTE admissible interval
    
    search_space = Tolerances_var_bound + K_values_var_bound + KTE_var_bound
    search_space = Space(search_space)

    ## Sampling methods
    if sampling_model == 'random':
        random_data = search_space.rvs(pop_s)
           
    if sampling_model == 'LHS': ## A Latin Hyperbulic sampling method which has a better distribution method, but consuming more time to generate random number
#        lhs = Lhs(criterion="ratio", iterations=1000) ## Number of iterations has a significant impact on the outputs
        lhs = skopt.sampler.Lhs(lhs_type="centered", criterion="maximin", iterations=1000)
        random_data = lhs.generate(search_space.dimensions, pop_s )
        
    if sampling_model == 'hammersly': ## A hammersly sampling method, fast and efficient
        hammersly = Hammersly()
        random_data = hammersly.generate(search_space.dimensions, pop_s)
    
    if sampling_model == 'halton':
        hal = skopt.sampler.Halton()
        random_data = hal.generate(search_space.dimensions, pop_s)
                            
    random_data = np.asarray(random_data)
 
    return random_data

        
start_time = time.time()
## Generate set of experiments 
number_of_iterations = int(1000000)

  
data = design_of_experience (number_of_iterations,'hammersly')

def calculate(func, args):
    result = func(*args)
    return result

def calculatestar(args):
    return calculate(*args)

result_list = []
def log_result(result):
    # This is called whenever Gd.Optimization_interface returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def test():
    PROCESSES = multiprocess.cpu_count()
    print('Creating pool with %d processes\n' % PROCESSES)
    
    with multiprocess.Pool(PROCESSES) as pool:
        iter_index = int(number_of_iterations/48)

        TASKS = [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index),int(iter_index*2))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index*2),int(iter_index*3))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index*3),int(iter_index*4))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index*4),int(iter_index*5))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index*5),int(iter_index*6))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index*6),int(iter_index*7))] + \
                [(Gd.Optimization_interface, (data[i,0:7], data[i,7:14], data[i, 14],'truncated' )) for i in range(int(iter_index*7),number_of_iterations)]
    
                
        results = [pool.apply_async(calculate, t, callback = log_result) for t in TASKS]

        print(result_list)
        i=0
        print('Ordered results using pool.apply_async():')
        for r in results:
            i = i+1
            print('\t', "Iteration index %s" % i, "/%s" %number_of_iterations, r.get())

        print()
        
        pool.close()
        pool.join()

if __name__ == '__main__':
    multiprocess.freeze_support()
    test()
    
    print("--- DOE RunTime = %s seconds ---" % int(round((time.time() - start_time))))
    result_list = np.asarray(result_list)
    Train_data = np.concatenate((data , result_list), axis=1)
    np.save('DOE', Train_data)

print("--- DOE RunTime = %s seconds ---" % int(round((time.time() - start_time))))



