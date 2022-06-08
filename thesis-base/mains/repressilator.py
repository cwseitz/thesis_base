from arwn import dynamics
import matplotlib.pyplot as plt
import numpy as np
import numba 

@numba.njit
def repressilator_propensity(propensities, population, t, 
                             kmu,
                             kmo,
                             kp,
                             gamma_m,
                             gamma_p,
                             kr,
                             ku1,
                             ku2):
    m1, p1, m2, p2, m3, p3, n1, n2, n3 = population
    
    propensities[0] = kmu if n3 == 0 else kmo
    propensities[1] = kmu if n1 == 0 else kmo
    propensities[2] = kmu if n2 == 0 else kmo
    propensities[3] = kp * m1                
    propensities[4] = kp * m2                
    propensities[5] = kp * m3                
    propensities[6] = gamma_m * m1           
    propensities[7] = gamma_m * m2           
    propensities[8] = gamma_m * m3           
    propensities[9] = gamma_p * p1           
    propensities[10] = gamma_p * p2          
    propensities[11] = gamma_p * p3          
    propensities[12] = gamma_p * n1          
    propensities[13] = gamma_p * n2          
    propensities[14] = gamma_p * n3          
    propensities[15] = kr * p3 * (n3 < 2)    
    propensities[16] = kr * p1 * (n1 < 2)    
    propensities[17] = kr * p2 * (n2 < 2)    
    propensities[18] = ku1*(n3==1) + 2*ku2*(n3==2)
    propensities[19] = ku1*(n1==1) + 2*ku2*(n1==2)
    propensities[20] = ku1*(n2==1) + 2*ku2*(n2==2)
    
#stoichiometric matrix
repressilator_update = np.array([
    # 0   1   2   3   4   5   6   7   8
    [ 1,  0,  0,  0,  0,  0,  0,  0,  0], # 0
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0], # 1
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0], # 2
    [ 0,  1,  0,  0,  0,  0,  0,  0,  0], # 3
    [ 0,  0,  0,  1,  0,  0,  0,  0,  0], # 4
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0], # 5
    [-1,  0,  0,  0,  0,  0,  0,  0,  0], # 6
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0], # 7
    [ 0,  0,  0,  0, -1,  0,  0,  0,  0], # 8
    [ 0, -1,  0,  0,  0,  0,  0,  0,  0], # 9
    [ 0,  0,  0, -1,  0,  0,  0,  0,  0], # 10
    [ 0,  0,  0,  0,  0, -1,  0,  0,  0], # 11
    [ 0,  0,  0,  0,  0,  0, -1,  0,  0], # 12
    [ 0,  0,  0,  0,  0,  0,  0, -1,  0], # 13
    [ 0,  0,  0,  0,  0,  0,  0,  0, -1], # 14
    [ 0,  0,  0,  0,  0, -1,  0,  0,  1], # 15
    [ 0, -1,  0,  0,  0,  0,  1,  0,  0], # 16
    [ 0,  0,  0, -1,  0,  0,  0,  1,  0], # 17
    [ 0,  0,  0,  0,  0,  1,  0,  0, -1], # 18
    [ 0,  1,  0,  0,  0,  0, -1,  0,  0], # 19
    [ 0,  0,  0,  1,  0,  0,  0, -1,  0], # 20
    ], dtype=int)
    
# Parameter values
kmu = 0.5
kmo = 5e-4
kp = 0.167
gamma_m = 0.005776
gamma_p = 0.001155
kr = 1.0
ku1 = 224.0
ku2 = 9.0

repressilator_args = (kmu,
                      kmo,
                      kp,
                      gamma_m,
                      gamma_p,
                      kr,
                      ku1,
                      ku2)
                      
                      
# State with 10 copies of everything, nothing bound to operators
repressilator_pop_0 = np.array([10, 10, 10, 10, 10, 10, 0, 0, 0], dtype=int)

repressilator_time_points = np.linspace(0, 80000, 4001)

# Perform the Gillespie simulation
pop = dynamics.gillespie_ssa(repressilator_propensity, 
                                repressilator_update, 
                                repressilator_pop_0, 
                                repressilator_time_points, 
                                args=repressilator_args,
                                progress_bar=True)


fig, ax = plt.subplots(1,3)
ax[0].plot(pop[0,:,0])
ax[0].plot(pop[0,:,2])
ax[0].plot(pop[0,:,4])
ax[1].plot(pop[0,:,1])
ax[1].plot(pop[0,:,3])
ax[1].plot(pop[0,:,5])
ax[2].plot(pop[0,:,6])
ax[2].plot(pop[0,:,7])
ax[2].plot(pop[0,:,8])
plt.show()
