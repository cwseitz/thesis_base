import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Likelihood:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def eval(self,params):
        prod = 1
        for xi,yi in zip(self.x,self.y):
            s = multivariate_normal.pdf(yi,mean=params[1]+params[0]*xi,cov=1)
            prod *= s
        return prod

class Prior:
    def __init__(self,mu,cov):
        self.mu = mu
        self.cov = cov
    def eval(self,m,b):
        f = multivariate_normal(mean=self.mu, cov=self.cov)
        return f.pdf([m,b])

class Proposal:
    def __init__(self,mu,cov):
        self.mu = mu
        self.cov = cov
    def sample(self):
        dr = np.random.multivariate_normal(mean=self.mu, cov=self.cov)
        return dr     

#define linear model
x = np.linspace(0,1,100)
theta_gt = np.array([1,1]) #m0,b0
eps = np.random.normal(0,0.1,size=x.shape)
y = theta_gt[1] + theta_gt[0]*x + eps

#prior and proposal mean and covariance
var_prop = 1
mu_prop = np.array([0,0])
cov_prop = var_prop*np.eye(2)

var_prior = 1
mu_prior = np.array([0,0])
cov_prior = var_prior*np.eye(2)

#parameters
theta = np.array([0,0])
niters = 1000
prop = Proposal(mu_prop,cov_prop)
prior = Prior(mu_prior,cov_prior)
like = Likelihood(x,y)
params = []

for i in range(niters):
    #draw new parameters from proposal
    dtheta = prop.sample()
    theta_new = theta + dtheta
    
    #compute prior and likelihood under new parameters
    like1 = like.eval(theta_new)
    prior1 = prior.eval(theta_new[0],theta_new[1])
    
    #compute prior and likeihood under old parameters
    like2 = like.eval(theta)
    prior2 = prior.eval(theta[0],theta[1])
    
    #compute acceptance ratio
    a1 = (like1*prior1)/(like2*prior2)
    a2 = 1 #proposal is symmetric
    a = a1*a2
    
    #accept or reject the new params
    if a >= 1:
        theta = theta_new
        u = None
    else:
        u = np.random.uniform(0,1)
        if u <= a:
            theta = theta_new 
    
    #store params
    params.append(theta)
    print(f'Iteration: {i}, m={theta[0]}, b={theta[1]}')
    print(like1,prior1,like2,prior2)
    
params = np.array(params)
mvals, mbins = np.histogram(params[:,0],density=True)
bvals, bbins = np.histogram(params[:,1],density=True)
fig,ax = plt.subplots(1,3,figsize=(12,3))
ax[0].scatter(x,y,color='gray')
ax[0].plot(x,theta_gt[0]*x+theta_gt[1],color='purple')
ax[1].plot(mbins[:-1],mvals,color='red')
ax[2].plot(bbins[:-1],bvals,color='cyan')
plt.tight_layout()
plt.show()
    
