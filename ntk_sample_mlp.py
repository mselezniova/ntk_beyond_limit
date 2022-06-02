import numpy as np
import jax
from jax import lax, random, numpy as jnp
from flax import linen as nn

from typing import Any, Callable, Sequence, Optional
from tqdm import tqdm
import pickle


file_prefix = "ntk_init"

activation = nn.relu # activation function

MC_it = 500   # number of samples

M = 100        # width parameter
n0 = 100       # input dimension

L = int(M/2)  # depth


var_w_s = [0.5,0.8,1.0,1.2,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.5,3.0,3.5]  # variance parameter \sigma_w^2
var_b = 0.                                                                   # variance parameter \sigma_b^2

# custom fully-connected network (MLP) class
class MLP(nn.Module):
    widths: Sequence[int] # We need to specify all the layer width (including input and output widths)
    v_w: float # variance parameter \sigma_w^2
    v_b: float # variance parameter \sigma_b^2
        
        
    activation: Callable # activation function (the same in all the hidden layers)
    kernel_init: Callable = jax.nn.initializers.normal # Gaussian initialization
    bias_init: Callable = jax.nn.initializers.normal

    def setup(self):
        self.layers = [nn.Dense(self.widths[l+1],
                               kernel_init = self.kernel_init(jnp.sqrt(self.v_w/self.widths[l])),
                               bias_init = self.bias_init(jnp.sqrt(self.v_b))
                               ) for l in range(len(self.widths)-1)]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers[:-1]):
            x = lyr(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
    

# the NTK on a single pair of samples (x1,x2)
def K(model):
    
    def K(x1,x2,params):
        f1 = jax.jacobian(lambda p: model.apply(p,x1))(params)
        f2 = jax.jacobian(lambda p: model.apply(p,x2))(params)
        leaves, struct = jax.tree_util.tree_flatten(jax.tree_multimap(jnp.multiply,f1,f2))

        return sum([jnp.sum(leaf) for leaf in leaves])
    
    return jax.jit(K)

# the NTK matrix (vectorization of K)
def K_matr(model):
    _K = K(model)
    
    def K_matr(X,Y,params):
        f = lambda x1,x2: _K(x1,x2,params)
        return jax.vmap(jax.vmap(f,(None,0)),(0,None))(X,Y)
    
    return jax.jit(K_matr)



key = random.PRNGKey(0)
init_keys_w = jax.random.split(key, num=len(var_w_s)) # array of keys for every var_w value 


for var_w, init_key in tqdm(zip(var_w_s,init_keys_w)):
    
    K_l = []
    
    for l in tqdm(range(1,L)):

        n = [n0] + [M]*l + [1]
         
        model = MLP(widths = n, v_w=var_w, v_b=var_b, activation = activation)
        K_ = K_matr(model)
        
        mc_keys = jax.random.split(init_key, num=MC_it) # array of keys for each random initialization sample 
        K_MC = []
        
        for i, key in enumerate(mc_keys):

            subkey1, subkey2, subkey3 = jax.random.split(key, num=3)

            params = model.init(subkey1, x)		# random initialization (every time new key)

            x = jax.random.normal(subkey2,[n0]) # Generate random input (every time new key)
            x = x/jnp.linalg.norm(x)			# and normalize
            y = jax.random.normal(subkey3,[n0]) # Generate another independent input (every time new key)
            y = y/jnp.linalg.norm(y)			
            

            X = jnp.array([r*x + jnp.sqrt(1.-r**2)*y for r in jnp.arange(1.0,-0.1,-0.1)]) # array of inputs [x0,x1,...,x10] correlating with x0
            																			  # such that <x0,xi> = (10-i)/10

            K_vals = K_(X[0].reshape(1,-1),X,params).ravel() # Compute the NTK matrix on (x0, [x0,...x10])

            K_MC.append(K_vals)
        
        K_l.append(K_MC)
            
    pickle.dump(jnp.array(K_l), open( "ntk_dispersion/"+file_prefix+"_w"+str(int(var_w*10))+"M"+str(M)+"n"+str(n0), "wb" ) ) # Save results (separate file for every var_w value)
    