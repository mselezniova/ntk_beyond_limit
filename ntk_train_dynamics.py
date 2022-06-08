import numpy as np

import jax
from jax import lax, random, numpy as jnp

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

from typing import Any, Callable, Sequence, Optional
import pickle

from tensorflow import keras

file_prefix = "struct"

activation = nn.relu # activation function

M = 300  # width parameter
L = 20  # depth

alpha = 10e-5 # learning rate
epochs = 3000
kernel_steps = [0,100,500,1000,2000,3000] # epochs at which the NTK is computed

var_w_s = [1.0,2.0,2.2]  # variance parameter \sigma_w^2
var_b = 0.               # variance parameter \sigma_b^2

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

# MSE loss function
def mse(x_batched, y_batched):
    def mse(params):
        # MSE on a single pair (x,y)
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y-pred, y-pred)/2.0
        return jnp.mean(jax.vmap(squared_error)(x_batched,y_batched), axis=0) #Vectorized MSE
    return jax.jit(mse) 



 # Load and preprocess MNIST

n_class = 10
ker_size_per_class = 10 
mnist_n0 = 28*28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 
x_train = x_train.reshape(x_train.shape[0], mnist_n0)
x_test = x_test.reshape(x_test.shape[0], mnist_n0)

# choose subset of data with ker_size_per_class samples from each class
ind = []
for k in range(ker_size_per_class): 
    ind += list(np.random.choice(np.argwhere(y_train==k).ravel(), size=ker_size_per_class, replace=False)) 

x_train, x_test = x_train/255.,x_test/255.
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)

x_ker = x_train[ind] # We compute the NTK only on a subset of samples
y_ker = y_train[ind]

# -------

key = random.PRNGKey(0)
subkeys = jax.random.split(key, num=len(var_w_s))

widths = [mnist_n0]+[M]*L+[n_class]
optimizer_def = optim.Adam(learning_rate=alpha) # Define Adam optimizer

loss = mse(x_train, y_train) # train loss function
loss_grad_fn = jax.value_and_grad(loss) # function to get loss value and gradient 
test_loss = mse(x_test, y_test) # test loss function


for var_w, subkey in zip(var_w_s, subkeys):
    
    model = MLP(widths = widths, v_w=var_w, v_b=var_b, activation = activation) # Define MLP model
    params = model.init(subkey, x_train) # Initialize model 
    optimizer = optimizer_def.create(params) # Create optimizer with initial parameters
    
    K_t = []
    loss_t = []
    test_loss_t = []

    K_func = K_matr(model)

    for i in range(epochs+1):
        loss_val, grad = loss_grad_fn(optimizer.target) # Get gradient and train loss value
        test_loss_val = test_loss(optimizer.target)  # Get test loss value

        test_loss_t.append(test_loss_val)
        loss_t.append(loss_val)

        # Compute the NTK for the chosen epochs
        if i in kernel_steps:
            print('Loss step {}: '.format(i), loss_val, test_loss_val)
            K_t.append(K_func(x_ker,x_ker,optimizer.target))
            
        optimizer = optimizer.apply_gradient(grad) # Update optimizer parameters
    
    # Save the results
    pickle.dump(jnp.array(K_t), open( "ntk_dynamics/"+file_prefix+"_w"+str(int(var_w*10))+"M"+str(M)+"L"+str(L), "wb" ) )
    pickle.dump(jnp.array(loss_t), open( "ntk_dynamics/"+file_prefix+"_loss_w"+str(int(var_w*10))+"M"+str(M)+"L"+str(L), "wb" ) )
    pickle.dump(jnp.array(test_loss_t), open( "ntk_dynamics/"+file_prefix+"_test_loss_w"+str(int(var_w*10))+"M"+str(M)+"L"+str(L), "wb" ) )


