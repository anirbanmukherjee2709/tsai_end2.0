# tsai_end2.0_Session_1
The School of AI repo for END 2.0 Session 1

# What is a neural network neuron?
It is a mathematical implementation of a biological neuron. This is represented as model where in the input data is combined with weights and biases to pass on the input to the next layer and compute an output for it. These inputs are  converted into outputs using a  mathematical function which is called an activation function.

# What is the use of the learning rate?
Learning rate is a hyperparameter that can be configured to train a neural network. It should be in the range (0,1); excluding the outer-bounds. Count of training epochs will be larger for a very small learning-rate while a larger learning-rate can provide a suboptimal solution in very few epochs as there will be rapid change in change in weights.

# How are weights initialized?
Weight initialization is a procedure to set the weights of a neural network to small random values that define the starting point for the optimization (learning or training) of the neural network model. There are multiple ways to initialise weights such as:
- sampling from a normal distribution with (mean =0, standard deviation = 1). 
- Sampling from a uniform distribution, 
- zeroes, ones
- Or taking a constant number for the weight

Listed above are the most basic ways to initialise the weights, however, there can be multiple more ways to do the same.
The code to initialise weights is below:
```
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

weights_init(model)
```

# What is "loss" in a neural network?
Generally, a loss function tells the optimisation algorithm how 'good' the predictions are by seeing the difference between actual and predicted values. These losses are used as a feedback signal, thus updating the weights and biases to have a lower loss in the next epoch. 


# What is the "chain rule" in gradient flow?
