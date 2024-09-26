# Odin ML

Collection of AI/Machine Learning code written in Odin.

## Micrograd
Odin implementation of [micrograd](https://github.com/karpathy/micrograd/tree/master/micrograd).
Written by following along Andrej Karparthy's "Neural Networks: Zero to Hero" course [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

### Credits
1. [@karpathy](https://github.com/karpathy)



## WIP TODO's
- return params for Neuron. Layer, MLP etc (slice concat?)
- loss function of values (MSE)
- parameters function for neuron, layer, mlp
- raylib visualisation

- investigate if I can just put the params as part of the struct definition during initialisation instead of create a procedure to generate them... They are just pointers to `Value` structs anyway.....
