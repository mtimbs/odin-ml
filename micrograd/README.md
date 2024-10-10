## Micrograd
Odin implementation of [micrograd](https://github.com/karpathy/micrograd/tree/master/micrograd).
Written by following along Andrej Karparthy's "Neural Networks: Zero to Hero" course [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

### Credits
1. [@karpathy](https://github.com/karpathy)

### TODO:
- raylib visualisation
- investigate if I can just put the params as part of the struct definition during initialisation instead of create a procedure to generate them... They are just pointers to `Value` structs anyway.....


If I want this to be usable for something like MNIST then I need to add support for matmuls and softmax and ReLU etc. Representing a non trivial matrix multiplication as individual multiplication and addition operations creates an insanely large computational graph.

E.g. a 728 pixel MNIST image is something like 1M nodes per image. Creating the topology for a backprop just takes an ungodly amount of CPU time. I need to squash the matmul into a single operation that has a derivative so it results in a single node
