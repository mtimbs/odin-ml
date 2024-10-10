package micrograd

import "core:fmt"
import "core:math/rand"
import "core:testing"

@(private)
Layer :: struct {
	neurons:       []^Neuron,
	num_neurons:   i64,
	num_params:    i64,
	activation_fn: Activation,
}


@(private)
layer :: proc(num_inputs: i64, num_neurons: i64, activation_fn: Activation) -> ^Layer {
	neurons := make([]^Neuron, num_neurons)
	for i in 0 ..< num_neurons {
		neurons[i] = neuron(num_inputs)
	}
	layer := new(Layer)
	layer^ = Layer{neurons, num_neurons, 0, activation_fn}
	return layer
}

@(test)
test_layer_initialisation :: proc(t: ^testing.T) {
	// TODO
}

@(private)
l_forward :: proc(layer: ^Layer, xs: []^Value) -> []^Value {
	outputs := make([]^Value, layer.num_neurons)

	for &neuron, i in layer.neurons {
		outputs[i] = n_forward(neuron, xs, layer.activation_fn)
	}

	return outputs
}

@(test)
test_l_forward :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	x1 := value(1.0)
	x2 := value(2.0)
	l := layer(2, 3, .tanh)
	outputs := l_forward(l, {x1, x2})

	// This is what we expect the activation to be before running through tanh
	testing.expect_value(t, len(outputs), 3)
}

@(private)
l_params :: proc(layer: ^Layer) -> []^Value {
	num_params := i64(0)
	for n in layer.neurons {
		num_params += (n.num_weights + 1)
	}

	params := make([]^Value, num_params)
	i := 0
	for n in layer.neurons {
		for p in n_params(n) {
			params[i] = p
			i += 1
		}
	}

	return params
}

@(test)
test_l_params :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	// TODO
}
