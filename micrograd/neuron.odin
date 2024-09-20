package micrograd

import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:testing"

Neuron :: struct {
	weights:     []Value,
	num_weights: i64,
	bias:        Value,
}

neuron :: proc(num_weights: i64, allocator := context.allocator) -> Neuron {
	weights := make([]Value, num_weights, allocator)
	for i in 0 ..< num_weights {
		weights[i] = value(rand.float64_uniform(-1.0, 1.0))
	}
	bias := value(rand.float64_uniform(-1.0, 1.0))
	return Neuron{weights, num_weights, bias}
}


@(test)
test_nueron_initialisation :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	neuron_1 := neuron(1, test_allocator)
	neuron_5 := neuron(5, test_allocator)

	testing.expect_value(t, len(neuron_1.weights), 1)
	testing.expect_value(t, neuron_1.num_weights, 1)
	testing.expect(t, neuron_1.weights[0].val > -1.0)
	testing.expect(t, neuron_1.weights[0].val < 1.0)
	testing.expect(t, neuron_1.bias.val > -1.0)
	testing.expect(t, neuron_1.bias.val < 1.0)
	testing.expect_value(t, len(neuron_5.weights), 5)
	testing.expect_value(t, neuron_5.num_weights, 5)
}

n_forward :: proc(neuron: ^Neuron, xs: []f64) -> Value {
	assert(
		len(xs) == len(neuron.weights),
		"values must be of same length as Neuron weights ({},{})",
	)
	// We could initialise this to 0 and add a bias later but can also just
	// initialise this to the bias immediately and save a layer of recursion
	activation := value(neuron.bias.val)

	for i in 0 ..< neuron.num_weights {
		tmp := value(xs[i])
		prod := mult(&neuron.weights[i], &tmp)
		activation = add(&activation, &prod)
	}

	return tanh(&activation)
}

@(test)
test_n_forward :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	n := neuron(2, test_allocator)
	w1 := n.weights[0].val
	w2 := n.weights[1].val
	x1 := f64(1.0)
	x2 := f64(2.0)
	bias := n.bias.val

	sum := n_forward(&n, []f64{x1, x2})

	// This is what we expect the activation to be before running through tanh
	activation := value((x1 * w1 + x2 * w2) + bias)
	expected := tanh(&activation).val
	testing.expect_value(t, expected, sum.val)
}
