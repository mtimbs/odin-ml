package micrograd

import "core:fmt"
import "core:math/rand"
import "core:testing"

MLP :: struct {
	layers:     []^Layer,
	num_layers: i64,
}

LayerOutputs :: struct {
	num_outputs:   i64,
	activation_fn: Activation,
}

mlp :: proc(num_inputs: i64, layer_outs: []LayerOutputs) -> MLP {
	layers := make([]^Layer, len(layer_outs))
	layers[0] = layer(num_inputs, layer_outs[0].num_outputs, layer_outs[0].activation_fn)
	for i in 1 ..< len(layer_outs) {
		layers[i] = layer(
			layer_outs[i - 1].num_outputs,
			layer_outs[i].num_outputs,
			layer_outs[i].activation_fn,
		)
	}
	num_layers := i64(len(layer_outs))
	return MLP{layers, num_layers}
}

mlp_forward :: proc(mlp: ^MLP, xs: []^Value) -> []^Value {
	outs := xs
	for &layer, i in mlp.layers {
		outs = l_forward(layer, outs)

	}
	return outs
}

@(test)
test_mlp_initialisation :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	nn := mlp(2, {LayerOutputs{4, .tanh}, LayerOutputs{4, .tanh}, LayerOutputs{1, .tanh}})

}

params :: proc(mlp: ^MLP) -> []^Value {
	num_params := 0
	for l in mlp.layers {
		num_params += len(l_params(l))
	}

	params := make([]^Value, num_params)
	i := 0
	for l in mlp.layers {
		for p in l_params(l) {
			params[i] = p
			i += 1
		}
	}

	return params
}


@(test)
test_mlp_params :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	// TODO
}
