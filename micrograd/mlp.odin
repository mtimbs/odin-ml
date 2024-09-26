package micrograd

import "core:fmt"
import "core:math/rand"
import "core:testing"

MLP :: struct {
	layers:     []^Layer,
	num_layers: i64,
}

mlp :: proc(num_inputs: i64, layer_outs: []i64) -> MLP {
	layers := make([]^Layer, len(layer_outs))
	layers[0] = layer(num_inputs, layer_outs[0])
	for i in 1 ..< len(layer_outs) {
		layers[i] = layer(layer_outs[i - 1], layer_outs[i])
	}
	num_layers := i64(len(layer_outs))
	return MLP{layers, num_layers}
}

@(private)
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
	nn := mlp(2, {4, 4, 1})

}

@(private)
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
