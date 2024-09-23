package micrograd

import "core:fmt"
import "core:math/rand"
import "core:testing"

MLP :: struct {
	layers:     []Layer,
	num_layers: i64,
}

// mlp(3, [4,4,1])
mlp :: proc(num_inputs: i64, layer_outs: []i64, allocator := context.allocator) -> MLP {
	layers := make([]Layer, len(layer_outs) + 1, allocator)
	layers[0] = layer(num_inputs, layer_outs[1], allocator)
	for i in 1 ..< len(layer_outs) {
		layers[i] = layer(layer_outs[i - 1], layer_outs[i], allocator)
	}
	num_layers := i64(len(layers))
	return MLP{layers, num_layers}
}

mlp_forward :: proc(mlp: ^MLP, xs: []Value, allocator := context.allocator) -> []Value {
	outs := xs
	for &layer in mlp.layers {
		outs = l_forward(&layer, outs)
	}
	return outs
}

@(test)
test_mlp_initialisation :: proc(t: ^testing.T) {
	test_allocator := context.allocator
	defer free_all(test_allocator)
	nn := mlp(2, {4, 4, 1})

}
