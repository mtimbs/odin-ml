package micrograd

import utils "../utils"
import "core:fmt"
import "core:math"
import "core:testing"

@(private)
tanh :: proc(val: ^Value) -> ^Value {
	x := val.val
	t := (math.exp(2.0 * x) - 1.0) / (math.exp(2 * x) + 1)
	// Second child for tanh is nil as it only applies to a single value
	value := new(Value)
	value^ = Value{t, 0.00, {val, nil}, .tanh}
	return value
}


@(private)
backward_tanh :: proc(val: ^Value) {
	val.children[0].grad += (1 - math.pow(val.val, 2)) * val.grad
}


@(test)
test_backward_tanh :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(0.88137358701954316)
	b := tanh(a)
	b.grad = 1.0

	// Act
	backward_tanh(b)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 0.5))
}
