package micrograd

import utils "../utils"
import "core:fmt"
import "core:math"
import "core:testing"

@(private)
relu :: proc(val: ^Value) -> ^Value {
	value := new(Value)
	value^ = Value{max(0, val.val), 0.00, {val, nil}, .relu}
	return value
}


@(private)
backward_relu :: proc(val: ^Value) {
	val.children[0].grad += (val.val > 0 ? val.val : 0) * val.grad
}


@(test)
test_backward_relu_zero :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(0)
	b := relu(a)
	b.grad = 1.0

	// Act
	backward_relu(b)

	// Assert
	// The gradient should not change for a value of 0
	testing.expect(t, utils.compare_f64(a.grad, 0.0))
}

@(test)
test_backward_relu_non_zero :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(0.5)
	b := relu(a)
	b.grad = 2.5

	// Act
	backward_relu(b)

	// Assert
	// The gradient should be b.grad * b.val
	testing.expect(t, utils.compare_f64(a.grad, 1.25))
}
