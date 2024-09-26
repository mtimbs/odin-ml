package micrograd

import utils "../utils"
import "core:fmt"
import "core:math"
import "core:testing"

@(private)
sub :: proc(a: ^Value, b: ^Value) -> ^Value {
	value := new(Value)
	value^ = Value{a.val - b.val, 0.0, {a, b}, .subtract}
	return value
}

@(private)
backward_subtract :: proc(val: ^Value) {
	// dv/da = 1
	// dv/db = -1
	val.children[0].grad += val.grad
	val.children[1].grad -= val.grad
}


@(test)
test_backward_subtract :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(10.0)
	b := value(4.0)
	s := sub(a, b)
	s.grad = 0.5

	// Act
	backward_subtract(s)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 0.5))
}
