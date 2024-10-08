package micrograd

import utils "../utils"
import "core:testing"

@(private)
mult :: proc(a: ^Value, b: ^Value) -> ^Value {
	value := new(Value)
	value^ = Value{a.val * b.val, 0.0, {a, b}, .multiply}
	return value
}

@(test)
test_mult :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := 1.0
	b := 3.0
	a_val := value(a)
	b_val := value(b)
	// Act
	output := mult(a_val, b_val)
	// Assertw
	testing.expect_value(t, output.val, a * b)
	testing.expect_value(t, output.grad, 0.0)
	testing.expect_value(t, output.children, [2]^Value{a_val, b_val})
}


@(private)
backward_mult :: proc(val: ^Value) {
	// dv/da = b
	// dv/db = a
	a := val.children[0].val
	b := val.children[1].val

	val.children[0].grad += b * val.grad
	val.children[1].grad += a * val.grad
}

@(test)
test_backward_mult :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(2.0)
	b := value(-3.0)
	m := mult(a, b)
	m.grad = 0.5

	// Act
	backward_mult(m)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, -1.5))
	testing.expect(t, utils.compare_f64(b.grad, 1.0))
}
