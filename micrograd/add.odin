package micrograd

import utils "../utils"
import "core:testing"

add :: proc(a: ^Value, b: ^Value) -> ^Value {
	value := new(Value)
	value^ = Value{a.val + b.val, 0.0, {a, b}, .add}
	return value
}

@(test)
test_add :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := 1.0
	b := 3.0
	a_val := value(a)
	b_val := value(b)
	// Act
	output := add(a_val, b_val)
	// Assert
	testing.expect_value(t, output.val, a + b)
	testing.expect_value(t, output.grad, 0.0)
	testing.expect_value(t, output.children, [2]^Value{a_val, b_val})
	testing.expect_value(t, output.op, Op.add)
}

@(private)
backward_add :: proc(val: ^Value) {
	// dv/da = 1
	// dv/db = 1
	val.children[0].grad += val.grad
	val.children[1].grad += val.grad
}


@(test)
test_backward_add :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(6.8814)
	b := value(-6.0)
	s := add(a, b)
	s.grad = 0.5

	// Act
	backward_add(s)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 0.5))
	testing.expect(t, utils.compare_f64(b.grad, 0.5))
}
