package micrograd

import "core:fmt"
import "core:testing"


Value :: struct {
	val:      f64,
	grad:     f64,
	children: [2]^Value,
}

value :: proc(x: f64) -> Value {
	return Value{x, 0.0, {}}
}

print_value :: proc(val: Value) {
	fmt.printfln("Value(val: {}, grad: {}", val.val, val.grad)
}

@(test)
test_value_initialisation :: proc(t: ^testing.T) {
	// Assemble
	input := 5.0
	// Act
	output := value(input)
	// Assert
	testing.expect_value(t, output.val, input)
	testing.expect_value(t, output.grad, 0.0)
	testing.expect_value(t, output.children, [2]^Value{})
}


add :: proc(a: ^Value, b: ^Value) -> Value {
	return Value{a.val + b.val, 0.0, {a, b}}
}

@(test)
test_add :: proc(t: ^testing.T) {
	// Assemble
	a := 1.0
	b := 3.0
	a_val := value(a)
	b_val := value(b)
	// Act
	output := add(&a_val, &b_val)
	// Assert
	testing.expect_value(t, output.val, a + b)
	testing.expect_value(t, output.grad, 0.0)
	testing.expect_value(t, output.children, [2]^Value{&a_val, &b_val})
}


mult :: proc(a: ^Value, b: ^Value) -> Value {
	return Value{a.val * b.val, 0.0, {a, b}}
}

@(test)
test_mult :: proc(t: ^testing.T) {
	// Assemble
	a := 1.0
	b := 3.0
	a_val := value(a)
	b_val := value(b)
	// Act
	output := mult(&a_val, &b_val)
	// Assert
	testing.expect_value(t, output.val, a * b)
	testing.expect_value(t, output.grad, 0.0)
	testing.expect_value(t, output.children, [2]^Value{&a_val, &b_val})
}
