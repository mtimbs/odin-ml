package micrograd

import utils "../utils"
import "core:fmt"
import "core:testing"

@(private)
MAXIMUM_NODE_COUNT :: 1000000

Op :: enum {
	add,
	divide,
	multiply,
	power,
	relu,
	subtract,
	tanh,
	value,
}

Activation :: enum {
	relu,
	tanh,
}

Value :: struct {
	val:      f64,
	grad:     f64,
	children: [2]^Value,
	op:       Op,
}

value :: proc(x: f64) -> ^Value {
	value := new(Value)
	value^ = Value{x, 0.0, {}, .value}
	return value
}

print_value :: proc(val: ^Value) {
	fmt.printfln("Value(val: {}, grad: {}", val.val, val.grad)
}

@(test)
test_value_initialisation :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	input := 5.0
	// Act
	output := value(input)
	// Assert
	testing.expect_value(t, output.val, input)
	testing.expect_value(t, output.grad, 0.0)
	testing.expect_value(t, output.children, [2]^Value{})
	testing.expect_value(t, output.op, Op.value)
}
