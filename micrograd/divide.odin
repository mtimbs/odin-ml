package micrograd

import utils "../utils"
import "core:math"
import "core:testing"

div :: proc(a: ^Value, b: ^Value) -> ^Value {
	assert(b.val != 0.0, "Divide by zero error")
	value := new(Value)
	value^ = Value{a.val / b.val, 0.0, {a, b}, .divide}
	return value
}


@(private)
backward_divide :: proc(val: ^Value) {

	// dv/da  = 1/b
	// dv/db  = -a/(b^2)
	a := val.children[0].val
	b := val.children[1].val
	assert(b != 0.0, "Cannot divide by zero")

	val.children[0].grad += (1.0 / b) * val.grad
	val.children[1].grad += ((-1.0 * a) / (math.pow(b, 2))) * val.grad
}

@(test)
test_backward_divide :: proc(t: ^testing.T) {
	// TODO: I'm not 100% on the derivatives here. GO back and revisit
}
