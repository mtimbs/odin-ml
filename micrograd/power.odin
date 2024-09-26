package micrograd

import utils "../utils"
import "core:math"
import "core:testing"

@(private)
pow :: proc(a: ^Value, b: ^Value) -> ^Value {
	value := new(Value)
	value^ = Value{math.pow(a.val, b.val), 0.0, {a, b}, .power}
	return value
}


@(private)
backward_power :: proc(val: ^Value) {
	// dv/da = b * a^(b-1)
	// dv/db = a^b * log(a)
	//
	// v = a^b
	// dv/db =
	a := val.children[0].val
	b := val.children[1].val

	val.children[0].grad += (b * math.pow(a, b - 1)) * val.grad
	val.children[1].grad += math.pow(a, b) * math.log10(a) * val.grad
}


@(test)
test_backward_pow :: proc(t: ^testing.T) {
	// TODO: I'm not 100% on the derivatives here. GO back and revisit
}
