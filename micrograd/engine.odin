package micrograd

import utils "../utils"
import "core:fmt"
import "core:math"
import "core:testing"

MAXIMUM_NODE_COUNT :: 100

Op :: enum {
	None,
	Add,
	Multiply,
	Tanh,
}
Value :: struct {
	val:      f64,
	grad:     f64,
	children: [2]^Value,
	op:       Op,
}

value :: proc(x: f64) -> Value {
	return Value{x, 0.0, {}, .None}
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
	testing.expect_value(t, output.op, Op.None)
}


add :: proc(a: ^Value, b: ^Value) -> Value {
	return Value{a.val + b.val, 0.0, {a, b}, .Add}
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
	testing.expect_value(t, output.op, Op.Add)
}


mult :: proc(a: ^Value, b: ^Value) -> Value {
	return Value{a.val * b.val, 0.0, {a, b}, .Multiply}
}

tanh :: proc(val: ^Value) -> Value {
	x := val.val
	t := (math.exp(2.0 * x) - 1.0) / (math.exp(2 * x) + 1)
	// Second child for tanh is nil as it only applies to a single value
	return Value{t, 0.00, {val, nil}, .Tanh}
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

@(private)
build_topology :: proc(
	val: ^Value,
	topo: ^[]^Value,
	topo_size: ^i32,
	visited: ^[]^Value,
	visited_size: ^i32,
) {
	// Check if we have already visited this node
	for i := i32(0); i < visited_size^; i += 1 {
		if (visited[i] == val) {
			return
		}
	}

	// Mark it as visited and increment our visited size counter
	visited[visited_size^] = val
	visited_size^ += 1

	for child in val.children {
		// recurse for each child
		if (child != nil) {
			build_topology(child, topo, topo_size, visited, visited_size)
		}
	}
	// Add to topology and icnrement size counter
	topo[topo_size^] = val
	topo_size^ += 1
}

@(test)
test_build_topology :: proc(t: ^testing.T) {
	// Assemble
	a := value(1.0)
	b := value(2.0)
	c := add(&a, &b)

	topo := make([]^Value, 3)
	visited := make([]^Value, 3)
	defer delete(topo)
	defer delete(visited)
	visit_count := i32(0)
	topo_count := i32(0)

	// Act
	build_topology(&c, &topo, &topo_count, &visited, &visit_count)

	// Assert
	testing.expect(t, topo[0]^ == a)
	testing.expect(t, topo[1]^ == b)
	testing.expect(t, topo[2]^ == c)
}


backward :: proc(val: ^Value) {
	topo := make([]^Value, MAXIMUM_NODE_COUNT)
	visited := make([]^Value, MAXIMUM_NODE_COUNT)
	defer delete(topo)
	defer delete(visited)
	visit_count := i32(0)
	topo_count := i32(0)

	build_topology(val, &topo, &topo_count, &visited, &visit_count)

	// need to set the gradient to 1.0 so that we can backprop
	val.grad = 1.0
	#reverse for node in topo[:topo_count] {
		switch node.op {
		case .None:
		case .Add:
			backward_add(node)
		case .Multiply:
			backward_mult(node)
		case .Tanh:
			backward_tanh(node)
		}
	}
}

@(private)
backward_add :: proc(val: ^Value) {
	if (len(val.children) == 2) {
		val.children[0].grad += 1.0 * val.grad
		val.children[1].grad += 1.0 * val.grad
	}
}


@(private)
backward_mult :: proc(val: ^Value) {
	val.children[0].grad += val.children[1].val * val.grad
	val.children[1].grad += val.children[0].val * val.grad
}


@(private)
backward_tanh :: proc(val: ^Value) {
	val.children[0].grad += (1 - math.pow_f64(val.val, 2)) * val.grad
}

@(test)
test_backward_on_leaf_no_ops :: proc(t: ^testing.T) {
	// Assemble
	a := value(5.0)

	// Act
	backward(&a)

	// Assert
	// Grad is 1.0 because thats what we default the root node grad to in
	// backward procedure
	testing.expect(t, utils.compare_f64(a.grad, 1.0))
}

@(test)
test_backward_add :: proc(t: ^testing.T) {
	// Assemble
	a := value(6.8814)
	b := value(-6.0)
	s := add(&a, &b)
	s.grad = 0.5

	// Act
	backward_add(&s)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 0.5))
	testing.expect(t, utils.compare_f64(b.grad, 0.5))
}

@(test)
test_backward_same_node :: proc(t: ^testing.T) {
	// Assemble
	a := value(3.0)
	s := add(&a, &a)

	// Act
	backward(&s)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 2.0))
}


@(test)
test_backward_shared_nodes :: proc(t: ^testing.T) {
	// Assemble
	a := value(-2.0)
	b := value(3.0)
	d := mult(&a, &b)
	e := add(&a, &b)
	f := mult(&d, &e)

	// Act
	backward(&f)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, -3.0))
	testing.expect(t, utils.compare_f64(b.grad, -8.0))
	testing.expect(t, utils.compare_f64(d.grad, 1.0))
	testing.expect(t, utils.compare_f64(e.grad, -6.0))
}

@(test)
test_backward_mult :: proc(t: ^testing.T) {
	// Assemble
	a := value(2.0)
	b := value(-3.0)
	m := mult(&a, &b)
	m.grad = 0.5

	// Act
	backward_mult(&m)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, -1.5))
	testing.expect(t, utils.compare_f64(b.grad, 1.0))
}

@(test)
test_backward_tanh :: proc(t: ^testing.T) {
	// Assemble
	a := value(0.88137358701954316)
	b := tanh(&a)
	b.grad = 1.0
	print_value(a)
	print_value(b)

	// Act
	backward_tanh(&b)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 0.5))
}
