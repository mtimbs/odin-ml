package micrograd

import utils "../utils"
import "core:math"
import "core:testing"

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
	// add to topology and icnrement size counter
	topo[topo_size^] = val
	topo_size^ += 1
}

@(test)
test_build_topology :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(1.0)
	b := value(2.0)
	c := add(a, b)

	topo := make([]^Value, 3)
	visited := make([]^Value, 3)
	defer delete(topo)
	defer delete(visited)
	visit_count := i32(0)
	topo_count := i32(0)

	// Act
	build_topology(c, &topo, &topo_count, &visited, &visit_count)

	// Assert
	testing.expect(t, topo[0] == a)
	testing.expect(t, topo[1] == b)
	testing.expect(t, topo[2] == c)
}


backward :: proc(val: ^Value, allocator := context.allocator) {
	topo := make([]^Value, MAXIMUM_NODE_COUNT, allocator)
	visited := make([]^Value, MAXIMUM_NODE_COUNT, allocator)
	visit_count := i32(0)
	topo_count := i32(0)

	build_topology(val, &topo, &topo_count, &visited, &visit_count)

	// need to set the gradient to 1.0 so that we can backprop
	val.grad = 1.0
	#reverse for node in topo[:topo_count] {
		switch node.op {
		case .add:
			backward_add(node)
		case .divide:
			backward_divide(node)
		case .multiply:
			backward_mult(node)
		case .power:
			backward_power(node)
		case .subtract:
			backward_subtract(node)
		case .tanh:
			backward_tanh(node)
		case .value:

		}
	}
}


@(test)
test_backward_value_no_ops :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(5.0)

	// Act
	backward(a, test_allocator)

	// Assert
	// Grad is 1.0 because thats what we default the root node grad to in
	// backward procedure
	testing.expect(t, utils.compare_f64(a.grad, 1.0))
}


@(test)
test_backward_same_node :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(3.0)
	s := add(a, a)

	// Act
	backward(s, test_allocator)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, 2.0))
}


@(test)
test_backward_shared_nodes :: proc(t: ^testing.T) {
	// Assemble
	test_allocator := context.allocator
	defer free_all(test_allocator)
	a := value(-2.0)
	b := value(3.0)
	d := mult(a, b)
	e := add(a, b)
	f := mult(d, e)

	// Act
	backward(f, test_allocator)

	// Assert
	testing.expect(t, utils.compare_f64(a.grad, -3.0))
	testing.expect(t, utils.compare_f64(b.grad, -8.0))
	testing.expect(t, utils.compare_f64(d.grad, 1.0))
	testing.expect(t, utils.compare_f64(e.grad, -6.0))
}
