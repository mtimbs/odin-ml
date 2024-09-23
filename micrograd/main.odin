package micrograd
import "core:fmt"
import "core:math"
import "core:mem"

MAX_MEMORY_USAGE_BYTES :: 1 * 1024 * 1024 * 1024
main :: proc() {
	arena: mem.Arena
	arena_buffer := make([]byte, MAX_MEMORY_USAGE_BYTES)
	mem.arena_init(&arena, arena_buffer[:])
	context.allocator = mem.arena_allocator(&arena)

	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}


	// xs := []Value{value(2.0), value(3.0), value(-1.0)}
	// n := mlp(3, {4, 4, 1}, context.allocator)
	// mlp_forward(&n, xs)

	xs := [][]Value {
		{value(2.0), value(3.0), value(-1.0)},
		{value(3.0), value(-1.0), value(0.5)},
		{value(0.5), value(1.0), value(1.0)},
		{value(1.0), value(1.0), value(-1.0)},
	}
	ys := []Value{value(1.0), value(-1.0), value(-1.0), value(1.0)}

	y_pred: [4]Value
	for &x, i in xs {
		n := neuron(3, context.allocator)
		y_pred[i] = n_forward(&n, x)

	}

	loss: Value
	diff: [4]Value
	two := value(2.0)
	fmt.println("y_pred:", y_pred)
	// TODO: This is all broken. Refer to readme for todo fixes
	for &y, i in y_pred {
		diff[i] = sub(&y, &ys[i])
		loss = pow(&diff[i], &two)
	}
	fmt.println("loss:", loss)

	free_all(context.allocator)
}
