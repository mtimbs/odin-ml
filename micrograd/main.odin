package micrograd

import "core:fmt"
import "core:mem"

main :: proc() {
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

	// // Inputes
	x1 := value(2.0)
	x2 := value(0.0)
	// weights
	w1 := value(-3.0)
	w2 := value(1.0)
	// neuron bias
	b := value(6.8813735870195432)

	x1w1 := mult(&x1, &w1)
	x2w2 := mult(&x2, &w2)
	x1w1_x2w2 := add(&x1w1, &x2w2)
	n := add(&x1w1_x2w2, &b)
	fmt.print("n:")
	print_value(n)
	o := tanh(&n)
	o.grad = 1.0
	print_value(o)
	backward(&o)
	fmt.print("n:")
	print_value(n)
	//
}
