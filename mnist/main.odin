package mnist

import mg "../micrograd"
import "core:fmt"
import "core:mem"

MAX_MEMORY_USAGE_BYTES :: 1 * 1024 * 1024 * 1024
NUM_INPUTS :: 28 * 28 // 28x28 pixels
NUM_OUTPUTS :: 10 // 0-9

TRAINING_DATA_SIZE :: 60000
BATCH_SIZE :: 256
NUM_EPOCHS :: 5
STEP_SIZE :: 0.001

train :: proc(inputs: ^[][28 * 28]u8, labels: ^[]u8) {
	nn := mg.mlp(
		NUM_INPUTS,
		{
			mg.LayerOutputs{200, .tanh},
			mg.LayerOutputs{100, .tanh},
			mg.LayerOutputs{NUM_OUTPUTS, .relu},
		},
	)
	nn_params := mg.params(&nn)

	context.allocator = context.temp_allocator
	xs := make([][NUM_INPUTS]^mg.Value, min(len(labels), BATCH_SIZE), context.temp_allocator)
	ys := make([]^mg.Value, min(len(labels), BATCH_SIZE), context.temp_allocator)
	y_pred := make([]^mg.Value, min(len(labels), BATCH_SIZE), context.temp_allocator)

	for i in 0 ..< NUM_EPOCHS {
		batch_start := 0

		// Convert raw data to Value's
		for batch_start < len(labels) {
			clamped_batch_size := min(len(labels) - batch_start, BATCH_SIZE)
			defer batch_start += clamped_batch_size
			fmt.println("batch start:", batch_start, "batch_size:", clamped_batch_size)

			xs_raw := inputs[batch_start:batch_start + clamped_batch_size]
			ys_raw := labels[batch_start:batch_start + clamped_batch_size]

			fmt.println("Converting raw to Values")
			for i in 0 ..< clamped_batch_size {
				ys[i] = mg.value(f64(ys_raw[i]))
				for j in 0 ..< NUM_INPUTS {
					xs[i][j] = mg.value(f64(xs_raw[i][j]))
				}
			}

			fmt.println("Forward Pass")
			// Forward pass
			for &x, i in xs {
				y_pred[i] = mg.mlp_forward(&nn, x[:])[0]
			}

			// Calculate Loss
			fmt.println("Calc loss")
			loss := mg.value(0.0)
			defer free(loss)
			for y, i in y_pred {
				loss = mg.add(loss, mg.pow(mg.sub(y, ys[i]), mg.value(2.0)))
			}

			fmt.println("Epoch:", i, "Loss:", loss.val)

			// Back prop
			fmt.println("zero grad")
			for p in nn_params {
				p.grad = 0.0
			}
			fmt.println("Back prop")
			mg.backward(loss)

			fmt.println("Gradient descent")
			// Gradient Descent
			for p in nn_params {
				p.val += STEP_SIZE * p.grad * -1.0
			}

			fmt.println("Next Batch")
		}
		fmt.println("Next Epoch")

	}


	for i in 0 ..< len(labels) {
		fmt.println("y(", i, "): ", ys[i].val, "y_pred(", i, "): ", y_pred[i].val)
	}
	free_all(context.temp_allocator)
}

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

	labels := new([TRAINING_DATA_SIZE]u8)
	images := new([TRAINING_DATA_SIZE][28 * 28]u8)

	label_ok := read_mnist_labels(TRAINING_LABEL_PATH, labels)
	if (!label_ok) {
		fmt.println("Error loading label data")
		return
	}
	images_ok := read_mnist_images(TRAINING_IMAGES_PATH, images)
	if (!images_ok) {
		fmt.println("Error loading image data")
		return
	}

	sliced_images := images[:10]
	sliced_labels := labels[:10]
	train(&sliced_images, &sliced_labels)


	free_all(context.allocator)
}
