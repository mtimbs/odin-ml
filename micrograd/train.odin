package micrograd

import "core:fmt"
import "core:math"
import "core:mem"

STEP_SIZE :: 0.05

train :: proc(inputs: [][]^Value, labels: []^Value, mlp: ^MLP) {
	nn_params := params(mlp)

	y_pred := make([]^Value, len(labels))

	for epoch in 0 ..< 1000 {
		// Forward pass
		for x, i in inputs {
			y_pred[i] = mlp_forward(mlp, x)[0]
		}

		// Calculate Loss
		loss := value(0.0)
		for y, i in y_pred {
			loss = add(loss, pow(sub(y, labels[i]), value(2.0)))
		}
		fmt.println("loss(", epoch, "): ", loss.val)

		// Back prop
		for p in nn_params {
			p.grad = 0.0
		}
		backward(loss)

		// Gradient Descent
		for p in nn_params {
			p.val += STEP_SIZE * p.grad * -1.0
		}

	}

	for i in 0 ..< len(labels) {
		fmt.println("y_pred(", i, "): ", y_pred[i].val)
	}


}
