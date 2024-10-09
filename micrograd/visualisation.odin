package micrograd

import rl "vendor:raylib"

WINDOW_W :: 1080
WINDOW_H :: 720
FPS :: 120

VALUE_WIDTH :: 125
VALUE_HEIGHT :: 50
ARROW_LENGTH :: 90

CameraTarget :: struct {
	x: f32,
	y: f32,
}


// This is going to take a *^Value and then basically recurse its children and create a graph similar to what we do in build_topology
visualise :: proc() {
	zoom := f32(1.0)
	x_0 := i32(900)
	y_0 := i32(360)
	x_scale := f32(1.0)
	y_scale := f32(1.0)
	target := CameraTarget {
		x = 0,
		y = 0,
	}

	camera := rl.Camera2D {
		target = {target.x, target.y},
		zoom   = zoom,
	}

	rl.InitWindow(WINDOW_W, WINDOW_H, "Micrograd Visualisation")
	defer rl.CloseWindow()
	rl.SetWindowState(rl.ConfigFlags{.WINDOW_RESIZABLE})
	rl.SetTargetFPS(FPS)
	for !rl.WindowShouldClose() {
		rl.ClearBackground({0, 0, 0, 255})

		camera.zoom = clamp(camera.zoom + rl.GetMouseWheelMove() * 0.05, 1, 3)


		if (rl.IsKeyDown(.LEFT)) {
			target.x -= 20
		}
		if (rl.IsKeyDown(.RIGHT)) {
			target.x += 20
		}
		if (rl.IsKeyDown(.UP)) {
			target.y -= 20
		}
		if (rl.IsKeyDown(.DOWN)) {
			target.y += 20
		}
		if (rl.IsKeyDown(.R)) {
			target.x = 0
			target.y = 0
		}
		rl.BeginDrawing()


		camera.offset = {target.x, target.y}
		rl.BeginMode2D(camera)

		// Shape for values
		rl.DrawRectangleLines(x_0, y_0, VALUE_WIDTH, VALUE_HEIGHT, rl.LIGHTGRAY)
		rl.DrawLine(
			x_0 + VALUE_WIDTH / 2,
			y_0,
			x_0 + VALUE_WIDTH / 2,
			y_0 + VALUE_HEIGHT,
			rl.LIGHTGRAY,
		)
		rl.DrawText(
			rl.TextFormat("%.4f", 0.1024),
			x_0 + 10,
			y_0 + VALUE_HEIGHT / 2 - 8,
			16,
			rl.WHITE,
		)
		rl.DrawText(
			rl.TextFormat("%.4f", 1.0000),
			x_0 + VALUE_WIDTH / 2 + 10,
			y_0 + VALUE_HEIGHT / 2 - 8,
			16,
			rl.BLUE,
		)

		// Shape for operation
		rl.DrawEllipseLines(
			x_0 - ARROW_LENGTH - (VALUE_WIDTH / 4 - VALUE_HEIGHT / 2),
			y_0 + VALUE_HEIGHT / 2,
			VALUE_WIDTH / 4,
			VALUE_HEIGHT / 2,
			rl.LIGHTGRAY,
		)
		rl.DrawText(
			rl.TextFormat("%s", "ReLU"),
			x_0 - ARROW_LENGTH - VALUE_WIDTH / 4 + 6,
			y_0 + VALUE_HEIGHT / 2 - 8,
			16,
			rl.GREEN,
		)

		// Shape for arrow connecting operations and values
		rl.DrawLine(
			x_0 - ARROW_LENGTH + VALUE_HEIGHT / 2,
			y_0 + VALUE_HEIGHT / 2,
			x_0,
			y_0 + VALUE_HEIGHT / 2,
			rl.LIGHTGRAY,
		)
		rl.DrawTriangle(
			{f32(x_0 - 8), f32(y_0 + VALUE_HEIGHT / 2 - 7)},
			{f32(x_0 - 8), f32(y_0 + VALUE_HEIGHT / 2 + 7)},
			{f32(x_0), f32(y_0 + VALUE_HEIGHT / 2)},
			rl.LIGHTGRAY,
		)

		rl.EndMode2D()
		rl.EndDrawing()
	}

}
