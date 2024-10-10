package mnist

import "core:encoding/endian"
import "core:fmt"
import "core:os"

TRAINING_LABEL_PATH :: "mnist/data/train-labels.idx1-ubyte"
TRAINING_IMAGES_PATH :: "mnist/data/train-images.idx3-ubyte"


read_mnist_labels :: proc(filepath: string, labels: ^[60000]u8) -> (ok: bool) {
	label_data := os.read_entire_file(filepath, context.allocator) or_return
	defer delete(label_data, context.allocator)

	magic_number := endian.get_u32(label_data[:4], endian.Byte_Order.Big) or_return
	assert(magic_number == 2049)

	num_labels := endian.get_u32(label_data[4:8], endian.Byte_Order.Big) or_return
	assert(num_labels == 60000)


	for i in 0 ..< 60000 {
		labels[i] = label_data[8 + i]
	}

	return true
}

read_mnist_images :: proc(filepath: string, images: ^[60000][28 * 28]u8) -> (ok: bool) {
	image_data := os.read_entire_file(filepath, context.allocator) or_return
	defer delete(image_data, context.allocator)

	magic_number := endian.get_u32(image_data[:4], endian.Byte_Order.Big) or_return
	assert(magic_number == 2051)

	num_labels := endian.get_u32(image_data[4:8], endian.Byte_Order.Big) or_return
	assert(num_labels == 60000)

	num_rows := endian.get_u32(image_data[8:12], endian.Byte_Order.Big) or_return
	assert(num_rows == 28)

	num_cols := endian.get_u32(image_data[12:16], endian.Byte_Order.Big) or_return
	assert(num_cols == 28)


	for i in 0 ..< 60000 {
		for j in 0 ..< 28 * 28 {
			val := image_data[(16 + i) * j]
			images[i][j] = val
		}
	}

	return true

}
