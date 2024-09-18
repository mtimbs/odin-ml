package micrograd

main :: proc() {
	a := value(4.0)
	b := value(5.0)
	c := add(&a, &b)
	print_value(c)
}
