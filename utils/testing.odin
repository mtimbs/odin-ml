package utils

compare_f64 :: proc(a: f64, b: f64, tolerance := 0.00001) -> bool {
    return abs(a - b) < tolerance
}
