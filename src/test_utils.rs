pub fn assert_eq_close(x: f32, y: f32, atol: f32) {
    if (x - y).abs() > atol {
        panic!("abs({} - {}) = {}", x, y, (x - y).abs());
    }
}
