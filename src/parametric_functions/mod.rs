mod linear;

pub fn linear(in_size: usize, out_size: usize) -> Box<linear::Linear> {
    Box::new(linear::Linear::new(in_size, out_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backward;
    use crate::variable::Variable;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn linear_forward_backward() {
        let fc1 = linear(100, 200);
        let fc2 = linear(200, 10);

        let x = Rc::new(RefCell::new(Variable::rand(vec![32, 100])));
        let h = fc1.call(x);
        let output = fc2.call(h);

        assert_eq!(output.borrow().shape[0], 32);
        assert_eq!(output.borrow().shape[1], 10);

        backward(output);
    }
}
