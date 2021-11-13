use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct MatMul {}

impl MatMul {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let y = inputs[1].borrow();
        let output = outputs[0].borrow();

        // supports only 2-dim tensors
        assert_eq!(x.shape.len(), 2);
        assert_eq!(y.shape.len(), 2);
        assert_eq!(output.shape.len(), 2);
        assert_eq!(x.shape[1], y.shape[0]);
        assert_eq!(output.shape[0], x.shape[0]);
        assert_eq!(output.shape[1], y.shape[1]);
    }
}

fn matmul_impl(
    x: &Vec<f32>,
    x_shape: &Vec<u32>,
    y: &Vec<f32>,
    y_shape: &Vec<u32>,
    output: &mut Vec<f32>,
    transpose_x: bool,
    transpose_y: bool,
) {
    let x_rows = if transpose_x { x_shape[1] } else { x_shape[0] };
    let x_cols = if transpose_x { x_shape[0] } else { x_shape[1] };
    let y_cols = if transpose_y { y_shape[0] } else { y_shape[1] };
    for i in 0..x_rows as usize {
        for j in 0..y_cols as usize {
            let out_index = i * (y_cols as usize) + j;
            for k in 0..x_cols as usize {
                let x_index = i * (x_cols as usize) + k;
                let y_index = k * (y_cols as usize) + j;
                output[out_index] += x[x_index] * y[y_index];
            }
        }
    }
}

impl FunctionImpl for MatMul {
    fn forward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let y = inputs[1].borrow();
        let mut output = outputs[0].borrow_mut();

        // set zeros in output tensor
        output.zeros();

        // x @ y = output
        matmul_impl(
            &x.data,
            &x.shape,
            &y.data,
            &y.shape,
            &mut output.data,
            false,
            false,
        );
    }

    fn backward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let mut y = inputs[1].borrow_mut();
        let output = outputs[0].borrow();

        // gradients for x
        // g_out @ g_y^T = g_x
        matmul_impl(
            &output.grad,
            &output.shape,
            &y.data,
            &y.shape,
            &mut x.grad,
            false,
            true,
        );

        // gradients for y
        // g_x^T @ g_out = g_y
        matmul_impl(
            &x.data,
            &x.shape,
            &output.grad,
            &output.shape,
            &mut y.grad,
            true,
            false,
        );
    }

    fn get_name(&self) -> &str {
        "MatMul"
    }
}

pub fn matmul(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(vec![
        x.borrow().shape[0],
        y.borrow().shape[1],
    ])));
    let function = Box::new(MatMul {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x, y],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backward;

    #[test]
    fn matmul_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![2, 2])));
        let y = Rc::new(RefCell::new(Variable::new(vec![2, 2])));
        x.borrow_mut().set_data(&[1.0, 0.0, 0.0, 1.0]);
        y.borrow_mut().set_data(&[1.0, 0.0, 0.0, 1.0]);
        let output = matmul(x, y);
        let data = &output.borrow().data;
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 1.0);
    }

    #[test]
    fn matmul_backward() {
        let x = Rc::new(RefCell::new(Variable::new(vec![2, 2])));
        let y = Rc::new(RefCell::new(Variable::new(vec![2, 2])));
        let output = matmul(x, y);
        backward(output);
    }
}
