use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct MatMul {}

impl MatMul {
    fn validate(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]) {
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

fn transpose(x: &[f32], y: &mut [f32], shape: &[usize]) {
    for (i, v) in x.iter().enumerate().take(x.len()) {
        let orig_rows = i / shape[1];
        let orig_cols = i % shape[1];
        let transpose_index = shape[0] * orig_cols + orig_rows;
        y[transpose_index] = *v;
    }
}

fn matmul_impl(x: &[f32], x_shape: &[usize], y: &[f32], y_shape: &[usize], output: &mut [f32]) {
    let x_rows = x_shape[0];
    let x_cols = x_shape[1];
    let y_cols = y_shape[1];
    for i in 0..x_rows {
        for j in 0..y_cols {
            let out_index = i * y_cols + j;
            for k in 0..x_cols {
                let x_index = i * (x_cols) + k;
                let y_index = k * (y_cols) + j;
                output[out_index] += x[x_index] * y[y_index];
            }
        }
    }
}

impl FunctionImpl for MatMul {
    fn forward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let y = inputs[1].borrow();
        let mut output = outputs[0].borrow_mut();

        // set zeros in output tensor
        output.zeros();

        // x @ y = output
        matmul_impl(&x.data, &x.shape, &y.data, &y.shape, &mut output.data);
    }

    fn backward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let mut y = inputs[1].borrow_mut();
        let output = outputs[0].borrow();

        // gradients for x
        // g_out @ g_y^T = g_x
        let mut transposed_y = vec![0.0; y.data.len()];
        transpose(&y.data, &mut transposed_y, &y.shape);
        matmul_impl(
            &output.grad,
            &output.shape,
            &transposed_y,
            &[y.shape[1], y.shape[0]],
            &mut x.grad,
        );

        // gradients for y
        // g_x^T @ g_out = g_y
        let mut transposed_x = vec![0.0; x.data.len()];
        transpose(&x.data, &mut transposed_x, &x.shape);
        matmul_impl(
            &transposed_x,
            &[x.shape[1], x.shape[0]],
            &output.grad,
            &output.shape,
            &mut y.grad,
        );
    }

    fn get_name(&self) -> &str {
        "MatMul"
    }
}
