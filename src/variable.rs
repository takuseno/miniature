use std::rc::Rc;
use std::cell::RefCell;

use crate::function::CgFunction;


#[derive(Debug)]
pub struct Variable {
    pub parent: Option<Rc<RefCell<CgFunction>>>,
    pub shape: Vec<u32>,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Variable {
    pub fn new(shape: Vec<u32>) -> Self {
        let mut size = 1;
        for i in 0..shape.len() {
            size *= shape[i];
        }

        let mut data = Vec::new();
        let mut grad = Vec::new();

        data.resize(size as usize, 0.0);
        grad.resize(size as usize, 0.0);

        Self { parent: None, shape: shape, data: data, grad: grad }
    }

    pub fn size(&self) -> u32 {
        let mut size = 1;
        for i in 0..self.shape.len() {
            size *= self.shape[i];
        }
        size
    }

    pub fn set_data(&mut self, data: &[f32]) {
        assert_eq!(self.size() as usize, data.len());
        for i in 0..data.len() {
            self.data[i] = data[i];
        }
    }

    pub fn set_grad(&mut self, grad: &[f32]) {
        assert_eq!(self.size() as usize, grad.len());
        for i in 0..grad.len() {
            self.grad[i] = grad[i];
        }
    }

    pub fn zeros(&mut self) {
        self.data.fill(0.0);
    }

    pub fn zero_grads(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn set_parent(&mut self, parent: Rc<RefCell<CgFunction>>) {
        self.parent = Some(parent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_variable() {
        let mut variable = Variable::new(vec![1, 2, 3]);
        assert_eq!(variable.size(), 6);

        variable.data[0] = 1.0;
        variable.zeros();
        assert_eq!(variable.data[0], 0.0);

        variable.grad[0] = 1.0;
        variable.zero_grads();
        assert_eq!(variable.grad[0], 0.0);
    }
}
