use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::variable::Variable;

pub mod add;
pub mod argmax;
pub mod broadcast;
pub mod div;
pub mod log;
pub mod matmul;
pub mod mean;
pub mod mul;
pub mod neg;
pub mod onehot;
pub mod relu;
pub mod softmax;
pub mod square;
pub mod sub;

use add::Add;
use argmax::Argmax;
use broadcast::Broadcast;
use div::Div;
use log::Log;
use matmul::MatMul;
use mean::Mean;
use mul::Mul;
use neg::Neg;
use onehot::Onehot;
use relu::ReLu;
use softmax::Softmax;
use square::Square;
use sub::Sub;

pub fn add(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Add {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x, y],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn argmax(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let shape = vec![x.borrow().shape[0]];
    let output = Rc::new(RefCell::new(Variable::new(shape)));
    let function = Box::new(Argmax {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output.borrow_mut().set_need_grad(false);
    output
}

pub fn broadcast(x: Rc<RefCell<Variable>>, shape: Vec<u32>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(shape.clone())));
    let function = Box::new(Broadcast { shape: shape });
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn div(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Div {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x, y],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn log(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Log {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
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

pub fn mean(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(vec![1])));
    let function = Box::new(Mean {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn mul(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Mul {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x, y],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn neg(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Neg {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn onehot(x: Rc<RefCell<Variable>>, num_classes: u32) -> Rc<RefCell<Variable>> {
    let shape = vec![x.borrow().shape[0], num_classes];
    let output = Rc::new(RefCell::new(Variable::new(shape)));
    let function = Box::new(Onehot {
        num_classes: num_classes,
    });
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output.borrow_mut().set_need_grad(false);
    output
}

pub fn relu(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(ReLu {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn square(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Square {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn softmax(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Softmax {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn sub(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Sub {});
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
    fn add_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let h = add(x, y);
        let z = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = add(h, z);
        backward(output);
    }

    #[test]
    fn argmax_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![10, 5])));
        let output = argmax(x);
        assert_eq!(output.borrow().shape.len(), 1);
        assert_eq!(output.borrow().shape[0], 10);
    }

    #[test]
    fn broadcast_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = broadcast(x, vec![3, 2, 3]);
        backward(output);
    }

    #[test]
    fn div_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        x.borrow_mut().ones();
        y.borrow_mut().ones();
        let h = div(x, y);
        let z = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        z.borrow_mut().ones();
        let output = div(h, z);
        backward(output);
    }

    #[test]
    fn log_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        x.borrow_mut().ones();
        let output = log(x);
        backward(output);
    }

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

    #[test]
    fn mean_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = mean(x);
        backward(output);
    }

    #[test]
    fn mul_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let h = mul(x, y);
        let z = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = mul(h, z);
        backward(output);
    }

    #[test]
    fn neg_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = neg(x);
        backward(output);
    }

    #[test]
    fn onehot_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![10])));
        let output = onehot(x, 20);
        assert_eq!(output.borrow().shape[0], 10);
        assert_eq!(output.borrow().shape[1], 20);
    }

    #[test]
    fn relu_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = relu(x);
        backward(output);
    }

    #[test]
    fn softmax_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 10])));
        let output = softmax(x);
        backward(output);
    }

    #[test]
    fn sub_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let h = sub(x, y);
        let z = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = sub(h, z);
        backward(output);
    }
}
