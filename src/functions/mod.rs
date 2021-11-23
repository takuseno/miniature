use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::variable::Variable;

mod add;
mod argmax;
mod broadcast;
mod div;
mod log;
mod log_softmax;
mod matmul;
mod mean;
mod mul;
mod neg;
mod onehot;
mod relu;
mod softmax;
mod square;
mod sub;

use add::Add;
use argmax::Argmax;
use broadcast::Broadcast;
use div::Div;
use log::Log;
use log_softmax::LogSoftmax;
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
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x, y],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn argmax(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let shape = vec![x.borrow().shape[0]];
    let output = Rc::new(RefCell::new(Variable::new(shape)));
    let function = Box::new(Argmax {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output.borrow_mut().set_need_grad(false);
    output
}

pub fn broadcast(x: Rc<RefCell<Variable>>, shape: Vec<usize>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(shape.clone())));
    let function = Box::new(Broadcast { shape });
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn div(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Div {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x, y],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn log(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Log {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn log_softmax(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(LogSoftmax {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
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
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x, y],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn mean(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(vec![1])));
    let function = Box::new(Mean {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn mul(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Mul {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x, y],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn neg(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Neg {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn onehot(x: Rc<RefCell<Variable>>, num_classes: u32) -> Rc<RefCell<Variable>> {
    let shape = vec![x.borrow().shape[0], num_classes as usize];
    let output = Rc::new(RefCell::new(Variable::new(shape)));
    let function = Box::new(Onehot { num_classes });
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output.borrow_mut().set_need_grad(false);
    output
}

pub fn relu(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(ReLu {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn square(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Square {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn softmax(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Softmax {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn sub(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Sub {});
    let cg_function = Rc::new(RefCell::new(CgFunction::new(
        vec![x, y],
        vec![output.clone()],
        function,
    )));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

pub fn cross_entropy_loss(
    x: Rc<RefCell<Variable>>,
    t: Rc<RefCell<Variable>>,
) -> Rc<RefCell<Variable>> {
    assert_eq!(x.borrow().shape, t.borrow().shape);
    mean(neg(mul(t, log_softmax(x))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backward;
    use rand::Rng;

    fn assert_eq_close(x: f32, y: f32, atol: f32) {
        if (x - y).abs() > atol {
            panic!("abs({} - {}) = {}", x, y, (x - y).abs());
        }
    }

    #[test]
    fn add_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let output = add(x.clone(), y.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let y_data = &y.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            assert_eq!(x_data[i] + y_data[i], output_data[i]);
        }
    }

    #[test]
    fn argmax_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![10, 5])));
        let output = argmax(x.clone());
        assert_eq!(output.borrow().shape.len(), 1);
        assert_eq!(output.borrow().shape[0], 10);

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().shape[0] {
            let offset = i * x.borrow().shape[1];
            let mut max = x_data[offset];
            let mut max_index = 0;
            for j in 1..x.borrow().shape[1] {
                if x_data[j + offset] > max {
                    max = x_data[j + offset];
                    max_index = j;
                }
            }
            assert_eq!(output_data[i] as usize, max_index);
        }
    }

    #[test]
    fn broadcast_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let output = broadcast(x.clone(), vec![3, 2, 3]);
        backward(output.clone());

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..output.borrow().shape[0] {
            let offset = i * x.borrow().size();
            for j in 0..x.borrow().size() {
                assert_eq!(output_data[j + offset], x_data[j]);
            }
        }
    }

    #[test]
    fn div_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let output = div(x.clone(), y.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let y_data = &y.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            assert_eq!(output_data[i], x_data[i] / y_data[i]);
        }
    }

    #[test]
    fn log_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let size = x.borrow().size();
        for i in 0..size {
            x.borrow_mut().data[i] += 1.0; // to prevent log(0)
        }
        let output = log(x.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            assert_eq!(output_data[i], x_data[i].ln());
        }
    }

    #[test]
    fn log_softmax_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![32, 10])));
        let output = log_softmax(x.clone());
        backward(output.clone());

        let test_output = log(softmax(x.clone()));
        let output_data = &output.borrow().data;
        let test_output_data = &test_output.borrow().data;
        for i in 0..output.borrow().size() {
            assert_eq_close(output_data[i], test_output_data[i], 0.001);
        }
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
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let output = mean(x.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        let mut sum = 0.0;
        for i in 0..x.borrow().size() {
            sum += x_data[i];
        }
        assert_eq!(output_data[0], sum / x.borrow().size() as f32);
    }

    #[test]
    fn mul_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let output = mul(x.clone(), y.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let y_data = &y.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            assert_eq!(output_data[i], x_data[i] * y_data[i]);
        }
    }

    #[test]
    fn neg_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let output = neg(x.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            assert_eq!(output_data[i], -x_data[i]);
        }
    }

    #[test]
    fn onehot_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![10])));
        let mut rng = rand::thread_rng();
        let size = x.borrow().size();
        for i in 0..size {
            x.borrow_mut().data[i] = rng.gen_range(0, 20) as f32;
        }
        let output = onehot(x.clone(), 20);
        assert_eq!(output.borrow().shape[0], 10);
        assert_eq!(output.borrow().shape[1], 20);

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            let offset = i * 20;
            for j in 0..20 as usize {
                if x_data[i] == j as f32 {
                    assert_eq!(output_data[j + offset], 1.0);
                } else {
                    assert_eq!(output_data[j + offset], 0.0);
                }
            }
        }
    }

    #[test]
    fn relu_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = relu(x.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            if x_data[i] > 0.0 {
                assert_eq!(output_data[i], x_data[i]);
            } else {
                assert_eq!(output_data[i], 0.0);
            }
        }
    }

    #[test]
    fn softmax_variables() {
        let x = Rc::new(RefCell::new(Variable::rand(vec![2, 10])));
        let output = softmax(x);
        for i in 0..output.borrow().shape[0] {
            let offset = i * output.borrow().shape[1];
            let mut sum = 0.0;
            for j in 0..output.borrow().shape[1] {
                sum += output.borrow().data[j + offset];
            }
            assert_eq_close(sum, 1.0, 0.01);
        }
        backward(output);
    }

    #[test]
    fn sub_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = sub(x.clone(), y.clone());
        backward(output.clone());

        let x_data = &x.borrow().data;
        let y_data = &y.borrow().data;
        let output_data = &output.borrow().data;
        for i in 0..x.borrow().size() {
            assert_eq!(output_data[i], x_data[i] - y_data[i]);
        }
    }

    #[test]
    fn cross_entropy_loss_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![32, 10])));
        let y = Rc::new(RefCell::new(Variable::new(vec![32, 10])));
        let output = cross_entropy_loss(x.clone(), y.clone());
        backward(output.clone());

        assert_eq!(output.borrow().shape.len(), 1);
        assert_eq!(output.borrow().shape[0], 1);
    }
}
