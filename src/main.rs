mod function;
mod functions;
mod graph;
mod optimizer;
mod parametric_functions;
mod variable;

use std::rc::Rc;
use std::cell::RefCell;

use optimizer::OptimizerImpl;

fn main() {
    let fc1 = parametric_functions::linear::Linear::new(16, 32);
    let fc2 = parametric_functions::linear::Linear::new(32, 2);

    let mut optim = optimizer::SGD::new(0.001);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());

    for i in 0..1000 {
        let x = Rc::new(RefCell::new(variable::Variable::rand(vec![32, 16])));
        let t = Rc::new(RefCell::new(variable::Variable::new(vec![32, 2])));
        t.borrow_mut().zeros();

        // forward
        let h = functions::relu(fc1.call(x));
        let output = fc2.call(h);

        // loss
        let loss = functions::mean(functions::square(functions::sub(output, t)));
        println!("{}", loss.borrow().data[0]);

        optim.zero_grad();
        graph::backward(loss);
        optim.update();
    }
}
