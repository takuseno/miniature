# miniature
A miniature is a toy deep learning library written in Rust.

The miniature is:
- implemented for author's Rust practice.
- designed as simple as possible.

The miniature is NOT:
- supporting CUDA.
- optimized for computational costs.
- a product-ready library.


## example
```rs
use std::rc::Rc;
use std::cell::RefCell;

use miniature::functions as F;
use miniature::parametric_functions as PF;
use miniature::optimizer::SGD;
use miniature::optimizer::OptimizerImpl;
use miniature::variable::Variable;
use miniature::graph::backward;

fn main() {
    let fc1 = PF::linear::Linear::new(16, 32);
    let fc2 = PF::linear::Linear::new(32, 2);

    let mut optim = SGD::new(0.001);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());

    for i in 0..1000 {
        let x = Rc::new(RefCell::new(Variable::rand(vec![32, 16])));
        let t = Rc::new(RefCell::new(Variable::new(vec![32, 2])));
        t.borrow_mut().zeros();

        // forward
        let h = fc1.call(x);
        let output = fc2.call(h);

        // loss
        let loss = F::mean(F::square(F::sub(output, t)));
        println!("{}", loss.borrow().data[0]);

        optim.zero_grad();
        backward(loss);
        optim.update();
    }
}
```
