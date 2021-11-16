mod adam;
mod sgd;

use crate::optimizer::Optimizer;

pub fn sgd(lr: f32) -> Box<Optimizer> {
    let sgd_impl = Box::new(sgd::Sgd { lr });
    Box::new(Optimizer::new(sgd_impl))
}

pub fn adam(lr: f32, betas: (f32, f32), eps: f32) -> Box<Optimizer> {
    let adam_impl = Box::new(adam::Adam::new(lr, betas, eps));
    Box::new(Optimizer::new(adam_impl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::Variable;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn sgd_update() {
        let mut optim = sgd(0.1);

        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        optim.set_params(vec![x, y]);

        optim.zero_grad();
        optim.update();
    }

    #[test]
    fn adam_update() {
        let mut optim = adam(0.1, (0.9, 0.999), 1e-8);

        let x = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::rand(vec![1, 2, 3])));
        optim.set_params(vec![x, y]);

        optim.zero_grad();
        optim.update();
    }
}
