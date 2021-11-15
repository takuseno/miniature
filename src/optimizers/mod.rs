mod sgd;

pub fn sgd(lr: f32) -> Box<sgd::SGD> {
    Box::new(sgd::SGD::new(lr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;
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
}