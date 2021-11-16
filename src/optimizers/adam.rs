use std::cell::RefCell;
use std::rc::Rc;

use crate::optimizer::OptimizerImpl;
use crate::variable::Variable;

pub struct Adam {
    pub lr: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    means: Vec<Vec<f32>>,
    vars: Vec<Vec<f32>>,
    t: u32,
}

impl Adam {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            means: vec![],
            vars: vec![],
            t: 0,
        }
    }

    fn init_states(&mut self, params: &[Rc<RefCell<Variable>>]) {
        for param in params {
            let mean = vec![0.0; param.borrow().size() as usize];
            let var = vec![0.0; param.borrow().size() as usize];
            self.means.push(mean);
            self.vars.push(var);
        }
    }
}

impl OptimizerImpl for Adam {
    fn update(&mut self, params: &[Rc<RefCell<Variable>>]) {
        if self.means.is_empty() {
            self.init_states(params);
        }

        self.t += 1;
        let mut bias_correction = (1.0 - self.betas.1.powf(self.t as f32)).sqrt();
        bias_correction /= 1.0 - self.betas.0.powf(self.t as f32);
        let alpha = self.lr * bias_correction;

        for (i, param) in params.iter().enumerate().take(params.len()) {
            let mut param = param.borrow_mut();
            for j in 0..param.size() as usize {
                let grad = param.grad[j];

                // update states
                let old_mean = self.means[i][j];
                let old_var = self.vars[i][j];
                let new_mean = self.betas.0 * old_mean + (1.0 - self.betas.0) * grad;
                let new_var = self.betas.1 * old_var + (1.0 - self.betas.1) * grad * grad;

                // update parameter
                param.data[j] -= alpha * new_mean / (new_var.sqrt() + self.eps);
                self.means[i][j] = new_mean;
                self.vars[i][j] = new_var;
            }
        }
    }
}
