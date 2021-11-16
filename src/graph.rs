use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::variable::Variable;

pub fn backward(variable: Rc<RefCell<Variable>>) {
    if variable.borrow().parent.is_none() {
        return;
    }

    // initialize leaf gradient with ones
    variable.borrow_mut().one_grads();

    let mut queue: VecDeque<Rc<RefCell<CgFunction>>> = VecDeque::new();
    queue.push_back(variable.borrow().parent.as_ref().unwrap().clone());
    loop {
        if queue.is_empty() {
            break;
        }
        let function = queue.pop_front().unwrap();

        function.borrow_mut().backward();

        for input in function.borrow_mut().get_inputs().iter() {
            if !input.borrow().need_grad {
                continue;
            }
            let borrowed_input = input.borrow();
            let parent = borrowed_input.parent.as_ref();
            if let Some(p) = parent {
                queue.push_back(p.clone());
            }
        }
    }
}
