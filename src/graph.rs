use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;

use crate::variable::Variable;
use crate::function::CgFunction;

pub fn backward(variable: Rc<RefCell<Variable>>) {
    if variable.borrow().parent.is_none() {
        return
    }

    let mut queue: VecDeque<Rc<RefCell<CgFunction>>> = VecDeque::new();
    queue.push_back(variable.borrow().parent.as_ref().unwrap().clone());
    loop {
        if queue.is_empty() {
            break
        }
        let function = queue.pop_front().unwrap();

        function.borrow_mut().backward();

        for input in function.borrow_mut().inputs.iter() {
            let borrowed_input = input.borrow();
            let parent = borrowed_input.parent.as_ref();
            if parent.is_some() {
                queue.push_back(parent.unwrap().clone());
            }
        }
    }
}
