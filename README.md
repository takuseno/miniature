# miniature: a toy deep learning library written in Rust
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![test](https://github.com/takuseno/miniature/actions/workflows/test.yaml/badge.svg)](https://github.com/takuseno/miniature/actions/workflows/test.yaml)

A miniature is a toy deep learning library written in Rust.

The miniature is:
- implemented for author's Rust practice.
- designed as simple as possible.

The miniature is NOT:
- supporting CUDA.
- optimized for computational costs.
- a product-ready library.

## features
- define-by-run style API
- easy as Python libraries (e.g. TensorFlow, PyTorch, nnabla)
- easy to add more features (e.g. layers, optimizers)

## run MNIST
Download MNIST dataset for the first time.
```
$ python scripts/download_mnist.py
```

Then, run.
```
$ cargo run --release
```


## example
```rs
use miniature::functions as F;
use miniature::graph::backward;
use miniature::optimizer as S;
use miniature::parametric_functions as PF;
use miniature::variable::Variable;

use std::rc::Rc;
use std::cell::RefCell;

fn main() {
    // define layers
    let fc1 = PF::linear(28 * 28, 256);
    let fc2 = PF::linear(256, 256);
    let fc3 = PF::linear(256, 10);

    // define optimizer
    let mut optim = S::adam(0.001, (0.9, 0.999), 1e-8);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());
    optim.set_params(fc3.get_params());

    let x = Rc::new(RefCell::new(Variable::rand(vec![32, 28 * 28])));
    let t = Rc::new(RefCell::new(Variable::rand(vec![32])));

    // forward
    let h1 = F::relu(fc1.call(x));
    let h2 = F::relu(fc2.call(h1));
    let y = fc3.call(h2);

    // loss
    let loss = F::cross_entropy_loss(y, F::onehot(t, 10));

    // update
    optim.zero_grad();
    backward(loss);
    optim.update();
}
```
