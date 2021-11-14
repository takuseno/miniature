# miniature
A miniature is a toy deep learning library written in Rust.

The miniature is:
- implemented for author's Rust practice.
- designed as simple as possible.

The miniature is NOT:
- supporting CUDA.
- optimized for computational costs.
- a product-ready library.


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
use miniature::datasets::MNISTLoader;
use miniature::functions as F;
use miniature::graph::backward;
use miniature::optimizer as S;
use miniature::optimizer::OptimizerImpl;
use miniature::parametric_functions as PF;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MNISTLoader::new("datasets")?;
    let (test_x, test_t) = dataset.get_test_data();

    let fc1 = PF::linear::Linear::new(28 * 28, 128);
    let fc2 = PF::linear::Linear::new(128, 128);
    let fc3 = PF::linear::Linear::new(128, 10);

    let mut optim = S::SGD::new(0.001);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());
    optim.set_params(fc3.get_params());

    let mut iter = 0;
    loop {
        let (x, t) = dataset.sample(128);
        let onehot_t = F::onehot(t, 10);

        // forward
        let h1 = F::relu(fc1.call(x));
        let h2 = F::relu(fc2.call(h1));
        let output = fc3.call(h2);

        // loss
        let cross_entropy = F::neg(F::mul(onehot_t, F::log(F::softmax(output))));
        let loss = F::mean(cross_entropy);

        optim.zero_grad();
        backward(loss);
        optim.update();

        iter += 1;
        if iter % 100 == 0 {
            // test
            let h1 = F::relu(fc1.call(test_x.clone()));
            let h2 = F::relu(fc2.call(h1));
            let output = F::argmax(fc3.call(h2));

            let mut count = 0;
            let test_size = output.borrow().shape[0];
            for i in 0..test_size as usize {
                if output.borrow().data[i] == test_t.borrow().data[i] {
                    count += 1;
                }
            }
            let accuracy = (count as f32) / (test_size as f32);
            println!("Iteration {}: Accuracy={}", iter, accuracy);
        }
    }

    Ok(())
}
```
