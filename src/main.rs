mod datasets;
mod function;
mod functions;
mod graph;
mod optimizer;
mod parametric_functions;
mod variable;

use optimizer::OptimizerImpl;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = datasets::MNISTLoader::new("datasets")?;
    let (test_x, test_t) = dataset.get_test_data();

    let fc1 = parametric_functions::linear::Linear::new(28 * 28, 128);
    let fc2 = parametric_functions::linear::Linear::new(128, 128);
    let fc3 = parametric_functions::linear::Linear::new(128, 10);

    let mut optim = optimizer::SGD::new(0.001);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());
    optim.set_params(fc3.get_params());

    let mut iter = 0;
    loop {
        let (x, t) = dataset.sample(128);
        let onehot_t = functions::onehot(t, 10);

        // forward
        let h1 = functions::relu(fc1.call(x));
        let h2 = functions::relu(fc2.call(h1));
        let output = fc3.call(h2);

        // loss
        let cross_entropy = functions::neg(functions::mul(
            onehot_t,
            functions::log(functions::softmax(output)),
        ));
        let loss = functions::mean(cross_entropy);

        optim.zero_grad();
        graph::backward(loss);
        optim.update();

        iter += 1;
        if iter % 100 == 0 {
            // test
            let h1 = functions::relu(fc1.call(test_x.clone()));
            let h2 = functions::relu(fc2.call(h1));
            let output = functions::argmax(fc3.call(h2));

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
