use miniature::datasets::MNISTLoader;
use miniature::functions as F;
use miniature::graph::backward;
use miniature::optimizers as S;
use miniature::parametric_functions as PF;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MNISTLoader::new("datasets")?;
    let (test_x, test_t) = dataset.get_test_data();

    let fc1 = PF::linear(28 * 28, 256);
    let fc2 = PF::linear(256, 10);

    let mut optim = S::sgd(0.001);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());

    let mut iter = 0;
    loop {
        let (x, t) = dataset.sample(32);
        let onehot_t = F::onehot(t, 10);

        // forward
        let h = F::relu(fc1.call(x));
        let output = fc2.call(h);

        // loss
        let loss = F::cross_entropy_loss(output, onehot_t);

        optim.zero_grad();
        backward(loss);
        optim.update();

        iter += 1;
        if iter % 100 == 0 {
            // test
            let h = F::relu(fc1.call(test_x.clone()));
            let output = F::argmax(fc2.call(h));

            let mut count = 0;
            let test_size = output.borrow().shape[0];
            for i in 0..test_size as usize {
                let pred_label = output.borrow().data[i] as u8;
                let test_label = test_t.borrow().data[i] as u8;
                if pred_label == test_label {
                    count += 1;
                }
            }
            let accuracy = (count as f32) / (test_size as f32);
            println!("Iteration {}: Accuracy={}", iter, accuracy);
        }

        if iter == 100000 {
            break;
        }
    }

    Ok(())
}
