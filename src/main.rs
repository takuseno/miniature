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

    let fc1 = parametric_functions::linear::Linear::new(28 * 28, 128);
    let fc2 = parametric_functions::linear::Linear::new(128, 128);
    let fc3 = parametric_functions::linear::Linear::new(128, 10);

    let mut optim = optimizer::SGD::new(0.001);
    optim.set_params(fc1.get_params());
    optim.set_params(fc2.get_params());
    optim.set_params(fc3.get_params());

    for _ in 0..10000 {
        let (x, t) = dataset.sample(128);

        // forward
        let h1 = functions::relu(fc1.call(x));
        let h2 = functions::relu(fc2.call(h1));
        let output = fc3.call(h2);

        // loss
        let loss = functions::mean(functions::neg(functions::mul(
            t,
            functions::log(functions::softmax(output)),
        )));
        println!("{}", loss.borrow().data[0]);

        optim.zero_grad();
        graph::backward(loss);
        optim.update();
    }

    Ok(())
}
