use rand::Rng;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufReader, Read};
use std::rc::Rc;

use crate::variable::Variable;

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const MNIST_TRAIN_IMAGE_FILE: &str = "mnist-train-images";
const MNIST_TRAIN_LABEL_FILE: &str = "mnist-train-labels";
const MNIST_TEST_IMAGE_FILE: &str = "mnist-test-images";
const MNIST_TEST_LABEL_FILE: &str = "mnist-test-labels";

fn load_mnist_image_file(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);

    // read header
    let mut buf = [0; 4];

    // read magic number
    file.read_exact(&mut buf)?;

    // read total number
    file.read_exact(&mut buf)?;
    let total_number = i32::from_be_bytes(buf);

    // read height
    file.read_exact(&mut buf)?;
    let height = i32::from_be_bytes(buf);

    // read width
    file.read_exact(&mut buf)?;
    let width = i32::from_be_bytes(buf);

    // read pixel data
    let mut images: Vec<Vec<f32>> = Vec::new();
    for _ in 0..total_number {
        let mut image = vec![0.0; (height * width) as usize];
        for i in 0..(height * width) as usize {
            let mut buf = [0; 1];
            file.read_exact(&mut buf)?;
            image[i] = u8::from_be_bytes(buf) as f32 / 255.0;
        }
        images.push(image);
    }

    Ok(images)
}

fn load_mnist_label_file(path: &str) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);

    // read header
    let mut buf = [0; 4];

    // read magic number
    file.read_exact(&mut buf)?;

    // read total number
    file.read_exact(&mut buf)?;
    let total_number = i32::from_be_bytes(buf);

    // read label data
    let mut labels: Vec<i32> = vec![0; total_number as usize];
    for i in 0..total_number as usize {
        let mut buf = [0; 1];
        file.read_exact(&mut buf)?;
        labels[i] = u8::from_be_bytes(buf) as i32;
    }

    Ok(labels)
}

pub struct MNISTLoader {
    train_images: Vec<Vec<f32>>,
    train_labels: Vec<i32>,
    test_images: Vec<Vec<f32>>,
    test_labels: Vec<i32>,
    train_size: usize,
    test_size: usize,
}

impl MNISTLoader {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let base_dir = String::from(path);

        println!("Loading training images...");
        let train_image_path = &(base_dir.clone() + "/" + MNIST_TRAIN_IMAGE_FILE);
        let train_images = load_mnist_image_file(train_image_path)?;

        println!("Loading training labels...");
        let train_label_path = &(base_dir.clone() + "/" + MNIST_TRAIN_LABEL_FILE);
        let train_labels = load_mnist_label_file(train_label_path)?;

        println!("Loading test images...");
        let test_image_path = &(base_dir.clone() + "/" + MNIST_TEST_IMAGE_FILE);
        let test_images = load_mnist_image_file(test_image_path)?;

        println!("Loading test labels...");
        let test_label_path = &(base_dir + "/" + MNIST_TEST_LABEL_FILE);
        let test_labels = load_mnist_label_file(test_label_path)?;

        let train_size = train_images.len();
        let test_size = test_images.len();

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            train_size,
            test_size,
        })
    }

    pub fn sample(&self, batch_size: usize) -> (Rc<RefCell<Variable>>, Rc<RefCell<Variable>>) {
        let mut images = vec![0.0; batch_size * MNIST_IMAGE_SIZE];
        let mut labels = vec![0.0; batch_size];

        let mut rng = rand::thread_rng();
        for i in 0..batch_size {
            let index = rng.gen_range(0, self.train_size) as usize;

            // set image
            let image_start = MNIST_IMAGE_SIZE * i;
            let image_end = image_start + MNIST_IMAGE_SIZE;
            images[image_start..image_end].copy_from_slice(&self.train_images[index]);

            // set label
            labels[i] = self.train_labels[index] as f32;
        }

        let image_batch = Rc::new(RefCell::new(Variable::new(vec![
            batch_size,
            MNIST_IMAGE_SIZE,
        ])));
        let label_batch = Rc::new(RefCell::new(Variable::new(vec![batch_size])));
        image_batch.borrow_mut().set_data(&images);
        label_batch.borrow_mut().set_data(&labels);
        (image_batch, label_batch)
    }

    pub fn get_test_data(&self) -> (Rc<RefCell<Variable>>, Rc<RefCell<Variable>>) {
        let mut images = vec![0.0; self.test_size * MNIST_IMAGE_SIZE];
        let mut labels = vec![0.0; self.test_size];

        for (i, label) in self.test_labels.iter().enumerate().take(self.test_size) {
            // set image
            let image_start = MNIST_IMAGE_SIZE * i;
            let image_end = image_start + MNIST_IMAGE_SIZE;
            images[image_start..image_end].copy_from_slice(&self.test_images[i]);

            // set label
            labels[i] = *label as f32;
        }

        let image_batch = Rc::new(RefCell::new(Variable::new(vec![
            self.test_size,
            MNIST_IMAGE_SIZE,
        ])));
        let label_batch = Rc::new(RefCell::new(Variable::new(vec![self.test_size])));
        image_batch.borrow_mut().set_data(&images);
        label_batch.borrow_mut().set_data(&labels);
        (image_batch, label_batch)
    }
}
