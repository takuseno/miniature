use rand::Rng;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufReader, Read};
use std::rc::Rc;

use crate::variable::Variable;

pub struct MNISTLoader {
    train_images: Vec<Vec<f32>>,
    train_labels: Vec<i32>,
    test_images: Vec<Vec<f32>>,
    test_labels: Vec<i32>,
    train_size: i32,
    test_size: i32,
}

const MNIST_NUM_CLASSES: u32 = 10;
const MNIST_IMAGE_SIZE: u32 = 28 * 28;
const MNIST_TRAIN_IMAGE_FILE: &str = "mnist-train-images";
const MNIST_TRAIN_LABEL_FILE: &str = "mnist-train-labels";
const MNIST_TEST_IMAGE_FILE: &str = "mnist-test-images";
const MNIST_TEST_LABEL_FILE: &str = "mnist-test-labels";

fn load_mnist_image_file(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);

    // read header
    let mut buf = [0; 4];

    // read magic number
    file.read(&mut buf)?;

    // read total number
    file.read(&mut buf)?;
    let total_number = i32::from_be_bytes(buf);

    // read height
    file.read(&mut buf)?;
    let height = i32::from_be_bytes(buf);

    // read width
    file.read(&mut buf)?;
    let width = i32::from_be_bytes(buf);

    // read pixel data
    let mut images: Vec<Vec<f32>> = Vec::new();
    for _ in 0..total_number {
        let mut image: Vec<f32> = Vec::new();
        image.resize((height * width) as usize, 0.0);
        for i in 0..(height * width) as usize {
            let mut buf = [0; 1];
            file.read(&mut buf)?;
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
    file.read(&mut buf)?;

    // read total number
    file.read(&mut buf)?;
    let total_number = i32::from_be_bytes(buf);

    // read label data
    let mut labels: Vec<i32> = Vec::new();
    labels.resize(total_number as usize, 0);
    for i in 0..total_number as usize {
        let mut buf = [0; 1];
        file.read(&mut buf)?;
        labels[i] = u8::from_be_bytes(buf) as i32;
    }

    Ok(labels)
}

impl MNISTLoader {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let base_dir = String::from(path);

        println!("Loading training images...");
        let train_images =
            load_mnist_image_file(&(base_dir.clone() + "/" + MNIST_TRAIN_IMAGE_FILE))?;

        println!("Loading training labels...");
        let train_labels =
            load_mnist_label_file(&(base_dir.clone() + "/" + MNIST_TRAIN_LABEL_FILE))?;

        println!("Loading test images...");
        let test_images = load_mnist_image_file(&(base_dir.clone() + "/" + MNIST_TEST_IMAGE_FILE))?;

        println!("Loading test labels...");
        let test_labels = load_mnist_label_file(&(base_dir.clone() + "/" + MNIST_TEST_LABEL_FILE))?;

        let train_size = train_images.len() as i32;
        let test_size = test_images.len() as i32;

        Ok(Self {
            train_images: train_images,
            train_labels: train_labels,
            test_images: test_images,
            test_labels: test_labels,
            train_size: train_size,
            test_size: test_size,
        })
    }

    pub fn sample(&self, batch_size: u32) -> (Rc<RefCell<Variable>>, Rc<RefCell<Variable>>) {
        let mut images: Vec<f32> = Vec::new();
        let mut labels: Vec<f32> = Vec::new();
        images.resize((batch_size * MNIST_IMAGE_SIZE) as usize, 0.0);
        labels.resize((batch_size * MNIST_NUM_CLASSES) as usize, 0.0);

        let mut rng = rand::thread_rng();
        for i in 0..batch_size as usize {
            let index = rng.gen_range(0, self.train_size);

            let image_start = MNIST_IMAGE_SIZE as usize * i;
            let image_end = image_start + MNIST_IMAGE_SIZE as usize;
            images[image_start..image_end].copy_from_slice(&self.train_images[index as usize]);

            let label_offset = MNIST_NUM_CLASSES as usize * i;
            for j in 0..MNIST_NUM_CLASSES as usize {
                labels[j + label_offset] = if self.train_labels[index as usize] == j as i32 {
                    1.0
                } else {
                    0.0
                }
            }
        }

        let image_batch = Rc::new(RefCell::new(Variable::new(vec![batch_size, 28 * 28])));
        let label_batch = Rc::new(RefCell::new(Variable::new(vec![batch_size, 10])));
        image_batch.borrow_mut().set_data(&images);
        label_batch.borrow_mut().set_data(&labels);
        (image_batch, label_batch)
    }
}
