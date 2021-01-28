# Gravel

Gravel: A Gradient Divergence-Aware Approach to Improved Robustness of Deep Learning Models

## Usage
To run the deep learning model, use the following format:

`python main.py --config <config.yml> --dataset <dataset> --mode <mode> --version <ver>`

* `--config`: the configuration `.yml` file
* `--dataset`: the dataset (`cifar10` or `mnist` in our experiment)
* `--mode`: `1` for the standard training, `2` for the `FREE` training, `3` for the `Gravel` training
* `--version`: version number

For example:

`python main.py --config config.yml --dataset cifar10 --mode 3 --version 1`

To evaluate the model under the PGD attack, please use:

`python main.py --config config.yml --dataset cifar10 --mode 3 --version 1 --evaluate`
