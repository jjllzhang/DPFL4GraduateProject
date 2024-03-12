# Differential Privacy and Shuffler in Federated Learning System

This project as my graduate project, aims to enhance the privacy protection capabilities of federated learning systems by integrating Differential Privacy and the Shuffler mechanism. It's designed for distributed machine learning scenarios with stringent privacy requirements.

## Motivation and Background

In the era of ubiquitous data, protecting individual privacy has become a significant challenge. Although federated learning offers a way to conduct machine learning tasks without sharing raw data among participants, it still faces potential privacy leakage risks. This project introduces Differential Privacy and the Shuffler mechanism to federated learning to minimize these risks and enhance the system's overall privacy protection.

## Installation Guide

This project requires Python 3.6 or higher. Follow these steps to install the necessary dependencies:

```bash
git clone https://github.com/jjllzhang/DPFL4GraduateProject.git
cd DPFL4GraduateProject
conda env create -f environment.yml
conda activate DPFL
```

## Usage Example

Here's a simple example of how to start a federated learning task with differential privacy and Shuffler mechanism enabled:

- First check the parameters in [`config.yml`](./config.yml)
- Modify the parameters as you want
- Then run the following bash to start training the model

```bash
python main.py
```

## Technical Architecture and Key Technologies

### Architecture overview

![The system architecture](./imgs/architecture.png)

### Differential Privacy

implement the following:

- Perlayer DP
- Auto DP
- DP with Shuffler

Differential Privacy (DP) ensures that the removal or addition of a single database item does not significantly affect the outcome of any analysis, providing strong privacy guarantees for individuals' data.

### Shuffler Mechanism

The Shuffler mechanism adds an additional layer of privacy by randomly permuting data points. This process helps in breaking the link between the data and its source, further enhancing privacy.

### Federated Learning

The core of our project, Federated Learning, is a distributed machine learning approach that enables multiple participants to collaboratively learn a shared model while keeping their data local.

## Features

- **Privacy-Preserving**: Implements 3 kinds of Differential Privacy Federated Learning and Shuffler mechanisms.
- **Scalability**: Designed to efficiently handle large-scale federated learning tasks.
- **Flexibility**: Supports various federated learning scenarios and configurations.
- **Privacy-Accountant**: Uses Renyi Differential Privacy to account for the privacy loss to get tighter bounds

## Performance Metrics and Advantages

The system has been rigorously tested under different conditions, demonstrating significant improvements in privacy protection without compromising learning efficiency.

TODO: show performance results

## Copyright and License

This project is licensed under the Apache License - see the [`LICENSE`](./LICENSE) file for details.

## Acknowledgments

- A special thanks to [@JeffffffFu](https://github.com/JeffffffFu) whose DP videos uploaded at [BiliBili](https://bilibili.com) and codes implemention help me lot.
- Gratitude to my lab who provides me a server for experimental test.
- Appreciation for the open-source community for the tools and libraries that made this project possible.
