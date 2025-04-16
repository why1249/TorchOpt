# TorchOptLib Documentation

Welcome to the TorchOptLib documentation! TorchOptLib is a modular optimization library based on PyTorch for implementing and experimenting with various optimization algorithms.

## About TorchOptLib

TorchOptLib is a lightweight, extensible framework that provides:

- A collection of classic benchmark functions for testing optimization algorithms
- Easy-to-extend base classes for creating custom optimization algorithms
- GPU acceleration support through PyTorch's device management
- Built-in visualization and tracking of optimization progress

## Contents

- [Installation Guide](installation.md)
- [Quick Start](quick_start.md)
- [Benchmark Functions](benchmarks.md)
- [Optimization Algorithms](algorithms.md)
- [Extending TorchOptLib](extending.md)

## Motivation

As an ordinary college student learning optimization algorithms, I found that existing libraries for optimization and benchmark testing were often fragmented, lacked compatibility, and made it difficult to compare different algorithms effectively. Additionally, parallelizing these algorithms was not straightforward.

During my studies, I wanted to create a project that would help me better understand these algorithms while also providing a practical tool for experimentation. By leveraging PyTorch's powerful tensor operations and parallel computing capabilities, I created this unified framework to aid my learning process.

The goal of TorchOptLib is to:

- Provide a modular design with base classes that can easily integrate various test functions and optimization algorithms.
- Enable seamless GPU acceleration for faster computation.
- Simplify the process of comparing and experimenting with different optimization techniques.
- Serve as a learning tool for students like me who are interested in optimization algorithms.
