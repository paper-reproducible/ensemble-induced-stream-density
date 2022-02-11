# Ensemble-induced Density Estimators for Data Stream Clustering
Benchmark test and demo of ensemble-induced density estimators for data stream clustering.

Written by Xiaoyu Qin, Monash University, February 2022, version 1.0.

This software is under MPLv2. Please communicate with the author of this repository if you need to redistribute an altered version of this software.

To successfully run the demo, please make sure that python 3.8+ was installed.

This software uses Python libraries include Numpy, SciPy, Scikit-Learn, Pandas, MatplotLib and CuPy (Optional). 

To install the mandatory libraries with **pip**

```bash
pip install -r requirements.txt
```

If **CUDA** is supported, it is recommended to install **CuPy** to accelerate some parts of the algorithm.

If running **python** provided by a **conda** environment is preferred, it is recommended to use **miniforge**, especially when using Apple Silicon MacOS.

Running the following command starts the benchmark test with an artificial data stream.

```bash
python main.py
```

Please notice that the artificial data stream is generated in real-time and is not same for each time of running the benchmark.
Running the following command starts an animation demonstrating how the artificial data stream is generated, 
in which each data point is coloured by one out of four Gaussian probability density functions generating it, 
and the proper clustering result after the end of the data stream.

```bash
python main.py demo
```

Running the following command starts the benchmark test with manipulated AGNews dataset, which has one significant concept drift of new class emerging in.

```bash
python main.py agnews
```

A paper of this project is under review. Citation information will be given and please cite when the paper is published. 

