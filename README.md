
## Summary
This repository contains several Python implementations of popular matrix-based spike-deconvolution algorithms, making use of the open source cvxpy solver. A good comparison of such methods is included in [4]. Please see the [writeup](./doc/deconv-notes.md) giving a brief description of the deconvolution problem and FOOPSI variants.


## Features

* object-oriented design
* multiprocesses support
* friendly runtime info
* Discretization via threshold binning
* Deconvolution via Wiener filter
* FOOPSI [1]
* Constrained FOOPSI [2]
* Thresholded. FOOPSI [3]
* Automatic parameter selection [4]

## Getting Started

### Prerequisites

Use `config.json` to set running parameters.

### Installing and Running

```shell
# run under `pyexperimentparser` directory
# create vitual environment
python3 -m virtualenv .env
# enter virtual environment
source .env/bin/activate
# install required packages
pip3 install -r requirements.txt
# run the project
python3 main.py
# exit virtual environment
deactivate
```

The output including data, figure and other analysis is put under path `out/` by default. The following figure demonstrates the result of the current config file.

![output example](figures/example.png)

### Debug

For debugging purpose or more running time info, please run with `--debug` flag. Otherwise, errors generated by subprocesses may not be revealed.

```
# this may heavily increase running time
python3 main.py --debug
```

## Running the Tests

```shell
# run all unit tests
python3 -m unittest discover
```

## License

## Acknowledgments

* This project is inspired by [CaImAn](https://github.com/flatironinstitute/CaImAn) which is a Python library for Calcium imaging analysis.
* This readme file is following the style of [README-Template.md](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2).

##  \& Resources

* [1] Vogelstein, J.T., Packer, A.M., Machado, T.A., Sippy, T., Babadi, B., Yuste, R. and Paninski, L., 2010. Fast nonnegative deconvolution for spike train inference from population calcium imaging. Journal of neurophysiology,104(6), pp.3691-3704.

* [2] Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T.A., Merel, J., Pfau, D., Reardon, T., Mu, Y., Lacefield, C., Yang, W. and Ahrens, M., 2016. Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron, 89(2), pp.285-299.

* [3] Friedrich, J., Zhou, P. and Paninski, L., 2017. Fast online deconvolution of calcium imaging data. PLoS computational biology, 13(3), p.e1005423.

* [4] Marius Pachitariu, Carsen Stringer, and Kenneth D. Harris. Robustness of spike deconvolution for neuronal calcium imaging. The Journal of Neuroscience 38(37):3339-17

* Connectomics Kaggle challenge: https://www.kaggle.com/c/connectomics

## TODO

* Direct L0 minimization: Sean W Jewell, Toby Dylan Hocking, Paul Fearnhead, Daniela M Witten., Fast nonconvex deconvolution of calcium imaging data., Biostatistics, kxy083, https://doi.org/10.1093/biostatistics/kxy083.

* Eftychios A. Pnevmatikakis and Liam Paninski., Joint low-rank \& sparse estimation of neuron locations and spikes in caclcium images: Sparse nonnegative deconvolution for compressive calcium imaging: algorithms and phase transitions., NeurIPS 2013.

* Complete OASIS implementation


## Contributors

* Chester Holtz, chholtz@eng.ucsd.edu

