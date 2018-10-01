
GeneSSA Overview
================

GeneSSA provides a framework for stochastic simulation of gene regulatory networks.


Installation
============

First, download the [latest distribution](https://github.com/sebastianbernasek/genessa/archive/v0.1.tar.gz).

Before attempting to install GeneSSA, make sure you have installed all necessary dependencies.


System Requirements
-------------------

 - Python 3.6+
 - [NumPy](https://www.scipy.org/): ``pip install numpy``
 - [Cython](http://cython.org/): ``pip install cython`` (optional)


Install Binding
---------------

The simplest method is to install via ``pip``:

    pip install genessa-0.1.tar.gz

The core solver is implemented in cython, with the relevant extension modules residing in ``genessa/solver/*.pyx`` and ``genessa/solver/*.pxd``. These extension modules must be compiled prior to runtime. Upon installation of ``genessa``, the package installer will attempt to use a local cython installation to compile the extension modules. If no cython installation is found, pre-compiled versions are automatically imported from the ``genessa`` source distribution. Note that compilation has only been tested in macOS.

To manually compile the GeneSSA package, unpack the tarball and build inplace:

    tar -xzf genessa-0.1.tar.gz
    cd genessa-0.1
    python setup.py build_ext --inplace



GeneSSA Modules
===============

GeneSSA consists of a core model supported by several additional tools.

The core modeling components are implemented as cython extension modules in ``binding.model``. These extension modules include:

  * ``binding.model.elements`` provides an Element base class for constructing individual binding elements.

  * ``binding.model.trees`` provides a Tree base class for constructing a microstate enumeration tree.

  * ``binding.model.partitions`` provides PartitionFunction base class for evaluating the statistical frequency of all binding microstates for a given binding element.

  * ``binding.model.parallel`` provides an interface to the python ``multiprocessing`` module. Microstate enumeration is parallelized by subprocessing individual branches of a microstate enumeration tree below a specified cut depth.


The supporting python modules include:

  * ``binding.analysis`` provides an interface for generating equilibrium binding site occupancy phase diagrams and titration contours.

  * ``binding.utilities`` provides tools for converting experimentally measured equilibrium dissociation constants to binding energies that may be used as model parameters.



Example Usage
=============

Define transcription factor binding energies:

    # binding energy to strong sites for TF species 1, 2, ... (kcal/mol)
    alpha = [-10, -10, ...]

    # binding energy to weak sites for TF species 1, 2, ... (kcal/mol)
    beta = [-6, -6, ...]

    # stabilization energy from adjacent sites bound by same TF species (kcal/mol)
    gamma = [-8, -8, ...]

    binding_energies = dict(alpha=alpha, beta=beta, gamma=gamma)


Define a DNA binding element:

    from binding.model.elements import Element

    element_size = 12
    strong_sites = (0,)

    # instantiate binding element
    element = Element(Ns=element_size,
                      params=binding_energies,
                      ets=strong_sites)


Evaluate equilibrium binding site occupancies or a range of TF concentrations:

    from binding.model.partitions import PartitionFunction
    import numpy as np

    # define TF concentrations
    C = np.linspace(0, 100, 100) * 1E-9
    xx, yy = np.meshgrid(*(C, C), indexing='xy')
    concentrations = np.stack((xx.T, yy.T)).reshape(2, -1).T

    # instantiate partition function
    pf = PartitionFunction(element, concentrations)

    # evaluate equilibrium binding site occupancies
    occupancies = pf.c_get_occupancies()


Further Examples
----------------

For an additional usage example, please refer to Figure 3 of our [study](https://github.com/sebastianbernasek/pnt_yan_ratio) of Pnt and Yan expression in the *Drosophila* eye.
