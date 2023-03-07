
GeneSSA Overview
================

GeneSSA provides a framework for stochastic simulation of dynamic processes, with a particular emphasis on chemical reaction networks and gene regulatory networks. We acknowledge that there are many other excellent tools designed for simulating stochastic dynamics. The main advantage of GeneSSA is its barebones implementation, which allow for rapid large scale simulation of a wide variety of systems. While GeneSSA was developed with biological circuits in mind; the code is well suited to any system exhibiting Markovian dynamics.

Please keep in mind that this package was developed for internal use, and as such we offer no guarantees regarding its performance.


Installation
============

First, download the [latest distribution](https://github.com/sebastianbernasek/genessa/archive/0.3.tar.gz).

Before attempting to install GeneSSA, we suggest creating a clean virtual environment and installing all necessary dependencies ahead of time. While the distribution includes a pre-compiled version of the cythonized solver code, we can't guarantee that it will run correctly on all platforms. For best results, we recommend that you install cython before installing GeneSSA.


System Requirements
-------------------

 - Python 3.6+
 - [NumPy](https://www.scipy.org/): ``pip install numpy``
 - [Cython](http://cython.org/): ``pip install cython`` (optional, but recommended)


Install GeneSSA
---------------

The simplest method is to install via ``pip``:

    pip install genessa-0.3.tar.gz

The core solver is implemented in cython, with the relevant extension modules residing in ``genessa/solver/*.pyx`` and ``genessa/solver/*.pxd``. These extension modules must be compiled prior to runtime. Upon installation of ``genessa``, the package installer will attempt to use a local cython installation to compile the extension modules. If no cython installation is found, pre-compiled versions are automatically imported from the ``genessa`` source distribution. Note that compilation has only been tested in macOS.

To manually compile the GeneSSA package, unpack the tarball and build inplace:

    tar -xzf genessa-0.3.tar.gz
    cd genessa-0.3
    python setup.py build_ext --inplace



GeneSSA Modules
===============

The GeneSSA simulation platform consists of several core modules:

  * ``genessa.kinetics`` provides objects for representing various types of interactions.

  * ``genessa.networks`` provides objects for constructing a network of interactions.

  * ``genessa.signals`` provides objects for constructing exogenous signals (e.g. perturbations).

  * ``genessa.solver`` provides a basic implementation of the stochastic simulation algorithm (Gillespie 1977).

  * ``genessa.timeseries`` provides objects for storing and analyzing multidimensional timeseries.

Additionally, GeneSSA includes templates for common gene regulatory networks:

  * ``genessa.models`` provides base classes for easy construction of several different types of GRNs.

  * ``genessa.demo`` provides some fully functional example networks.



Example Usage
=============

We have included a series of [Jupyter notebooks](https://github.com/sebastianbernasek/genessa/tree/master/notebooks) with some examples of how to get started with using GeneSSA. These examples include:

  * [First-order decay](https://github.com/sebastianbernasek/genessa/tree/master/notebooks/first_order_decay.ipynb)

  * [A birth-death process](https://github.com/sebastianbernasek/genessa/tree/master/notebooks/birth_death_process.ipynb)

  * [Coupled oscillators](https://github.com/sebastianbernasek/genessa/tree/master/notebooks/oscillators.ipynb)


Further Examples
----------------

For more detailed usage examples, please refer to the [simulations](https://github.com/sebastianbernasek/gram) we performed as part of our [study](https://www.cell.com/cell/pdf/S0092-8674(19)30686-5.pdf) of the relationship between metabolic conditions and developmental gene expression.

