# teareduce

Utilities for astronomical data reduction.

This package is not intended to be a general-purpose image reduction code. It
only includes specific operations required in certain steps of the traditional
astronomical image reduction process that, at the time of its creation, were
not available in more established packages such as
[ccdproc](https://ccdproc.readthedocs.io/en/latest/). In addition, it also
offers alternative ways to perform certain tasks that we have found to be more
practical for use in Master’s level classes.


## Documentation

The documentation for this package is available at [this
link](https://nicocardiel.github.io/teareduce-cookbook/intro.html).
It includes Juypter notebooks that can be easily downloaded and demonstrate the
practical use of the defined functionality.

## Installing the code

In order to keep your Python installation clean, it is highly recommended to 
first build a specific Python 3 *virtual enviroment*

### Creating and activating the Python virtual environment

```shell
$ python3 -m venv venv_teareduce
$ . venv_teareduce/bin/activate
(venv_teareduce) $ 
```

If you are planning to use **tea-cleanest**, you need to install this package
with extra dependencies. In this case employ:

```shell
(venv_teareduce) $ pip install 'teareduce[cleanest]'
```

### Installing the package

The latest stable version is available via de [PyPI repository](https://pypi.org/project/teareduce/):

```shell
(venv_teareduce) $ pip install teareduce
```

**Note**: This command can also be employed in a Windows terminal opened through the 
``CMD.exe prompt`` icon available in Anaconda Navigator.

The latest development version is available through [GitHub](https://github.com/nicocardiel/teareduce):

```shell
(venv_teareduce) $ pip install git+https://github.com/nicocardiel/teareduce.git@main#egg=teareduce
```

### Testing the installation

```shell
(venv_teareduce) $ pip show teareduce
```

```shell
(venv_teareduce) $ ipython
In [1]: import teareduce as tea
In [2]: print(tea.__version__)
0.5.9
```

Note that in PyPI there is a package called **tea** that provides utilities
unrelated to **teareduce**. However, throughout the examples described
in the documentation we are making use of ``import teareduce as tea``
to define a convenient alias.
