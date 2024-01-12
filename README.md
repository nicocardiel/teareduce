# teareduce
Utilities for astronomical data reduction

## Installing the code

In order to keep your Python installation clean, it is highly recommended to 
first build a specific Python 3 *virtual enviroment*

### Creating and activating the Python virtual environment

```shell
$ python3 -m venv venv_teareduce
$ . venv_teareduce/bin/activate
(venv_teareduce) $ 
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
0.2.1
```
