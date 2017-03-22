============
Installation
============

The OASIS package currently supports Python 3. Legacy Python (2.x) is not
supported.

Installation from source
------------------------
Currently the package must be installed from source, although in future it may
be available on PyPI.

To clone the source from Github, execute the following command::

    $ git clone https://github.com/ngmarchant/oasis.git

You can then install the package to your user site directory as follows::

    $ cd oasis
    $ python3 setup.py install --user

Dependencies
------------
OASIS depends on the following Python packages:

* numpy
* scipy
* sklearn
* tables (for experiments)

These dependencies should be automatically resolved during installation.