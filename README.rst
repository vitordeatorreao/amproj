AM Proj
--------

.. image:: https://travis-ci.org/vitordeatorreao/amproj.svg?branch=master
    :target: https://travis-ci.org/vitordeatorreao/amproj

This is the code repository for the 2017.1 Machine Learning class project for
the UFPE's MSc in Computer Science programme.

==============
Usage
==============

To install the project, simply run::

	$ python setup.py install

This will install the amproj library and the amproj_cmd tool in your current
Python environment. Note that this is probably **not** what you want if you
intend to contribute or develop for the project.

To simply install the project's dependencies, run::

	$ python setup.py develop

That will install the all dependencies the project needs to your current Python
environment.

=======================
Develop for the Project
=======================

If you are a collaborator or you simply want to develop the project, the
recommended way is run::

	$ pip install -e .

That will install the project and its dependencies into your current Python
environment, but the project itself will be installed with a symlink, that way,
if you make any changes to the source files will be immediately made available
to any code that uses the library.

=============
Run the tests
=============

To run the unit tests in this project, simply run::

	$ python setup.py test

That should also install ``nose`` on your current Python envinroment and
execute ``nosetests``.
