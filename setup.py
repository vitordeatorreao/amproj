from setuptools import setup

description = 'The 2017 UFPE\'s machine learning class project'
authors = ['Vitor Torreao <vat@cin.ufpe.br>',
           'Avyner Lucena <ahbfl@cin.ufpe.br>',
           'Jorge Linhares <jhcl@cin.ufpe.br>']


def readme():
    try:
        with open('README.rst') as f:
            return f.read()
    except(IOError):
        return description


setup(name='amproj',
      version='0.1',
      description=description,
      long_description=readme(),
      url='http://github.com/vitordeatorreao/amproj',
      author=', '.join(authors),
      author_email='',
      license='GPL-2.0',
      packages=['amproj'],
      zip_safe=False)
