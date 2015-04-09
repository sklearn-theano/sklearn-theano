import os
import setuptools

setuptools.setup(
    name='sklearn-theano',
    version='0.0.1',
    packages=setuptools.find_packages(),
    package_data={"": ['*.jpg', '*.png', '*.json', '*.txt']},
    author='Kyle Kastner',
    author_email='kastnerkyle@gmail.com',
    description='Scikit-learn compatible estimators using Theano',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.rst')).read(),
    license='BSD 3-clause',
    url='http://github.com/sklearn-theano/sklearn-theano/',
    install_requires=['numpy',
                      'scipy',
                      'Theano',
                      'Pillow',
                      'scikit-learn'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
