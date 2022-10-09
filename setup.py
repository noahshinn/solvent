from setuptools import setup


setup(
    name='solvent',
    version='0.0.0',
    description='Solvent is an open-source code for training highly accurate equivariant deep learning interatomic potentials for multiple electronic states.',
    author='Noah Shinn, Sulin Liu',
    packages=['solvent'],
    python_requires='>=3.7',
    install_requires=[
        'e3nn',
        'joblib'
    ]
)

