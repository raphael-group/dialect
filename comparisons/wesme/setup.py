from setuptools import setup, find_packages

setup(
    name='wesme',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'networkx==2.7.1',
        'numpy==1.26.4',
        'pandas==1.4.2',
        'scipy==1.13.0',
        'setuptools==69.0.2',  # Note: Remove duplicate entries
        'statsmodels==0.13.2',
        'tqdm==4.64.0'
    ],
)
