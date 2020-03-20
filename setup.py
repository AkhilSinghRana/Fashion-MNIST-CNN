import os 
from setuptools import setup, find_packages

#Use README file for long description of the project

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def parse_requiremetns(fname):
    with open(fname) as f:
        required = f.read().splitlines()
    return required
    
setup(
    name = "FASHION-MNIST-CNN",
    author = "Akhil Singh Rana",
    author_email = "er.akhil.singh.rana@gmail.com",
    description = ("This is a solution for the benchmarking of FASHION-MNIST"
                    "Classification of different objects in Fashion MNIST dataste"),
    long_description = read("README.md"),
    install_requires = parse_requiremetns("requirements.txt"),    
    packages=find_packages(),
)