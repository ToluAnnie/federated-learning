from setuptools import setup, find_packages

setup(
    name="federated_learning_diabetes",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "flwr",
        "flwr",
        "numpy"
    ],
)