from setuptools import setup, find_packages

setup(
    name='fish_tracker',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-python",
        "scipy",
        
    ],
)