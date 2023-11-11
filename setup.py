from setuptools import setup

setup(
    name="clearbox",
    version="0.1",
    description="A set of educational implementations of machine learning algorithms.",
    url="https://kgbench.info",
    author="Peter Bloem",
    author_email="clearbox@peterbloem.nl",
    packages=["clearbox"],
    install_requires=[
        "numpy",
        "matplotlib",
        "colorama",
        'tqdm',
        'fire',
        'torch',
        'torchvision'
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': ['clearbox=clearbox.command_line:main'],
    }
)