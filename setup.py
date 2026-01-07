from setuptools import setup, find_packages

setup(
    name="PyPoli",
    version="0.1.0",
    description="Pauli Propagation for Python with JAX/Flax support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PyPoli",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "flax",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
