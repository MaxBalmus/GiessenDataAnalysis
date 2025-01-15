from setuptools import setup, find_packages

setup(
    name="my_package",                  # Package name
    version="0.1.0",                    # Version
    author="Your Name",                 # Author
    author_email="your_email@example.com",
    description="A short description of the package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",  # Repository URL
    packages=find_packages(),           # Automatically find sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",            # Minimum Python version
    install_requires=[                  # Dependencies
        "numpy", 
        "scipy",
        "pandas",
        "matplotlib",
        "ipykernel",
        "scikit-learn"
    ],
)
