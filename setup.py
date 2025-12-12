from setuptools import setup, find_packages

setup(
    name="ww_pgd",
    version="0.1.0",
    description="WeightWatcher Projected Gradient Descent (WW-PGD): spectral tail projection add-on for PyTorch optimizers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Charles H. Martin, PhD",
    url="https://github.com/<YOUR_GITHUB_USERNAME>/ww_pgd",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "pandas",
        "weightwatcher",
    ],
    extras_require={
        "dev": ["pytest", "twine", "build"],
    },
    include_package_data=True,
    license="MIT",
)
