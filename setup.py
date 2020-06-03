import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformerocr", # Replace with your own username
    version="0.1.14",
    author="pbcquoc",
    author_email="pbcquoc@gmail.com",
    description="Transformer base text detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pbcquoc/transformerocr",
    packages=setuptools.find_packages(),
    install_requires=[
        'einops==0.2.0',
        'gdown==3.11.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
