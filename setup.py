import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vietocr",
    version="0.3.13",
    author="pbcquoc",
    author_email="pbcquoc@gmail.com",
    description="Transformer base text detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pbcquoc/vietocr",
    packages=setuptools.find_packages(),
    install_requires=[
        'einops==0.2.0',
        'gdown==4.4.0',
        'prefetch_generator==1.0.1',
        'imgaug==0.4.0',
        'albumentations==1.4.2',
        'lmdb>=1.0.0',
        'scikit-image>=0.21.0',
        'pillow==10.3.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
