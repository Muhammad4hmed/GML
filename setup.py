import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="GML",
    version="3.0.0",
    description="Automating Data Science",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Muhammad4hmed/Ghalat-Machine-Learning",
    author="Muhammad Ahmed & Naman Tuli",
    author_email="m.ahmed.memonn@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["GML"],
    include_package_data=True,
    install_requires=['scikit-learn','xgboost','fastai==1.0.61','catboost','Keras','lightgbm', 'torch', 'torchvision', 'category_encoders','Pint',
                     'string', 'pandas', 'numpy', 'torchvision', 'torch', 'albumentations','transformers',
                     'efficientnet_pytorch', 'matplotlib', 'seaborn', 'tqdm', 'requests', 'beautifulsoup4',
                     'ftfy', 'tensorflow', 'sympy'],
    
)