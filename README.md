# pytorch-tabr

## Overview

pytorch-tabr is a Python package that provides a PyTorch implementation of TabR, a deep learning model for tabular data. This package allows for easy and efficient modeling of both classification and regression tasks using tabular data. It includes support for various kinds of embeddings and customizations to cater to different types of tabular datasets.

## Features

- **TabR Model**: Core deep learning model for tabular data.
- **Classification and Regression**: Support for both classification (`TabRClassifier`) and regression (`TabRRegressor`) tasks.
- **Custom Embeddings**: Supports categorical, numerical, and other types of embeddings.
- **Efficient Handling of Data**: Efficient data loaders and utilities for handling tabular data.

## Installation

```bash
pip install pytorch-tabr
```

## Usage

### Basic example

```python
from tabr import TabRClassifier, TabRRegressor

# For a classification task
classifier = TabRClassifier(cat_indices=[0, 2], cat_cardinalities=[3, 5])
# Training and prediction...

# For a regression task
regressor = TabRRegressor(cat_indices=[1, 3], cat_cardinalities=[4, 2])
# Training and prediction...
```

## API Overview
- **TabRClassifier**: Model for classification tasks.
- **TabRRegressor**: Model for regression tasks.
- **TabR**: Base module implementing the TabR architecture.
