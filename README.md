# OGRRE Data Cleaning

A Python package for cleaning OGRRE data.

## Installation
To install from source

1. Clone the repository
2. Navigate to the repository directory
3. Run the following command:

```bash
pip install .
```

To include the optional LLM cleaning dependencies:

```bash
pip install '.[llm]'
```

## Usage

To use the package, import the functions you need directly from the package.

```python
from ogrre_data_cleaning import CLEANING_FUNCTIONS, string_to_float, string_to_int
```

`CLEANING_FUNCTIONS` contains the cleaning functions available to callers by schema/configuration name.

Note: `llm_clean` requires the optional `llm` extra. Model checkpoints will be automatically downloaded when first using the `llm_clean` function.

## Overview

A final but necessary step in the OGRRE digitization project is to clean the data before the data is exported for its final use. In order to accomplish this, we have developed a Python package that utilizes a variety of techniques to clean the data. These techniques range from simple string replacements to more complex techniques such as using Large Language Models (LLMs) to clean the data. The main goals of the package are to:

1. Ensure the data is of the correct type. For example, dates should be of type `datetime` and booleans should be of type `bool`.
2. Standardize the data to ensure that it is in a consistent format. For example, all dates should be in the format `mm/dd/yyyy`.
3. Suggest corrections for the data where possible. For extracted text that reads "12-1/4'" should be converted to the float `12.25`.

## Functions for Families of Data Fields

The package is designed so that any data field can be cleaned using generic data conversion functions. For example, the `string_to_float` function will attempt to convert a string to a float. However, this function is basic and will not be able to make suggestions for corrections. For some data fields of low priority, this may not be an issue. However, for data fields of high priority, this may be a problem. Thus the package aims to provide more sophisticated functions for families of data fields which are of higher priority. Some examples of families of data fields are Hole Sizes and Dates. Since the valid values for hole sizes and dates are well known, we can use this information to provide suggestions for corrections. For Hole Sizes we assume that a hole size must be in 1/8" increments and that the only valid fractional values are 1/8", 1/4", 3/8", 1/2", 5/8", 3/4", 7/8", and 1". For dates we assume that oil and gas records should occur in the 20th and 21st centuries. With this information we can provide a function to clean Hole Size for a "Family of Data Fields" such as Casing Size, Open Hole Size, Tubing Size, etc. 

## LLM Functions and Future Work

In the current implementation, the majority of the functions used to clean the data are generic or use simple logic to clean the data and make suggestions. However, as more data becomes available, more and more sophisticated functions, such as the LLM functions, can be used to clean the data. This would be particularly useful in a high priority data fields that often require cleaning suggestions. It is known that it is difficult to **accurately** extract data from un-digitized oil and gas records due to human error, forms being filled out incorrectly, or limitations of OCR software. Thus as this technology improves, the LLM functions can assist in cleaning the data to ensure that the data is as accurate as possible.
