# MNIST Digit Viewer

A Streamlit web application that displays random samples from the MNIST handwritten digits dataset.

## Features

- Interactive web interface built with Streamlit
- Load and display MNIST dataset (70,000 handwritten digits)
- User input for selecting digits (0-9)
- Display exactly 5 random samples of the selected digit
- Beautiful visualization with matplotlib
- Dataset statistics and information

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run mnist_viewer.py
```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`)

3. Select a digit (0-9) using the dropdown or number input

4. Click "Generate 5 Random Samples" to view random samples of that digit

## Dataset Information

The MNIST dataset contains:
- 60,000 training images
- 10,000 testing images
- 28×28 pixel grayscale images
- Handwritten digits from 0 to 9

## Citation

```
@article{lecun2010mnist,
         title={MNIST handwritten digit database},
         author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
         journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
         volume={2},
         year={2010}
}
```
