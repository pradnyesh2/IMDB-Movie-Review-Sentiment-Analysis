<h1>IMDB Movie Review Sentiment Analysis using RNN</h1>

This project demonstrates sentiment analysis of movie reviews from the IMDB dataset using a Recurrent Neural Network (RNN). The model is trained to classify reviews as either positive or negative.

<h2>Overview</h2>

This project utilizes the internal IMDB dataset provided by TensorFlow Keras. The dataset consists of movie reviews labeled as positive or negative. An RNN model, specifically an Embedding layer followed by a SimpleRNN layer and a Dense output layer, is implemented to learn the sentiment expressed in the text data.

This application has also been deployed on Streamlit Community Cloud, allowing users to interact with the trained model through a web interface.

**Live Demo:** [https://imdb-movie-review-sentiment-analysis-pradnyesh.streamlit.app/](https://imdb-movie-review-sentiment-analysis-pradnyesh.streamlit.app/)

## Technologies Used

* **Python:** Programming language
* **TensorFlow:** Open-source machine learning framework
* **Keras:** High-level API for building and training neural networks (part of TensorFlow)
* **NumPy:** Library for numerical computations
* **Streamlit:** Python library for creating interactive web applications

## Dataset

* **IMDB Dataset:** A dataset of 50,000 movie reviews from the Internet Movie Database (IMDb) for sentiment analysis. The dataset is split into 25,000 reviews for training and 25,000 reviews for testing. Each review is labeled as either positive (1) or negative (0).
* **Source:** This project uses the built-in IMDB dataset accessible through `tensorflow.keras.datasets.imdb`.

## Model Architecture

The sentiment analysis model consists of the following layers:

1.  **Embedding Layer:** This layer converts integer-encoded words into dense vector representations. It learns the semantic relationships between words during training.
    * **Input Dimension:** Determined by the vocabulary size of the dataset.
    * **Output Dimension:** The size of the dense word vectors (hyperparameter).

2.  **SimpleRNN Layer:** A basic recurrent layer that processes the sequence of word embeddings. It maintains an internal state that captures information about the preceding elements in the sequence.
    * **Units:** The number of hidden units in the RNN layer (hyperparameter).

3.  **Dense Layer:** A fully connected layer with a single output unit and a sigmoid activation function. This layer outputs the probability of the review being positive (between 0 and 1).
    * **Activation Function:** Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$) to produce a probability.

## Getting Started

### Prerequisites

* Python 3.12
* TensorFlow (>= 2.x)
* NumPy
* Streamlit (if you want to run the web application locally)

You can install the necessary libraries using pip:

```bash
pip install tensorflow numpy streamlit
