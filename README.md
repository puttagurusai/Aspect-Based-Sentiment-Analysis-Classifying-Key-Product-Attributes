# BERT Layer Freezing with Trainable LSTM for Text Classification

## Introduction
This project explores a hybrid approach to text classification using a pretrained BERT model where its layers are frozen, and a custom LSTM network is trained on top. By leveraging BERT's feature extraction capabilities and combining it with the sequential modeling power of LSTM, this approach aims to balance performance and computational efficiency. The notebook implements this architecture, processes datasets, and provides tools for fine-tuning.

## Description
The notebook focuses on:
- Freezing all layers of a pretrained BERT model.
- Training a custom LSTM network on top of BERT's output.
- Implementing multi-label classification for textual data.
- Visualizing the training process and evaluating performance on validation data.

This architecture is particularly useful when you want to utilize BERT's pretrained embeddings without the computational cost of fine-tuning its layers, while still capturing sequential dependencies in the data with an LSTM layer.

## Features
- **Data Preprocessing**: Tokenization using `BertTokenizer` and data transformation for multi-label classification.
- **Model Architecture**: Combines BERT as a feature extractor with a trainable Bidirectional LSTM.
- **Training Optimization**: Uses techniques like model checkpoints and validation splitting.
- **Visualization**: Includes plots for training loss and accuracy.

## Installation

### Dependencies
Ensure you have the following Python libraries installed:
```
from google.colab import drive
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer, TFBertForSequenceClassification, TFBertModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
```

To install missing dependencies, use:
```bash
pip install tensorflow transformers scikit-learn matplotlib pandas
```

## Usage

### Data Preparation
The dataset is expected to have textual data and associated labels. For example:
- Input text column: `X_lable`
- Label column: `Y_lable`

Split the data into training and validation sets:
```python
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### Model Training
Load the notebook, configure paths to your dataset, and run the cells in sequence. Below is a snippet showing dataset loading and preprocessing:
```python
import pandas as pd

df_train = pd.read_excel("path to train data.xlsx")
df_test = pd.read_excel("path to test data.xlsx")

```

Start training by running the notebook cells. Model checkpoints will save the best-performing model based on validation loss.

## Libraries Used
The following libraries are utilized in this project:

1. **Google Colab Drive**  
   - `from google.colab import drive`  
   - Used for accessing and managing files in Google Colab.  

2. **Scikit-learn**  
   - `from sklearn.model_selection import train_test_split`  
   - Used for splitting the dataset into training and validation sets.  

3. **TensorFlow and Keras**  
   - `from tensorflow.keras.callbacks import ModelCheckpoint`  
   - `from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional`  
   - `from tensorflow.keras.models import Sequential`  
   - `from tensorflow.keras.preprocessing.sequence import pad_sequences`  
   - `from tensorflow.keras.preprocessing.text import Tokenizer`  
   - Used for building and training the model, including callbacks and preprocessing.  

4. **Transformers**  
   - `from transformers import BertTokenizer, TFBertForSequenceClassification, TFBertModel`  
   - Used for utilizing pretrained BERT models for text classification.  

5. **Matplotlib**  
   - `import matplotlib.pyplot as plt`  
   - Used for visualizing training progress and results.  

6. **NumPy**  
   - `import numpy as np`  
   - Used for numerical operations and data handling.  

7. **Pandas**  
   - `import pandas as pd`  
   - Used for data manipulation and preprocessing.  

### Additional Notes
If your notebook contains specific versions of these libraries (e.g., for compatibility), you can include those in the README or a `requirements.txt` file. For example:
```bash
pip install tensorflow==2.12.0 transformers==4.28.1 pandas==1.5.3 numpy==1.23.5 matplotlib==3.6.2 scikit-learn==1.2.2
```

## Examples
Here are some examples of visual outputs and results:
- **Loss and Accuracy Plots**: Graphs showing the training progress.



![image](https://github.com/user-attachments/assets/b94f08ab-2118-4b39-9699-53b1add62421)




## Contribution
Feel free to contribute by submitting pull requests or reporting issues. For major changes, please open an issue first to discuss your ideas.

