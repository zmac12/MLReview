# Recurrent Neural Network #

# RNN = ANN with a loop

# Recurrent means that the output at the current time step becomes input to next time step. Therefore it's not just the current input but what it remembers about the preceding element.

# Example: Considering the entire sentence as a whole rather than each individual word separately when forming a response.

# RNN should be able to see the words "But" and "Terrible Exciting" And realize that the meaning switches from negative to positive

# RNNs are made up of LSTM cells which maintain a cell state as well as a carry for ensuring that the signal (information in the form of a gradient) is not lost

# At each time step, LSTM cells consider the current word, the carry and the cell state.

# Tokenize (each word becomes its own feature)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, CuDNNLSTM

model = Sequential()

model.add(
    Embedding(input_dim=num_words,
    input_length = training_length,
    output_dim = 100,
    weights = [embedding_matrix],
    trainable = False,
    mask_zero = True)
)
model.add(Masking(mask_value=0.0))
model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(64, activation='relu'))