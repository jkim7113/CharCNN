import string
import numpy as np
import pandas as pd
import random
from scipy import stats

def create_vocab_set():
    alphabet = set(list(string.ascii_lowercase))
    vocab_size = len(alphabet) + 1 # Incorporating 0 for characters not in vocab
    vocab = {}
    for letter in alphabet:
        vocab[letter] = ord(letter) - 96

    return vocab, vocab_size

# v, s = create_vocab_set()
# print(v, s)

def encode_data(x, maxlen=21):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen, 27)
    vocab, vocab_size = create_vocab_set()
    input_data = np.zeros((len(x), vocab_size, maxlen), dtype=np.int8)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c.lower(), 0)  # get index from vocab dictionary, if not in vocab, return 0
                input_data[dix, ix, counter] = 1
                counter += 1
    return input_data

def normalize(data, mean=0.5, std=(0.5/3)):
    return mean + (data - data.mean())*(std / data.std())
    
def load_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna()
    #data.drop(data.columns[[2, 5, 6, 7]], axis=1, inplace=True)
    # data["Word"] = data["Word"].str.lower().replace("'", "")
    
    # i_zscore = (data['I_Zscore'])
    # data["Normalized_HAL_Freq"] = stats.zscore(data['Log_Freq_HAL'])
    # inversed_freq = -data["Normalized_HAL_Freq"]

    # data["Difficulty"] = normalize((i_zscore + 2 * inversed_freq) / 2)
    
    i_zscore = stats.zscore(data['I_Mean_RT']) 
    freq_normalized = stats.zscore(data['Log_Freq_HAL'])
    inversed_freq = -freq_normalized
    difficulty = (i_zscore + inversed_freq) / 2

    data["Difficulty"] = (stats.rankdata(difficulty, method="average") / len(data))
    
    data.to_csv("data.csv", index=False)
    data_x = data["Word"]
    data_x = np.array(data_x)
    data_y = data["Difficulty"]
    data_y = np.array(data_y, dtype=np.float32)

    print(data.sort_values(by='Difficulty', ascending=False)[['Word', 'Difficulty']].head(20))
    return data_x, data_y

filepath = "./data/I159729-refined.csv"
x, y = load_data(filepath)
# print(x.shape, y.shape)
# # print(x.shape, y.shape)
# print(x[12968], y[12968])

# for i in range(10):
#      num = random.randint(0, len(x))
#      print(num, x[num], y[num])