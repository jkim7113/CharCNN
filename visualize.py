import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from preprocess import normalize
import matplotlib.pyplot as plt

data = pd.read_csv('./data/I159729-refined.csv')
data = data.dropna()

i_zscore = stats.zscore(data['I_Mean_RT'])
i_zaccuracy = -stats.zscore(data['I_Mean_Accuracy']) 
freq_normalized = stats.zscore(data['Log_Freq_HAL'])
inversed_freq = -freq_normalized

difficulty = (i_zscore + inversed_freq + i_zaccuracy) / 3
difficulty = difficulty - difficulty.min() + 1e-5

difficulty = normalize(stats.boxcox(difficulty)[0])

i_zscore = stats.zscore(data['I_Mean_RT']) 
freq_normalized = stats.zscore(data['Log_Freq_HAL'])
inversed_freq = -freq_normalized
difficulty = (i_zscore + inversed_freq) / 2

data["Difficulty"] = (stats.rankdata(difficulty, method="average") / len(data))
    

_, axis = plt.subplots(ncols=3)

sns.histplot(i_zscore, kde=True, bins=30, color="blue", ax=axis[0])
sns.histplot(data["Difficulty"], kde=True, bins=30, color="green", ax=axis[1])
sns.histplot(difficulty, kde=True, bins=30, color="red", ax=axis[2])

axis[0].axvline(x=i_zscore.mean(), color='red', linestyle='--', label="mean")
axis[0].axvline(x=np.median(i_zscore), color='green', linestyle='--', label="median")
axis[2].axvline(x=difficulty.mean(), color='red', linestyle='--', label=difficulty.mean())

# Add titles and labels
axis[0].set_title('Distribution of I-Zscore')
axis[0].set_xlabel('I-Zscore')
axis[0].set_ylabel('Frequency') 
axis[1].set_title('Distribution of Log Freq HAL Score')
axis[1].set_xlabel('Log_Freq_HAL')
axis[1].set_ylabel('Frequency') 
axis[2].set_title('Distribution of word difficulty (theta)')
axis[2].set_xlabel('Difficulty')
axis[2].set_ylabel('Frequency') 
# Show the plot
plt.show()
