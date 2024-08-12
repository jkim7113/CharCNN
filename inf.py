import torch
import preprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

save_path = "./model/charCNN.pt"

torch.manual_seed(123) 
torch.backends.cudnn.deterministic = True # for reproducibility
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Loading model...')
model = torch.load(save_path)
model.to(device)
model.eval()

inputs = ["America", "Korea", "John", "Jane", 
            "Susan", "David", "Nicholas", "Lucas", "JCPenny", 
            "speedy", "fast", "rapid", "agile", "swift", 
            "expeditious", "alacritous", "simultaneous", "true", 
            "false", "cost-effective", "open-minded", "self-conscious"
        ]

inputs = input("Enter words: ").split()

encoded = preprocess.encode_data(inputs)
encoded = torch.from_numpy(encoded).to(device).float()

with torch.no_grad():
    outputs = model(encoded)
    for i in range(len(inputs)):
        print(f'{inputs[i]}: {outputs[i].item()}')

def show_distribution():
    data = pd.read_csv('./data/sat-words.csv')
    data = data.dropna()
    #data.drop(data.columns[[1, 2, 4, 6, 7, 8]], axis=1, inplace=True)
    data["Word"] = data["Word"].str.lower().replace("'", "")
    words = data["Word"].tolist()
    
    encoded = preprocess.encode_data(words)
    encoded = torch.from_numpy(encoded).to(device).float()
    
    with torch.no_grad():
        outputs = model(encoded)
        data["Difficulty"] = outputs.cpu().detach().numpy()
        print(data["Difficulty"].mean()) # 0.4998382
        data.to_csv("irt_sample.csv", index=False)
        sns.histplot(data["Difficulty"], kde=True, bins=30, color="blue")
        plt.title('Distribution of Inferred Word Difficulty')
        plt.xlabel('Word Difficulty')
        plt.ylabel('Frequency') 
        plt.show()
    
    print(data.sort_values(by='Difficulty', ascending=False)[['Word', 'Difficulty']].head(20))
#show_distribution()