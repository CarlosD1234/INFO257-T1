from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from sklearn.preprocessing import LabelEncoder

class PokemonImages(Dataset):
    
    def __init__(self, data_path):
        metadata_path = Path(data_path) / "pokedex_Ver9.csv"
        special_parsers = {'Height': lambda x: float(x[:-1]), 'Weight': lambda x: float(x[:-2])}
        df = pd.read_csv(metadata_path, converters=special_parsers).set_index("No")
        self.categories = sorted(df["Type1"].unique())
        self.attribute_names = ['Height', 'Weight', 'HP', 'Attack', 'Defense', 'Speed']
        le = LabelEncoder().fit(self.categories)
        self.images, self.labels, self.names, self.attributes = [], [], [], []
        for no in range(1, 610):
            image_path = Path(data_path) / 'pokemon_jpg' / f'{no}.jpg'
            if not image_path.exists():
                continue
            image = torchvision.io.read_image(str(image_path)).type(torch.float32)/255.
            self.images.append(image)
            metadata = df.loc[no]
            if metadata.ndim > 1:
                metadata = metadata.iloc[0]
            self.names.append(metadata['Name'])
            self.labels.append(le.transform([metadata['Type1']])[0])
            self.attributes.append(torch.from_numpy(metadata[self.attribute_names].values.astype('float32')))
            
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.names[idx], self.attributes[idx]
    
    def __len__(self):
        return len(self.labels)

