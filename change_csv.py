import pandas as pd
import os

# Stap 1: Laad bestandsnamen uit map (alleen de namen, geen paden)
dir = 'esc50_extra/train_val_audio'
filenames = set(os.listdir(dir))  # Zorgt voor snelle lookup

# Stap 2: Lees originele CSV in
csv_path = 'esc50_extra/meta/esc59.csv'
df = pd.read_csv(csv_path)

# Stap 3: Filter de rijen waarvan 'filename' in de map zit
df = df[df['filename'].isin(filenames)]

# Stap 4: Opslaan naar nieuwe CSV
df.to_csv('esc50_extra/meta/esc59_train.csv', index=False)
