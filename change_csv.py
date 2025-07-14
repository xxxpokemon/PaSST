import pandas as pd
import os

# Stap 1: Laad bestandsnamen uit map (alleen de namen, geen paden)
train_map = 'esc50_extra/train_val_audio'
train_filenames = set(os.listdir(train_map))  # Zorgt voor snelle lookup

# Stap 2: Lees originele CSV in
csv_path = 'esc50_extra/meta/esc52.csv'
df = pd.read_csv(csv_path)

# Stap 3: Filter de rijen waarvan 'filename' in de map zit
df_train = df[df['filename'].isin(train_filenames)]

# Stap 4: Opslaan naar nieuwe CSV
df_train.to_csv('esc50_extra/meta/esc52_40pct.csv', index=False)
