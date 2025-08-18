import pandas as pd
import os

# Stap 1: Laad bestandsnamen uit map (alleen de namen, geen paden)
dir = 'bahmei_test'
filenames = set(os.listdir(dir))  # Zorgt voor snelle lookup

# Stap 2: Lees originele CSV in
csv_path = 'esc50_extra/meta/bahmei_sounds.csv'
df = pd.read_csv(csv_path)

# Stap 3: Filter de rijen waarvan 'filename' in de map zit
df = df[df['filename'].isin(filenames)]

# Stap 4: Opslaan naar nieuwe CSV
df.to_csv('esc50_extra/meta/bahmei_test.csv', index=False)
