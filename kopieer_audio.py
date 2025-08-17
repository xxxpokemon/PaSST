import os
import shutil
import csv

# === Instellingen ===
input_dir = "esc50_extra/bahmei_sounds"     # map met originele bestanden
output_dir = "esc50_extra/bahmei_sounds_copy"   # map waar gekopieerd wordt
csv_path = "esc50_extra/meta/bahmei_sounds_copy.csv"         # output csv

# mapping van oude -> nieuwe labels
label_map = {
    23: 0,
    24: 1,
    28: 2,
    29: 3,
    31: 4,
    32: 5,
    38: 6,
    50: 7,
}

# mapping van oude labels naar tekstuele categorie
category_map = {
    23: "breathing",
    24: "coughing",
    28: "snoring",
    29: "drinking_sipping",
    31: "mouse_click",
    32: "keyboard_typing",
    38: "clock_tick",
    50: "chewing",
}

# maak output map aan
os.makedirs(output_dir, exist_ok=True)

# open CSV voor schrijven
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # header
    writer.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
    
    # doorloop alle bestanden
    for fname in os.listdir(input_dir):
        if not fname.endswith(".wav"):
            continue
        
        # voorbeeld: 1-20545-A-28.wav
        parts = fname[:-4].split("-")
        if len(parts) != 4:
            print(f"Onverwacht formaat: {fname}")
            continue
        
        fold, src_file, take, old_label_str = parts
        old_label = int(old_label_str)
        
        if old_label not in label_map:
            print(f"Label {old_label} niet in mapping, bestand {fname} wordt overgeslagen")
            continue
        
        new_label = label_map[old_label]
        category = category_map[old_label]
        
        # maak nieuwe bestandsnaam (categorie vervangen)
        new_fname = f"{fold}-{src_file}-{take}-{new_label}.wav"
        
        # kopieer bestand naar output_dir
        shutil.copy(os.path.join(input_dir, fname),
                    os.path.join(output_dir, new_fname))
        
        # schrijf rij naar csv
        writer.writerow([
            new_fname,         # filename
            fold,              # fold
            new_label,         # target
            category,          # category
            False,             # esc10
            src_file,          # src_file
            take               # take
        ])

print(f"Klaar! Bestanden staan in {output_dir}, CSV in {csv_path}")
