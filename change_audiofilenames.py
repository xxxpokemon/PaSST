import os
import re

dir = "wrong_name_audio"
names = ['Ritsel', 'Bestek', 'Pen', 'Schraap', 'Snuif', 'Knak', 'Achtergrond', 'Neurie']
name_codes = {'Ritsel': 1, 'Bestek': 2, 'Pen': 3, 'Schraap': 4, 'Snuif': 5, 'Knak': 6, 'Achtergrond': 7, 'Neurie': 8}

name_pattern = "|".join(re.escape(name) for name in names)
pattern = re.compile(rf"^({name_pattern})(\d{{1,2}})\.wav$")

def legal_file_name(filename):
    match = pattern.match(filename)
    if match:
        return match.group(1), int(match.group(2))
    return "", 0

for file in os.listdir(dir):
    old_path = os.path.join(dir, file)
    if not os.path.isfile(old_path):
        continue

    num = legal_file_name(file)
    if num[1] != 0:
        n = str(num[1]) if num[1] > 9 else "0" + str(num[1])
        new_name = f"6-300{name_codes[num[0]]}{n}-A-{name_codes[num[0]] + 50}.wav"
        new_path = os.path.join(dir, new_name)
        os.rename(old_path, new_path)
        print(f"{file} => {new_name}")