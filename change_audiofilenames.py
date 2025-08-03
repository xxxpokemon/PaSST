import os
import re

dir = "wrong_name_audio"
names = ['Ritsel', 'Bestek', 'Pen', 'Schraap', 'Snuif', 'Knak', 'Tik', 'Neurie']
name_codes = {'Ritsel': 1, 'Bestek': 2, 'Pen': 3, 'Schraap': 4, 'Snuif': 5, 'Knak': 6, 'Tik': 7, 'Neurie': 8}

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

    info = legal_file_name(file)
    if info[1] != 0:
        num = str(info[1]) if info[1] > 9 else "0" + str(info[1])
        new_name = f"6-300{name_codes[info[0]]}{num}-A-{name_codes[info[0]] + 50}.wav"
        new_path = os.path.join(dir, new_name)
        os.rename(old_path, new_path)
        print(f"{file} => {new_name}")