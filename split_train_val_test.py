import os
import random as rd
import shutil

train_val_dir = "esc50_extra/train_val_audio"
test_dir = "test_audio"
complete_audio_dir = "esc50_extra/audio_32k"
target_classes = [23, 24, 28, 29, 31, 32, 38, 50, 51, 52, 53, 54, 55, 56, 57, 58]

num_of_total_classes = 59
samples_per_class = 40
ratio_test_total = 0.4
test_per_class = int(samples_per_class * ratio_test_total)
train_val_per_class = samples_per_class - test_per_class

test_dir_counters = [0] * num_of_total_classes
total_counters = [0] * num_of_total_classes

for file in os.listdir(complete_audio_dir):
    category = file[-6:-4]
    cat_int = int(category) if category[0] != '-' else int(category[1])
    if total_counters[cat_int] >= samples_per_class:
        continue
    audio_path = os.path.join(complete_audio_dir, file)
    if (test_dir_counters[cat_int] < test_per_class and rd.random() < ratio_test_total) or \
        (samples_per_class - total_counters[cat_int] <= test_per_class - test_dir_counters[cat_int]):

        test_dir_counters[cat_int] += 1
        if cat_int in target_classes:
            shutil.copy(audio_path, test_dir)
    else:
        shutil.copy(audio_path, train_val_dir)
    total_counters[cat_int] += 1

print('Added audiofiles to test directory from categories:')
for i in range(len(test_dir_counters)):
    if i in target_classes and test_dir_counters[i] != 0:
        print(f"{i} ({test_dir_counters[i]}), ", end='')
print()
print('Added audiofiles to train/val directory from categories:')
for i in range(len(total_counters)):
    if total_counters[i] - test_dir_counters[i] != 0:
        print(f"{i} ({total_counters[i] - test_dir_counters[i]}), ", end='')
print()