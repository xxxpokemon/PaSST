import os
import random as rd
import shutil

train_val_dir = "esc50_extra/bahmei_train_val"
test_dir = "bahmei_test"
complete_audio_dir = "esc50_extra/bahmei_sounds_copy"
target_classes = [0, 1, 2, 3, 4, 5, 6, 7]

num_of_total_classes = 8
samples_per_class = 40
ratio_test_total = 0.4
test_per_class = int(samples_per_class * ratio_test_total)
train_val_per_class = samples_per_class - test_per_class

# for file in os.listdir(complete_audio_dir):
#     category = file[-6:-4]
#     cat_int = int(category) if category[0] != '-' else int(category[1])
#     if cat_int in target_classes:
#         audio_path = os.path.join(complete_audio_dir, file)
#         shutil.copy(audio_path, test_dir)

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