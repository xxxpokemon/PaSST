names = ['rustling', 'utensils', 'pen_clicking', 'clearing_throat', 'sniffing', 'cracking_joints', 'finger_ticking', 'humming']
for i in range(1,9):
    for ii in range(1,41):
        num = str(ii) if ii > 9 else "0" + str(ii)
        line = f"6-300{i}{num}-A-{i + 50}.wav,6,{i + 50},{names[i - 1]},False,300{i}{num},A"
        print(line)