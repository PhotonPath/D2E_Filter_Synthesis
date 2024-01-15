import matplotlib.pyplot as plt
import numpy as np
import yaml

Band = 'C'
Lattice_Order = 5
gains = np.linspace(10, 30, 5).astype(int)
dLs = np.linspace(30.7, 30.7, 1)
dL0s = np.linspace(5, 30, 26)

# file_name_save = f"Band {Band} - Order {Lattice_Order} - High Range ({dLs[0]} - {dLs[-1]})"
file_name_save = "Band C - Order 5 - Variable dL - High Range (30.7- 30.7) - dL0 (4.0 - 30.0)"
file_path_save = "Data C-Band/"

with open(file_path_save + file_name_save + ".yaml", 'r') as file:
    load_data = yaml.load(file, yaml.BaseLoader)
max_ripples_sum = load_data['max_ripples_sum']
max_ripples = load_data['max_ripples']

color_start = np.array([1, 0, 0])
color_end = np.array([0, 0, 1])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
plt.xlabel("dL [um]", fontsize=23)
ax1.set_title(f"Band {Band} - Order {Lattice_Order}", fontsize=23)
colors = [color_start * p / len(gains) + color_end * (1 - p / len(gains)) for p in range(len(gains))]
for idL0, dL0 in enumerate(dL0s):
    for idL, dL in enumerate(dLs):
        code_max = f"_dL0{dL0}_dL{dL}"
        ax2.scatter(dL0, float(max_ripples_sum[code_max]), color=colors[0])
        for idg, gain in enumerate(gains):
            code = f"gain{gain}_dL0{dL0}_dL{dL}"
            if idL0 == 0:
                ax1.scatter(dL0, float(max_ripples[code]), color=colors[idg], label=f"Gain {gain}")
            else:
                ax1.scatter(dL0, float(max_ripples[code]), color=colors[idg])
for idx in range(2):
    if idx == 0:
        ax1.set_ylabel("Max Ripple [dB]", fontsize=23)
        ax1.grid()
        plt.xticks(fontsize=23)
        ax1.tick_params(axis='y', labelsize=23)
        ax1.legend(fontsize=14)
    elif idx == 1:
        ax2.set_ylabel("Sum Max Ripple [dB]", fontsize=23)
        ax2.grid()
        ax2.tick_params(axis='y', labelsize=23)

plt.show()