#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 6:
    print("Usage: python script.py <start_image> <end_image> <file1_to_read_name> <file2_to_read_name> <file3_to_read_name>")
    sys.exit(1)

first_image = int(sys.argv[1])
last_image = int(sys.argv[2])
image_range = range(first_image, last_image + 1)
filename1 = str(sys.argv[3])
filename2 = str(sys.argv[4])
filename3 = str(sys.argv[5])

dc_L = []
dc_U = []

col_profile_L=[]
col_profile_U=[]

with open(filename1, "r") as file:
    for line in file:
        num1, num2 = map(float, line.strip().split())
        dc_L.append(num1)
        dc_U.append(num2)

with open(filename2, "r") as file:
    for line in file:
        num = line.strip()
        col_profile_L.append(float(num))

with open(filename3, "r") as file:
    for line in file:
        num = float(line.strip())
        col_profile_U.append(num)

plt.plot(image_range, dc_L, label="Dark Current L-side")
plt.plot(image_range, dc_U, label="Dark Current U-side")
plt.xlabel("image id")
plt.ylabel("Dark Current (e$^-$/pixel)")
plt.legend()
plt.show()

plt.bar(np.arange(np.size(col_profile_L)),col_profile_L,3,color='teal')
plt.xlabel('column number')
plt.ylabel('counts')
#plt.yscale('log')
plt.legend()
plt.tick_params(axis='both', which='both', length=10, direction='in')
plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
plt.title('L-side column charge profile')
plt.show()

plt.bar(np.arange(np.size(col_profile_U)),col_profile_U,3,color='teal')
plt.xlabel('column number')
plt.ylabel('counts')
#plt.yscale('log')
plt.legend()
plt.tick_params(axis='both', which='both', length=10, direction='in')
plt.grid(color='grey', linestyle=':', linewidth=1, which='both')
plt.title('U-side column charge profile')
plt.show()

import os
os.remove(filename1)
os.remove(filename2)
os.remove(filename3)
