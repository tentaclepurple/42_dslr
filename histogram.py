import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

try:
    filesys = sys.argv[1]
except IndexError:
    print('No file provided')
    sys.exit(1)

try:
    df = pd.read_csv(filesys)
except FileNotFoundError:
    print('File not found')
    sys.exit(1)

courses=[ 
    'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
    'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
]

house_column = ['Hogwarts House']

df = df.dropna(subset=house_column+ courses) #Filtrar las filas con datos numericos validos

# Crear una figura con subplots
num_courses = len(courses)
cols = 4 
rows = (num_courses + cols - 1) // cols  # Calcula el número de filas necesarias

fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
axes = axes.flatten()  # Convertir a una lista plana para facilitar el acceso

for i, course in enumerate(courses):
    ax = axes[i]
    for house in df[house_column[0]].unique():
        subset = df[df[house_column[0]] == house]
        ax.hist(subset[course], bins=20, alpha=0.5, label=house)
    
    ax.set_title(course)
    ax.set_xlabel('Scores')
    ax.set_ylabel('Number of Students')
    ax.grid(True)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize='xx-large')
plt.tight_layout()

output_dir = 'histograms'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'all_courses_histograms.png')
plt.savefig(output_path)
plt.close()
