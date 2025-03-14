import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Create a figure with black background
fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
ax.set_facecolor('black')

# Remove axes
ax.set_axis_off()

# Set the limits of the plot
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# Create simplified continent shapes (blurred)
continents = [
    patches.Ellipse((2, 3), 1.5, 1, angle=30, alpha=0.2, facecolor='#333333'),
    patches.Ellipse((3.5, 2.5), 2, 1.2, angle=0, alpha=0.2, facecolor='#333333'),
    patches.Ellipse((5.5, 3), 1.8, 1, angle=-20, alpha=0.2, facecolor='#333333'),
    patches.Ellipse((7, 2.8), 1.5, 0.8, angle=15, alpha=0.2, facecolor='#333333'),
    patches.Ellipse((3, 4), 1.2, 0.6, angle=10, alpha=0.2, facecolor='#333333'),
]

for continent in continents:
    ax.add_patch(continent)

# Add caption text
ax.text(5, 0.5, 'Each light represents casualties in a conflict zone', 
        horizontalalignment='center', color='#888888', fontsize=8)

# Define casualty locations
locations = [
    {'name': 'Iraq', 'x': 5.5, 'y': 3.0, 'start_frame': 10},
    {'name': 'Syria', 'x': 5.7, 'y': 3.3, 'start_frame': 20},
    {'name': 'Afghanistan', 'x': 6.2, 'y': 3.0, 'start_frame': 30},
    {'name': 'Ukraine', 'x': 5.3, 'y': 3.7, 'start_frame': 40},
    {'name': 'Yemen', 'x': 5.9, 'y': 2.7, 'start_frame': 50},
    {'name': 'Bosnia', 'x': 4.9, 'y': 3.5, 'start_frame': 60}
]

# Create the dots (initially invisible)
dots = []
for loc in locations:
    dot = ax.scatter(loc['x'], loc['y'], s=100, color='white', alpha=0, 
                    edgecolor=None, zorder=10)
    dots.append({'dot': dot, 'start_frame': loc['start_frame']})

# Animation update function
def update(frame):
    for i, dot_info in enumerate(dots):
        dot = dot_info['dot']
        start_frame = dot_info['start_frame']
        
        # Calculate alpha value
        if frame >= start_frame and frame < start_frame + 15:
            # Fade in and then out over 15 frames
            if frame - start_frame < 5:
                alpha = (frame - start_frame) / 5  # Fade in
            else:
                alpha = 1 - (frame - start_frame - 5) / 10  # Fade out
            dot.set_alpha(alpha)
        elif frame >= start_frame + 15 or frame < start_frame:
            dot.set_alpha(0)
    
    return [dot_info['dot'] for dot_info in dots]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=90, interval=100, blit=True)

# Save as mp4
writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save('war_casualties_map.mp4', writer=writer)

print("Video saved as 'war_casualties_map.mp4'")