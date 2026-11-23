import pyFC
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.ion()

# Create a LogNormalFractalCube with specified parameters
fc = pyFC.LogNormalFractalCube(ni=3, nj=128, nk=128, kmin=10, mean=1)

# Generate the cube
fc.gen_cube()

# Plot the field statistics
pyFC.plot_field_stats(fc, scaling='log', vmin=-2.1, vmax=2.1, cmap=cm.jet)

# Keep the plot window open
plt.show()