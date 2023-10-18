import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv(r'C:\Users\HUAWEI\Desktop\data.csv')

# Create a figure and an axis
fig, ax1 = plt.subplots()

# Plot the epoch - IOU line chart
ax1.plot(df['epoch'], df['iou'], color='blue', label='IOU')
ax1.set_xlabel('Epoch')  # Set the x-axis label
ax1.set_ylabel('IOU', color='blue')  # Set the y-axis label
ax1.tick_params(axis='y', labelcolor='blue')  # Set the tick parameters for y-axis

# Create a second y-axis
ax2 = ax1.twinx()  
ax2.plot(df['epoch'], df['loss'], color='red', label='Loss')  # Plot the epoch - loss line chart
ax2.set_ylabel('Loss', color='red')  # Set the y-axis label for the second y-axis
ax2.tick_params(axis='y', labelcolor='red')  # Set the tick parameters for the second y-axis

# Add a legend
fig.tight_layout()  # Adjust the layout to avoid overlap
lines, labels = ax1.get_legend_handles_labels()  # Get the lines and labels for the first axis
lines2, labels2 = ax2.get_legend_handles_labels()  # Get the lines and labels for the second axis
ax2.legend(lines + lines2, labels + labels2, loc=0)  # Display the legend

# Show the plot
plt.show()
