import matplotlib.pyplot as plt

# Data
barrier_types = ['Personal Barriers', 'Economic Barriers', 'Social Barriers']
percentages = [52, 37, 21]

# Create the bar plot
plt.figure(figsize=(7, 5))
bars = plt.bar(barrier_types, percentages, color=['#66c2a5', '#fc8d62', '#8da0cb'])

# Add percent labels on top of each bar
for bar, percentage in zip(bars, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{percentage}%", ha='center', va='bottom', fontsize=11)

plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Barriers to Seeking Eye Treatment', fontsize=14)

# Minimal style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(0, max(percentages) + 10)

plt.tight_layout()
plt.savefig('barriers_to_eye_treatment.png', format='png', dpi=300)
plt.show()