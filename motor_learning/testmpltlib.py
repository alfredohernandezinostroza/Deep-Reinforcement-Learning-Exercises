import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
plt.show(block=False)
plt.pause(5)  # Wait 5 seconds