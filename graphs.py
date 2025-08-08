import matplotlib.pyplot as plt


x = [3, 9, 14]
y = [2, 7, 30]

plt.plot(x, y, c='red', linewidth=2, label="Line 1")

x2 = [1, 15, 18]
y2 = [0, 3, 12]

plt.plot(x2, y2, c="green", linewidth=0.5, label="Line 2", linestyle="dashed",
         marker='o', markerfacecolor="blue")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Two lines")

# plt.ylim(1, 10)
# plt.xlim(0,30)

plt.legend()

plt.show()