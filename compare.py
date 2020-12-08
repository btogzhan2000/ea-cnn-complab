import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker

df = pd.read_csv('data_graph.csv')
df2 = pd.read_csv('data_graph2.csv')

locator = matplotlib.ticker.MultipleLocator(2)
plt.gca().xaxis.set_major_locator(locator)

plt.xlabel('Generation number')
plt.ylabel('Accuracy')

plt.plot(range(1,20), 'mean', data=df, label="mean without inception")
# plt.plot(range(1,20), 'min',  data=df, label="min without inception")
# plt.plot(range(1,20), 'max',  data=df, label="max without inception")

plt.plot(range(1,20), 'mean', data=df2, label="mean with inception")
# plt.plot(range(1,20), 'min',  data=df2, label="min with inception")
# plt.plot(range(1,20), 'max',  data=df2, label="max with inception")

plt.legend()
plt.show()