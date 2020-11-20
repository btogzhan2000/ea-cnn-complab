import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
#with open("data_graph.csv", "w") as csv_file:
csv_file = open("data_graph.csv", "w")
fieldnames = ['gen_no', 'mean', 'min', 'max']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()
writer.writerow({'gen_no':"00", 'mean':0.58164, 'min':0.36733, 'max':0.62922})

gen_no = 1
while gen_no < 19:
    if gen_no < 10:
        gen_no_str = "0"+str(gen_no)
    else:
        gen_no_str = str(gen_no)

    f = open("populations/after_"+ gen_no_str +".txt", "r")

    sum = 0
    count = 0
    max1 = 0
    min1 = 1
    for line in f.readlines():
        record_indi = float(line.split("=")[1])
        if record_indi != 0.00000:
            sum += record_indi
            count += 1
            max1 = max(max1, record_indi)
            min1 = min(min1, record_indi)
    mean = sum / count
    #with open("data_graph.csv", "w") as csv_file:
        #writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writerow({'gen_no':gen_no_str, 'mean':mean, 'min':min1, 'max':max1})

    print(gen_no_str) 
    print(mean)
    print(min1)
    print(max1)
    gen_no += 1
    f.close()

csv_file.close()

df = pd.read_csv('data_graph.csv')

locator = matplotlib.ticker.MultipleLocator(2)
plt.gca().xaxis.set_major_locator(locator)

plt.xlabel('Generation number')
plt.ylabel('Accuracy')

#plt.plot(range(1,20), 'mean', data=df, label="mean")
#plt.plot(range(1,20), 'min',  data=df, label="min")
plt.plot(range(1,20), 'max',  data=df, label="max")

plt.legend()
plt.show()
