import gzip
import csv
from collections import defaultdict
import average
import lpputils
import scipy
import scipy.sparse as sp
import linear
import numpy as np
import time
import datetime

# Registration	Driver ID	Route	Route Direction	Route description	First station	Departure time	Last station	Arrival time

class Homework3:

    def __init__(self):
        parsed_data = self.parse_training_data()
        self.y = parsed_data[0]
        self.train_data = parsed_data[1]
        self.test_data = self.parse_test_data()

    def parse_training_data(self):
        f = gzip.open("ucni-podatki/train.csv.gz", "rt", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        dataY = defaultdict(list)
        dataX = defaultdict(list)
        raw = [d for d in reader]
        print("Parsing train_data")
        for d in raw:
            dataY[average.linekey(d)].append(float(lpputils.tsdiff(d[8], d[6])))
            # dataX[average.linekey(d)].append([float(time.mktime(lpputils.parsedate(d[6]).timetuple())), float(time.mktime(lpputils.parsedate(d[8]).timetuple()))])
            dataX[average.linekey(d)].append([float(time.mktime(lpputils.parsedate(d[6]).timetuple()))])

        return [dataY, dataX]

    def parse_test_data(self):
        f = gzip.open("ucni-podatki/test.csv.gz", "rt", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        data = defaultdict(list)
        raw = [d for d in reader]
        print("Parsing test_data")
        for d in raw:
            data[average.linekey(d)].append([float(time.mktime(lpputils.parsedate(d[6]).timetuple()))])

        return data

    def do_it(self):
        models = defaultdict(list)
        for m in self.train_data:
            X = np.array(self.train_data[m])
            Xsp = scipy.sparse.csr_matrix(X)
            y = np.array(self.y[m])
            lr = linear.LinearLearner(lambda_=0.05)
            models[m] = lr(Xsp, y)

        i = 0
        f = open("result.txt", "wt")
        for a in self.test_data:
            print(''.join(a))
            for b in self.test_data[a]:
                # print(b)
                # if a in self.train_data:
                #     # print(models[a](np.array(float(b[0]))), i)
                #     trip_len = models[a](np.array(float(b[0])))
                #     arrival_time = lpputils.tsadd(datetime.datetime.fromtimestamp(timestamp=int(b[0])).strftime("%Y-%m-%d %H:%M:%S.%f"), trip_len)
                #     print('Departure:', datetime.datetime.fromtimestamp(timestamp=int(b[0])).strftime("%Y-%m-%d %H:%M:%S.%f"), 'Trip Len:', trip_len, 'Arrival:', arrival_time)
                #     # with open("test.txt", "wt") as out_file:
                #     #     out_file.write("This Text is going to out file\nLook at it and see!")
                #     f.write(arrival_time + "\n")

                trip_len = models[a](np.array(float(b[0])))
                arrival_time = lpputils.tsadd(datetime.datetime.fromtimestamp(timestamp=int(b[0])).strftime("%Y-%m-%d %H:%M:%S.%f"), trip_len)
                print('Departure:', datetime.datetime.fromtimestamp(timestamp=int(b[0])).strftime("%Y-%m-%d %H:%M:%S.%f"), 'Trip Len:', trip_len, 'Arrival:', arrival_time)
                f.write(arrival_time + "\n")
                i+=1
        f.close()
            # # print(a, len(self.test_data[a]), len(self.y[a]))
            # if len(self.test_data[a]) > 0 and len(self.y[a]) > 0:
            # print(models[a](np.array(self.test_data[a])))


hw = Homework3().do_it()
