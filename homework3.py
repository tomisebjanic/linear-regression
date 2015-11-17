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
        # self.test_data = self.parse_test_data()

    def linekey(self, d):
        return tuple(d[2:4])

    def parse_training_data(self):
        f = gzip.open("ucni-podatki/train.csv.gz", "rt", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        dataY = defaultdict(list)
        dataX = defaultdict(list)
        raw = [d for d in reader]
        print("===Parsing train_data===")
        i = 465562
        for d in raw:
            print(i)
            id_trip = tuple(d[2:4])
            dataY[id_trip].append(float(lpputils.tsdiff(d[8], d[6])))
            dataX[id_trip].append([float(time.mktime(lpputils.parsedate(d[6]).timetuple()))])
            i -= 1

        return [dataY, dataX]

    # def parse_test_data(self):
    #     f = gzip.open("ucni-podatki/test.csv.gz", "rt", encoding="utf-8")
    #     reader = csv.reader(f, delimiter="\t")
    #     next(reader)
    #     data = defaultdict(list)
    #     raw = [d for d in reader]
    #     print("Parsing test_data")
    #     for d in raw:
    #         data[average.linekey(d)].append([float(time.mktime(lpputils.parsedate(d[6]).timetuple()))])
    #
    #     return data

    def do_it(self):
        models = defaultdict(list)
        for m in self.train_data:
            X = np.array(self.train_data[m])
            Xsp = scipy.sparse.csr_matrix(X)
            y = np.array(self.y[m])
            lr = linear.LinearLearner(lambda_=0.05)
            models[m] = lr(Xsp, y)

        f = gzip.open("ucni-podatki/test.csv.gz", "rt", encoding="utf-8")
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        raw = [d for d in reader]
        f.close()
        fo = open("result.txt", "wt")
        i = 23043
        print("===Predicting===")
        for line in raw:
            print(i)
            id_trip = tuple(line[2:4])
            departure_time = float(time.mktime(lpputils.parsedate(line[6]).timetuple()))
            trip_time = models[id_trip](np.array(float(departure_time)))
            arrival_time = lpputils.tsadd(datetime.datetime.fromtimestamp(timestamp=int(departure_time)).strftime("%Y-%m-%d %H:%M:%S.%f"), trip_time)
            print('Departure:', datetime.datetime.fromtimestamp(timestamp=int(departure_time)).strftime("%Y-%m-%d %H:%M:%S.%f"), 'Trip Len:', trip_time, 'Arrival:', arrival_time)
            fo.write(arrival_time + "\n")
            i -= 1
        fo.close()





hw = Homework3().do_it()
