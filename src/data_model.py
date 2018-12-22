import numpy as np
import pandas as pd
import time
import random
import os

random.seed(time.time())

class stockDataSet(object):
    def __init__(self,
                 stock_symbol,
                 inp=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price=True):
        self.stock_symbol = stock_symbol
        self.input_size = inp
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.normalized = normalized
        self.close_price = close_price

        #import CSV
        df = pd.read_csv(os.path.join("data", "%s.csv" % stock_symbol))

        if close_price:
            self.raw_seq = df['Close'].toList()
        else:
            self.raw_seq = [price for tup in df[['Open','Close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq)
        self.trainX, self.trainY, self.testX, self.testY = self._prepare_data(self.raw_seq)

    def info(self):
        return 'StockDataSet [%s] train: %d test: %d' % (self.stock_symbol, len(self.trainX), len(self.testY))

    def _prepare_data(self, seq):
        seq = [np.array(seq[i * self.input_size: (i+1) * self.input_size])
               for i in range(len(seq) // self.input_size)]
        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        x = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(x) * (1.0 - self.test_ratio))
        trainX, testX = x[:train_size], x[train_size:]
        trainY, testY = y[:train_size], y[train_size:]

        return trainX, trainY, testX, testY

    def generate_epoch(self, batch_size):
        num_batches = int(len(self.trainX))
        if batch_size * num_batches < len(self.trainX):
            num_batches += 1

        batch_ind = range(num_batches)
        random.shuffle(batch_ind)

        for k in batch_ind:
            batch_x = self.trainX[k * batch_size: (j+1)*batch_size]
            batch_y = self.trainY[k * batch_size: (j+1)*batch_size]
            assert set(map(len, batch_x)) == {self.num_steps}
            yield batch_x, batch_y

