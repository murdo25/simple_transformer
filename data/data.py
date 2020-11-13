from tqdm import tqdm


class Data:
    def __init__(self, data_path, BATCH_SIZE=32):
        self.BATCH_SIZE = BATCH_SIZE
        self.vocab = {}
        self.train = []
        self.test = []

        self.data = self.clean(data_path)
        print(len(self.data))

    def clean(self, data_path):
        open_file = open(data_path, 'r').readlines()

        lines = []
        # Strips the newline character 
        for line in tqdm(open_file): 
            line = line.strip()
            lines += [line]
            # print("Line {}: {}".format(count, line))
            for c in line:
                if c in self.vocab:
                    self.vocab[c] += 1
                else:
                    self.vocab[c] = 0
        return lines


# d = Data('mini_train_set.txt') 
# d = Data('full_train_set.txt') 

# print("VOCAB:", d.vocab)
# print("NUM TOKENS", len(d.vocab))
# print("SIZE DATASET:", len(d.data))

# print("sample:",d.data[0])
