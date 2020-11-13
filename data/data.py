from tqdm import tqdm
import torchtext
from torchtext.data.utils import get_tokenizer

class Data:
    def __init__(self, data_path, BATCH_SIZE=32):
        self.BATCH_SIZE = BATCH_SIZE
        self.vocab = {}
        self.train = []
        self.test = []
        self.eval = []
        self.TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

        self.data = self.clean(data_path)
        self.TEXT.build_vocab(self.data)
        print(self.TEXT.vocab.stoi)
        self.max_seq_len = self.find_max_seq_len()

        print(len(self.data))

    def clean(self, data_path):
        open_file = open(data_path, 'r').readlines()

        lines = []
        # Strips the newline character 
        for line in tqdm(open_file): 
            line = line.strip()
            lines += [line]
        return lines

    def getData(self):
        return self.train, self.test, self.eval
    
    def find_max_seq_len(self):
        longest = 0
        for d in self.data:
            if d.shape[1] > longest:
                longest = d.shape[1]
        print("longest: ", longest)
        return longest


    def batchify(self, data, batch_size):
        data = [self.TEXT.numericalize(d) for d in data]

        exit()
        data = data.cat(data)
        print(data.shape)
        # Divide the dataset into bsz parts.
        nbatch = []
        for i in range(0, len(data), bsz):
            nbatch.append(data[i:i + bsz])
        print(nbatch)

        data = torch.cat(nbatch)
        print("shape",data.shape)
        # nbatch = data.size(0) // bsz
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

# d = Data('mini_train_set.txt') 
# d = Data('full_train_set.txt') 


# train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

# print(type(train_txt.examples[0].text))
# print(train_txt.examples[0].text)
# print("my data:", datasetLoader.data[0])

# data = TEXT.numericalize(datasetLoader.data[0])
# print(data)
# print(data.shape)
# print("my data:", len(datasetLoader.data[0]))
# print(TEXT.vocab_cls)


# def batchify(data, bsz):
#     #data = TEXT.numericalize([data.examples[0].text])
#     data = datasetLoader.data
#     print("data", type(data), data)
#     # Divide the dataset into bsz parts.
#     nbatch = []
#     for i in range(0, len(data), bsz):
#         nbatch.append(data[i:i + bsz])
#     print('nbatch', nbatch)
#     exit()
#     # nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)

