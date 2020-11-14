from tqdm import tqdm
import torchtext
from torchtext.data.utils import get_tokenizer

class Data:
    def __init__(self, data_path, BATCH_SIZE=32):
        self.batch_size = BATCH_SIZE
        self.vocab = {}
        self.max_seq_len = 0 
        # self.train = []
        self.test = [] # split(split_ratio=0.7, stratified=False, strata_field='label', random_state=None)
        self.eval = []
        self.TEXT = torchtext.data.ReversibleField(tokenize=get_tokenizer("basic_english"),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
        # self.TEXT_REV = torchtext.data.ReversibleField( lower=True, include_lengths=True)

        self.data = self.clean(data_path)
        self.max_seq_len = self.find_max_seq_len(self.data)
        self.TEXT.build_vocab(self.data)
        print("vocab", self.TEXT.vocab.stoi)
        self.train = self.batchify(self.data, self.batch_size)


    def clean(self, data_path):
        open_file = open(data_path, 'r').readlines()

        lines = []
        # Strips the newline character 
        for line in tqdm(open_file): 
            line = line.strip()
            lines.append(line)
        return lines

    def getData(self):
        return self.train, self.test, self.eval
    
    def find_max_seq_len(self, data):
        max_seq_len = 0 
        for d in self.data:
            if len(d) > max_seq_len:
                max_seq_len = len(d)
        print("max_seq_len: ", max_seq_len)
        return max_seq_len

    def toString(self, T):
        print(T)
        charArray = [c for c in T]
        print(charArray)

        for indx in charArray:
            for key in self.TEXT.vocab.stoi.keys():
                # print(indx, key)
                if key == indx:
                    print(key)

        # charArray = [self.TEXT.vocab.stoi[int(c)] for c in T]
        # print(charArray)


    def batchify(self, data, batch_size):
        # print("input to numericalize", data)
        # print("vocab", self.TEXT.vocab.stoi)
        print("\n\n")
        
        # data = [self.TEXT.numericalize(d) for d in data]
        data = self.TEXT.pad(data)
        data = self.TEXT.numericalize(data)
        data = data
        print(data)
        print(self.TEXT.reverse(data))
        # print(self.TEXT_REV.reverse(data[0]))
        print(data.shape)

        # self.toString(data[0])

        # print(data[0])
        # print(data[0][0].shape)
        # print(data[0][0][0].shape)

        # data = self.TEXT.pad(data)
        # print(data[0][0])
        # print(data)
        exit()

        data = data.cat(data)
        print(data.shape)
        # Divide the dataset into bsz parts.
        nbatch = []
        # for i in range(0, len(data), bsz):
        #     nbatch.append(data[i:i + bsz])
        print(nbatch)

        data = torch.cat(nbatch)
        print("shape",data.shape)
        # nbatch = data.size(0) // bsz
        # Evenly divide the data across the bsz batches.
        # data = data.view(bsz, -1).t().contiguous()
        # return data.to(device)

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

