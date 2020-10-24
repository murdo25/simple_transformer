from tqdm import tqdm

BATCH_SIZE = 32

#full token set:
{'(': 1717012, 
'7': 646954, 
'-': 2932950, 
'3': 950997, 
'*': 6296347, 
'z': 245598, 
')': 1717012, 
'5': 803946, 
'9': 500411, 
'=': 999999, 
'1': 1551638, 
'2': 2739471, 
'8': 799494, 
'6': 854152, 
's': 568437, 
'n': 566388, 
'+': 1249604, 
'4': 952515, 
'x': 243915,
'c': 284520, 
'0': 621754, 
'k': 245041, 
'o': 283087,
'j': 244093, 
'h': 244131, 
'y': 246023,
'i': 528182, 
't': 285214, 
'a': 284687}


file1 = open('mini_train_set.txt', 'r') 
# file1 = open('full_train_set.txt', 'r') 
Lines = file1.readlines() 

vocab = {}
lines = []

count = 0
# Strips the newline character 
for line in tqdm(Lines): 
    line = line.strip()
    # print("Line {}: {}".format(count, line))
    count += 1
    for c in line:
        if c in vocab:
            vocab[c] += 1
        else:
            vocab[c] = 0
    lines += line    

print("VOCAB:", vocab)
print("NUM TOKENS", len(vocab))
print("SIZE DATASET:", len(lines))

vocab_size = len(vocab)

