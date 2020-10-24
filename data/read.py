
file1 = open('mini_train_set.txt', 'r') 
Lines = file1.readlines() 


characters = {}


count = 0
# Strips the newline character 
for line in Lines: 
    line = line.strip()
    print("Line {}: {}".format(count, line))
    count += 1
    for c in line:
        if c in characters:
            characters[c] += 1
        else:
            characters[c] = 0
    print(characters)

