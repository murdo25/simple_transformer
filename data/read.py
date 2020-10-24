


file1 = open('mini_train_set.txt', 'r') 
Lines = file1.readlines() 
  
count = 0
# Strips the newline character 
for line in Lines: 
    print("Line {}: {}".format(count, line.strip()))
    count += 1
