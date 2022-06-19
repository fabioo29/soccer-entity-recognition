import os

# read dataset/train,test,val/labels
labels = open(os.path.join('dataset', 'train', 'labels.txt'), 'r').readlines()
labels += open(os.path.join('dataset', 'val', 'labels.txt'), 'r').readlines()
labels += open(os.path.join('dataset', 'test', 'labels.txt'), 'r').readlines()

# read dataset/train/sentences
sentences = open(os.path.join('dataset', 'train', 'sentences.txt'), 'r').readlines()
sentences += open(os.path.join('dataset', 'val', 'sentences.txt'), 'r').readlines()
sentences += open(os.path.join('dataset', 'test', 'sentences.txt'), 'r').readlines()

# print each word with label != 'O' to check the dataset accuracy manually
for idx1, (label, sentence) in enumerate(zip(labels, sentences)):
    for idx, (l, s) in enumerate(zip(label.split(), sentence.split())):
        if l not in ('O'): 
            try:
                bef = sentence.split()[idx-1]
                aft = sentence.split()[idx+1]
                print(f'{idx1}, {l}, {bef} {s} {aft} ')
            except:
                print(f'{idx1}, {l}, {s}')