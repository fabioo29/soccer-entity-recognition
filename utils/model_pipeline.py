import os
import trax 
import shutil
import numpy as np
import random as rnd

from trax import layers as tl
from trax.supervised import training

rnd.seed(33)

def get_vocab(vocab_path, tags_path):
    vocab = {}
    with open(vocab_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i  # to avoid the 0
        # loading tags (we require this to map tags to their indices)
    vocab['<PAD>'] = len(vocab) # 35180
    tag_map = {}
    with open(tags_path) as f:
        for i, t in enumerate(f.read().splitlines()):
            tag_map[t] = i 
    
    return vocab, tag_map

def get_params(vocab, tag_map, sentences_file, labels_file):
    sentences = []
    labels = []

    with open(sentences_file) as f:
        for sentence in f.read().splitlines():
            # replace each token by its index if it is in vocab
            # else use index of UNK_WORD
            s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.split(' ')]
            sentences.append(s)

    with open(labels_file) as f:
        for sentence in f.read().splitlines():
            # replace each label by its index
            l = [tag_map[label] for label in sentence.split()] # I added plus 1 here
            labels.append(l) 
    return sentences, labels, len(sentences)

def data_generator(batch_size, x, y, pad, shuffle=False, verbose=False):
    '''
    Input: 
        batch_size - integer describing the batch size
        x - list containing sentences where words are represented as integers
        y - list containing tags associated with the sentences
        shuffle - Shuffle the data order
        pad - an integer representing a pad character
        verbose - Print information during runtime
    Output:
        a tuple containing 2 elements:
        X - np.ndarray of dim (batch_size, max_len) of padded sentences
        Y - np.ndarray of dim (batch_size, max_len) of tags associated with the sentences in X
    '''
    
    # count the number of lines in data_lines
    num_lines = len(x)
    
    # create an array with the indexes of data_lines that can be shuffled
    lines_index = [*range(num_lines)]
    
    # shuffle the indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(lines_index)
    
    index = 0 # tracks current location in x, y
    while True:
        buffer_x = [0] * batch_size # Temporal array to store the raw x data for this batch
        buffer_y = [0] * batch_size # Temporal array to store the raw y data for this batch
        
        # Copy into the temporal buffers the sentences in x[index : index + batch_size] 
        # along with their corresponding labels y[index : index + batch_size]
        # Find maximum length of sentences in x[index : index + batch_size] for this batch. 
        # Reset the index if we reach the end of the data set, and shuffle the indexes if needed.
        max_len = 0
        for i in range(batch_size):
            # if the index is greater than or equal to the number of lines in x
            if index >= num_lines:
                # then reset the index to 0
                index = 0
                # re-shuffle the indexes if shuffle is set to True
                if shuffle:
                    rnd.shuffle(lines_index)
            
            # The current position is obtained using `lines_index[index]`
            # Store the x value at the current position into the buffer_x
            buffer_x[i] = x[lines_index[index]]
            
            # Store the y value at the current position into the buffer_y
            buffer_y[i] = y[lines_index[index]]
            
            lenx = len(x[lines_index[index]])    #length of current x[]
            if lenx > max_len:
                max_len = lenx                   #max_len tracks longest x[]
            
            # increment index by one
            index += 1


        # create X,Y, NumPy arrays of size (batch_size, max_len) 'full' of pad value
        X = np.full((batch_size, max_len), pad)
        Y = np.full((batch_size, max_len), pad)

        # copy values from lists to NumPy arrays. Use the buffered values
        for i in range(batch_size):
            # get the example (sentence as a tensor)
            # in `buffer_x` at the `i` index
            x_i = buffer_x[i]
            
            # similarly, get the example's labels
            # in `buffer_y` at the `i` index
            y_i = buffer_y[i]
            
            # Walk through each word in x_i
            for j in range(len(x_i)):
                # store the word in x_i at position j into X
                X[i, j] = x_i[j]
                
                # store the label in y_i at position j into Y
                Y[i, j] = y_i[j]

        if verbose: print("index=", index)
        yield((X,Y))

def train_model(NER, train_generator, eval_generator, train_steps=1, output_dir='model'):
    '''
    Input: 
        NER - the model you are building
        train_generator - The data generator for training examples
        eval_generator - The data generator for validation examples,
        train_steps - number of training steps
        output_dir - folder to save your model
    Output:
        training_loop - a trax supervised training Loop
    '''
    train_task = training.TrainTask(
        train_generator, # A train data generator
        loss_layer = tl.CrossEntropyLoss(), # A cross-entropy loss function
        optimizer = trax.optimizers.Adam(0.01),  # The adam optimizer
    )

    eval_task = training.EvalTask(
        labeled_data = eval_generator, # A labeled data generator
        metrics = [tl.CrossEntropyLoss(), tl.Accuracy()], # Evaluate with cross-entropy loss and accuracy
        n_eval_batches = 10  # Number of batches to use on each evaluation
    )

    training_loop = training.Loop(
        NER, # A model to train
        train_task, # A train task
        eval_tasks = [eval_task], # The evaluation task
        output_dir = output_dir) # The output directory

    # Train with train_steps
    training_loop.run(n_steps = train_steps)
    return training_loop

def evaluate_prediction(pred, labels, pad):
    """
    Inputs:
        pred: prediction array with shape 
            (num examples, max sentence length in batch, num of classes)
        labels: array of size (batch_size, seq_len)
        pad: integer representing pad character
    Outputs:
        accuracy: float
    """
    outputs = np.argmax(pred, axis=2)
    print("outputs shape:", outputs.shape)

    mask = labels != pad
    print("mask shape:", mask.shape, "mask[0][20:30]:", mask[0][20:30])

    accuracy = np.sum(outputs == labels) / float(np.sum(mask))

    return accuracy

def predict(sentence, model, vocab, tag_map):
    # create tensor with info about each word in the sentence
    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.replace('\n', '').split(' ')]
    batch_data = np.ones((1, len(s)))
    batch_data[0][:] = s
    sentence = np.array(batch_data).astype(int)

    # predict the tags for each word
    output = model(sentence)
    
    # for each word get the label with the max value
    outputs = np.argmax(output, axis=2)

    # convert each label (number) to its string value
    labels = list(tag_map.keys())
    pred = []
    for i in range(len(outputs[0])):
        idx = outputs[0][i] 
        pred_label = labels[idx]
        pred.append(pred_label)
    return pred

def model_pipeline(batch_size, train_steps, vocab_size, d_model):
    vocab, tag_map = get_vocab('dataset/words.txt', 'dataset/tags.txt')
    t_sentences, t_labels, t_size = get_params(vocab, tag_map, 'dataset/train/sentences.txt', 'dataset/train/labels.txt')
    v_sentences, v_labels, v_size = get_params(vocab, tag_map, 'dataset/val/sentences.txt', 'dataset/val/labels.txt')
    test_sentences, test_labels, test_size = get_params(vocab, tag_map, 'dataset/test/sentences.txt', 'dataset/test/labels.txt')

    vocab_size = len(vocab)
    embedded_size = len(t_sentences[0])
    tags = tag_map

    # initializing your model
    model = tl.Serial(
        tl.Embedding(vocab_size, embedded_size), # Embedding layer
        tl.LSTM(embedded_size), # LSTM layer
        tl.Dense(len(tags)), # Dense layer with len(tags) units
        tl.LogSoftmax()  # LogSoftmax layer
    )
    # display your model
    print(model)

    if os.path.exists('model'):
        shutil.rmtree('model')

    # Create training data, mask pad id=35180 for training.
    train_generator = trax.data.inputs.add_loss_weights(
        data_generator(batch_size, t_sentences, t_labels, vocab['<PAD>'], True),
        id_to_mask=vocab['<PAD>'])

    # Create validation data, mask pad id=35180 for training.
    eval_generator = trax.data.inputs.add_loss_weights(
        data_generator(batch_size, v_sentences, v_labels, vocab['<PAD>'], True),
        id_to_mask=vocab['<PAD>'])

    # Train the model
    training_loop = train_model(model, train_generator, eval_generator, train_steps)

    """
    model = tl.Serial(
        tl.Embedding(vocab_size, embedded_size),    # Embedding layer
        tl.LSTM(embedded_size),                     # LSTM layer
        tl.Dense(len(tags)),                        # Dense layer with len(tags) units
        tl.LogSoftmax()                             # LogSoftmax layer
    )
    model.init(trax.shapes.ShapeDtype((1, 1), dtype=np.int32))
    model.init_from_file('model/model.pkl.gz', weights_only=True)
    """

    # create the evaluation inputs
    x, y = next(data_generator(len(test_sentences), test_sentences, test_labels, vocab['<PAD>']))
    print("input shapes", x.shape, y.shape)

    # sample prediction
    tmp_pred = model(x)
    print(type(tmp_pred))
    print(f"tmp_pred has shape: {tmp_pred.shape}")

    accuracy = evaluate_prediction(model(x), y, vocab['<PAD>'])
    print("accuracy: ", accuracy)

    sentence = """
    Hello and welcome to the SANTIAGO BERNABEU that will be the stage for 
    one of the most anticipated games of the season between REAL MADRID and 
    BARCELONA. The referee for this match is Danny Makkelie.
    """.lower().replace('\n', '')

    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.split(' ')]
    t = [token if token in vocab else 'UNK' for token in sentence.split(' ')]
    predictions = predict(sentence, model, vocab, tag_map)
    print([(word,token,token_word,tag) for (word,token,token_word,tag) in list(zip(sentence.split(' '), s, t, predictions)) if tag != 'O'])

if __name__ == "__main__":
    model_pipeline(64, 100, None, None)