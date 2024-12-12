import json
import torch as tc
import time

def preprocess(filename, x_name, y_name):
    graph_count = {}
    phone_count = {'<sos>': 1}
    corpus = {}
    expected = {}
    word_idx = 0

    with open(filename, 'r') as f:
        for row in f:
            graph, phone = row.split('\t')
            graph_tokens = [g for g in list(graph) if g != '-']
            phone_tokens = [p.strip() for p in phone.split(' ')]
            
            if word_idx not in corpus:
                corpus[word_idx] = graph_tokens
                expected[word_idx] = ['<sos>'] + phone_tokens + ['<eos>']
                

            for g_token in graph_tokens:
                # count occurrences of each grapheme
                if g_token not in graph_count:
                    graph_count[g_token] = 1
                else:
                    graph_count[g_token] += 1

            for p_token in phone_tokens:
                # count occurrences of each phoneme
                if p_token not in phone_count:
                    phone_count[p_token] = 1
                else:
                    phone_count[p_token] += 1
            

            word_idx += 1

    phone_count['<eos>'] = 1
    phone_count['<pad>'] = 1
    graph2idx = {graph: idx for idx, graph in enumerate(graph_count)}
    phone2idx = {phone: idx for idx, phone in enumerate(phone_count)}
    idx2phone = {int(idx): phone for idx, phone in enumerate(phone_count)}

    max_input_steps = max([len(value) for _, value in corpus.items()])
    max_expected_steps = max([len(value) for _, value in expected.items()])

    X = tc.zeros((word_idx, max_input_steps, len(graph2idx)))
    Y = tc.zeros((word_idx, max_expected_steps ,len(phone2idx)))
    
    # loop through each sample
    for sample in range(word_idx):
        for time_step in range(len(corpus[sample])):
            g_token = corpus[sample][time_step]
            X[sample][time_step][graph2idx[g_token]] = 1
        
        
        for time_step in range(max_expected_steps):
            if time_step < len(expected[sample]):
                p_token = expected[sample][time_step]
                Y[sample][time_step][phone2idx[p_token]] = 1
            else:
                Y[sample][time_step][phone2idx['<pad>']] = 1

    print(X.shape)
    print(Y.shape)

    tc.save(X, x_name)
    tc.save(Y, y_name)

    return X, Y, graph2idx, phone2idx, idx2phone, max_input_steps, max_expected_steps

def process(filename, x_name, y_name,  graph2idx, phone2idx, max_input_steps, max_expected_steps):
    corpus = {}
    expected = {}
    word_idx = 0
    with open(filename, 'r') as f:
        for row in f:
            graph, phone = row.split('\t')
            graph_tokens = [g for g in list(graph) if g != '-']
            phone_tokens = [p.strip() for p in phone.split(' ')]

            if word_idx not in corpus:
                corpus[word_idx] = graph_tokens
                expected[word_idx] = ['<sos>'] + phone_tokens + ['<eos>']
        
            word_idx += 1

    
    X = tc.zeros((word_idx, max_input_steps, len(graph2idx)))
    Y = tc.zeros((word_idx, max_expected_steps ,len(phone2idx)))

    # loop through each sample
    for sample in range(word_idx):
        for time_step in range(len(corpus[sample])):
            g_token = corpus[sample][time_step]
            X[sample][time_step][graph2idx[g_token]] = 1
        
        for time_step in range(max_expected_steps):
            if time_step < len(expected[sample]):
                p_token = expected[sample][time_step]
                Y[sample][time_step][phone2idx[p_token]] = 1
            else:
                Y[sample][time_step][phone2idx['<pad>']] = 1


    print(X.shape)
    print(Y.shape)
    tc.save(X, x_name)
    tc.save(Y, y_name)


def main():
    print('Start processing the data')
    start_time = time.time()
    _, _, graph2idx, phone2idx, idx2phone, max_input_steps, max_expected_steps = preprocess('data/train.dict', 'data_mx/X_train.pt', 'data_mx/Y_train.pt')
    process('data/dev.dict', 'data_mx/X_dev.pt', 'data_mx/Y_dev.pt', graph2idx, phone2idx, max_input_steps, max_expected_steps)
    process('data/test.dict', 'data_mx/X_test.pt', 'data_mx/Y_test.pt', graph2idx, phone2idx, max_input_steps, max_expected_steps)
    
    # save the positional idex
    with open('graph2idx.json', 'w') as f:
        json.dump(graph2idx, f)

    with open('phone2idx.json', 'w') as f:
        json.dump(phone2idx, f)

    with open('idx2phone.json', 'w') as f:
        json.dump(idx2phone, f)
    

    print("--- %s seconds ---" % (time.time() - start_time))
    print('Done')

if __name__ == "__main__":
    main()