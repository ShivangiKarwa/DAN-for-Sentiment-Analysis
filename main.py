# models.py

import torch
from torch import nn
import torch.nn.functional as F
import random
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, DANWG, SentimentDataBPE, BPE


torch.manual_seed(42)

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        #X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
       # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy



def collate_fn(batch):
    sentence_embeddings, labels = zip(*batch)

    


    #sentence_embeddings = [sent.long() for sent in sentence_embeddings]
    # Pad sequences
    batch_sentences = torch.nn.utils.rnn.pad_sequence(
        sentence_embeddings, 
        batch_first=True, 
        padding_value=0
    )

   

    # Stack labels into a batch tensor
    batch_labels = torch.stack(labels)

    return batch_sentences, batch_labels

def prepare_data(tokenized_sentences, vocab):
    input_data = []
    for sentence in tokenized_sentences:
        indices = tokenize_bpe_to_indices(sentence, vocab)
        input_data.append(indices)
    return input_data


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds\n")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":

        start_time = time.time()

        print('\nDAN model with 50 dim: ')
        word_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        train_data_DAN = SentimentDatasetDAN("data/train.txt",word_embeddings)
        dev_data_DAN = SentimentDatasetDAN("data/dev.txt",word_embeddings)
        train_loader_DAN = DataLoader(train_data_DAN, batch_size=16, collate_fn=collate_fn, shuffle=True)
        test_loader_DAN = DataLoader(dev_data_DAN, batch_size=16, collate_fn=collate_fn, shuffle=False)

        
        dan_train_accuracy, dan_test_accuracy = experiment(DAN(word_embeddings,hidden_size=100, num_classes=2, dropout=0.2), train_loader_DAN, test_loader_DAN)
        

        end_time = time.time()

        print('\nTotal time taken by the DAN model (in secs): ', end_time - start_time)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='Training accuracy with DAN: ')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_dan.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='Dev Accuracy with DAN:')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_DAN.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

    elif args.model == "DANWG":

        start_time = time.time()
        word_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        train_data_DAN = SentimentDatasetDAN("data/train.txt",word_embeddings)
        dev_data_DAN = SentimentDatasetDAN("data/dev.txt",word_embeddings)
        train_loader_DAN = DataLoader(train_data_DAN, batch_size=16, collate_fn=collate_fn, shuffle=True)
        test_loader_DAN = DataLoader(dev_data_DAN, batch_size=16, collate_fn=collate_fn, shuffle=False)
        vocab_size = 14923  # Number of unique words/tokens in your dataset
        embedding_dim = 50 
        print('\nDAN model without GloVe: ')
        danWG_train_accuracy, danWG_test_accuracy = experiment(DANWG(vocab_size, embedding_dim, hidden_size=100, num_classes=2, dropout=0.2), train_loader_DAN, test_loader_DAN)
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(danWG_train_accuracy, label='Train Accuracy for DAN without GloVe')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Train Accuracy for DAN without GloVe')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_danWG.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(danWG_test_accuracy, label='Dev Accuracy for DAN without GloVe')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN without GloVe')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_danWG.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
    
    elif args.model == "SUBWORDDAN":

        dataset = SentimentDataBPE("data/train.txt",vocab_size = 1000,is_training=True)

        train_loader_BPE = DataLoader(dataset, batch_size=32, shuffle=True)
        #vocab_size = len(dataset.get_vocab())

        dev_data = SentimentDataBPE("data/dev.txt",bpe=dataset.bpe,is_training=False)
        test_loader_BPE = DataLoader(dev_data, batch_size=32, shuffle=False)

        #vocab_size = 1000
        vocab_size = len(dataset.get_vocab())
        embedding_dim = 300
        danBPE_train_accuracy, danBPE_test_accuracy = experiment(DANWG(vocab_size, embedding_dim, hidden_size=100, num_classes=2, dropout=0.3), train_loader_BPE, test_loader_BPE)

        plt.figure(figsize=(8, 6))
        plt.plot(danBPE_train_accuracy, label='Train Accuracy for DAN with BPE')

        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Train Accuracy for DAN with BPE')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_danBPE.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(danBPE_test_accuracy, label='Dev Accuracy for DAN with BPE')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN with BPE')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy_danBPE.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

       

if __name__ == "__main__":
    main()
