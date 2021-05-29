#!/usr/bin/env python
import os
import csv
import time
import random
import datetime

import torch
import numpy as np
import pandas as pd
from transformers import (BertTokenizer, BertForSequenceClassification, AdamW,
                          BertConfig, get_linear_schedule_with_warmup)
from torch.utils.data import (TensorDataset, random_split, DataLoader,
                              RandomSampler, SequentialSampler)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, matthews_corrcoef)

#############
# Functions #
#############

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def getDevice(quiet=True):
    '''
    Returns either a CPU or GPU device, depending on what is available.
    '''
    if torch.cuda.is_available():    
        device = torch.device('cuda')
        if not quiet:
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        if not quiet:
            print('WARNING: Using CPU')
        device = torch.device('cpu')
    return device

def printParams(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

#######################
# Decorator Functions #
#######################

def train_required(func):
    def wrapper(*args):
        if not args[0].training_done: # args[0] is the same as the "self" variable
            raise RuntimeError('Need to run train() first')
        func(*args)
    return wrapper

def test_required(func):
    def wrapper(*args):
        if not args[0].testing_done: # args[0] is the same as the "self" variable
            raise RuntimeError('Need to run test() first')
        return func(*args)
    return wrapper

###########
# Classes #
###########

class CVSSMetric():
    '''Model for a given CVSS metric.
    metric:     string metric name
    train_df:   dataframe to save for training purposes
    test_df:    dataframe to save for testing purposes (optional)
    add_label:  If False, a 'label' column will not be added to the input dfs
    '''

    label_to_id = {
        'attack_vector': {
            'NETWORK':          0,
            'ADJACENT_NETWORK': 1,
            'LOCAL':            2,
            'PHYSICAL':         3,
        },
        'attack_complexity': {
            'LOW':  0,
            'HIGH': 1,
        },
        'privileges_required': {
            'NONE': 0,
            'LOW':  1,
            'HIGH': 2,
        },
        'user_interaction': {
            'NONE':     0,
            'REQUIRED': 1,
        },
        'scope': {
            'UNCHANGED':  0,
            'CHANGED':    1,
        },
        'confidentiality': {
            'NONE': 0,
            'LOW':  1,
            'HIGH': 2,
        },
        'integrity': {
            'NONE': 0,
            'LOW':  1,
            'HIGH': 2,
        },
        'availability': {
            'NONE': 0,
            'LOW':  1,
            'HIGH': 2,
        },
    }
    metrics = label_to_id.keys()

    def __init__(self, metric, train_df, test_df=None, add_label=True):
        if metric not in CVSSMetric.metrics:
            raise ValueError(f'Invalid CVSS metric: {metric}')

        self.metric = metric
        self.train_df = train_df
        self.test_df = test_df

        # Create label column by converting the text label to an integer
        to_id = CVSSMetric.label_to_id[metric]

        if add_label:
            self.train_df['label'] = self.train_df[metric].apply(lambda x: to_id[x])
            if self.test_df is not None:
                self.test_df['label'] = self.test_df[metric].apply(lambda x: to_id[x])

        self.testing_done = False
        self.training_done = False

    def getIdToLabel(self):
        d = CVSSMetric.label_to_id[self.metric]
        return dict((v,k) for k, v in d.items())

    @train_required
    def saveModel(self, output_dir=''):
        '''Save trained model'''

        if not output_dir: 
            output_dir = f'{self.metric}_model'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = self.model
        tokenizer = self.tokenizer

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    @test_required
    def getClassificationStats(self):
        ret = {}
        ret['accuracy']  = accuracy_score(self.flat_true_labels, self.flat_predictions)
        ret['precision'] = precision_score(self.flat_true_labels, self.flat_predictions, average='micro')
        ret['recall']    = recall_score(self.flat_true_labels, self.flat_predictions, average='micro')
        ret['f1']        = f1_score(self.flat_true_labels, self.flat_predictions, average='micro')
        ret['mcc']       = matthews_corrcoef(self.flat_true_labels, self.flat_predictions)
        ret['cm']        = confusion_matrix(self.flat_true_labels, self.flat_predictions,
                                            list(CVSSMetric.label_to_id[self.metric].values()))
        return ret

    # Note that the two output path arguments are concatenated.
    @test_required
    def savePredictions(self, sentences=None, output_dir='', output_file=''):
        '''Save predictions on test data'''

        # Setup
        if not sentences:
            sentences = self.test_df['description'].values
        if not output_dir: 
            output_dir = f'{self.metric}_model'
        if not output_file:
            output_file = f'{self.metric}_predictions.csv'

        output_file = os.path.join(output_dir, output_file)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        id_to_label = self.getIdToLabel()

        # Write to the file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['True Label', 'Predicted Label', 'Correct Prediction', 'Text'])

            for i in range(len(sentences)):
                expected = self.flat_true_labels[i]
                actual = self.flat_predictions[i]

                if (expected == actual):
                    col = 'True'
                else:
                    col = 'False'

                writer.writerow([id_to_label[expected], id_to_label[actual], col, sentences[i]])


    def createDataset(self, sentences, labels, tokenizer, max_length):
        '''Tokenize all of the sentences and map the tokens to their word IDs.
        sentences:  sentences to tokenize
        labels:     labels to use
        tokenizer:  tokenizer to use
        max_length: max sentence length in tokens (0-512)
        '''

        assert max_length <= 512 and max_length > 0

        input_ids = []
        attention_masks = []

        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_length,   # Pad & truncate all sentences.
                                padding = 'max_length',
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                                truncation=True,
                           )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Combine the training inputs into a TensorDataset.
        return TensorDataset(input_ids, attention_masks, labels)

    def getDataLoaders(self, dataset, train_ratio, batch_size):
        '''Returns training and validation data loaders
        dataset:        dataset to split
        train_ratio:    float from 0-1 (ex: .9 means 90% train 10% validation)
        batch_size:     batch size for training
        '''

        assert train_ratio <= 1 and train_ratio > 0

        # Calculate the number of samples to include in each set.
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

        return train_dataloader, validation_dataloader

    def train(self,
              max_length=512,
              batch_size=8,
              epochs=2,
              train_ratio=0.9,
              optimizer=None,
              lr=3e-5,
              eps=1e-8,
              tokenizer=None,
              labels=None,
              sentences=None,
              num_labels=None,
              seed_val=22,
              ):
        '''Giant custom training loop
        max_length:     max sentence length in tokens (0-512)
        batch_size:     batch size for training
        epochs:         epoch count for training
        train_ratio:    float from 0-1 (ex: .9 means 90% train 10% validation)
        optimizer:      overrides default AdamW optimizer
        lr:             learning rate for default optimizer
        eps:            epsilon for default optimizer
        tokenizer:      overrides default tokenizer
        labels:         overrides default labels for training
        sentences:      overrides default sentences for training
        num_labels:     overrides internally calculated num_labels (necessary
                        if 'labels' is specified)
        seed_val:       value to seed random operations
        '''

        model_str = 'bert-base-uncased'

        # Fill out missing default arguments
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(model_str, do_lower_case=True)

        if labels is None:
            labels = self.train_df['label'].values
        if sentences is None:
            sentences = self.train_df['description'].values
        if num_labels is None:
            num_labels = len(CVSSMetric.label_to_id[self.metric])

        # Tokenize all of the sentences and map the tokens to their word IDs.
        dataset = self.createDataset(sentences, labels, tokenizer, max_length)

        # Create a train-validation split.
        train_dataloader, validation_dataloader = self.getDataLoaders(dataset,
                                                                      train_ratio,
                                                                      batch_size)

        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            model_str,
            num_labels = num_labels, # The number of output labels
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU, if needed.
        device = getDevice()
        if str(device) == 'cuda':
            model.cuda()

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        if not optimizer:
            optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = 0, # Default in run_glue.py
                                                    num_training_steps = total_steps)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # In PyTorch, calling `model` will in turn call the model's `forward`
                # function and pass down the arguments. The `forward` function is
                # documented here:
                # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
                # The results are returned in a results object, documented here:
                # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
                # Specifically, we'll get the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

                loss = result.loss
                logits = result.logits

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print('')
            print('  Average training loss: {0:.2f}'.format(avg_train_loss))
            print('  Training epoch took: {:}'.format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print('')
            print('Running Validation...')

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    result = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels,
                                   return_dict=True)

                # Get the loss and "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like the
                # softmax.
                loss = result.loss
                logits = result.logits

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)


            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print('  Accuracy: {0:.2f}'.format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print('  Validation Loss: {0:.2f}'.format(avg_val_loss))
            print('  Validation took: {:}'.format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print('')
        print('Training complete!')

        print('Total training took {:} (h:mm:ss)'.format(format_time(time.time()-total_t0)))

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')

        # Save variables needed for later
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        self.df_stats = df_stats

        self.training_done = True

        return df_stats

    def test(self, labels=None, sentences=None):
        '''Run test data through model. Uses values saved during training.
        labels:     override default labels
        sentences:  override default sentences
        '''
        if self.test_df is None:
            raise RuntimeError('No test data to use')

        if labels is None:
            labels = self.test_df['label'].values
        if sentences is None:
            sentences = self.test_df['description'].values

        model = self.model
        batch_size = self.batch_size
        device = self.device
        tokenizer = self.tokenizer
        max_length = self.max_length

        # Tokenize all of the sentences and map the tokens to their word IDs.
        prediction_data = self.createDataset(sentences, labels, tokenizer, max_length)

        # Create the DataLoader.
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Put model in evaluation mode
        model.eval()

        # Tracking variables 
        predictions, true_labels = [], []

        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
  
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
  
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model(b_input_ids, 
                               token_type_ids=None, 
                               attention_mask=b_input_mask,
                               return_dict=True)

            logits = result.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
  
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Combine the results across all batches.
        flat_predictions = np.concatenate(predictions, axis=0)

        # For each sample, pick the label with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)

        # Save results
        self.flat_predictions = flat_predictions
        self.flat_true_labels = flat_true_labels 

        self.testing_done = True

        return accuracy_score(flat_true_labels, flat_predictions)

if __name__ == '__main__':
    column = 'integrity'

    train_data = pd.read_csv('data/cve_train.csv', index_col=0)
    test_data = pd.read_csv('data/cve_test.csv', index_col=0)

    a = CVSSMetric(column, train_data, test_data)

    a.train(max_length=256)

