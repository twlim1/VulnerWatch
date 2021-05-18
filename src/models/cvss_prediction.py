"""
:Title: CVSS prediction
:Description: Load all pre-trained BERT models to predict CVSS scores
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import textwrap

from datetime import datetime
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error


class BaseScore:
    @staticmethod
    def round_up(d):
        int_input = round(d * 100000)
        if int_input % 10000 == 0:
            return int_input / 100000.0
        else:
            return (math.floor(int_input / 10000) + 1) / 10.0

    @staticmethod
    def get_av_score(metric):
        if metric == 'network':
            return 0.85
        elif metric == 'adjacent_network':
            return 0.62
        elif metric == 'local':
            return 0.55
        elif metric == 'physical':
            return 0.20
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_ac_score(metric):
        if metric == 'low':
            return 0.77
        elif metric == 'high':
            return 0.44
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_pr_score(metric, s):
        if metric == 'none':
            return 0.85
        elif metric == 'low':
            return 0.68 if s == 'changed' else 0.62
        elif metric == 'high':
            return 0.50 if s == 'changed' else 0.27
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_ui_score(metric):
        if metric == 'none':
            return 0.85
        elif metric == 'required':
            return 0.62
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_c_score(metric):
        if metric == 'high':
            return 0.56
        elif metric == 'low':
            return 0.22
        elif metric == 'none':
            return 0
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_i_score(metric):
        if metric == 'high':
            return 0.56
        elif metric == 'low':
            return 0.22
        elif metric == 'none':
            return 0
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_a_score(metric):
        if metric == 'high':
            return 0.56
        elif metric == 'low':
            return 0.22
        elif metric == 'none':
            return 0
        else:
            raise ValueError('Invalid metric value')

    def calculate_iss(self, c, i, a):
        return 1 - (1-self.get_c_score(c)) * (1-self.get_i_score(i)) * (1-self.get_a_score(a))

    def calculate_impact(self, s, c, i, a):
        iss = self.calculate_iss(c, i, a)
        if s == 'unchanged':
            return 6.42 * iss
        elif s == 'changed':
            return (7.52 * (iss - 0.029)) - (3.25 * (iss - 0.02)**15)
        else:
            raise ValueError('Invalid metric value')

    def calculate_exploitability(self, av, ac, pr, ui, s):
        return 8.22 * self.get_av_score(av) * self.get_ac_score(ac) * self.get_pr_score(pr, s) * self.get_ui_score(ui)

    def calculate_scores(self, av, ac, pr, ui, s, c, i, a):
        av = av.lower()
        ac = ac.lower()
        pr = pr.lower()
        ui = ui.lower()
        s = s.lower()
        c = c.lower()
        i = i.lower()
        a = a.lower()

        impact = self.calculate_impact(s, c, i, a)
        exploitability = self.calculate_exploitability(av, ac, pr, ui, s)
        if impact <= 0:
            base = 0
        else:
            if s == 'unchanged':
                base = min((impact + exploitability), 10)
            elif s == 'changed':
                base = min(1.08 * (impact + exploitability), 10)
        return self.round_up(base), round(impact, 1), round(exploitability, 1)


class CVSS:
    def __init__(self, enable_print=True):
        file_path = '../../data/raw/cve.json'
        av_path = '../../models/AV'
        ac_path = '../../models/AC'
        ui_path = '../../models/UI'
        pr_path = '../../models/PR'
        s_path = '../../models/SC'
        c_path = '../../models/CI'
        i_path = '../../models/II'
        a_path = '../../models/AI'

        self.enabled = enable_print
        self.scorer = BaseScore()
        self.device = self.get_device()

        self.av_tokenizer, self.av_model = self.load_model(av_path)
        self.ac_tokenizer, self.ac_model = self.load_model(ac_path)
        self.pr_tokenizer, self.pr_model = self.load_model(pr_path)
        self.ui_tokenizer, self.ui_model = self.load_model(ui_path)
        self.s_tokenizer, self.s_model = self.load_model(s_path)
        self.c_tokenizer, self.c_model = self.load_model(c_path)
        self.i_tokenizer, self.i_model = self.load_model(i_path)
        self.a_tokenizer, self.a_model = self.load_model(a_path)

    @staticmethod
    def get_device():
        # If there's a GPU available...
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            device = torch.device('cuda')
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device('cpu')
        return device

    def load_model(self, model_path):
        token = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
        model.to(self.device)
        return token, model

    def text_to_embedding(self, tokenizer, model, max_len, in_text):
        encoded_dict = tokenizer.encode_plus(
                            in_text,                     # Sentence to encode.
                            add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                            max_length=max_len,          # Pad & truncate all sentences.
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,  # Construct attn. masks.
                            return_tensors='pt',         # Return pytorch tensors.
                        )
        input_ids = encoded_dict['input_ids']
        attn_mask = encoded_dict['attention_mask']

        model.eval()

        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        with torch.no_grad():
            result = model(input_ids=input_ids,
                           token_type_ids=None,
                           attention_mask=attn_mask,
                           )

        layer_i = 12
        batch_i = 0
        token_i = 0
        logits = result.logits
        logits = logits.detach().cpu().numpy()

        vec = result.hidden_states[layer_i][batch_i][token_i]
        vec = vec.detach().cpu().numpy()

        return logits, vec

    def print_custom(self, text):
        if self.enabled:
            print(text)

    @staticmethod
    def logits_2_confidence(logits):
        return math.exp(logits) / (1 + math.exp(logits))

    def predict(self, text, confidence_threshold=None):
        wrapper = textwrap.TextWrapper(initial_indent='  ', subsequent_indent='  ', width=120)
        self.print_custom('Description: \n\n{}'.format(wrapper.fill(text)))

        self.print_custom('\nPredictions:\n')
        logits, vec = self.text_to_embedding(self.av_tokenizer, self.av_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_av = 'network'
        elif np.argmax(logits, axis=1) == 1:
            pred_av = 'adjacent_network'
        elif np.argmax(logits, axis=1) == 2:
            pred_av = 'local'
        else:
            pred_av = 'physical'
        conf_av = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  AV: {}\t\tConfidence: {:.4f}'.format(pred_av.capitalize(), conf_av))

        logits, vec = self.text_to_embedding(self.ac_tokenizer, self.ac_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_ac = 'low'
        else:
            pred_ac = 'high'
        conf_ac = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  AC: {}\t\tConfidence: {:.4f}'.format(pred_ac.capitalize(), conf_ac))

        logits, vec = self.text_to_embedding(self.pr_tokenizer, self.pr_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_pr = 'none'
        elif np.argmax(logits, axis=1) == 1:
            pred_pr = 'low'
        else:
            pred_pr = 'high'
        conf_pr = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  PR: {}\t\tConfidence: {:.4f}'.format(pred_pr.capitalize(), conf_pr))

        logits, vec = self.text_to_embedding(self.ui_tokenizer, self.ui_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_ui = 'none'
        else:
            pred_ui = 'required'
        conf_ui = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  UI: {}\t\tConfidence: {:.4f}'.format(pred_ui.capitalize(), conf_ui))

        logits, vec = self.text_to_embedding(self.s_tokenizer, self.s_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_sc = 'unchanged'
        else:
            pred_sc = 'changed'
        conf_sc = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  S : {}\t\tConfidence: {:.4f}'.format(pred_sc.capitalize(), conf_sc))

        logits, vec = self.text_to_embedding(self.c_tokenizer, self.c_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_ci = 'none'
        elif np.argmax(logits, axis=1) == 1:
            pred_ci = 'low'
        else:
            pred_ci = 'high'
        conf_ci = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  C : {}\t\tConfidence: {:.4f}'.format(pred_ci.capitalize(), conf_ci))

        logits, vec = self.text_to_embedding(self.i_tokenizer, self.i_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_ii = 'none'
        elif np.argmax(logits, axis=1) == 1:
            pred_ii = 'low'
        else:
            pred_ii = 'high'
        conf_ii = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  I : {}\t\tConfidence: {:.4f}'.format(pred_ii.capitalize(), conf_ii))

        logits, vec = self.text_to_embedding(self.a_tokenizer, self.a_model, 512, text)
        if np.argmax(logits, axis=1) == 0:
            pred_ai = 'none'
        elif np.argmax(logits, axis=1) == 1:
            pred_ai = 'low'
        else:
            pred_ai = 'high'
        conf_ai = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  A : {}\t\tConfidence: {:.4f}'.format(pred_ai.capitalize(), conf_ai))

        pred_b, pred_i, pred_e = self.scorer.calculate_scores(pred_av, pred_ac, pred_pr, pred_ui,
                                                              pred_sc, pred_ci, pred_ii, pred_ai)
        self.print_custom('')
        self.print_custom('  Base score: {}'.format(pred_b))
        self.print_custom('  Impact score: {}'.format(pred_i))
        self.print_custom('  Exploitability score: {}'.format(pred_e))

        if confidence_threshold:
            if conf_av < confidence_threshold or conf_ac < confidence_threshold or \
                    conf_pr < confidence_threshold or conf_ui < confidence_threshold or \
                    conf_sc < confidence_threshold or conf_ci < confidence_threshold or \
                    conf_ii < confidence_threshold or conf_ai < confidence_threshold:
                return None, None
        return (pred_b, pred_i, pred_e), (pred_av, pred_ac, pred_pr, pred_ui, pred_sc, pred_ci, pred_ii, pred_ai)

    def batch_prediction(self, df, confidence_threshold=None):
        pred_b_labels = list()
        pred_i_labels = list()
        pred_e_labels = list()

        pred_av_labels = list()
        pred_ac_labels = list()
        pred_pr_labels = list()
        pred_ui_labels = list()
        pred_sc_labels = list()
        pred_ci_labels = list()
        pred_ii_labels = list()
        pred_ai_labels = list()

        pred_confidence = list()

        start_time = datetime.now()
        for idx, row in df.iterrows():
            if (idx + 1) % 1000 == 0:
                print('Processing index: {}'.format(idx + 1))

            scores, metrics = self.predict(row['description'], confidence_threshold)
            if scores is None and metrics is None:
                pred_confidence.append(False)
                continue
            else:
                pred_confidence.append(True)

            (b, i, e), (av, ac, pr, ui, sc, ci, ii, ai) = scores, metrics
            pred_b_labels.append(b)
            pred_i_labels.append(i)
            pred_e_labels.append(e)

            pred_av_labels.append(av)
            pred_ac_labels.append(ac)
            pred_pr_labels.append(pr)
            pred_ui_labels.append(ui)
            pred_sc_labels.append(sc)
            pred_ci_labels.append(ci)
            pred_ii_labels.append(ii)
            pred_ai_labels.append(ai)

        print('Total Time: {}'.format((datetime.now() - start_time).seconds))
        print('Confidence threshold: {}'.format(confidence_threshold))
        print('Total Predicted: {}'.format(len(df)))
        print('Total above threshold: {}'.format(len(pred_b_labels)))


if __name__ == '__main__':
    # scorer = BaseScore()
    # scores = scorer.calculate_scores('Network', 'High', 'Low', 'Required', 'Unchanged', 'Low', 'Low', 'Low')
    # print(scores)

    input_text = 'A malicious unauthenticated user could abuse the lack of authentication check on a particular ' \
                 'web service exposed by default in SAP Netweaver JAVA stack, allowing them to fully compromise ' \
                 'the targeted system.'
    cvss = CVSS()
    cvss.predict(input_text)
