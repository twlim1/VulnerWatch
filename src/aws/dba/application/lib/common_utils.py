import os
import numpy as np
import math
import torch
import textwrap

from transformers import BertForSequenceClassification, BertTokenizer


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
    def get_ci_score(metric):
        if metric == 'high':
            return 0.56
        elif metric == 'low':
            return 0.22
        elif metric == 'none':
            return 0
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_ii_score(metric):
        if metric == 'high':
            return 0.56
        elif metric == 'low':
            return 0.22
        elif metric == 'none':
            return 0
        else:
            raise ValueError('Invalid metric value')

    @staticmethod
    def get_ai_score(metric):
        if metric == 'high':
            return 0.56
        elif metric == 'low':
            return 0.22
        elif metric == 'none':
            return 0
        else:
            raise ValueError('Invalid metric value')

    def calculate_iss(self, ci, ii, ai):
        return 1 - (1-self.get_ci_score(ci)) * (1-self.get_ii_score(ii)) * (1-self.get_ai_score(ai))

    def calculate_impact(self, sc, ci, ii, ai):
        iss = self.calculate_iss(ci, ii, ai)
        if sc == 'unchanged':
            return 6.42 * iss
        elif sc == 'changed':
            return (7.52 * (iss - 0.029)) - (3.25 * (iss - 0.02)**15)
        else:
            raise ValueError('Invalid metric value')

    def calculate_exploitability(self, av, ac, pr, ui, sc):
        return 8.22 * self.get_av_score(av) * self.get_ac_score(ac) * self.get_pr_score(pr, sc) * self.get_ui_score(ui)

    def calculate_scores(self, av, ac, pr, ui, sc, ci, ii, ai):
        av = integer_to_value('AV', av)
        ac = integer_to_value('AC', ac)
        pr = integer_to_value('PR', pr)
        ui = integer_to_value('UI', ui)
        sc = integer_to_value('SC', sc)
        ci = integer_to_value('CI', ci)
        ii = integer_to_value('II', ii)
        ai = integer_to_value('AI', ai)

        impact = self.calculate_impact(sc, ci, ii, ai)
        exploitability = self.calculate_exploitability(av, ac, pr, ui, sc)
        if impact <= 0:
            base = 0
        else:
            if sc == 'unchanged':
                base = min((impact + exploitability), 10)
            elif sc == 'changed':
                base = min(1.08 * (impact + exploitability), 10)
        return self.round_up(base), round(impact, 1), round(exploitability, 1)


class CVSS:
    def __init__(self, model_path, enable_print=True):
        av_path = os.path.join(model_path, 'AV')
        ac_path = os.path.join(model_path, 'AC')
        ui_path = os.path.join(model_path, 'UI')
        pr_path = os.path.join(model_path, 'PR')
        sc_path = os.path.join(model_path, 'SC')
        ci_path = os.path.join(model_path, 'CI')
        ii_path = os.path.join(model_path, 'II')
        ai_path = os.path.join(model_path, 'AI')

        self.enabled = enable_print
        self.scorer = BaseScore()
        self.device = self.get_device()

        self.av_tokenizer, self.av_model = self.load_model(av_path)
        self.ac_tokenizer, self.ac_model = self.load_model(ac_path)
        self.pr_tokenizer, self.pr_model = self.load_model(pr_path)
        self.ui_tokenizer, self.ui_model = self.load_model(ui_path)
        self.sc_tokenizer, self.si_model = self.load_model(sc_path)
        self.ci_tokenizer, self.ci_model = self.load_model(ci_path)
        self.ii_tokenizer, self.ii_model = self.load_model(ii_path)
        self.ai_tokenizer, self.ai_model = self.load_model(ai_path)

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

    def predict(self, text):
        wrapper = textwrap.TextWrapper(initial_indent='  ', subsequent_indent='  ', width=120)
        self.print_custom('Description: \n\n{}'.format(wrapper.fill(text)))

        self.print_custom('\nPredictions:\n')
        logits, vec = self.text_to_embedding(self.av_tokenizer, self.av_model, 512, text)
        pred_av = np.argmax(logits, axis=1)[0]
        conf_av = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  AV: {}\t\tConfidence: {:.4f}'.format(integer_to_value('AV', pred_av).capitalize(),
                                                                  conf_av))

        logits, vec = self.text_to_embedding(self.ac_tokenizer, self.ac_model, 512, text)
        pred_ac = np.argmax(logits, axis=1)[0]
        conf_ac = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  AC: {}\t\tConfidence: {:.4f}'.format(integer_to_value('AC', pred_ac).capitalize(),
                                                                  conf_ac))

        logits, vec = self.text_to_embedding(self.pr_tokenizer, self.pr_model, 512, text)
        pred_pr = np.argmax(logits, axis=1)[0]
        conf_pr = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  PR: {}\t\tConfidence: {:.4f}'.format(integer_to_value('PR', pred_pr).capitalize(),
                                                                  conf_pr))

        logits, vec = self.text_to_embedding(self.ui_tokenizer, self.ui_model, 512, text)
        pred_ui = np.argmax(logits, axis=1)[0]
        conf_ui = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  UI: {}\t\tConfidence: {:.4f}'.format(integer_to_value('UI', pred_ui).capitalize(),
                                                                  conf_ui))

        logits, vec = self.text_to_embedding(self.sc_tokenizer, self.si_model, 512, text)
        pred_sc = np.argmax(logits, axis=1)[0]
        conf_sc = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  SC: {}\t\tConfidence: {:.4f}'.format(integer_to_value('SC', pred_sc).capitalize(),
                                                                  conf_sc))

        logits, vec = self.text_to_embedding(self.ci_tokenizer, self.ci_model, 512, text)
        pred_ci = np.argmax(logits, axis=1)[0]
        conf_ci = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  CI: {}\t\tConfidence: {:.4f}'.format(integer_to_value('CI', pred_ci).capitalize(),
                                                                  conf_ci))

        logits, vec = self.text_to_embedding(self.ii_tokenizer, self.ii_model, 512, text)
        pred_ii = np.argmax(logits, axis=1)[0]
        conf_ii = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  II: {}\t\tConfidence: {:.4f}'.format(integer_to_value('II', pred_ii).capitalize(),
                                                                  conf_ii))

        logits, vec = self.text_to_embedding(self.ai_tokenizer, self.ai_model, 512, text)
        pred_ai = np.argmax(logits, axis=1)[0]
        conf_ai = self.logits_2_confidence(np.max(logits[0]))
        self.print_custom('  AI: {}\t\tConfidence: {:.4f}'.format(integer_to_value('AI', pred_ai).capitalize(),
                                                                  conf_ai))

        pred_b, pred_i, pred_e = self.scorer.calculate_scores(pred_av, pred_ac, pred_pr, pred_ui,
                                                              pred_sc, pred_ci, pred_ii, pred_ai)
        self.print_custom('')
        self.print_custom('  Base score: {}'.format(pred_b))
        self.print_custom('  Impact score: {}'.format(pred_i))
        self.print_custom('  Exploitability score: {}'.format(pred_e))

        return (pred_b, pred_i, pred_e), \
               (pred_av, pred_ac, pred_pr, pred_ui, pred_sc, pred_ci, pred_ii, pred_ai), \
               (conf_av, conf_ac, conf_pr, conf_ui, conf_sc, conf_ci, conf_ii, conf_ai)


def value_to_integer(metric, value):
    metric = metric.lower()
    value = value.lower()
    if metric == 'av':
        if value == 'network':
            return 0
        elif value in ('adjacent', 'adjacent_network'):
            return 1
        elif value == 'local':
            return 2
        elif value == 'physical':
            return 3
    if metric == 'ac':
        if value == 'low':
            return 0
        elif value == 'high':
            return 1
    if metric == 'ui':
        if value == 'none':
            return 0
        elif value == 'required':
            return 1
    if metric == 'sc':
        if value == 'unchanged':
            return 0
        elif value == 'changed':
            return 1
    if metric in ('pr', 'ci', 'ii', 'ai'):
        if value == 'none':
            return 0
        elif value == 'low':
            return 1
        elif value == 'high':
            return 2
    raise ValueError('Invalid metric: {} value: {}'.format(metric, value))


def integer_to_value(metric, integer):
    metric = metric.lower()
    if metric == 'av':
        if integer == 0:
            return 'network'
        elif integer == 1:
            return 'adjacent_network'
        elif integer == 2:
            return 'local'
        elif integer == 3:
            return 'physical'
    if metric == 'ac':
        if integer == 0:
            return 'low'
        elif integer == 1:
            return 'high'
    if metric == 'ui':
        if integer == 0:
            return 'none'
        elif integer == 1:
            return 'required'
    if metric == 'sc':
        if integer == 0:
            return 'unchanged'
        elif integer == 1:
            return 'changed'
    if metric in ('pr', 'ci', 'ii', 'ai'):
        if integer == 0:
            return 'none'
        elif integer == 1:
            return 'low'
        elif integer == 2:
            return 'high'
    raise ValueError('Invalid metric: {} value: {}'.format(metric, integer))
