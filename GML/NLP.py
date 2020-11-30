import warnings
warnings.filterwarnings("ignore")

import tqdm 

import requests
from bs4 import BeautifulSoup

import logging
import re

from ftfy import fix_text

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

from transformers import (AutoTokenizer, TFAutoModel, AutoModelWithLMHead, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel,
                          AdamW, get_cosine_with_hard_restarts_schedule_with_warmup)

import base64

import csv

import os

import argparse

import torch
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import random
import string
import unicodedata

# add new languages here
specials = {
    "de": {
        "case_insensitive": [["ä", "ae"], ["ü", "ue"], ["ö", "oe"]],
        "case_sensitive": [["ß", "ss"]],
    }
}
escape_sequence = "xxxxx"


def norm(text):
    return unicodedata.normalize("NFC", text)


def save_replace(text, lang, back=False):
    # perserve the casing of the original text
    # TODO: performance of matching

    # normalize the text to make sure to really match all occurences
    text = norm(text)

    possibilities = (
        specials[lang]["case_sensitive"]
        + [[norm(x[0]), x[1]] for x in specials[lang]["case_insensitive"]]
        + [
            [norm(x[0].upper()), x[1].upper()]
            for x in specials[lang]["case_insensitive"]
        ]
    )
    for pattern, target in possibilities:
        if back:
            text = text.replace(escape_sequence + target + escape_sequence, pattern)
        else:
            text = text.replace(pattern, escape_sequence + target + escape_sequence)
    return text

import re
import sys
import unicodedata

CURRENCIES = {
    "$": "USD",
    "zł": "PLN",
    "£": "GBP",
    "¥": "JPY",
    "฿": "THB",
    "₡": "CRC",
    "₦": "NGN",
    "₩": "KRW",
    "₪": "ILS",
    "₫": "VND",
    "€": "EUR",
    "₱": "PHP",
    "₲": "PYG",
    "₴": "UAH",
    "₹": "INR",
}
CURRENCY_REGEX = re.compile(
    "({})+".format("|".join(re.escape(c) for c in CURRENCIES.keys()))
)

PUNCT_TRANSLATE_UNICODE = dict.fromkeys(
    (i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")),
    "",
)

ACRONYM_REGEX = re.compile(
    r"(?:^|(?<=\W))(?:(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|[0-9]s?))|(?:[0-9](?:\-?[A-Z])+))(?:$|(?=\W))",
    flags=re.UNICODE,
)

EMAIL_REGEX = re.compile(
    r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}(?:$|(?=\b))",
    flags=re.IGNORECASE | re.UNICODE,
)

PHONE_REGEX = re.compile(
    r"(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))"
)

NUMBERS_REGEX = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))"
)

LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
MULTI_WHITESPACE_TO_ONE_REGEX = re.compile(r"\s+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")

# source: https://gist.github.com/dperini/729294
# @jfilter: I guess it was changed
URL_REGEX = re.compile(
    r"(?:^|(?<![\w\/\.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?" r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r"|" r"(?:(localhost))" r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:\/[^\)\]\}\s]*)?",
    # r"(?:$|(?![\w?!+&\/\)]))",
    # @jfilter: I removed the line above from the regex because I don't understand what it is used for, maybe it was useful?
    # But I made sure that it does not include ), ] and } in the URL.
    flags=re.UNICODE | re.IGNORECASE,
)


strange_double_quotes = [
    "«",
    "‹",
    "»",
    "›",
    "„",
    "“",
    "‟",
    "”",
    "❝",
    "❞",
    "❮",
    "❯",
    "〝",
    "〞",
    "〟",
    "＂",
]
strange_single_quotes = ["‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"]

DOUBLE_QUOTE_REGEX = re.compile("|".join(strange_double_quotes))
SINGLE_QUOTE_REGEX = re.compile("|".join(strange_single_quotes))

log = logging.getLogger()

# fall back to `unicodedata`
try:
    from unidecode import unidecode
except:
    from unicodedata import normalize

    unidecode = lambda x: normalize("NFKD", x)
    log.warning(
        "Since the GPL-licensed package `unidecode` is not installed, using Python's `unicodedata` package which yields worse results."
    )


def fix_strange_quotes(text):
    """
    Replace strange quotes, i.e., 〞with a single quote ' or a double quote " if it fits better.
    """
    text = SINGLE_QUOTE_REGEX.sub("'", text)
    text = DOUBLE_QUOTE_REGEX.sub('"', text)
    return text


def fix_bad_unicode(text, normalization="NFC"):
    """
    Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
    this includes mojibake, HTML entities and other code cruft,
    and non-standard forms for display purposes.
    Args:
        text (str): raw text
        normalization ({'NFC', 'NFKC', 'NFD', 'NFKD'}): if 'NFC',
            combines characters and diacritics written using separate code points,
            e.g. converting "e" plus an acute accent modifier into "é"; unicode
            can be converted to NFC form without any change in its meaning!
            if 'NFKC', additional normalizations are applied that can change
            the meanings of characters, e.g. ellipsis characters will be replaced
            with three periods
    Returns:
        str
    """
    # fix if the unicode is fucked up
    try:
        text = text.encode().decode("unicode-escape")
    except:
        pass

    return fix_text(text, normalization=normalization)


def to_ascii_unicode(text, lang="en"):
    """
    Try to represent unicode data in ascii characters similar to what a human
    with a US keyboard would choose.
    Works great for languages of Western origin, worse the farther the language
    gets from Latin-based alphabets. It's based on hand-tuned character mappings
    that also contain ascii approximations for symbols and non-Latin alphabets.
    """
    # normalize quotes before since this improves transliteration quality
    text = fix_strange_quotes(text)

    lang = lang.lower()
    # special handling for German text to preserve umlauts
    if lang == "de":
        text = save_replace(text, lang=lang)

    text = unidecode(text)

    # important to remove utility characters
    if lang == "de":
        text = save_replace(text, lang=lang, back=True)
    return text


def normalize_whitespace(text, no_line_breaks=False):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more line breaks with a single newline. Also strip leading/trailing whitespace.
    """
    if no_line_breaks:
        text = MULTI_WHITESPACE_TO_ONE_REGEX.sub(" ", text)
    else:
        text = NONBREAKING_SPACE_REGEX.sub(
            " ", LINEBREAK_REGEX.sub(r"\n", text)
        )
    return text.strip()


def replace_urls(text, replace_with="<URL>"):
    """Replace all URLs in ``text`` str with ``replace_with`` str."""
    return URL_REGEX.sub(replace_with, text)


def replace_emails(text, replace_with="<EMAIL>"):
    """Replace all emails in ``text`` str with ``replace_with`` str."""
    return EMAIL_REGEX.sub(replace_with, text)


def replace_phone_numbers(text, replace_with="<PHONE>"):
    """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
    return PHONE_REGEX.sub(replace_with, text)


def replace_numbers(text, replace_with="<NUMBER>"):
    """Replace all numbers in ``text`` str with ``replace_with`` str."""
    return NUMBERS_REGEX.sub(replace_with, text)


def replace_digits(text, replace_with="0"):
    """Replace all digits in ``text`` str with ``replace_with`` str, i.e., 123.34 to 000.00"""
    return re.sub(r"\d", replace_with, text)


def replace_currency_symbols(text, replace_with="<CUR>"):
    """
    Replace all currency symbols in ``text`` str with string specified by ``replace_with`` str.
    Args:
        text (str): raw text
        replace_with (str): if None (default), replace symbols with
            their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
            otherwise, pass in a string with which to replace all symbols
            (e.g. "*CURRENCY*")
    Returns:
        str
    """
    if replace_with is None:
        for k, v in CURRENCIES.items():
            text = text.replace(k, v)
        return text
    else:
        return CURRENCY_REGEX.sub(replace_with, text)


def remove_punct(text):
    """
    Replace punctuations from ``text`` with whitespaces.
    Args:
        text (str): raw text
    Returns:
        str
    """
    return text.translate(PUNCT_TRANSLATE_UNICODE)

class MyDataset(Dataset):
	def __init__(self, data_file_name):
		super().__init__()

		data_path = data_file_name

		self.data_list = []
		self.end_of_text_token = " <|endoftext|> "
		
		with open(data_path) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			
			for row in csv_reader:
				data_str = f"{row[0]}: {row[1]}{self.end_of_text_token}"
				self.data_list.append(data_str)
		
	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, item):
		return self.data_list[item]

def get_data_loader(data_file_name):
	dataset = MyDataset(data_file_name)
	data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
	return data_loader

def train(epochs, data_loader, batch_size, tokenizer, model, device, optimizer, scheduler):	
	batch_counter = 0
	sum_loss = 0.0

	for epoch in range(epochs):
		print (f'Running {epoch+1} epoch')

		for idx, txt in enumerate(data_loader):
			txt = torch.tensor(tokenizer.encode(txt[0]))
			txt = txt.unsqueeze(0).to(device)
			outputs = model(txt, labels=txt)
			loss, _ = outputs[:2]
			loss.backward()
			sum_loss += loss.data

			if idx%batch_size==0:
				batch_counter += 1
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				model.zero_grad()

			if batch_counter == 10:
				print(f"Total Loss is {sum_loss}") #printed after every 10*batch_size
				batch_counter = 0
				sum_loss = 0.0

	return model

def save_model(model, name):
	"""
	Summary:
		Saving model to the Disk
	Parameters:
		model: Trained model object
		name: Name of the model to be saved
	"""
	print ("Saving model to Disk")
	torch.save(model.state_dict(), f"{name}.pt")
	return

def load_models():
	"""
	Summary:
		Loading Pre-trained model
	"""
	print ('Loading/Downloading GPT-2 Model')
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
	model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
	return tokenizer, model

def choose_from_top_k_top_n(probs, k=50, p=0.8):
	ind = np.argpartition(probs, -k)[-k:]
	top_prob = probs[ind]
	top_prob = {i: top_prob[idx] for idx,i in enumerate(ind)}
	sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}
	
	t=0
	f=[]
	pr = []
	for k,v in sorted_top_prob.items():
	  t+=v
	  f.append(k)
	  pr.append(v)
	  if t>=p:
				break
	top_prob = pr / np.sum(pr)
	token_id = np.random.choice(f, 1, p = top_prob)

	return int(token_id)

def load_tokenizer(model_name):
	"""
	Summary:
		Loading the trained model
	"""
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
	return tokenizer

def generate(tokenizer, model, sentences, label, device):
	generated_text = []
	with torch.no_grad():
	  for idx in range(sentences):
		  finished = False
		  cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to(device)
		  for i in range(100):
			  outputs = model(cur_ids, labels=cur_ids)
			  loss, logits = outputs[:2]

			  softmax_logits = torch.softmax(logits[0,-1], dim=0)

			  if i < 5:
				  n = 10
			  else:
				  n = 5

			  next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy()) #top-k-top-n sampling
			  cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)

			  if next_token_id in tokenizer.encode('<|endoftext|>'):
				  finished = True
				  break

		  if finished:	          
			  output_list = list(cur_ids.squeeze().to('cpu').numpy())
			  output_text = tokenizer.decode(output_list)
			  generated_text.append(output_text)
		  else:
			  output_list = list(cur_ids.squeeze().to('cpu').numpy())
			  output_text = tokenizer.decode(output_list)
			  generated_text.append(output_text)
	return generated_text

class AutoNLP:
  def __init__(self):
    pass
  def clean( self,
    text,
    fix_unicode=True,
    to_ascii=True,
    lower=True,
    no_line_breaks=True,
    no_urls=True,
    no_emails=True,
    no_phone_numbers=True,
    no_numbers=True,
    no_digits=True,
    no_currency_symbols=True,
    no_punct=True,
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en",):
    """
    Normalize various aspects of a raw text. A convenience function for applying all other preprocessing functions in one go.
    Args:
        text (str): raw text to preprocess
        fix_unicode (bool): if True, fix "broken" unicode such as
            mojibake and garbled HTML entities
        to_ascii (bool): if True, convert non-to_ascii characters
            into their closest to_ascii equivalents
        lower (bool): if True, all text is lower-cased
        no_line_breaks (bool): if True, strip line breaks from text
        no_urls (bool): if True, replace all URL strings with a special URL token
        no_emails (bool): if True, replace all email strings with a special EMAIL token
        no_phone_numbers (bool): if True, replace all phone number strings
            with a special PHONE token
        no_numbers (bool): if True, replace all number-like strings
            with a special NUMBER token
        no_digits (bool): if True, replace all digits with a special DIGIT token
        no_currency_symbols (bool): if True, replace all currency symbols
            with a special CURRENCY token
        no_punct (bool): if True, remove all punctuation (replace with
            empty string)
        replace_with_url (str): special URL token, default "<URL>",
        replace_with_email (str): special EMAIL token, default "<EMAIL>",
        replace_with_phone_number (str): special PHONE token, default "<PHONE>",
        replace_with_number (str): special NUMBER token, default "<NUMBER>",
        replace_with_digit (str): special DIGIT token, default "0",
        replace_with_currency_symbol (str): special CURRENCY token, default "<CUR>",
        lang (str): special language-depended preprocessing. Besides the default English ('en'), only German ('de') is supported
    Returns:
        str: input ``text`` processed according to function args
    Warning:
        These changes may negatively affect subsequent NLP analysis performed
        on the text, so choose carefully, and preprocess at your own risk!
    """

    if text is None:
        return ""

    text = str(text)

    if fix_unicode:
        text = fix_bad_unicode(text)
    if no_currency_symbols:
        text = replace_currency_symbols(text, replace_with_currency_symbol)
    if to_ascii:
        text = to_ascii_unicode(text, lang=lang)
    if no_urls:
        text = replace_urls(text, replace_with_url)
    if no_emails:
        text = replace_emails(text, replace_with_email)
    if no_phone_numbers:
        text = replace_phone_numbers(text, replace_with_phone_number)
    if no_numbers:
        text = replace_numbers(text, replace_with_number)
    if no_digits:
        text = replace_digits(text, replace_with_digit)
    if no_punct:
        text = remove_punct(text)
    if lower:
        text = text.lower()

    # always normalize whitespace
    text = normalize_whitespace(text, no_line_breaks)

    return text

  def Model_Names(self):
    """
    Simply call this function to get the list of available models (internet required)
    """
    page = requests.get("https://huggingface.co/models?filter=tf,text-classification")
    soup = BeautifulSoup(page.content, 'html.parser')
    for element in soup.find_all('ul')[-1]:
        try:
            name = str(element).split('code>')[1].strip().replace('>','').replace('<','').replace('"','')[:-1]
            print(name)
        except:
            continue

  def tokenize(self, text):
    """
    Input: Text
    Return: Tokenized 
    """
    tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    encoded = tokenizer.batch_encode_plus(
      text,
      return_token_type_ids = False,
      pad_to_max_length = True,
      max_length = self.MAX_LEN
    )
    return np.array(encoded['input_ids'])
  
  def set_params(self, text, tokenizer_name = 'lschneidpro/distilbert_uncased_imdb', BATCH_SIZE = 32, model_name = 'lschneidpro/distilbert_uncased_imdb',
                 MAX_LEN = 100, epochs = 1, column = 'Text'):
    """
    Inputs:
        text: text
        tokenizer_name: Tokenizer name (call Model_Names() function to get all the names)
        BATCH_SIZE: Batch size
        model_name: Model's name (call Model_Names() function to get all the names)
        MAX_LEN: Maximum length of each sequence in text frame
        epochs: Number of training epochs
        Column: name of the text column
    """
    self.tokenizer_name = tokenizer_name
    self.BATCH_SIZE = BATCH_SIZE
    self.model_name = model_name
    self.MODEL_NAME = model_name
    self.MAX_LEN = MAX_LEN
    self.epochs = epochs
    self.column = column
    self.n_steps = text.shape[0] // BATCH_SIZE

  def train_model(self, text, y):
    """
    Input:
        text: text
        y: labels (numeric form)
    Return:
        model
    """
    transformer = TFAutoModel.from_pretrained(self.model_name)
    input_word_ids = Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    t_h = model.fit(text, y, epochs = self.epochs, steps_per_epoch = self.n_steps)
    return model

  def augmentation_train(self, DATA_FILE, BATCH_SIZE = 2, EPOCHS = 1, LEARNING_RATE = 0.01, WARMUP_STEPS = 100, MAX_SEQ_LEN = 100, MODEL_NAME = 'gpt2'):
    """
    DATA_FILE: Path of the data file
    BATCH_SIZE:  Batch size
    EPOCHS: Number of augmentation epochs
    LEARNING_RATE: Learning Rate
    WARMUP_STEPS: Warm up steps
    MAX_SEQ_LEN: Maximum sequence length in each of text
    MODEL_NAME: For now, only gpt2 is supported. more will be added in future.
    """
    TOKENIZER, MODEL = load_models()
    LOADER = get_data_loader(DATA_FILE)

    DEVICE = 'cpu'
    if torch.cuda.is_available():
      DEVICE = 'cuda'

    model = MODEL.to(DEVICE)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)

    model = train(EPOCHS, LOADER, BATCH_SIZE, TOKENIZER, MODEL, DEVICE, optimizer, scheduler)
    self.MODEL = model

  def augmentation_generate(self, y, SENTENCES = 10):
    """
    RETURNS AUGMENTED TEXT 
    
    y: label column (numeric form)
    SENTENCES: Sentences to generate for each of the label. eg, if there are 5 labels and SENTENCES = 10 then total 50 sentences will be generated
    """
    aug_text = pd.DataFrame(columns=['Text','labels'], index = np.arange(SENTENCES*len(np.unique(y))))
    j = 0

    for LABEL in tqdm.tqdm(np.unique(y)):
      TOKENIZER = load_tokenizer(self.MODEL_NAME)
      DEVICE = 'cpu'
      if torch.cuda.is_available():
        DEVICE = 'cuda'
      texts = generate(TOKENIZER, self.MODEL, SENTENCES, [int(LABEL)], DEVICE)
      
      aug_text.loc[j:j+SENTENCES-1,'Text'] = texts
      aug_text.loc[j:j+SENTENCES-1,'labels'] = int(LABEL)

      j += SENTENCES

    return aug_text