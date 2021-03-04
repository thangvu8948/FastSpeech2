""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""


# Regular expression matching whitespace:
import re
from unidecode import unidecode
from unicodedata import normalize
from .numbers import normalize_numbers
from num2words import num2words
from dataset.texts.vicleaners import cleaners

_whitespace_re = re.compile(r"\s+")
#punctuations = """+-!()[]{};:'"\<>/?@#^&*~,"""
punctuations = """+-!()[]{};'"\<>/?@#^&*~"""
# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s" % x[0], re.UNICODE), x[1])
    for x in [
        ("[ ]*[%]", " phần trăm "),
        ("csgt", " cảnh sát giao thông "),
        ("gvcn", " giáo viên chủ nhiệm "),
        ("[ ]*[+]", " cộng "),
        ("[ ]*[/]", " phần "),
        ("[ ]*[*]", " nhân "),
        ("[ ]*[=]", " bằng "),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]
def vi_num2words(num):
   return num2words(num, lang='vi')

def convert_time_to_text(time_string):
   # Support only hh:mm format
   try:
       h, m = time_string.split(":")
       time_string = vi_num2words(int(h)) + " giờ " + vi_num2words(int(m)) + " phút"
       return time_string
   except:
       return None

def replace_time(text):
   # Define regex to time hh:mm
   result = re.findall(r'\d{1,2}:\d{1,2}|', text)
   match_list = list(filter(lambda x : len(x), result))

   for match in match_list:
       if convert_time_to_text(match):
           text = text.replace(match, convert_time_to_text(match))
   return text

def replace_number(text):
   return re.sub('(?P<id>\d+)', lambda m: vi_num2words(int(m.group('id'))), text)

def normalize_text(text):
   text = normalize("NFC", text)
   text = text.lower()
   #text = expand_abbreviations(text)
   #text = replace_time(text)
   #text = replace_number(text)
   #text = collapse_whitespace(text)
   text = cleaners(text)
   return text

def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = normalize_text(text)
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)




def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def punctuation_removers(text):
    no_punct = ""
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct
