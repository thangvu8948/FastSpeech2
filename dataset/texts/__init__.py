""" from https://github.com/keithito/tacotron """
import re
from dataset.texts import cleaners
from dataset.texts.symbols import (
    symbols,
    _eos,
    phonemes_symbols,
    PAD,
    EOS,
    _PHONEME_SEP,
)
from dataset.texts.dict_ import symbols_
import nltk
from g2p_en import G2p

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

symbols_inv = {v: k for k, v in symbols_.items()}

valid_symbols = [
    "WB", "a1_T1", "a1_T2", "a1_T3", "a1_T4", "a1_T5", "a1_T6", "a2_T1", "a2_T2", "a2_T3", "a2_T4", "a2_T5", "a2_T6",
    "a3_T1", "a3_T2", "a3_T3", "a3_T4", "a3_T5", "a3_T6", "ai_T1", "ai_T2", "ai_T3", "ai_T4",
    "ai_T5", "ai_T6", "ao_T1", "ao_T2", "ao_T3", "ao_T4", "ao_T5", "ao_T6", "au3_T1", "au3_T2", "au3_T3",
    "au3_T4", "au3_T5", "au3_T6", "au_T1", "au_T2", "au_T3", "au_T4", "au_T5", "au_T6", "ay3_T1", "ay3_T2",
    "ay3_T3", "ay3_T4", "ay3_T5", "ay3_T6", "ay_T1", "ay_T2", "ay_T3", "ay_T4", "ay_T5", "ay_T6", "b", "ch",
    "d1", "d2", "e1_T1", "e1_T2", "e1_T3", "e1_T4", "e1_T5", "e1_T6", "e2_T1", "e2_T2", "e2_T3", "e2_T4", "e2_T5",
    "e2_T6", "eo_T1", "eo_T2", "eo_T3", "eo_T4", "eo_T5", "eo_T6", "eu_T2", "eu_T3", "eu_T4", "eu_T5", "eu_T6",
    "g", "h", "i_T1", "i_T2", "i_T3", "i_T4", "i_T5", "i_T6", "ie2_T1", "ie2_T2", "ie2_T3", "ie2_T4", "ie2_T5",
    "ie2_T6", "ieu_T1", "ieu_T2", "ieu_T3", "ieu_T4", "ieu_T5", "ieu_T6", "iu_T1", "iu_T2", "iu_T3", "iu_T4",
    "iu_T5", "iu_T6", "j", "k", "kh", "l", "m", "n", "ng", "nh", "o1_T1", "o1_T2", "o1_T3", "o1_T4", "o1_T5", "o1_T6",
    "o2_T1", "o2_T2", "o2_T3", "o2_T4", "o2_T5", "o2_T6", "o3_T1", "o3_T2", "o3_T3", "o3_T4", "o3_T5", "o3_T6",
    "oa_T1", "oa_T2", "oa_T3", "oa_T4", "oa_T5", "oa_T6", "oe_T1", "oe_T2", "oe_T3", "oe_T4", "oe_T5", "oe_T6",
    "oi2_T1", "oi2_T2", "oi2_T3", "oi2_T4", "oi2_T5", "oi2_T6", "oi3_T1", "oi3_T2", "oi3_T3", "oi3_T4",
    "oi3_T5", "oi3_T6", "oi_T1", "oi_T2", "oi_T3", "oi_T4", "oi_T5", "oi_T6", "p", "ph", "r", "s", "sil", "t", "th", "tr",
    "u1_T1", "u1_T2", "u1_T3", "u1_T4", "u1_T5", "u1_T6", "u2_T1", "u2_T2", "u2_T3", "u2_T4", "u2_T5", "u2_T6",
    "ua2_T1", "ua2_T2", "ua2_T3", "ua2_T4", "ua2_T5", "ua2_T6", "ua_T1", "ua_T2", "ua_T3", "ua_T4", "ua_T5",
    "ua_T6", "ui2_T1", "ui2_T2", "ui2_T3", "ui2_T4", "ui2_T5", "ui2_T6", "ui_T1", "ui_T2", "ui_T3", "ui_T4",
    "ui_T5", "ui_T6", "uoi2_T1", "uoi2_T2", "uoi2_T3", "uoi2_T4", "uoi2_T5", "uoi2_T6", "uoi3_T1", "uoi3_T2",
    "uoi3_T3", "uoi3_T4", "uoi3_T5", "uoi3_T6", "uou_T1", "uou_T2", "uou_T3", "uou_T4", "uou_T6", "uu2_T1",
    "uu2_T2", "uu2_T3", "uu2_T4", "uu2_T5", "uu2_T6", "uy_T1", "uy_T2", "uy_T3", "uy_T4", "uy_T5", "uy_T6","v", "x", "pau","spn",
]


def pad_with_eos_bos(_sequence):
    return _sequence + [_symbol_to_id[_eos]]


def text_to_sequence(text, cleaner_names, eos):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    if eos:
        text = text + "~"
    try:
        sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
    except KeyError:
        print("text : ", text)
        exit(0)

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in symbols_inv:
            s = symbols_inv[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [symbols_[s.upper()] for s in symbols]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"


# For phonemes
_phoneme_to_id = {s: i for i, s in enumerate(valid_symbols)}
_id_to_phoneme = {i: s for i, s in enumerate(valid_symbols)}


def _should_keep_token(token, token_dict):
    return (
        token in token_dict
        and token != PAD
        and token != EOS
        and token != _phoneme_to_id[PAD]
        and token != _phoneme_to_id[EOS]
    )


def phonemes_to_sequence(phonemes):
    string = phonemes.split() if isinstance(phonemes, str) else phonemes
    # string.append(EOS)
    sequence = list(map(convert_phoneme_CMU, string))
    sequence = [_phoneme_to_id[s] for s in sequence]
    # if _should_keep_token(s, _phoneme_to_id)]
    return sequence


def sequence_to_phonemes(sequence, use_eos=False):
    string = [_id_to_phoneme[idx] for idx in sequence]
    # if _should_keep_token(idx, _id_to_phoneme)]
    string = _PHONEME_SEP.join(string)
    if use_eos:
        string = string.replace(EOS, "")
    return string


def convert_phoneme_CMU(phoneme):
    REMAPPING = {
        'AA0': 'AA1',
        'AA2': 'AA1',
        'AE2': 'AE1',
        'AH2': 'AH1',
        'AO0': 'AO1',
        'AO2': 'AO1',
        'AW2': 'AW1',
        'AY2': 'AY1',
        'EH2': 'EH1',
        'ER0': 'EH1',
        'ER1': 'EH1',
        'ER2': 'EH1',
        'EY2': 'EY1',
        'IH2': 'IH1',
        'IY2': 'IY1',
        'OW2': 'OW1',
        'OY2': 'OY1',
        'UH2': 'UH1',
        'UW2': 'UW1',
    }
    return REMAPPING.get(phoneme, phoneme)


def text_to_phonemes(text, custom_words={}):
    """
    Convert text into ARPAbet.
    For known words use CMUDict; for the rest try 'espeak' (to IPA) followed by 'listener'.
    :param text: str, input text.
    :param custom_words:
        dict {str: list of str}, optional
        Pronounciations (a list of ARPAbet phonemes) you'd like to override.
        Example: {'word': ['W', 'EU1', 'R', 'D']}
    :return: list of str, phonemes
    """
    g2p = G2p()

    """def convert_phoneme_CMU(phoneme):
        REMAPPING = {
            'AA0': 'AA1',
            'AA2': 'AA1',
            'AE2': 'AE1',
            'AH2': 'AH1',
            'AO0': 'AO1',
            'AO2': 'AO1',
            'AW2': 'AW1',
            'AY2': 'AY1',
            'EH2': 'EH1',
            'ER0': 'EH1',
            'ER1': 'EH1',
            'ER2': 'EH1',
            'EY2': 'EY1',
            'IH2': 'IH1',
            'IY2': 'IY1',
            'OW2': 'OW1',
            'OY2': 'OY1',
            'UH2': 'UH1',
            'UW2': 'UW1',
        }
        return REMAPPING.get(phoneme, phoneme)
        """

    def convert_phoneme_listener(phoneme):
        VOWELS = ['A', 'E', 'I', 'O', 'U']
        if phoneme[0] in VOWELS:
            phoneme += '1'
        return phoneme  # convert_phoneme_CMU(phoneme)

    try:
        known_words = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download("cmudict")
        known_words = nltk.corpus.cmudict.dict()

    for word, phonemes in custom_words.items():
        known_words[word.lower()] = [phonemes]

    words = nltk.tokenize.WordPunctTokenizer().tokenize(text.lower())

    phonemes = []
    PUNCTUATION = "!?.,-:;\"'()"
    for word in words:
        if all(c in PUNCTUATION for c in word):
            pronounciation = ["pau"]
        elif word in known_words:
            pronounciation = known_words[word][0]
            pronounciation = list(
                pronounciation
            )  # map(convert_phoneme_CMU, pronounciation))
        else:
            pronounciation = g2p(word)
            pronounciation = list(
                pronounciation
            )  # (map(convert_phoneme_CMU, pronounciation))

        phonemes += pronounciation

    return phonemes
