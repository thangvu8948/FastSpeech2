import os
import subprocess
import regex as re
from dataset.texts.cleaners import punctuations
from helper import thirdparty_binary

fst_path = "D:\\FastSpeech2\\g2p\\model.fst"
word_list_path = "D:\\FastSpeech2\\g2p\\word_list.txt"
out_file = "phonemes.txt"
phonetisaurus = "D:\\FastSpeech2\\g2p\\phonetisaurus-g2pfst.exe"
def parse_output(output):
    for line in output.splitlines():
        line = line.strip().split("\t")
        if len(line) == 2:
            line += [None]
        yield line[0], line[2]

def parse_errors(error_output):
    missing_symbols = []
    line_regex = re.compile("Symbol: '(.+?)' not found in input symbols table")
    for line in error_output.splitlines():
        m = line_regex.match(line)
        if m is not None:
            missing_symbols.append(m.groups()[0])
    return missing_symbols

def text_to_phonemes (text):
    for c in ",.:?!;""/\\[]{}":
        text = text.replace(c, " " + c + " ")
    tokens = text.split(" ")
    tokens = list(filter(lambda x: x != "", tokens))
    with open(word_list_path, mode="w", encoding="utf-8") as f:
        for token in tokens:
            if token != "":
                f.write(token + "\n")
    proc = subprocess.Popen([phonetisaurus,
                             '--model=' + fst_path, '--wordlist=' + word_list_path],
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    results = stdout.decode('utf-8')
    errors = stderr.decode('utf-8')

    missing_symbols = parse_errors(errors)
    if missing_symbols:
        print("There were unmatched symbols in your transcriptions")

    phonemes = ""
    with open(out_file, "w", encoding='utf-8') as f:
        for word, pronunciation in parse_output(results):
            if pronunciation is None:
                f.write(word + "|" + "pau\n")
                continue
            f.write(word + "|" + pronunciation + "\n")
            phonemes += pronunciation + " "
    dict = {
        "," : "pau",
        '.' : "pau"
    }
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line[0] != 'S':
                line = line[0:]
            parts = line.strip().split('|')
            if (parts[0] not in ".,"):
                dict[parts[0]] = parts[1]

    res = ""
    for word in tokens:
        res += dict[word] + " "
    print(res)
    return res