import textgrid
import argparse
import os


def createFinalDict(args):
    with open(args.meta_dir, encoding="utf-8", mode="r") as meta:
        in_dir = os.path.join(args.base_dir, 'hcmus')
        for line in meta:
            if line[0] != 'S':
                line = line[0:]

            parts = line.strip().split(',')
            wav_name = parts[0]
            text = parts[1].replace('\t', ', ')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            if os.path.isfile(os.path.join(args.textgrid_dir, '%s.TextGrid' % parts[0])) == True:
                if (wav_name[:5] in ["SB001", "SB002", "SB003"]):
                    with open(args.output_valid, 'a', encoding="utf-8") as t:
                        tg = textgrid.TextGrid.fromFile(
                            os.path.join(args.textgrid_dir, '%s.TextGrid' % parts[0]))
                        content = parts[1] + " |" + joinText(textgrid.TextGrid.fromFile(
                            os.path.join(args.textgrid_dir, '%s.TextGrid' % parts[0]))) + " |" + parts[0]
                        t.write(content + '\n')
                else:
                    with open(args.output_train, 'a', encoding="utf-8") as d:
                        content = parts[1] + " |" + joinText(textgrid.TextGrid.fromFile(
                            os.path.join(args.textgrid_dir, '%s.TextGrid' % parts[0]))) + " |" + parts[0]
                        d.write(content + '\n')


def joinText(tgs):
    s = ""
    for tg in tgs[1]:
        if tg.mark != "":
            s += " " + tg.mark
    return s


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('D:\\tacotron2'))
    parser.add_argument(
        '--meta_dir', default="D:\\tacotron2\\hcmus\\metadata.txt")
    parser.add_argument('--textgrid_dir',
                        default="D:\\DatasetTTS\\mfa\\montreal-forced-aligner\\outdir\\wavs")
    parser.add_argument('--output_train', default='D:/FastSpeech2/filelists/train_filelist_old.txt')
    parser.add_argument('--output_valid', default='D:/FastSpeech2/filelists/valid_filelist_old.txt')
    args = parser.parse_args()
    createFinalDict(args)
# tg = textgrid.TextGrid.fromFile('D:\\DatasetTTS\\mfa\\montreal-forced-aligner\\outdir\\wavs\\SB001-0001.TextGrid')[1][0]
# print(tg.mark)
if __name__ == "__main__":
    main()

