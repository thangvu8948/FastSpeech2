import numpy as np
import librosa
import textgrid
from utils.files import get_files
def generate():
    n_fft = 1024
    sample_rate = 22050
    hop_length = 256
    #a = get_files("D:\\DatasetTTS\\mfa\\montreal-forced-aligner\\outdir\\wavs", extension=".TextGrid")
    a = get_files("D:\\FastSpeech2\\vivos\\out_vivos_f\\wavs_f", extension=".TextGrid")
    for i in range(0, len(a)):
        tgrid = textgrid.TextGrid.fromFile(a[i])
        time_list = []
        phonemes = []
        words = []
        seperator = ['|']
        diff_list = []
        frames = []
        wav = [".wav"]

        file_name = ((a[i].split("\\")[-1].split(".")[0]) + ".wav")
        time_list.append(0.0)
        for ph in tgrid[1]:
            time_list.append(ph.maxTime)
            if (ph.mark == "sp"):
                ph.mark = "pau"
            phonemes.append(ph.mark)

        frames = librosa.core.time_to_frames(np.array(time_list), sr=sample_rate, hop_length=hop_length,
                                             n_fft=n_fft)

        for j in range(1, frames.shape[0]):
            diff_list.append(frames[j] - frames[j - 1])
        for word in (tgrid[0]):
            words.append(word.mark)

        # raw_lists_concat = words + seperator + list(frames) + seperator + list(diff_list) + seperator + phonemes + seperator + file_name
        Filelist_content = ' '.join([str(elem) for elem in words]) + "|" + ' '.join(
            [str(elem) for elem in frames]) + "|" + ' '.join([str(elem) for elem in diff_list]) + "|" + ' '.join(
            [str(elem) for elem in phonemes]) + "|" + file_name + "\n"
        if (file_name[:8] in ["VIVOSDEV"]):
            with open("filelists/" + "valid_filelist.txt", encoding="utf8", mode="a") as f:
                f.write(Filelist_content)
        else:
            with open("filelists/" + "train_filelist.txt", encoding="utf8", mode="a") as f1:
                f1.write(Filelist_content)

def main():
    generate()
if __name__ == "__main__":
    main()