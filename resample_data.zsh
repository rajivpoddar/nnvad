#!/bin/zsh
zmodload zsh/mathfunc

function random_string() {
    echo `cat /dev/urandom | env LC_CTYPE=C tr -cd 'a-f0-9' | head -c 6`
}

for file in `ls data_20ms/audio/*.wav`
do
    fn=`basename $file`
    prefix=${fn:0:1}

    fn=`random_string`
    f1=data/audio/${prefix}_${fn}.wav
    f2=data/specs/${prefix}_${fn}.png
    sox $file $f1 trim 0 0.1 spectrogram -x 128 -y 128 -w Hamming -r -o $f2 1>/dev/null 2>&1

    fn=`random_string`
    f1=data/audio/${prefix}_${fn}.wav
    f2=data/specs/${prefix}_${fn}.png
    sox $file $f1 trim 0.1 0.1 spectrogram -x 128 -y 128 -w Hamming -r -o $f2 1>/dev/null 2>&1
done
