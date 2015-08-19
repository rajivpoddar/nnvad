#!/bin/zsh
zmodload zsh/mathfunc

function random_string() {
    echo `cat /dev/urandom | env LC_CTYPE=C tr -cd 'a-f0-9' | head -c 6`
}

for file in `find data_200ms/audio -name n_\*.wav -mmin -7`
do
    fn=`basename $file`
    prefix=${fn:0:1}

    start_sec=0
    step=0.025

    for i in {0..7}
    do
        fn=`random_string`
        f1=data/audio/${prefix}_${fn}.wav
        f2=data/specs/${prefix}_${fn}.png
        sox $file $f1 trim $start_sec $step spectrogram -x 128 -y 128 -w Hamming -r -o $f2 1>/dev/null 2>&1

        start_sec=$((start_sec + step))
        start_sec=`printf "%0.3f" start_sec`
    done
done
