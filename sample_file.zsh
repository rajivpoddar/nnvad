#!/bin/zsh
zmodload zsh/mathfunc

#set -x

if [ $# -lt 1 ]
then
    echo "usage: ./sample_file.sh file"
    exit 1
fi

file=$1

function random_string() {
    echo `cat /dev/urandom | env LC_CTYPE=C tr -cd 'a-f0-9' | head -c 6`
}

function die() {
    rm -rf $f 1>/dev/null 2>&1
    echo $1
    exit 1
}

if [ ! -f $file ] 
then
    echo "$file not found"
    exit 1
fi

f=`random_string`.wav
echo "downsampling..."
sox $file -r 8k -b 8 $f remix - 1>/dev/null 2>&1

start_sec=60
step=360
duration=$((int(`soxi -D $f`)))
while [ $((start_sec + step)) -lt $duration ]
do
    fn=`random_string`
    sox $f ${fn}.wav trim $start_sec $step 1>/dev/null 2>&1

    if [ ! -f ${fn}.wav ]
    then
        die "sox trim failed"
    fi

    ts=`python energy.py ${fn}.wav -n 5 -s 200 -w 0.20 | cut -d ' ' -f1`
    for t in `echo $ts`
    do
        fn2=`random_string`
        sox ${fn}.wav ${fn2}.wav trim $t 0.20 1>/dev/null 2>&1

        if [ ! -f ${fn2}.wav ]
        then
            die "sox trim failed"
        fi

        while true
        do
            echo "\nplaying ${fn2}.wav"
            play -q ${fn2}.wav gain -l 12 1>/dev/null 2>&1
            prediction=`python mlp_vad.py ${fn2}.wav`
            echo -n "is this $prediction? (S/n/r/d)? "
            old_stty_cfg=$(stty -g)
            stty raw -echo ; yn=$(head -c 1) ; stty $old_stty_cfg
            case ${yn:0:1} in
                ""|S|s ) 
                    mv ${fn2}.wav data_200ms/audio/s_${fn2}.wav
                    echo "marked ${fn2} as speech"
                    break
                ;;

                N|n ) 
                    mv ${fn2}.wav data_200ms/audio/n_${fn2}.wav
                    echo "marked ${fn2} as noise"
                    break
                ;;

                D|d ) 
                    rm ${fn2}.wav
                    echo "discarded ${fn2}"
                    break
                ;;
            esac
        done
    done

    rm ${fn}.wav 1>/dev/null 2>&1

    start_sec=$((start_sec + step))
done

rm $f
