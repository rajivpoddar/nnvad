#!/bin/zsh
zmodload zsh/mathfunc

function random_string() {
    echo `cat /dev/urandom | env LC_CTYPE=C tr -cd 'a-f0-9' | head -c 6`
}

files=`print -lr -- data/audio/*.wav(om)`
total=`echo $files | wc -l | tr -d ' '`

count=0
for file in `echo $files`
do
    count=$((count+1))
    fn=`basename $file`
    prefix=${fn:0:1}

    while true
    do
        echo "\nplaying ${fn}, $count/$total"
        play -q ${file} speed 0.05 gain -l 12 1>/dev/null 2>&1
        prediction=`python mlp_vad.py ${file}`
        echo -n "is this $prediction? (s/n/r/d)? "
        old_stty_cfg=$(stty -g)
        stty raw -echo ; yn=$(head -c 1) ; stty $old_stty_cfg
        case ${yn:0:1} in
            S|s ) 
                if [ $prefix != 's' ]
                then
                    f2=data/audio/s_`random_string`.wav
                    mv $file $f2
                    echo "marked ${fn} as speech, `basename $f2`"
                fi
                break
            ;;

            N|n ) 
                if [ $prefix != 'n' ]
                then
                    f2=data/audio/n_`random_string`.wav
                    mv $file $f2
                    echo "marked ${fn} as noise, `basename $f2`"
                fi
                break
            ;;

            D|d ) 
                rm ${file}
                echo "discarded ${fn}"
                break
            ;;
        esac
    done
done
