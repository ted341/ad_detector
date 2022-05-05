if [ "$1" = "1" ]
then
    ./myprogram.py ./dataset/Videos/data_test1.rgb ./dataset/Videos/data_test1.wav outputVideo.rgb outputAudio.wav  -d 1
elif [ "$1" = "2" ]
then 
    ./myprogram.py ./dataset2/Videos/data_test2.rgb ./dataset2/Videos/data_test2.wav outputVideo.rgb outputAudio.wav -d 2
elif [ "$1" = "3" ]
then
    ./myprogram.py ./dataset3/Videos/data_test3.rgb ./dataset3/Videos/data_test3.wav outputVideo.rgb outputAudio.wav -d 3
fi