mat=$1
data=$2

for folder in $(ls $mat)
do
    echo $folder
    cp -R $mat/$folder/Material $data/$folder/
done
