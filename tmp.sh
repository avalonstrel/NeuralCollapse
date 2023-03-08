
FILE=./exprs/tmp/tmp.txt

while [ ! -f "$FILE" ];
do
    sleep 10
done

echo "Finished."