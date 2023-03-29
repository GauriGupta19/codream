read -p "Enter number of users: " n
# create a string of "localhost," repeated n times
localhosts=""
for (( i=1; i<=$n; i++ ))
do
  localhosts+="localhost,"
done
localhosts=${localhosts::-1} # remove the trailing comma
# run the following command - mpirun -np $n -H $localhosts python main.py
mpirun -np $n -H $localhosts python main.py
