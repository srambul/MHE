echo "simAnnealing"; 
python ./main.py --iterations=500 --generateproblem --simannealing > simAnnealing.txt
echo "hillClimbingDeterministic"; 
python ./main.py --iterations=300 --generateproblem --hillclimbingdeterministic > hillClimbingDeterministic.txt
echo "hillClimbingRandomized"; 
python ./main.py --iterations=300 --generateproblem --hillclimbingrandomized > hillClimbingRandomized.txt
