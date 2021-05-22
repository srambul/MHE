echo "SimAnnealing"; 
python ./main.py --iterations=500 --generateproblem --simannealing > simAnnealing.txt
echo "HillClimbingDeterministic"; 
python ./main.py --iterations=300 --generateproblem --hillclimbingdeterministic > hillClimbingDeterministic.txt
echo "HillClimbingRandomized"; 
python ./main.py --iterations=300 --generateproblem --hillclimbingrandomized > hillClimbingRandomized.txt
