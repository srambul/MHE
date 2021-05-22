echo "SimAnnealing"; 
python ./main.py --iterations=500 --generateproblem --simannealing > rep_simAnnealing.txt
echo "HillClimbingDeterministic"; 
python ./main.py --iterations=300 --generateproblem --hillclimbingdeterministic > rep_hillClimbingDeterministic.txt
echo "HillClimbingRandomized"; 
python ./main.py --iterations=300 --generateproblem --hillclimbingrandomized > rep_hillClimbingRandomized.txt
