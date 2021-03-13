#include <iostream>
#include <vector>


using namespace std;

int temporaryInput;
int vectorSize;
bool isDividableByThree = false;
vector<int> inputVector;

int calculate(vector<int> inputVectorSet)
{
    
    cout << "Input Set = { ";
    for (int n : inputVectorSet) {
        std::cout << n << ", ";
    }
    std::cout << "}; \n";
    return 0;
}

int main()
{

    while (isDividableByThree == false)
    {
        
        
        while ((cin >> temporaryInput) && temporaryInput != 9999)
        {
            cout << "insert divided by 3 set of numbers ";
            inputVector.push_back(temporaryInput);
        }
            

        if (inputVector.size() % 3 == 0)
        {
            isDividableByThree = true;
        }
    }
    calculate(inputVector);

    return 0;
}

