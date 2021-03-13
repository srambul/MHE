#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int temporaryInput = 0;
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

int main(int argc, char* argv[])
{
    inputVector.clear();
    for (int x = 1; x < argc ;x++)
    {
        
        temporaryInput = atoi(argv[x]);
        inputVector.push_back(temporaryInput);
    }

    while (isDividableByThree == false)
    {

        if (inputVector.size() % 3 == 0)
        {
            isDividableByThree = true;
            
        }
        else
        {
            cout << "The set You provided isn't dividable by 3. Please run program once again and insert correct size set " << endl;
            cout << "The set size You provided is" + inputVector.size() << endl;

            return 0;
        }
    }
    calculate(inputVector);


    cout << "Type anything and press enter to close";
    int x;
    cin >>x;
    return 0;
}

