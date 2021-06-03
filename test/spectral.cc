#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>


using namespace std;
double const PI = 4.0*atan(1.0); // Value of PI


void write(const vector<double>& x, const vector<double>& u){

    //write solution to file:
    ofstream File("plot.dat", ios::out) ;
    File.flags( ios::dec | ios::scientific );
    File.precision(16);
    if(!File) {cerr<< "Error: Output file couldnot be opened.\n";}

    for(i = 0 ; i <= N ; i++) File << x[i] << "\t" << u[i] << "\t" << endl;
}


int main()
{
    std::cout.flags( std::ios::dec | std::ios::scientific );
    std::cout.precision(5);

    //create grid:
    int i = 0, N = 100;
    double L = 1.0, L/double(N-1);
    std::vector<double> x(N+1);
    for (i = 0; i <= N; ++i) x[i] = i*h;


    //initialize:
    std::vector<double> u(N+1, 0.0); 

    double lamda = 0.5;
    double k = 2*PI/lamda;
    auto f = [k](double x){return cos(k*x); };

    
    


    return 0;
}
