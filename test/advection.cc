/*solve advection equation using FTBS, CTCS, BTBS, CN schemes*/


#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

void write_matplotlib(const std::vector<double>& x, const std::vector<double>& u, const std::vector<double>& uexact)
{

    std::fstream file;
    file.flags( std::ios::dec | std::ios::scientific );
    file.precision(16);
    file.open("plot.dat", std::ios::out);
    if (!file) std::cerr << "File not created!";

    int N = x.size();
    for (int i = 0; i < N; ++i) file<<x[i]<<"\t\t"<<u[i]<<"\t\t"<<uexact[i]<<"\n";  

}

int main()
{
    std::cout.flags( std::ios::dec | std::ios::scientific );
    std::cout.precision(5);

    //simulation parameters:
    int N = 101;
    double h = 1.0/(double)(N - 1);
    double cfl = 0.5;
    double dt = cfl*h;
    double T = 5;
    double t = 0;

    //create grid:
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = i*h;

    //initial condition:
    std::vector<double> u(N, 0.0), unew(N, 0.0), uexact(N, 0.0);
    for (int i = 0; i < N; ++i) u[i] = cos(4*M_PI*x[i]);

    while (t < T - dt){

        //loop over grid points and update:
        for (int i = 0; i < N; i++){
            
            if (i == 0)     unew[0] = u[0] - cfl*(u[0] - u[N-1]); 
            else            unew[i] = u[i] - cfl*(u[i] - u[i-1]);  

        }

        //copy unew to u:
        u = unew;

        t+=dt;
    }

    //compute exact solution:
    for (int i = 0; i < N; ++i) uexact[i] = cos(4*M_PI*x[i] - 4*M_PI*t);

    //compute Linfty error:
    double error = 0.0;
    for (int i = 0; i < N; ++i){
        if(fabs(u[i]-uexact[i]) > error) error = fabs(u[i]-uexact[i]);
    }

    std::cout<<"Linfty error at t = "<<t<<"\tis "<<error<<std::endl;

    //plot solution:
    write_matplotlib(x, u, uexact);


    return 0;
}