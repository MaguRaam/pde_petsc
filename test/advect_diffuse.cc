#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>


using namespace std;
double const PI = 4.0*atan(1.0); // Value of PI


void thomas(int N,const vector<double>& b,const vector<double>& a,
                const vector<double>& c,vector<double>& x,const vector<double>& q){

    int i;
    vector<double> l(N,0.0);
    vector<double> u(N,0.0);
    vector<double> d(N,0.0);
    vector<double> y(N,0.0);
    
    // LU Decomposition:
    d[0] = a[0];
    u[0] = c[0];
    
    for(i=0;i<N-2;i++){
        l[i] = b[i]/d[i];
        d[i+1] = a[i+1] - l[i]*u[i];
        u[i+1] = c[i+1];
    }
    
    l[N-2] = b[N-2]/d[N-2];
    d[N-1] = a[N-1] - l[N-2]*u[N-2];
    
    //Forward substitution [L][y] = [q] :
    y[0] = q[0];
    for(i=1;i<N;i++)
        y[i] = q[i] - l[i-1]*y[i-1];
    
    /* Backward Substitution [U][x] = [y] */
    x[N-1] = y[N-1]/d[N-1];
    for(i=N-2;i>=0;i--)
        x[i] = (y[i] - u[i]*x[i+1])/d[i];
}


void FTCS(const std::vector<double>& u, std::vector<double>& un, double cfl, double gamma)
{
	int N = u.size() - 1;

	double alpha = (gamma - 0.5*cfl);
	double beta  = (gamma + 0.5*cfl);
	

    for (int i = 0; i <= N; ++i){
        
        if (i == 0)     un[i] = 0;     //left bcs
        if (i == N)     un[i] = 1.0;   //right bcs
        else            un[i] = u[i+1]*alpha + u[i-1]*beta + u[i]*(1.0 - 2.0*gamma); //interior grid point
    }

}


void BTCS(std::vector<double>& u, std::vector<double>& un, double cfl, double gamma)
{
    
    int N = u.size() - 1, i;

    std::vector<double> sol(N-1,0.0), RHS(N-1,0.0);

    //create rhs:
    for(i = 1 ; i <= N-1 ; i++) RHS[i-1] = u[i]; 


    //build A matrix:
    std::vector<double> a(N-1, 1.0 + 2*gamma), b(N-1, -0.5*cfl - gamma), c(N-1, 0.5*cfl - gamma);

    //solve Aun = u:
    RHS[0] -= 0.0;
    RHS[N-2] -= (0.5*cfl - gamma);

    thomas(N-1, b, a, c, sol, RHS);

    for(i = 0 ; i < N-1 ; i++) un[i+1] = sol[i];
    un[0] = 0.0;
    un[N] = 1.0;
}



int main()
{
    std::cout.flags( std::ios::dec | std::ios::scientific );
    std::cout.precision(5);

    //simulation parameters:
    int i = 0;
    int N = 1000;
    double h = 1.0/(double)N;
    double c  = 1;
    double nu = 0.01;
    double dt = 0.1;
    double T = 1000;
    double t = 0;
    double cfl    = c*dt/h; /*not a stability criteria but needed for computation*/


    //stability criteria:
    double gamma  = nu*dt/(h*h);
    double peclet = c*h/nu; 

    std::cout<<"peclet = "<<peclet<<"\t"<<"gamma = "<<gamma<<std::endl;
    std::cout<<"boundary layer thickness = "<<(nu/c)<<std::endl;
    std::cout<<"grid size h = "<<h<<std::endl;

    //allocate memory and initialize:
    std::vector<double> x(N+1), u(N+1, 0.0), un(N+1, 0.0); 

    //create grid:
    for (i = 0; i <= N; ++i) x[i] = i*h;

    //evolve in time:
    while (t < T){

        BTCS(u, un, cfl, gamma);

        u = un;
        t+=dt;
    }

    std::cout<<"end time = "<<t<<std::endl;

    //write solution to file:
    ofstream File("plot.dat", ios::out) ;
    File.flags( ios::dec | ios::scientific );
    File.precision(16) ;
    if(!File) {cerr<< "Error: Output file couldnot be opened.\n";}

    double L2Error = 0.0, MaxError = 0.0, Exact;
    for(i = 0 ; i <= N ; i++) {

        Exact = (1.0 - exp(c*x[i]/nu) )/ (1.0 - exp(c/nu) );

        if(fabs(u[i]-Exact) > MaxError) MaxError = fabs(u[i]-Exact);

        File << x[i] << "\t" << u[i] << "\t" << Exact << endl ;
    }
    File.close() ;
    L2Error = sqrt(L2Error) ; 
    cout << "Linfty error = "<<MaxError << endl ;



    return 0;
}
