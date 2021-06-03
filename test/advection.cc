/*solve advection equation using FTBS, CTCS, BTBS, CrankNicholson schemes*/


#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <vector>


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


void FTBS(const std::vector<double>& u, std::vector<double>& un, double cfl){

    int N = u.size() - 1;

    for (int i = 0; i <= N; ++i){
        
        if (i == 0)     un[0] = u[0] - cfl*(u[0] - u[N]);
        else            un[i] = u[i] - cfl*(u[i] - u[i-1]);
    }
}

void CTCS(const std::vector<double>& uold, const std::vector<double>& u, std::vector<double>& un, double cfl){

    int N = u.size() - 1;

    for (int i = 0; i <= N; ++i){
        
        if (i == 0)     un[0] = uold[0] - cfl*(u[1] - u[N]);
        else if(i == N) un[N] = uold[N] - cfl*(u[0] - u[N-1]);
        else            un[i] = uold[i] - cfl*(u[i+1] - u[i-1]);
    }
}

void print(const std::vector<double>& u){
    std::cout<<"\n";
    for (const auto& e : u) std::cout<<e<<" ";
    std::cout<<"\n";
}

inline
double dot(const std::vector<double>& x, const std::vector<double>& y){
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}




void BTBS(const std::vector<double>& B, std::vector<double>& X, double cfl){

    //N:
    int N = X.size() - 1;

    //build A matrix:
    std::vector<double> a(N+1, 1.0 + cfl), b(N+1, -cfl), c(N+1, 0.0);


    //solve Ay = B:
    std::vector<double> y(N+1, 0.0);
    thomas(N+1, b, a, c, y, B);

    //create u vector:
    std::vector<double> u(N+1, 0.0);
    u[0] = 1.0;

    //create v vector:
    std::vector<double> v(N+1, 0.0);
    v[N] = -cfl;

    //solve Az = u:
    std::vector<double> z(N+1, 0.0);
    thomas(N+1, b, a, c, z, u);    

    //now solve for  (A + uv)X = B using X = y - [vTy/(1 + vTz)].z
    double vTy = dot(v,y);    
    double vTz = dot(v,z);

    for (int i = 0; i <= N; ++i) X[i] = y[i] - ( vTy/(1 + vTz) )*z[i];

}

void CrankNicholson(const std::vector<double>& B, std::vector<double>& X, double cfl){
    
    //N:
    int N = X.size() - 1;    

    //build A matrix:
    std::vector<double> a(N+1, 1.0 + cfl), b(N+1, -cfl), c(N+1, 0.0);

    
    

}

 


int main()
{
    std::cout.flags( std::ios::dec | std::ios::scientific );
    std::cout.precision(5);

    //simulation parameters:
    int i = 0;
    int N = 100;
    double h = 1.0/(double)N;
    double cfl = 0.5;
    double dt = cfl*h;
    double T = 5;
    double t = 0;


    //allocate memory:
    std::vector<double> x(N+1), u(N+1), un(N+1); 

    //create grid:
    for (i = 0; i <= N; ++i) x[i] = i*h;

    //initial condition:
    std::transform(x.begin(), x.end(), u.begin(), [](double xi){return cos(4*PI*xi);});


    //evolve in time:
    while (t < T - dt){

        BTBS(u, un, cfl);

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

        Exact = cos(4*PI*x[i] - 4*PI*t);

        if(fabs(u[i]-Exact) > MaxError) MaxError = fabs(u[i]-Exact);

        File << x[i] << "\t" << u[i] << "\t" << Exact << endl ;
    }
    File.close() ;
    L2Error = sqrt(L2Error) ; 
    cout << "Linfty error = "<<MaxError << endl ;



    return 0;
}
