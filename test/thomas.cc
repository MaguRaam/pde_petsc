//  Sample code for problem 2 of assignment 3. 
#include<iostream>
#include<cstdlib>
#include<fstream>
#include<cstring>
#include<math.h>

using namespace std ;

double const PI = 4.0*atan(1.0); // Value of PI

// Solution of a tridiagonal system. Diagonal elements are d, a is to the right of diagonal and b is 
// to the left. C denotes the right hand side. U is the final solution. Note that values of d and C are
// overwritten inside the loop.
void Tridiagonal(double*& b,double*& d,double*& a,double*& C, double*& u,int n){
	
	int i;

	for( int i = 1 ; i < n ; i++) {
		d[i] = d[i] - b[i]*(a[i-1]/d[i-1]);
		C[i] = C[i] - b[i]*(C[i-1]/d[i-1]);
	}
	u[n-1] = C[n-1]/d[n-1];
	for(i = n-2;i>=0;i--) u[i] = ( C[i]-a[i]*u[i+1])/d[i];
}


 


int main() {

	cout.flags( ios::dec | ios::scientific );
	cout.precision(5);
    
	double *x, *b, *d, *a, *u, Exact, *RHS, *Sol ; 
	double L2Error = 0.0, MaxError = 0.0 ;
	double beta ;

	int N, i;

	cout <<"Enter number of computational nodes : "; cin >> N ;
	cout <<"Enter parameter beta : "; cin >> beta ;

	x = new double[N+1] ; u = new double[N+1] ; 
	RHS = new double[N-1] ; Sol = new double[N-1] ;
	b = new double[N-1] ; d = new double[N-1] ; a = new double[N-1] ; 

	// set a uniform grid.
	for(i = 0 ; i <= N ; i++) x[i] = i/double(N) ;	

	// set right hand side 
	for(i = 1 ; i <= N-1 ; i++) RHS[i-1] = -sin(beta*PI*x[i])/double(N*N) ; 
	
	// set linear system for the FEM problem
	for(i = 1 ; i <= N-1 ; i++) {
		if(i == 1) { // starting point
			d[i-1] = -2.0 ;
			a[i-1] = 1.0 ;
		} else if(i == N-1) { // end point
			d[i-1] = -2.0 ;
			b[i-1] = 1.0 ;
		} else { // interior points
			d[i-1] = -2.0 ;
			b[i-1] = 1.0 ;
			a[i-1] = 1.0 ;
		}
	}
	u[0] = u[N] = 0.0 ; // boundary condition

	// Account for the change in linear system due to boundary condition
	// Here there is no change because of homogeneous BC
	RHS[0] -= u[0] ;
	RHS[N-2] -= u[N] ;

	// Tridiagonal solver
	Tridiagonal(b,d,a,RHS,Sol,N-1) ;
	for(i = 0 ; i < N-1 ; i++) u[i+1] = Sol[i] ; // copy the solution into final temperature

	// Write the final solution and compute errors.
	ofstream File("Output.dat", ios::out) ;
	File.flags( ios::dec | ios::scientific );
	File.precision(16) ;
	if(!File) {cerr<< "Error: Output file couldnot be opened.\n";}

	L2Error = MaxError = 0.0 ;
	for(i = 0 ; i <= N ; i++) {

		Exact = sin(beta*PI*x[i])/(beta*beta*PI*PI)  ;

		L2Error += (u[i]-Exact)*(u[i]-Exact)/(N+1.0) ;
		if(fabs(u[i]-Exact) > MaxError) MaxError = fabs(u[i]-Exact);

		File << x[i] << "\t" << u[i] << "\t" << Exact << endl ;
	}
	File.close() ;
	L2Error = sqrt(L2Error) ; 
	cout << "\n Error for N = " << N +1 ;
	cout << "\n L2 Error : " << L2Error << "\n Max Error : " << MaxError << endl ;


	// Free all the memory and exit
	delete[] x ; delete [] u ; 
	delete [] RHS ; delete [] Sol ;
	delete[] b ; delete[] d ; delete[] a ; 
	return 0;
}