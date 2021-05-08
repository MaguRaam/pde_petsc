/*Solve Poisson equation using jacobi iteration*/

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

double const PI = 4.0 * atan(1.0);

void Allocate_2D_R(double **&m, int d1, int d2)
{
    m = new double *[d1];
    for (int i = 0; i < d1; ++i)
    {
        m[i] = new double[d2];
        for (int j = 0; j < d2; ++j)
            m[i][j] = 0.0;
    }
}


//compute residual norm:
double residue(double **u, double **b, double hx, int Nx){

	double residue = 0.0;
    for(int i = 1 ; i < Nx ; i++) {
		for(int j = 1 ; j < Nx ; j++) {
			residue += (hx*hx*b[i][j] - u[i+1][j] - u[i-1][j] - u[i][j+1] - u[i][j-1] + 4.0*u[i][j])*(hx*hx*b[i][j] - u[i+1][j] - u[i-1][j] - u[i][j+1] - u[i][j-1] + 4.0*u[i][j]);
	    }
    }
	return residue;
}

//write vtk:
void write_vtk(double **u, int Nx, int Ny, const double *x, const double *y, double t, int it)
{
	int i,j;
	//write vtk data:
    char filename[20];
    sprintf(filename, "sol-%08d.vtk", it);
    std::ofstream vtk(filename, ios::out);
    vtk.flags(ios::dec | ios::scientific);
    vtk.precision(16);
    if(!vtk) {cerr<< "Error: Output file couldnot be opened.\n";exit(1);}

    vtk << "# vtk DataFile Version 2.0" << "\n";
    vtk << "2D Jacobi" << "\n";
    vtk << "ASCII" << "\n";
    vtk << "\nDATASET STRUCTURED_GRID" << "\n";
    vtk << "\nFIELD FieldData 1" << "\n"; 
    vtk << "TIME 1 1 double" << "\n";
    vtk << t << "\n"; 
    vtk << "\nDIMENSIONS " << Ny + 1 << " " << Nx + 1 << " " << 1 << "\n";
    vtk << "POINTS " << (Nx + 1)*(Ny + 1) <<  " double" << "\n";
    vtk << "\n";

    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            vtk << x[i] << " " << y[j] << " " << 0.0 << "\n";
        }
    }

    vtk << "\nPOINT_DATA " << (Nx + 1)*(Ny + 1) << "\n"; 
    vtk << "\nSCALARS U double 1" << "\n";
    vtk << "LOOKUP_TABLE default" << "\n";
    vtk << "\n";

    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            vtk << u[i][j] << "\n";
        }
    }

    vtk.close();
}

//print error norm:
void error_norm(double **u, double **uExact, int Nx, int Ny)
{
	cout.flags(ios::dec | ios::scientific);
    cout.precision(5);
	
	int i, j;
	double Max_Error = 0.0 ,L2_Error = 0.0;

    for(i = 0 ; i <= Nx ; i++) {
		for(j = 0 ; j <= Ny ; j++) {
			if( fabs(u[i][j] - uExact[i][j] ) > Max_Error) Max_Error = fabs( u[i][j] - uExact[i][j] );
			L2_Error += ( u[i][j] - uExact[i][j] )*( u[i][j] - uExact[i][j] )/ ( ( Nx+1.0 )*(Ny+1.0) ) ;
		}
	}

    L2_Error = sqrt(L2_Error);
	cout << "\n L2 : " << L2_Error << "\t Max : " << Max_Error <<  endl ;

}


int main()
{
	cout.flags(ios::dec | ios::scientific);
    cout.precision(5);

    double *x, *y, **u, **unew, **uExact, **b;
    double residual = 0.0, hx, hy;

    //grid:
    int Nx = 100, Ny = 100, i, j, iter = 0, n = 20;

    //create grid:
    x = new double[Nx + 1];
    y = new double[Ny + 1];

    //set uniform grid:
    for(i = 0; i <= Nx; ++i) x[i] = i/double(Nx);
    for(i = 0 ; i <= Ny ; i++) y[i] = i/double(Ny);
    hx = 1.0/double(Nx) ; 
	hy = 1.0/double(Ny) ;

    //allocate memory:
    Allocate_2D_R(u, Nx+1, Ny+1); Allocate_2D_R(unew, Nx+1, Ny+1);
    Allocate_2D_R(b, Nx+1, Ny+1); Allocate_2D_R(uExact, Nx+1, Ny+1);

    //initialize u and b:
    for(i = 0 ; i <= Nx ; i++) {
		for(j = 0 ; j <= Ny ; j++) {
			u[i][j] = 0.0 ; 
			unew[i][j] = 0.0;
			uExact[i][j] = sin(n*PI*x[i])*sin(n*PI*y[j]);
			b[i][j] = -2.0*n*n*PI*PI*sin(n*PI*x[i])*sin(n*PI*y[j]);
		}
	}

	do{

		//jacobi update
		for(int i = 1 ; i < Nx ; i++) {
			for(int j = 1 ; j < Nx ; j++) {
				unew[i][j] = -0.25*(hx*hx*b[i][j] - u[i+1][j] - u[i-1][j] - u[i][j+1] - u[i][j-1] );
	    	}
    	}

    	//copy unew to u:
		for(int i = 1 ; i < Nx ; i++) {
			for(int j = 1 ; j < Nx ; j++) {
				u[i][j] = unew[i][j];
	    	}
    	}

		
		iter++;
		
		residual = residue(u, b, hx, Nx);

		//write data:
		//if (iter % 1000 == 0) write_vtk(u, Nx, Ny, x, y, 0, iter);


	} while (residual > 1e-10);

    //write no of iterations for given wave number
    std::ofstream Iter("Iter.dat", ios::app);
    if(!Iter) {cerr<< "Error: Output file couldnot be opened.\n";exit(1);}
	Iter<<n<<"\t"<<iter<<std::endl;
	error_norm(u, uExact, Nx, Ny);	

    delete[] x;
    delete[] y;

    return 0;
}