/*Solve diffusion equation with homogeneous boundary condition using matrix bi-diagonalization method*/

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



int main()
{
    cout.flags(ios::dec | ios::scientific);
    cout.precision(5);

    double *x, *y, **uExact, **P, **PINV, *PEig, **u, **RHS;
    double **RHS_Tilde, **u_Tilde, **Temp;
    double L2_Error = 0.0, Max_Error = 0.0, hx, hy;

    //grid:
    int Nx = 200, Ny = 200, i, j, k;

    //create grid:
    x = new double[Nx + 1];
    y = new double[Ny + 1];

    
    //allocate memory:
    Allocate_2D_R(uExact, Nx+1, Ny+1); Allocate_2D_R(u, Nx+1, Ny+1); Allocate_2D_R(P, Nx-1, Nx-1); Allocate_2D_R(PINV, Nx-1, Nx-1); PEig = new double[Nx-1];
    Allocate_2D_R(RHS, Nx-1, Ny-1); Allocate_2D_R(Temp, Nx-1, Ny-1); Allocate_2D_R(RHS_Tilde, Nx-1, Ny-1); Allocate_2D_R(u_Tilde, Nx-1, Ny-1);     
    

    //set uniform grid:
    for(i = 0; i <= Nx; ++i) x[i] = i/double(Nx);
    for(i = 0 ; i <= Ny ; i++) y[i] = i/double(Ny);
    hx = 1.0/double(Nx) ; 
	hy = 1.0/double(Ny) ;

    //time step:
    double dt = 0.001, t = 0.0, T = 1.0;
    int it = 0;


    //set initial and exact solution:
    for(i = 0 ; i <= Nx ; i++) {
		for(j = 0 ; j <= Ny ; j++) {
			u[i][j]      = 0.0;
			uExact[i][j] = 0.0;
		}
	}

    //set up eigenvalues and matrices:
    for(i = 1 ; i < Nx ; i++) {
        PEig[i-1] = -4.0*sin(PI*0.5*x[i])*sin(PI*0.5*x[i]);       
        for(j = 1 ; j < Nx ; j++) {
            P[i-1][j-1] = sin(i*PI*x[j]);
            PINV[i-1][j-1] = 2.0*hx*sin(j*PI*x[i]);
        }
    }


    //evolve in time:
    while (t <= T)
    {

        //set the right handside:
        for(i = 1 ; i < Nx ; i++) {
            for(j = 1 ; j < Ny ; j++) {
                
                RHS[i-1][j-1]  = u[i][j];
                RHS[i-1][j-1] += 0.5*dt*( sin(2*PI*x[i]) * sin(2*PI*y[i]) * sin(2*PI*t) );
                RHS[i-1][j-1] += 0.5*dt*( sin(2*PI*x[i]) * sin(2*PI*y[i]) * sin(2*PI*(t + dt)) );
                RHS[i-1][j-1] *= hx*hx ;
                RHS[i-1][j-1] += 0.5*dt*(u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4.0*u[i][j]);
            }
        }


        // F_Tilde = compute P^{-1} F Q^{-T}
        for(i = 1 ; i < Nx ; i++) {
            for(j = 1 ; j < Nx ; j++) {
                Temp[i-1][j-1] = 0.0 ;
                for(k = 1 ; k < Nx ; k++) Temp[i-1][j-1] += PINV[i-1][k-1]*RHS[k-1][j-1];
            }
        }

        for(i = 1 ; i < Nx ; i++) {
            for(j = 1 ; j < Nx ; j++) {
                RHS_Tilde[i-1][j-1] = 0.0 ;
                for(k = 1 ; k < Nx ; k++) RHS_Tilde[i-1][j-1] += Temp[i-1][k-1]*P[k-1][j-1] ;
            }
        }


        // compute U_Tilde
        for(i = 1 ; i < Nx ; i++) {
            for(j = 1 ; j < Nx ; j++) {
                RHS_Tilde[i-1][j-1] /= (-0.5*dt*PEig[i-1] - 0.5*dt*PEig[j-1] + hx*hx) ;
            }
        }



        //compute Unew = P \bar{U} Q^T
        for(i = 1 ; i < Nx ; i++) {
            for(j = 1 ; j < Nx ; j++) {
                Temp[i-1][j-1] = 0.0 ;
                for(k = 1 ; k < Nx ; k++) Temp[i-1][j-1] += P[i-1][k-1]*RHS_Tilde[k-1][j-1] ;
            }
        }

        for(i = 1 ; i < Nx ; i++) {
            for(j = 1 ; j < Nx ; j++) {
                u[i][j] = 0.0 ;
                for(k = 1 ; k < Nx ; k++) u[i][j] += Temp[i-1][k-1]*PINV[k-1][j-1];
            }
        }


        it++;
        t+=dt;


        //set exact:
        for(i = 0 ; i <= Nx ; i++) 
            for(j = 0 ; j <= Ny ; j++) 
                 uExact[i][j] = (exp(-8.0 * PI * PI * t) - ((cos(2.0 * PI * t) - 4.0 * PI * sin(2.0 * PI * t)) / ((2.0 * PI) + (32.0 * PI * PI * PI)))) * (sin(2.0 * PI * x[i]) * sin(2.0 * PI * y[i]));


        std::cout<<"it = "<<it<<std::endl;
        std::cout<<"t = "<<t<<std::endl;
    
        //write vtk file:
        if (it % 10 == 0){
            std::cout<<"write data at t = "<<t<<std::endl;

            //write vtk data:
            char filename[20];
            sprintf(filename, "sol-%08d.vtk", it);
            std::ofstream vtk(filename, ios::out);
            vtk.flags(ios::dec | ios::scientific);
            vtk.precision(16);
            if(!vtk) {cerr<< "Error: Output file couldnot be opened.\n";}

            vtk << "# vtk DataFile Version 2.0" << "\n";
            vtk << "2D Linear Convection" << "\n";
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
            vtk << "\nSCALARS UExact double 1" << "\n";
            vtk << "LOOKUP_TABLE default" << "\n";
            vtk << "\n";

            for (i = 0; i <= Nx; i++)
            {
                for (j = 0; j <= Ny; j++)
                {
                    vtk << uExact[i][j] << "\n";
                }
            }

             
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

    }

 
    delete[] x;
    delete[] y;

    return 0;
}
