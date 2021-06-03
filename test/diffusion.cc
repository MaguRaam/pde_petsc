/*Solve diffusion equation with homogeneous boundary condition using matrix bi-diagonalization method*/

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

double const PI = 4.0 * atan(1.0);

//write vtk:
void write_vtk(double **u, double **uExact, int Nx, int Ny, const double *x, const double *y, double t, int it)
{
    int i, j;

    //write vtk data:
    char filename[20];
    sprintf(filename, "sol-%08d.vtk", it);

    std::ofstream vtk(filename, ios::out);
    vtk.flags(ios::dec | ios::scientific);
    vtk.precision(16);

    if (!vtk)
    {
        cerr << "Error: Output file couldnot be opened.\n";
        exit(1);
    }

    vtk << "# vtk DataFile Version 2.0"
        << "\n";
    vtk << "2D diffusion"
        << "\n";
    vtk << "ASCII"
        << "\n";
    vtk << "\nDATASET STRUCTURED_GRID"
        << "\n";
    vtk << "\nFIELD FieldData 1"
        << "\n";
    vtk << "TIME 1 1 double"
        << "\n";
    vtk << t << "\n";
    vtk << "\nDIMENSIONS " << Ny + 1 << " " << Nx + 1 << " " << 1 << "\n";
    vtk << "POINTS " << (Nx + 1) * (Ny + 1) << " double"
        << "\n";
    vtk << "\n";

    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            vtk << x[i] << " " << y[j] << " " << 0.0 << "\n";
        }
    }

    vtk << "\nPOINT_DATA " << (Nx + 1) * (Ny + 1) << "\n";

    vtk << "\nSCALARS U double 1"
        << "\n";
    vtk << "LOOKUP_TABLE default"
        << "\n";
    vtk << "\n";

    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            vtk << u[i][j] << "\n";
        }
    }

    vtk << "\nSCALARS UExact double 1"
        << "\n";
    vtk << "LOOKUP_TABLE default"
        << "\n";
    vtk << "\n";

    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            vtk << uExact[i][j] << "\n";
        }
    }

    vtk.close();
}

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

double rhs(double x, double y, double t)
{
    return sin(2 * PI * x) * sin(2 * PI * y) * sin(2 * PI * t);
}

double exact(double x, double y, double t)
{
    return ( exp(-8.0*PI*PI*t) + (4.0*PI*sin(2*PI*t) - cos(2*PI*t) )/(32*PI*PI*PI + 2*PI) )*sin(2*PI*x)*sin(2*PI*y);
}

//print error norm:
void error_norm(double **u, double **uExact, int Nx, int Ny)
{
    cout.flags(ios::dec | ios::scientific);
    cout.precision(5);

    int i, j;
    double Max_Error = 0.0, L2_Error = 0.0;

    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            if (fabs(u[i][j] - uExact[i][j]) > Max_Error)
                Max_Error = fabs(u[i][j] - uExact[i][j]);
            L2_Error += (u[i][j] - uExact[i][j]) * (u[i][j] - uExact[i][j]) / ((Nx + 1.0) * (Ny + 1.0));
        }
    }

    L2_Error = sqrt(L2_Error);

    //write error in file
    std::ofstream File("Error.dat", ios::app);
    File.flags(ios::dec | ios::scientific);
    File.precision(16);
    File << Nx<<"\t" << L2_Error << "\t"<< Max_Error <<  endl ;
    if(!File) {cerr<< "Error: Output file couldnot be opened.\n"; exit(1);}
}

int main()
{
    cout.flags(ios::dec | ios::scientific);
    cout.precision(5);

    double *x, *y, **uExact, **P, **PINV, *PEig, **u, **RHS;
    double **RHS_Tilde, **u_Tilde, **Temp;
    double L2_Error = 0.0, Max_Error = 0.0, hx, hy;

    //grid:
    int Nx = 100, Ny = 100, i, j, k;

    //create grid:
    x = new double[Nx + 1];
    y = new double[Ny + 1];

    //allocate memory:
    Allocate_2D_R(uExact, Nx + 1, Ny + 1);
    Allocate_2D_R(u, Nx + 1, Ny + 1);
    Allocate_2D_R(P, Nx - 1, Nx - 1);
    Allocate_2D_R(PINV, Nx - 1, Nx - 1);
    PEig = new double[Nx - 1];
    Allocate_2D_R(RHS, Nx - 1, Ny - 1);
    Allocate_2D_R(Temp, Nx - 1, Ny - 1);
    Allocate_2D_R(RHS_Tilde, Nx - 1, Ny - 1);
    Allocate_2D_R(u_Tilde, Nx - 1, Ny - 1);

    //set uniform grid:
    for (i = 0; i <= Nx; ++i)
        x[i] = i / double(Nx);
    for (i = 0; i <= Ny; i++)
        y[i] = i / double(Ny);
    hx = 1.0 / double(Nx);
    hy = 1.0 / double(Ny);

    //time step:
    double dt = (0.5/256), t = 0.0, T = 0.5;
    int it = 0;


    //set up eigenvalues and matrices:
    for (i = 1; i < Nx; i++)
    {
        PEig[i - 1] = -4.0 * sin(PI * 0.5 * x[i]) * sin(PI * 0.5 * x[i]);
        for (j = 1; j < Nx; j++)
        {
            P[i - 1][j - 1] = sin(i * PI * x[j]);
            PINV[i - 1][j - 1] = 2.0 * hx * sin(j * PI * x[i]);
        }
    }

    //set initial and exact solution:
    for (i = 0; i <= Nx; i++)
    {
        for (j = 0; j <= Ny; j++)
        {
            u[i][j] = exact(x[i], y[j], t);
            uExact[i][j] = exact(x[i], y[j], t);
        }
    }

    //write initial condition:
    std::cout << "timestep = " << it << "\ttime = " << t << std::endl;
    write_vtk(u, uExact, Nx, Ny, x, y, t, it);

    //evolve in time:
    while (t < T)
    {
        //set the right handside:
        for (i = 1; i < Nx; i++)
        {
            for (j = 1; j < Ny; j++)
            {

                RHS[i - 1][j - 1] = u[i][j] * hx * hx;
                RHS[i - 1][j - 1] += 0.5 * dt * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4.0 * u[i][j]);
                RHS[i - 1][j - 1] += 0.5 * dt * hx * hx * rhs(x[i], y[j], t);
                RHS[i - 1][j - 1] += 0.5 * dt * hx * hx * rhs(x[i], y[j], t + dt);
            }
        }

        // F_Tilde = compute P^{-1} F Q^{-T}
        for (i = 1; i < Nx; i++)
        {
            for (j = 1; j < Nx; j++)
            {
                Temp[i - 1][j - 1] = 0.0;
                for (k = 1; k < Nx; k++)
                    Temp[i - 1][j - 1] += PINV[i - 1][k - 1] * RHS[k - 1][j - 1];
            }
        }

        for (i = 1; i < Nx; i++)
        {
            for (j = 1; j < Nx; j++)
            {
                RHS_Tilde[i - 1][j - 1] = 0.0;
                for (k = 1; k < Nx; k++)
                    RHS_Tilde[i - 1][j - 1] += Temp[i - 1][k - 1] * P[k - 1][j - 1];
            }
        }

        // compute U_Tilde
        for (i = 1; i < Nx; i++)
        {
            for (j = 1; j < Nx; j++)
            {
                RHS_Tilde[i - 1][j - 1] /= (-0.5 * dt * PEig[i - 1] - 0.5 * dt * PEig[j - 1] + hx * hx);
            }
        }

        //compute Unew = P \bar{U} Q^T
        for (i = 1; i < Nx; i++)
        {
            for (j = 1; j < Nx; j++)
            {
                Temp[i - 1][j - 1] = 0.0;
                for (k = 1; k < Nx; k++)
                    Temp[i - 1][j - 1] += P[i - 1][k - 1] * RHS_Tilde[k - 1][j - 1];
            }
        }

        for (i = 1; i < Nx; i++)
        {
            for (j = 1; j < Nx; j++)
            {
                u[i][j] = 0.0;
                for (k = 1; k < Nx; k++)
                    u[i][j] += Temp[i - 1][k - 1] * PINV[k - 1][j - 1];
            }
        }

        //update timestep and time:
        it++;
        t += dt;

        std::cout << "timestep = " << it << "\ttime = " << t << std::endl;

        //compute exact solution:
        for (i = 0; i <= Nx; i++)
        {
            for (j = 0; j <= Ny; j++)
            {
                uExact[i][j] = exact(x[i], y[j], t);
            }
        }

        //write vtk file:
        if (it % 10 == 0)
        {
            std::cout << "write data at t = " << t << std::endl;
            //write_vtk(u, uExact, Nx, Ny, x, y, t, it);
        }
    }

    //compute error:
    error_norm(u, uExact, Nx, Ny);

    std::cout<< "stability criteria = "<<(dt/(hx*hx)) << std::endl;

    delete[] x;
    delete[] y;
    return 0;
}