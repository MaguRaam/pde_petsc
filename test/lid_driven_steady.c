#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*Function declarations*/

/*allocate and free 2d array*/
double** create_array_2d(int, int, double);
void free_array_2d(double**, int);

/*create 1d grid*/
double* create_grid(int, double);

/*write data*/
void write(int, double, double**, double**, double**, double*, double*, int);

/*finite difference operators*/
double advect_u(double**,  double**, double, int, int);
double advect_v(double**,  double**, double, int, int);
double diffuse_u(double**, double, int, int);
double diffuse_v(double**, double, int, int);
double dp_dx(double**, double, int, int);
double dp_dy(double**, double, int, int);
double div_u(double**, double**, double, int, int);


int main(){

    /*grid*/
    int N = 100;
    double h = 1.0/(double)(N-1);

    double* x = create_grid(N, h);
    double* y = create_grid(N, h);

    /*time*/
    double dt = 0.001;

    /*delta and error*/
    double delta = 4.5;
    double error = 1.0;

    /*Reynolds number*/
    double Re = 100;
    double inv_Re = 1.0/(double)(Re);

    /*lid velocity*/
    double Ulid = 1.0;

    /*initialize pressure*/
    double** pc = create_array_2d(N, N, 0.0);
    double** p  = create_array_2d(N + 1, N + 1, 1.0);  /*initialize pressure with 1*/
    double** pn = create_array_2d(N + 1, N + 1, 0.0);
    double** e  = create_array_2d(N + 1, N + 1, 0.0); 

    /*initialize x velocity*/
    double** uc = create_array_2d(N, N, 0.0);
    double** u  = create_array_2d(N, N + 1, 0.0);
    double** un = create_array_2d(N, N + 1, 0.0);

    /*initialize y velocity*/
    double** vc = create_array_2d(N, N, 0.0);
    double** v  = create_array_2d(N + 1, N, 0.0);
    double** vn = create_array_2d(N + 1, N, 0.0);  

    /*Enforce velocity boundary condition at the top wall*/
    for (int i = 0; i <= N - 1; i++){ 
        u[i][N-1] = Ulid; 
        u[i][N]   = Ulid;
    }


    /*Solve for u, v and p*/
    int step = 1;

    while (error > 1.0e-8){

        /*Update x momentum*/
        for (int i = 1; i <= (N - 2); ++i)
            for (int j = 1; j <= (N - 1); ++j)
                un[i][j] = u[i][j] + dt*( - advect_u(u, v, h, i, j) + inv_Re*(diffuse_u(u, h, i, j)) - dp_dx(p, h, i, j) );

        /*Enforce boundary condition for x velocity*/
        for (int j = 1; j <= (N - 1); j++)
		{
			un[0][j] = 0.0; 
            un[N - 1][j] = 0.0;
		}

        for (int i = 0; i <= (N - 1); i++)
		{
			un[i][0] = -un[i][1];
			un[i][N] = 2 - un[i][N - 1];
		}

        /*Update y momentum*/
        for (int i = 1; i <= N - 1; ++i)
            for (int j = 1; j <= N - 2; ++j)
                vn[i][j] = v[i][j] + dt*( - advect_v(u, v, h, i, j) + inv_Re*(diffuse_v(v, h, i, j)) - dp_dy(p, h, i, j) );


        /*Enforce boundary condition for x velocity*/
		for (int j = 1; j <= (N - 2); j++)
		{
			vn[0][j] = -vn[1][j];
			vn[N][j] = -vn[N - 1][j];
		}

		for (int i = 0; i <= N; i++)
		{
			vn[i][0] = 0.0;
			vn[i][N - 1] = 0.0;
		}


        /*Solve continuity equation*/
        for (int i = 1; i <= (N - 1); i++)
			for (int j = 1; j <= (N - 1); j++)
				pn[i][j] = p[i][j] - dt * delta * div_u(un, vn, h, i, j);

        /*Enforce boundary conditions for pressure*/
		for (int i = 1; i <= (N - 1); i++)
		{
			pn[i][0] = pn[i][1];
			pn[i][N] = pn[i][N - 1];
		}

		for (int j = 0; j <= N; j++)
		{
			pn[0][j] = pn[1][j];
			pn[N][j] = pn[N - 1][j];
		}

        /*Check incompressibility is satisfied or not ?*/
        error = 0.0;
        for (int i = 1; i <= (N - 1); i++){
			for (int j = 1; j <= (N - 1); j++){
                e[i][j] = fabs( div_u(un, vn, h, i, j) );
                error = error + e[i][j];
            }
        }


        if (step % 1000 == 1) printf("Error is %5.8f for the step %d\n", error, step);

        // Iterating u
		for (int i = 0; i <= (N - 1); i++)
			for (int j = 0; j <= N; j++)
				u[i][j] = un[i][j];

		// Iterating v
		for (int i = 0; i <= N; i++)
			for (int j = 0; j <= (N - 1); j++)
				v[i][j] = vn[i][j];

		// Iterating p
		for (int i = 0; i <= N; i++)
			for (int j = 0; j <= N; j++)
				p[i][j] = pn[i][j];

		step = step + 1;

    }
    /*End of solution :)*/



    /*write data to grid points*/
    for (int i = 0; i <= (N - 1); i++)
	{
		for (int j = 0; j <= (N - 1); j++)
		{
			uc[i][j] = 0.5 * (u[i][j] + u[i][j + 1]);
			vc[i][j] = 0.5 * (v[i][j] + v[i + 1][j]);
			pc[i][j] = 0.25 * (p[i][j] + p[i + 1][j] + p[i][j + 1] + p[i + 1][j + 1]);
		}
	}

    write(1, 0, uc, vc, pc, x, y, N); 

    /**/
    FILE *f;
    f = fopen("U.plt", "w");
    fprintf(f, "VARIABLES=\"U\",\"Y\"\n");
    fprintf(f, "VARIABLES=\"U\",\"Y\"\n");
    fprintf(f, "ZONE F=POINT\n");
    fprintf(f, "I=%d\n", N);
    for (int j = 0; j < N; j++)
	{
		double ypos = (double)j * h;
		fprintf(f, "%5.8f\t%5.8f\n", (uc[N / 2][j] + uc[(N / 2) + 1][j]) / (2.), ypos);
	}


    /*free memory*/
    free(x);
    free(y);
    free_array_2d(p, N + 1);
    free_array_2d(pn, N + 1);
    free_array_2d(pc, N);
    free_array_2d(e, N+1);
    free_array_2d(u, N);
    free_array_2d(un, N);
    free_array_2d(uc, N);
    free_array_2d(v, N + 1);
    free_array_2d(vn, N + 1);
    free_array_2d(vc, N);

    return 0;
}


/*create 2d array and initialize with zero values*/ /*initialize pressure with 1*/
double** create_array_2d(int Nx, int Ny, double value){

    double** arr;
    int i, j;

    arr = (double**)malloc(sizeof(double*)*Nx);

    for (i = 0; i < Nx; ++i) arr[i] = (double*)malloc(sizeof(double)*Ny);

    for (i = 0; i < Nx; ++i)
        for (j = 0; j < Ny; ++j)
            arr[i][j] = value;

    return arr;
}

/*free array*/
void free_array_2d(double** arr, int Nx){

    int i;

    for (i = 0; i < Nx; ++i) free(arr[i]);

    free(arr);
}

/*create grid*/
double* create_grid(int N, double h){

    double* x;
    int i;

    x = (double*)malloc(N*sizeof(double));
    for (i = 0; i < N; ++i) x[i] = i*h;

    return x;
}

/*u advection*/
double advect_u( double** u,  double** v, double h, int i, int j){

    double inv_h = 1.0/h;
    return  0.5*inv_h*( u[i+1][j]*u[i+1][j] - u[i-1][j]*u[i-1][j] ) + 0.25*inv_h* ( ( u[i][j+1] + u[i][j])*(v[i][j] + v[i+1][j]) - (u[i][j-1] + u[i][j])*(v[i][j-1] + v[i+1][j-1])  );
}

/*u diffusion*/
double diffuse_u( double** u, double h, int i, int j){

    double inv_h2 = 1.0/(h*h);
    return inv_h2* ( u[i+1][j] - 2.0*u[i][j] + u[i-1][j]) + inv_h2*( u[i][j+1] - 2.0*u[i][j] + u[i][j-1] );
}

/*dp/dx*/
double dp_dx( double** p, double h, int i, int j){

    double inv_h = 1.0/h;
    return inv_h*(p[i+1][j] - p[i][j]);
}

/*v advection*/
double advect_v( double** u,  double** v, double h, int i, int j){

    double inv_h = 1.0/h;
    return  0.25* inv_h* ( ( u[i][j+1] + u[i][j] )*(v[i+1][j] + v[i][j]) - ( u[i-1][j+1] + u[i-1][j] )*(v[i-1][j] + v[i][j])  ) +
            0.5*inv_h*( v[i][j+1]*v[i][j+1] - v[i][j-1]*v[i][j-1] );
}

/*v diffusion*/
double diffuse_v( double** v, double h, int i, int j){

    double inv_h2 = 1.0/(h*h);
    return inv_h2* ( v[i+1][j] - 2.0*v[i][j] + v[i-1][j]) + inv_h2*( v[i][j+1] - 2.0*v[i][j] + v[i][j-1] );
}

/*dp/dx*/
double dp_dy( double** p, double h, int i, int j){

    double inv_h = 1.0/h;
    return inv_h*(p[i][j+1] - p[i][j]);
}

/*divergence of u*/
double div_u(double** u, double** v, double h, int i, int j){

    double inv_h = 1.0/h;

    return inv_h*(u[i][j] - u[i - 1][j]) +  inv_h*(v[i][j] - v[i][j - 1]);
}


/*write data*/
void write(int step, double t, double** u, double** v, double** p, double* x, double* y, int N){
    FILE *f;
    char filename[20];
    sprintf(filename, "plot/sol-%08d.dat", step);
    f = fopen(filename, "w");

    if (f == NULL){
        printf("Error opening file\n");
        exit(1);
    }

    fprintf(f, "TITLE = \"2D\"\n"); 
    fprintf(f, "VARIABLES=\"X\",\"Y\",\"U\",\"V\",\"P\"\n");
    fprintf(f, "Zone I=%d, J=%d\n",N,N);
    fprintf(f, "SOLUTIONTIME = %.16e\n",t);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            fprintf(f, "%.16e\t%.16e\t%.16e\t%.16e\t%.16e\t\n", x[i], y[j], u[i][j], v[i][j], p[i][j]);

    fclose(f);
}

