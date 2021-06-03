static char help[] = "Solves 1d poisson equation -d2u/dx2 = sin(x) bcs: x = 0 u = 0 and x = 2pi u = 0\n";

#include <petsc.h>

struct AppCtx
{
    PetscInt    nx;
    PetscReal   lx;
    PetscReal   hx;
};

PetscErrorCode BuildMatrix(Mat A, const DMDALocalInfo& info, const AppCtx& user)
{
    PetscErrorCode ierr;
    PetscInt    i;
    MatStencil  row, col[3];
    PetscReal   v[3];
    PetscInt    ncols;

    /*loop over grid points in the current process and fill the matrix rows in the current process*/
        for (i = info.xs; i < info.xs + info.xm; ++i){

            ncols = 0;

            //get row index of matrix:
            row.i = i;

            //boundary point:
            if (i == 0 || i == info.mx - 1)
            {
                //fill diagonal element i
                col[ncols].i = i; 
                v[ncols++] = 1;
            }
            
            //interior point
            else
            {
                //fill diagonal element i
                col[ncols].i = i; 
                v[ncols++] = 2;
                

                //fill left element i-1
                if (i-1 != 0)
                {
                    col[ncols].i = i-1;
                    v[ncols++] = -1;
                }

                if (i+1 != info.mx - 1)
                {
                    col[ncols].i = i+1;
                    v[ncols++] = -1;
                }

            } 

            //fill matrix row:
            ierr = MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES); CHKERRQ(ierr);
        }


    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return 0;
}

template <typename F>
PetscErrorCode BuildRHS(F f, Vec b, DM da, const DMDALocalInfo& info, const AppCtx& user)
{
    PetscErrorCode ierr;
    PetscScalar *arr;

    ierr = DMDAVecGetArray(da, b, &arr); CHKERRQ(ierr);

    //loop over locally owned grid points:
    for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) arr[i] = f(i*user.hx)*user.hx*user.hx; 

    ierr = DMDAVecRestoreArray(da, b, &arr); CHKERRQ(ierr);

    return 0;
}

template <typename F>
PetscErrorCode BuildExact(F f, Vec x, DM da, const DMDALocalInfo& info, const AppCtx& user)
{
    PetscErrorCode ierr;
    PetscScalar *arr;

    ierr = DMDAVecGetArray(da, x, &arr); CHKERRQ(ierr);

    //loop over locally owned grid points:
    for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) arr[i] = f(i*user.hx); 

    ierr = DMDAVecRestoreArray(da, x, &arr); CHKERRQ(ierr);

    return 0;
}



int main(int argc, char **argv)
{
    KSP                 ksp;
    DM                  da;
    DMDALocalInfo       info;
    AppCtx              user;
    Mat                 A;
    Vec                 b, x, xexact;
    PetscReal           error_norm;

    //Initialize petsc program:
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    //create grid:
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 4, 1, 1, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);

    //get grid info:
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    //Set solver context:
    user.nx = info.mx;
    user.lx = 2.0*M_PI;
    user.hx = user.lx/(PetscReal)(user.nx - 1);

    //Set grid coordinates:
    ierr = DMDASetUniformCoordinates(da,0,user.lx,0,0,0,0); CHKERRQ(ierr);

    //Build Matrix A:
    ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A);
    ierr = BuildMatrix(A, info, user); CHKERRQ(ierr);

    //Build Vector b:
    ierr = DMCreateGlobalVector(da, &b); CHKERRQ(ierr);
    ierr = BuildRHS(sin,b,da,info,user); CHKERRQ(ierr);

    //Solve Ax = b;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    //Compute error and print on screen:
    ierr = DMCreateGlobalVector(da, &xexact); CHKERRQ(ierr);
    ierr = BuildExact(sin,xexact,da,info,user); CHKERRQ(ierr);
    ierr = VecAXPY(x,-1,xexact); CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_INFINITY, &error_norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "error on grid nx = %d \t %g\n", user.nx, error_norm); CHKERRQ(ierr);

    //Destroy petsc objects:
    DMDestroy(&da);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&xexact);
    KSPDestroy(&ksp);

    return PetscFinalize();
}



 
/*
    visualize grid:
    mpirun -n 4 ./poisson -da_grid_x 16 -dm_view draw -draw_pause 3

    visualize matrix sparsity pattern:  
    mpirun -n 4 ./poisson -da_grid_x 16 -mat_view draw -draw_pause 3
    
    write matrix:
    mpirun -n 4 ./poisson -da_grid_x 16 -mat_view :mat.dat:ascii_dense

    ksp solve iterate:
    mpirun -n 4 ./poisson -da_refine 4 -ksp_monitor_solution draw -draw_pause 0.1
*/
