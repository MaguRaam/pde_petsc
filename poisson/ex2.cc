static char help[] = "Solves 2d poisson equation -d2u/dx2 -d2u/dy2 = f\n";

#include <petsc.h>
#include <cassert>
#include <iostream>

struct AppCtx
{
    PetscInt    nx, ny;
    PetscReal   lx, ly;
    PetscReal   h;
};
 
PetscErrorCode BuildMatrix(Mat A, const DMDALocalInfo& info)
{
    PetscErrorCode ierr;
    PetscInt    i, j;
    MatStencil  row, col[5];
    PetscReal   v[5];
    PetscInt    ncols;

    /*loop over grid points in the current process and fill the matrix rows in the current process*/
    for (j = info.ys; j < info.ys + info.ym; ++j){
        for (i = info.xs; i < info.xs + info.xm; ++i){

            //get row index of matrix:
            row.i = i; row.j = j;

            //initialize no of cols:
            ncols = 0;

            //grid point is on the boundary
            if (i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1)
            {
                col[ncols].i = i; col[ncols].j = j;   //fill diagonal element
                v[ncols++] = 1;
            } 

            //interior grid point
            else
            {
                col[ncols].i = i; col[ncols].j = j;   //fill diagonal element
                v[ncols++] = 4;

                if (i-1 != 0)
                {
                    col[ncols].i = i-1; col[ncols].j = j;   //fill i-1,j element
                    v[ncols++] = -1;
                }

                if (j-1 != 0)
                {
                    col[ncols].i = i; col[ncols].j = j-1;   //fill i,j-1 element
                    v[ncols++] = -1;
                }

                if (i+1 != info.mx - 1)
                {
                    col[ncols].i = i+1; col[ncols].j = j;   //fill i+1,j element
                    v[ncols++] = -1;
                }

                if (j+1 != info.my - 1)
                {
                    col[ncols].i = i; col[ncols].j = j+1;   //fill i,j+1 element
                    v[ncols++] = -1;
                }

            }

            //fill matrix row:
            ierr = MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES); CHKERRQ(ierr);

        }
    }


    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return 0;
}

template <typename F>
PetscErrorCode BuildRHS(F f, Vec b, DM da, const DMDALocalInfo& info, const AppCtx& user)
{
    PetscErrorCode ierr;
    PetscScalar **arr;
    PetscScalar h = user.h;

    ierr = DMDAVecGetArray(da, b, &arr); CHKERRQ(ierr);

    //loop over locally owned grid points:
    for (PetscInt j = info.ys; j < info.ys + info.ym; ++j)
        for (PetscInt i = info.xs; i < info.xs + info.xm; ++i)
            arr[j][i] = f(i*h, j*h)*h*h;
    
    ierr = DMDAVecRestoreArray(da, b, &arr); CHKERRQ(ierr);

    return 0;
}

template <typename F>
PetscErrorCode BuildExact(F f, Vec x, DM da, const DMDALocalInfo& info, const AppCtx& user)
{
    PetscErrorCode ierr;
    PetscScalar **arr;
    PetscScalar h = user.h;

    ierr = DMDAVecGetArray(da, x, &arr); CHKERRQ(ierr);

    //loop over locally owned grid points:
    for (PetscInt j = info.ys; j < info.ys + info.ym; ++j)
        for (PetscInt i = info.xs; i < info.xs + info.xm; ++i)
            arr[j][i] = f(i*h, j*h);
    
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

    auto rhs = [](auto x, auto y){return 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y)
                    + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) );};

    auto exact = [](auto x, auto y){return x*x * (1.0 - x*x) * y*y * (y*y - 1.0);};

    //Initialize petsc program:
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    //create 2d grid:
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 
                        9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);

    //get grid info:
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    //Set solver context:
    user.nx = info.mx;  
    user.ny = info.my;
    
    assert(user.nx == user.ny);  //only square cells:

    user.lx = 1.0; 
    user.ly = 1.0;
    user.h  = user.lx/(PetscReal)(user.nx - 1);

    //Set grid coordinates:
    ierr = DMDASetUniformCoordinates(da,0,user.lx,0,user.ly,0,0); CHKERRQ(ierr);

    //Build Matrix A:
    ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A);
    ierr = BuildMatrix(A, info); CHKERRQ(ierr);

    //Build Vector b:
    ierr = DMCreateGlobalVector(da, &b); CHKERRQ(ierr);
    ierr = BuildRHS(rhs,b,da,info,user); CHKERRQ(ierr);

    //Solve Ax = b;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &x); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    //Compute error and print on screen:
    ierr = DMCreateGlobalVector(da, &xexact); CHKERRQ(ierr);
    ierr = BuildExact(exact,xexact,da,info,user); CHKERRQ(ierr);
    ierr = VecAXPY(x,-1,xexact); CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_INFINITY, &error_norm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "error on grid %d x %d is\t %g\n", user.nx, user.ny, error_norm); CHKERRQ(ierr);


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
    mpirun -n 4 ./2d -da_grid_x 16 -da_grid_y 16 -dm_view draw -draw_pause 3

    visualize matrix sparsity pattern:  
    mpirun -n 4 ./2d -da_grid_x 16 -da_grid_y 16 -mat_view draw -draw_pause 3
    
    write matrix:
    mpirun -n 4 ./2d -da_grid_x 16 -da_grid_y 16 -mat_view :mat.dat:ascii_dense

    ksp solve iterate:
    mpirun -n 4 ./2d -da_refine 4 -ksp_monitor_solution draw -draw_pause 0.1

    grid convergence:
    for K in 0 1 2 3 4 5 6; do mpirun -n 4 ./2d -ksp_rtol 1.0e-12 -da_refine $K; done

    view ksp solver
    ./2d -ksp_view

    set solver, preconditioner and tolerance
    time ./2d -da_refine 5 -ksp_converged_reason -ksp_rtol 1.0e-10 -ksp_type gmres -pc_type none

*/
