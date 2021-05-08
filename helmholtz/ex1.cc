static char help[] = "solves Helmholtz equation using KSP solver\n";

#include <petsc.h>


typedef struct
{
    double lambda;
    double h;
    double(*rhs)(double, double);
    double(*exact)(double, double);
    double(*bcs[4])(double);
} AppCtx;

PetscErrorCode write_vts(DM da, Vec u, const char filename[])
{
    PetscErrorCode  ierr;
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
    ierr = DMView(da, viewer);
    VecView(u, viewer);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

    return 0;
}

PetscErrorCode set_exact(DM da, Vec u, AppCtx user){
    
    PetscErrorCode  ierr;
    double **au;
    DMDALocalInfo  info;
    double x, y, h = user.h;
    int i, j;

    ierr = DMDAVecGetArray(da, u, &au); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    //loop over locally owned part of grid:
    for (j = info.ys; j < info.ys + info.ym; ++j){
        y = j*h;
        for (i = info.xs; i < info.xs + info.xm; ++i){
            x = i*h;
            au[j][i] = user.exact(x,y);
        }
    }

    ierr = DMDAVecRestoreArray(da, u, &au); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode  create_matrix(Mat A, DM da, AppCtx user)
{
    PetscErrorCode  ierr;
    DMDALocalInfo   info;
    double h = user.h;
    double lambda = user.lambda;
    double v[5];
    int i,j, ncols;
    MatStencil  row, col[5];

    //get grid info:
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    //loop over locally owned part of grid:
    for (j = info.ys; j < info.ys + info.ym; ++j){
        for (i = info.xs; i < info.xs + info.xm; ++i){
            
            //get row index:
            row.i = i; row.j = j;

            //grid point at boundary:
            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1)
            {
                ncols = 1;
                col[0].i = i; col[0].j = j; v[0] = 1; 
            } else
            {
                ncols = 5;
                col[0].i = i;   col[0].j = j;   v[0] = -4 + lambda*h*h; 
                col[1].i = i+1; col[1].j = j;   v[1] = 1;
                col[2].i = i-1; col[2].j = j;   v[2] = 1;
                col[3].i = i;   col[3].j = j+1; v[3] = 1;
                col[4].i = i;   col[4].j = j-1; v[4] = 1;
            }

            ierr = MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return 0;
}

PetscErrorCode  create_rhs(DM da, Vec b, AppCtx user)
{
    PetscErrorCode  ierr;
    double **ab;
    DMDALocalInfo  info;
    double x, y, h = user.h;
    int i, j;

    ierr = DMDAVecGetArray(da, b, &ab); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    //loop over locally owned part of grid:
    for (j = info.ys; j < info.ys + info.ym; ++j){
        y = j*h;
        for (i = info.xs; i < info.xs + info.xm; ++i){
            x = i*h;

            //left boundary:
            if (i == 0)                     ab[j][i] = user.bcs[2](y);
            else if (i == info.mx - 1)      ab[j][i] = user.bcs[3](y);
            else if (j == 0)                ab[j][i] = user.bcs[0](x);
            else if (j == info.my - 1)      ab[j][i] = user.bcs[1](x);
            else                            ab[j][i] = user.rhs(x,y)*h*h;
        
        }
    }

    ierr = DMDAVecRestoreArray(da, b, &ab); CHKERRQ(ierr);

    return 0;
}

double exact(double x, double y){ return PetscSinReal(10*PETSC_PI*x)*PetscCosReal(10*PETSC_PI*y);}
double rhs(double x, double y){ return (1000 - 200*PETSC_PI*PETSC_PI)*PetscSinReal(10*PETSC_PI*x)*PetscCosReal(10*PETSC_PI*y);}
double top(double x){return PetscSinReal(10*PETSC_PI*x);} //top bcs
double bottom(double x){return PetscSinReal(10*PETSC_PI*x);} //bottom bcs
double left(double y){return 0;}
double right(double y){return 0;}

int main(int argc, char **argv)
{
    DM                  da;
    DMDALocalInfo       info;
    KSP                 ksp;
    Mat                 A;
    Vec                 b, u, uexact;
    AppCtx              user;
    double              error;

    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    //create grid:
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                        100, 100, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    //set solver context:
    user.h = 1.0/(double)(info.mx - 1);
    user.lambda = 1000;
    user.exact    =  exact;
    user.rhs      =  rhs; 
    user.bcs[0]   =  top;
    user.bcs[1]   =  bottom;
    user.bcs[2]   =  left;
    user.bcs[3]   =  right;


    //create Helmholtz matrix:
    ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetUp(A); CHKERRQ(ierr);
    ierr = create_matrix(A, da, user); CHKERRQ(ierr);

    //create rhs:
    ierr = DMCreateGlobalVector(da, &b); CHKERRQ(ierr);
    ierr = create_rhs(da, b, user); CHKERRQ(ierr);
    ierr = write_vts(da, b, "b.vts");
    
    //create KSP solve:
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = VecDuplicate(b, &u); CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);
    ierr = write_vts(da, u, "u.vts");

    //compute uexact:
    ierr = VecDuplicate(u, &uexact); CHKERRQ(ierr);
    ierr = set_exact(da, uexact, user); CHKERRQ(ierr);
    ierr = write_vts(da, uexact, "uexact.vts");

    //compute error:
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
    ierr = VecNorm(u,NORM_INFINITY,&error); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                "on %d x %d grid:  error |u-uexact|_inf = %g\n",
                info.mx,info.my,error); CHKERRQ(ierr);
    ierr = write_vts(da, u, "error.vts");


    //destroy:
    DMDestroy(&da);
    MatDestroy(&A);
    KSPDestroy(&ksp);
    VecDestroy(&u);
    VecDestroy(&b);

    return PetscFinalize();
}

/*
    visualize grid:
    mpirun -n 4 ./helmholtz -da_grid_x 16 -da_grid_y 16 -dm_view draw -draw_pause 3

    visualize matrix sparsity pattern:  
    mpirun -n 4 ./helmholtz -da_grid_x 16 -da_grid_y 16 -mat_view draw -draw_pause 3
    
    write matrix:
    mpirun -n 4 ./helmholtz -da_grid_x 16 -da_grid_y 16 -mat_view :mat.dat:ascii_dense

    ksp solve iterate:
    mpirun -n 4 ./helmholtz -da_refine 4 -ksp_monitor_solution draw -draw_pause 0.1

    grid convergence:
    for K in 0 1 2 3 4 5 6; do mpirun -n 4 ./helmholtz -ksp_rtol 1.0e-12 -da_refine $K; done

    view ksp solver
    ./helmholtz -ksp_view

    set solver, preconditioner and tolerance
    time ./helmholtz -da_refine 5 -ksp_converged_reason -ksp_rtol 1.0e-10 -ksp_type gmres -pc_type none

*/