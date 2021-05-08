static char help[] = "Solve diffusion equation with homogeneous boundary condition\n";

#include <petsc.h>

typedef struct {
    double c;
    double (*ics)(double , double);
    double (*rhs)(double, double, double);
} AppCtx;

PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
PetscErrorCode FormInitialSolution(DM,Vec,void*);
PetscErrorCode MonitorSolution(TS, int, double,Vec,void*);


double ics(double x, double y){
    double r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
    return (r < 0.125) ? PetscExpReal(r*r*r) : 0.0; 
}
double rhs(double x, double y, double t){
    return 0.0;
}


int main(int argc, char **argv)
{
    /*petsc objects*/
    AppCtx           user;
    DM               da;
    TS               ts;
    Vec              u, r;
    Mat              J;
    double           ftime, dt;

    /*initialize*/
    PetscErrorCode ierr; 
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    /*Create 2d distributed structured grid*/
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 100, 100, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da, 0, "u"); CHKERRQ(ierr);
    
    /*set simulation parameters*/
    user.c      = 1.0;
    user.ics    = ics;
    user.rhs    = rhs;

    /*create solution and residual vector*/
    ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
    ierr = VecDuplicate(u, &r); CHKERRQ(ierr);

    /*create ts object and solve*/
    ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
    ierr = TSSetDM(ts, da); CHKERRQ(ierr);
    ierr = TSSetType(ts, TSBEULER); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, r, RHSFunction, &user); CHKERRQ(ierr);

    /*Set Jacobian*/
    ierr = DMSetMatType(da, MATAIJ); CHKERRQ(ierr);
    ierr = DMCreateMatrix(da, &J); CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL); CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, MonitorSolution, &user, NULL); CHKERRQ(ierr);

    ftime = 1.0;
    ierr = TSSetMaxTime(ts, ftime); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);

    /*set initial condition*/
    ierr = FormInitialSolution(da,u,&user);CHKERRQ(ierr);
    dt = 0.01;
    ierr = TSSetTimeStep(ts, dt); CHKERRQ(ierr);

    /*set runtime options*/
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    /*solve nonlinear system*/
    ierr = TSSolve(ts, u); CHKERRQ(ierr);



    /*destroy*/
    DMDestroy(&da);
    TSDestroy(&ts);
    MatDestroy(&J);
    VecDestroy(&u);
    VecDestroy(&r);


    return PetscFinalize();
}

PetscErrorCode FormInitialSolution(DM da,Vec U,void* ctx)
{
    PetscErrorCode  ierr;
    AppCtx         *user=(AppCtx*)ctx;
    PetscInt       i,j,xs,ys,xm,ym,Mx,My;
    PetscScalar    **u;
    PetscReal      hx,hy,x,y;
    PetscReal      xymin[2], xymax[2];

    PetscFunctionBeginUser;

    /*get mx and my*/
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

    /*compute hx and hy*/
    ierr = DMGetBoundingBox(da, xymin, xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0])/(Mx - 1);
    hy = (xymax[1] - xymin[1])/(My - 1);

    /*get array*/
    ierr = DMDAVecGetArray(da, U, &u); CHKERRQ(ierr);

    /*compute function over the locally owned part of the grid*/
    for (j = ys; j < ys + ym; ++j){
        y = j*hy;
        for (i = xs; i < xs + xm; ++i){
            x = i*hx;
            u[j][i] = user->ics(x, y);
        }
    }

    ierr = DMDAVecRestoreArray(da, U, &u); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ptr){

    DM  da;
    AppCtx   *user=(AppCtx*)ptr;
    PetscErrorCode  ierr;
    PetscInt    i, j, xs, ys, xm, ym, Mx, My;
    PetscReal   hx, hy, sx, sy, x, y;
    Vec         Ulocal;
    PetscReal   **u, **f, uxx, uyy;
    PetscReal      xymin[2], xymax[2];

    PetscFunctionBeginUser;

    /*get da from ts*/
    ierr = TSGetDM(ts, &da); CHKERRQ(ierr);

    /*get mx and my*/
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

    /*get local grid boundaries*/
    ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

    /*compute hx and hy*/
    ierr = DMGetBoundingBox(da, xymin, xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0])/(Mx - 1);
    hy = (xymax[1] - xymin[1])/(My - 1);

    sx = 1.0/(hx*hx);
    sy = 1.0/(hy*hy);

    /*create local vector that has space for ghost values*/
    ierr = DMGetLocalVector(da, &Ulocal); CHKERRQ(ierr);

    /*scatter values from global to local vector*/
    ierr = DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ulocal); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, U, INSERT_VALUES, Ulocal); CHKERRQ(ierr);

    /*get array from vector*/
    ierr = DMDAVecGetArrayRead(da, Ulocal, &u); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, F, &f); CHKERRQ(ierr);

    /*compute function over the locally owned part of the grid*/
    for (j = ys; j < ys + ym; ++j){
        y = j*hy;
        for (i = xs; i < xs + xm; ++i){
            x = i*hx;
            if (i == 0 || j == 0 || i == Mx - 1 || j == My - 1){
                f[j][i] = u[j][i];
                continue;
            }
            uxx = (u[j][i+1] - 2.0*u[j][i] + u[j][i-1])*sx;
            uyy = (u[j+1][i] - 2.0*u[j][i] + u[j-1][i])*sy;

            f[j][i] = uxx + uyy + user->rhs(x, y, t);
        }
    }

    /*restore array*/
    ierr = DMDAVecRestoreArrayRead(da, Ulocal, &u); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, F, &f); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &Ulocal); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat J,Mat Jpre,void *ctx){

    PetscErrorCode  ierr;
    DM              da;
    DMDALocalInfo   info;
    PetscInt        i, j;
    PetscReal       hx, hy, sx, sy;
    PetscReal       xymin[2], xymax[2];
    MatStencil      row, col[5];
    PetscReal       v[5];

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &da); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    
    ierr = DMGetBoundingBox(da, xymin, xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0])/(info.mx - 1);
    hy = (xymax[1] - xymin[1])/(info.my - 1);

    sx = 1.0/(hx*hx);
    sy = 1.0/(hy*hy);

    for (j = info.ys; j < info.ys + info.ym; ++j){
        for (i = info.xs; i < info.xs + info.xm; ++i){
            
            /*get row index for i,j*/
            row.i = i; row.j = j;

            if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1){
                col[0].i = i; col[0].j = j; v[0] = 1.0;
                ierr = MatSetValuesStencil(Jpre, 1, &row, 1, col, v, INSERT_VALUES); CHKERRQ(ierr);
            }
            else{
                col[0].i = i;       col[0].j = j;       v[0] = -2*sx - 2*sy;
                col[1].i = i+1;     col[1].j = j;       v[1] = sx;
                col[2].i = i-1;     col[2].j = j;       v[2] = sx;
                col[3].i = i;       col[3].j = j+1;     v[3] = sy;
                col[4].i = i;       col[4].j = j-1;     v[4] = sy;
            
                ierr = MatSetValuesStencil(Jpre, 1, &row, 5, col, v, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }


    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode MonitorSolution(TS ts, int step, double time, Vec u, void *ctx)
{
    PetscErrorCode  ierr;
    DM  da;

    ierr = TSGetDM(ts, &da); CHKERRQ(ierr);
    if ( (step % 10) == 0 )
    {
        char filename[20];
        sprintf(filename, "sol-%08d.vts", step); CHKERRQ(ierr);
        PetscViewer viewer;
        ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
        ierr = DMView(da, viewer); CHKERRQ(ierr);
        ierr = VecView(u, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    }

    return 0;
}


