static char help[] = "solve simple diffusion equation with neumann and periodic boundary \n";

#include <petsc.h> 


struct AppCtx{ double D0;};

static PetscReal gamma_neumann(PetscReal y) {
    return PetscSinReal(6.0 * PETSC_PI * y);
}

double f_source(double x, double y){
    return 3.0 * PetscExpReal(-25.0 * (x-0.6) * (x-0.6)) * PetscSinReal(2.0*PETSC_PI*y);
}

PetscErrorCode spacing(DMDALocalInfo *info, PetscReal *hx, PetscReal *hy) {
    if (hx)  *hx = 1.0 / (PetscReal)(info->mx-1);
    if (hy)  *hy = 1.0 / (PetscReal)(info->my);   // periodic direction
    return 0;
}

PetscErrorCode  f(DMDALocalInfo *info, double t, double **au, double **aG, void *ctx)
{
    PetscErrorCode  ierr;
    AppCtx *user = (AppCtx*)ctx;
    double hx, hy;
    double uxx, uyy;
    double ul, ur;
    double x,y;
    int i,j, mx = info->mx;

    //get grid size:
    ierr = spacing(info, &hx, &hy); CHKERRQ(ierr);

    //loop over locally owned grid:
    for (j = info->ys; j < info->ys + info->ym; ++j){
        y = j*hy;
        for (i = info->xs; i < info->xs + info->xm; ++i){
            x = i*hx;

            //compute ul:
            ul = (i == 0) ? au[j][i+1] + 2.0 * hx * gamma_neumann(y) : au[j][i-1];
            
            //compute ur:
            ur = (i == mx-1) ? au[j][i-1] : au[j][i+1];

            uxx = (ul - 2.0 * au[j][i]+ ur) / (hx*hx);
            uyy = (au[j-1][i] - 2.0 * au[j][i]+ au[j+1][i]) / (hy*hy);

            aG[j][i] = user->D0 * (uxx + uyy) + f_source(x,y);

        }
    }

    return 0;
}

//compute jacobian:
PetscErrorCode  jacobian(DMDALocalInfo *info, double t, double **au, Mat J, Mat P, AppCtx *user)
{
    PetscErrorCode  ierr;
    int i, j, ncols;
    double D = user->D0;
    double hx, hy, hx2, hy2, v[5];
    MatStencil col[5], row;

    ierr = spacing(info, &hx, &hy); CHKERRQ(ierr);
    hx2 = hx * hx;  hy2 = hy * hy;

    for (j = info->ys; j < info->ys+info->ym; j++){
        row.j = j;  col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++){
            row.i = i; col[0].i = i;
            v[0] = - 2.0 * D * (1.0 / hx2 + 1.0 / hy2);
            col[1].j = j-1;  col[1].i = i;    v[1] = D / hy2;
            col[2].j = j+1;  col[2].i = i;    v[2] = D / hy2;
            col[3].j = j;    col[3].i = i-1;  v[3] = D / hx2;
            col[4].j = j;    col[4].i = i+1;  v[4] = D / hx2;
            ncols = 5;

            if (i == 0) {
                ncols = 4;
                col[3].j = j;  col[3].i = i+1;  v[3] = 2.0 * D / hx2;
            } else if (i == info->mx-1) {
                ncols = 4;
                col[3].j = j;  col[3].i = i-1;  v[3] = 2.0 * D / hx2;
            }

            ierr = MatSetValuesStencil(P,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    return 0;

}

//monitor function:
PetscErrorCode monitor_func(TS ts, int step, double time, Vec u, void *ctx)
{
    PetscErrorCode ierr; 
    AppCtx *Ctx = (AppCtx*)ctx;

    DM da;
    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    
    if((step%1) == 0)
    {
        char filename[20];
        sprintf(filename, "sol-%08d.vts", step); // 8 is the padding level, increase it for longer simulations 
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in vts format to %s at t = %f, step = %d\n", filename, time, step);CHKERRQ(ierr);
        PetscViewer viewer;  
        ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr); 
        ierr = DMView(da, viewer);
        VecView(u, viewer);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

    return 0;
}


int main(int argc, char **argv)
{
    AppCtx              user;
    TS                  ts;
    DM                  da;
    DMDALocalInfo       info;
    Vec                 u;
    double              t0, tf;

    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    //set diffusion constant:
    user.D0 = 1.0;

    //create grid:
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC,
                        DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0, 1, 0, 1, 0, 0); CHKERRQ(ierr);

    //create vector:
    ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);

    //create ts:
    ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts, da); CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts, &user); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES, (DMDATSRHSFunctionLocal)f,&user); CHKERRQ(ierr);
    ierr = DMDATSSetRHSJacobianLocal(da, (DMDATSRHSJacobianLocal)jacobian, &user); CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,monitor_func,&user,NULL); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSBDF); CHKERRQ(ierr);
    ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,.1); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,0.001); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetMaxTime(ts,&tf); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "solving on %d x %d grid for t0=%g to tf=%g ...\n",
           info.mx,info.my,t0,tf); CHKERRQ(ierr);
    
    //solve:
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    //destroy:
    DMDestroy(&da);
    TSDestroy(&ts);
    VecDestroy(&u);

    return PetscFinalize();
}

/* runtime options:

    //visualize solution
    mpirun -n 4 ./diffusion -da_refine 3 -ts_max_time 0.02 -ts_dt 1.0e-5 -ts_monitor -ts_monitor_solution draw



*/