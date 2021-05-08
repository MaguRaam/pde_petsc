static char help[] = "solve simple ode\n";


#include <petsc.h>


PetscErrorCode f(TS ts, double t, Vec x, Vec f, void *ctx)
{
    PetscErrorCode  ierr;
    double *aF;
    const double *aX;

    ierr = VecGetArrayRead(x, &aX); CHKERRQ(ierr);
    ierr = VecGetArray(f, &aF); CHKERRQ(ierr);

    aF[0] = aX[1];
    aF[1] = -aX[0] + t;

    ierr = VecRestoreArray(f, &aF); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x, &aX); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode  jacobian(TS ts, double t, Vec x, Mat J, Mat P, void *ptr)
{
    PetscErrorCode  ierr;
    int row[2] = {0,1}, col[2] = {0,1};
    double v[4] = {0.0, 1.0, -1.0, 0.0};

    ierr = MatSetValues(P, 2, row, 2, col, v, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P){
        ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode exact(Vec x, double t)
{
    PetscErrorCode  ierr;
    double *aX;
    ierr = VecGetArray(x, &aX); CHKERRQ(ierr);
    aX[0] = t - sin(t);
    aX[1] = 1 - cos(t);
    ierr = VecRestoreArray(x, &aX); CHKERRQ(ierr);

    return 0;
}



int main(int argc, char **argv)
{
    TS  ts;
    Mat J;
    Vec x, xexact;
    double error;

    PetscErrorCode  ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, help);

    //create solution vector:
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, 2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &xexact); CHKERRQ(ierr);

    //set TS solver:
    ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, f, NULL); CHKERRQ(ierr);
   

    //create jacobian matrix:
    ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
    ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2); CHKERRQ(ierr);
    ierr = MatSetFromOptions(J); CHKERRQ(ierr);
    ierr = MatSetUp(J); CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts, J, J, jacobian, NULL); CHKERRQ(ierr);
    ierr = TSSetType(ts, TSCN); CHKERRQ(ierr);

    //set time axis:
    ierr = TSSetTime(ts, 0.0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts, 20); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, 0.1); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    //intialize and solve:
    double t;
    ierr = TSGetTime(ts, &t); CHKERRQ(ierr);
    ierr = exact(x, t); CHKERRQ(ierr);
    ierr = TSSolve(ts, x); CHKERRQ(ierr);

    //compute error at final time:
    ierr = TSGetTime(ts, &t); CHKERRQ(ierr);
    ierr = exact(xexact, t); CHKERRQ(ierr);
    ierr = VecAXPY(x, -1, xexact); CHKERRQ(ierr);
    ierr = VecNorm(x, NORM_INFINITY, &error);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "error at tf = %.3f :  |x-x_exact|_inf = %g\n", t, error); CHKERRQ(ierr);


    //destroy:
    VecDestroy(&x);
    VecDestroy(&xexact);
    TSDestroy(&ts);

    return PetscFinalize();
}

/* runtime options

See adaptive timestep:
./ode -ts_monitor

To see solution at each time step:
./ode -ts_monitor_solution

To see solution method:
./ode -ts_view 

plot solution at each time;
./ode -ts_monitor -ts_monitor_lg_solution -draw_pause 0.1

write data in binary format read by Python script TODO
./ode -ts_monitor binary:t.dat -ts_monitor_solution binary:y.dat

control to tmax dt 
./ode -ts_init_time 1.0 -ts_max_time 2.0 -ts_dt 0.001 -ts_monitor

switch of adaptive time stepping
./ode -ts_adapt_type none -ts_monitor

list of ts types:
./ode -ts_type euler
./ode -ts_type rk -ts rk type 4

TODO add more

*/