static char help[] = "Solves simple nonlinear equation\n";

#include <petsc.h>



PetscErrorCode rhs_func(SNES snes, Vec x, Vec f, void *ctx)
{
    PetscErrorCode ierr;
    const PetscReal *a_x;
    PetscReal *a_f;
    const PetscReal b = 2.0;

    ierr = VecGetArrayRead(x, &a_x); CHKERRQ(ierr);
    ierr = VecGetArray(f, &a_f); CHKERRQ(ierr);

    a_f[0] = (1.0/b)*PetscExpReal(b*a_x[0]) - a_x[1];
    a_f[1] = a_x[0] * a_x[0] + a_x[1] * a_x[1] - 1.0;

    ierr = VecRestoreArrayRead(x, &a_x); CHKERRQ(ierr);
    ierr = VecRestoreArray(f, &a_f); CHKERRQ(ierr);
    return 0;
}


int main(int argc, char **argv)
{
    PetscErrorCode      ierr;
    SNES                snes;
    Vec                 x, r;

    //initialize petsc program:
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    //create and initialize solution vector:
    ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, 2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecSet(x, 1.0);

    //create residual vector r:
    ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

    //create ksp solver:
    ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes, r, rhs_func, NULL); CHKERRQ(ierr);
    ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    //destroy objects:
    VecDestroy(&x); 
    VecDestroy(&r);
    SNESDestroy(&snes);

    return PetscFinalize();
}