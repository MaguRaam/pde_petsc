static char help[] = "2d Multigrid Poisson solver with homogeneous boundary condition\n";



#include <petsc.h>


typedef struct {
    double(*rhs)(double, double);
    double(*exact)(double, double);
} AppCtx;


PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  AppCtx    *user = (AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,M,N,xm,ym,xs,ys;
  PetscScalar    Hx,Hy,x,y;
  PetscScalar    **array;
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr); /* Fine grid */
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  
  for (j=ys; j<ys+ym; j++) {
    y = ((PetscReal)j+0.5)*Hy;
    for (i=xs; i<xs+xm; i++) {
      x = ((PetscReal)i+0.5)*Hx;
      array[j][i] = user->rhs(x, y)*Hx*Hy;
    }
  }

  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp,Mat J, Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];
  DM             da;
  MatNullSpace   nullspace;

  PetscFunctionBeginUser;
  ierr  = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr  = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr  = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;

      if (i==0 || j==0 || i==M-1 || j==N-1) {
        num=0; numi=0; numj=0;
        if (j!=0) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j-1;
          num++; numj++;
        }
        if (i!=0) {
          v[num] = -HydHx;              col[num].i = i-1; col[num].j = j;
          num++; numi++;
        }
        if (i!=M-1) {
          v[num] = -HydHx;              col[num].i = i+1; col[num].j = j;
          num++; numi++;
        }
        if (j!=N-1) {
          v[num] = -HxdHy;              col[num].i = i;   col[num].j = j+1;
          num++; numj++;
        }
        v[num] = ((PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx); col[num].i = i;   col[num].j = j;
        num++;
        ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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

PetscErrorCode ComputeExact(DM da,Vec b,void *ctx)
{
  AppCtx    *user = (AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,M,N,xm,ym,xs,ys;
  PetscScalar    Hx,Hy,x,y;
  PetscScalar    **array;
  
  PetscFunctionBeginUser;
  
  ierr = DMDAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr); /* Fine grid */
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  
  for (j=ys; j<ys+ym; j++) {
    y = ((PetscReal)j+0.5)*Hy;
    for (i=xs; i<xs+xm; i++) {
      x = ((PetscReal)i+0.5)*Hx;
      array[j][i] = user->exact(x, y);
    }
  }

  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


double rhs(double x, double y){
    int n = 1;
    return 2.0*n*n*PETSC_PI*PETSC_PI*PetscCosScalar(n*PETSC_PI*x)*PetscCosScalar(n*PETSC_PI*y);
}

double exact(double x, double y){
    int n = 1;
    return PetscCosScalar(n*PETSC_PI*x)*PetscCosScalar(n*PETSC_PI*y);
}


int main(int argc, char **argv)
{
    /*petsc objects*/
    AppCtx           user;
    DM               da, da_after;
    KSP              ksp;
    Vec              p, pexact;
    double           error;

    /*initialize*/
    PetscErrorCode ierr; 
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    /*set simulation parameters*/    
    int mx = 800, my = 800;
    double hx = 1.0/(double)mx, hy = 1.0/(double)my;
    user.rhs    = rhs;
    user.exact  = exact;

    /*Create 2d distributed structured grid*/
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, mx, my, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0.5*hx, 1.0 - 0.5*hx, 0.5*hy, 1.0 - 0.5*hy, 0, 0); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da, 0, "p"); CHKERRQ(ierr);
    
    /*create ksp and solve Ax = b*/
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetDM(ksp, da); CHKERRQ(ierr);
    ierr = KSPSetComputeRHS(ksp, ComputeRHS, &user); CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp, ComputeMatrix, &user); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSetUp(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, NULL, NULL); CHKERRQ(ierr);

    /*get solution and plot*/
    ierr = KSPGetSolution(ksp, &p); CHKERRQ(ierr); 
    ierr = KSPGetDM(ksp, &da_after); CHKERRQ(ierr);
    ierr = write_vts(da_after, p, "p.vts"); CHKERRQ(ierr);

    /*get exact solution plot*/
    ierr = DMCreateGlobalVector(da_after, &pexact); CHKERRQ(ierr);
    ierr = ComputeExact(da_after, pexact, &user); CHKERRQ(ierr);
    ierr = write_vts(da_after, pexact, "pexact.vts"); CHKERRQ(ierr);

    /*compute and display error*/
    ierr = VecAXPY(p, -1.0, pexact); CHKERRQ(ierr);
    ierr = VecDestroy(&pexact); CHKERRQ(ierr);
    ierr = VecNorm(p, NORM_INFINITY, &error); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Linfty error on %d x %d grid = %e\n", mx, my, error); CHKERRQ(ierr);

    /*destroy*/
    DMDestroy(&da);
    KSPDestroy(&ksp);

    return PetscFinalize();
}

/*
  mpirun -n 4 ./ex3 -pc_type mg -ksp_type cg -ksp_rtol 1.0e-12 -ksp_monitor

*/

