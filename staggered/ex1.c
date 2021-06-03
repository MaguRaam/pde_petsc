static char help[] = "Solve a toy 2D problem on a staggered grid\n\n";
/*

  To demonstrate the basic functionality of DMStag, solves an isoviscous
  incompressible Stokes problem on a rectangular 2D domain, using a manufactured
  solution.

  u_xx + u_yy - p_x = f^x
  v_xx + v_yy - p_y = f^y
  u_x + v_y         = g

  g is 0 in the physical case.

  Boundary conditions give prescribed flow perpendicular to the boundaries,
  and zero derivative perpendicular to them (free slip).

  Use the -pinpressure option to fix a pressure node, instead of providing
  a constant-pressure nullspace. This allows use of direct solvers, e.g. to
  use UMFPACK,

     ./ex2 -pinpressure 1 -pc_type lu -pc_factor_mat_solver_type umfpack

  This example demonstrates the use of DMProduct to efficiently store coordinates
  on an orthogonal grid.

*/
#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h> /* Includes petscdmproduct.h */

/* Shorter, more convenient names for DMStagStencilLocation entries */
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT


static PetscErrorCode CreateReferenceSolution(DM,Vec*);
static PetscErrorCode CreateSystem(DM,Mat*,Vec*,PetscBool);

 
static PetscScalar uxRef(PetscScalar x,PetscScalar y){return 0.0*x + y*y - 2.0*y*y*y + y*y*y*y;}       /* no x-dependence  */
static PetscScalar uyRef(PetscScalar x,PetscScalar y) {return x*x - 2.0*x*x*x + x*x*x*x + 0.0*y;}      /* no y-dependence  */
static PetscScalar pRef (PetscScalar x,PetscScalar y) {return -1.0*(x-0.5) + -3.0/2.0*y*y + 0.5;}      /* zero integral    */
static PetscScalar fx   (PetscScalar x,PetscScalar y) {return 0.0*x + 2.0 -12.0*y + 12.0*y*y + 1.0;}   /* no x-dependence  */
static PetscScalar fy   (PetscScalar x,PetscScalar y) {return 2.0 -12.0*x + 12.0*x*x + 3.0*y;}
static PetscScalar g    (PetscScalar x,PetscScalar y) {return 0.0*x*y;}                                 /* identically zero */

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dmSol;
  Vec            sol,solRef,rhs;
  Mat            A;
  KSP            ksp;
  PC             pc;
  PetscBool      pinPressure;

  /* Initialize PETSc and process command line arguments */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  pinPressure = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-pinpressure",&pinPressure,NULL);CHKERRQ(ierr);

  /* Create 2D DMStag for the solution, and set up. */
  {
    const PetscInt dof0 = 0, dof1 = 1,dof2 = 1; /* 1 dof on each edge and element center */
    const PetscInt stencilWidth = 1;
    ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,9,9,PETSC_DECIDE,PETSC_DECIDE,dof0,dof1,dof2,DMSTAG_STENCIL_BOX,stencilWidth,NULL,NULL,&dmSol);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmSol);CHKERRQ(ierr);
    ierr = DMSetUp(dmSol);CHKERRQ(ierr);
  }

  /* Define uniform coordinates as a product of 1D arrays */
  ierr = DMStagSetUniformCoordinatesProduct(dmSol,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);

  /* Compute (manufactured) reference solution */
  ierr = CreateReferenceSolution(dmSol,&solRef);CHKERRQ(ierr);

  /* Clean up and finalize PETSc */
  ierr = VecDestroy(&solRef);CHKERRQ(ierr);
  ierr = DMDestroy(&dmSol);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

 

/* Create a reference solution.
   Here, we use the more direct method of iterating over arrays.  */
static PetscErrorCode CreateReferenceSolution(DM dmSol,Vec *pSolRef)
{
  PetscErrorCode ierr;
  PetscInt       startx,starty,nx,ny,nExtra[2],ex,ey;
  PetscInt       iuy,iux,ip,iprev,icenter;
  PetscScalar    ***arrSol,**cArrX,**cArrY;
  Vec            solRefLocal;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dmSol,pSolRef);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);

  /* Obtain indices to use in the raw arrays */
  ierr = DMStagGetLocationSlot(dmSol,DOWN,0,&iuy);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,LEFT,0,&iux);CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmSol,ELEMENT,0,&ip);CHKERRQ(ierr);

  /* Use high-level convenience functions to get raw arrays and indices for 1d coordinates */
  ierr = DMStagGetProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmSol,LEFT,&iprev);CHKERRQ(ierr);

  ierr = DMStagVecGetArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmSol,&startx,&starty,NULL,&nx,&ny,NULL,&nExtra[0],&nExtra[1],NULL);CHKERRQ(ierr);
  for (ey=starty; ey<starty + ny + nExtra[1]; ++ey) {
    for (ex=startx; ex<startx + nx + nExtra[0]; ++ex) {
      arrSol[ey][ex][iuy] = uyRef(cArrX[ex][icenter],cArrY[ey][iprev]);
      arrSol[ey][ex][iux] = uxRef(cArrX[ex][iprev],cArrY[ey][icenter]);
      arrSol[ey][ex][ip]  = pRef(cArrX[ex][icenter],cArrY[ey][icenter]);
    }
  }
  ierr = DMStagVecRestoreArray(dmSol,solRefLocal,&arrSol);CHKERRQ(ierr);
  ierr = DMStagRestoreProductCoordinateArraysRead(dmSol,&cArrX,&cArrY,NULL);CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dmSol,solRefLocal,INSERT_VALUES,*pSolRef);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmSol,&solRefLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 

static PetscErrorCode CreateSystem(DM dmSol, Mat *pA, Vec *pRhs, PetscBool pinPressure)
{
    PetscErrorCode  ierr;
    PetscInt        N[2];
    PetscInt        ex,ey,startx,starty,nx,ny;
    PetscInt        iprev,icenter,inext;
    Mat             A;
    Vec             rhs;
    PetscReal       hx,hy;
    PetscScalar     **cArrX,**cArrY;

    PetscFunctionBeginUser;

    ierr = DMStagGetProductCoordinateLocationSlot(dmSol,ELEMENT,&icenter);CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateLocationSlot(dmSol,LEFT,&iprev);CHKERRQ(ierr);
    ierr = DMStagGetProductCoordinateLocationSlot(dmSol,RIGHT,&inext);CHKERRQ(ierr);

    ierr = DMCreateMatrix(dmSol, pA); CHKERRQ(ierr);


    PetscFunctionReturn(0);
}