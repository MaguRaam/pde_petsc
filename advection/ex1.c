static char help[] = "Solve 2d advection problem\n"
"of the form ut + div(a(x,y)u) = g(x,y,u)";


#include <petsc.h>


typedef enum {STRAIGHT, ROTATION} ProblemType;
static const char *ProblemTypes[] = {"straight","rotation", "ProblemType", "", NULL};

typedef enum {STUMP, SMOOTH, CONE, BOX} InitialType;
static const char *InitialTypes[] = {"stump", "smooth", "cone", "box", "InitialType", "", NULL};


typedef enum {NONE, CENTERED, VANLEER, KOREN} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer","koren", "LimiterType", "", NULL};


typedef struct {
    ProblemType problem;
    PetscReal windx, windy;
    PetscReal (*initial_func)(PetscReal, PetscReal);
    PetscReal (*limiter_fcn)(PetscReal);

} AppCtx;


// equal to 1 in a disc of radius 0.2 around (-0.6,-0.6)
static PetscReal stump(PetscReal x, PetscReal y){
    const PetscReal r = PetscSqrtReal((x+0.6)*(x+0.6) + (y+0.6)*(y+0.6));
    return (r < 0.2) ? 1.0 : 0.0;
}

//smooth (C^6) version of stump
static PetscReal smooth(PetscReal x, PetscReal y) {
    const PetscReal r = PetscSqrtReal((x+0.6)*(x+0.6) + (y+0.6)*(y+0.6));
    if (r < 0.2)
        return PetscPowReal(1.0 - PetscPowReal(r / 0.2,6.0),6.0);
    else
        return 0.0;
}

//cone of height 1 of base radius 0.35 centered at (-0.45,0.0)
static PetscReal cone(PetscReal x, PetscReal y) {
    const PetscReal r = PetscSqrtReal((x+0.45)*(x+0.45) + y*y);
    return (r < 0.35) ? 1.0 - r / 0.35 : 0.0;
}

//equal to 1 in square of side-length 0.5 (0.1,0.6) x (-0.25,0.25)
static PetscReal box(PetscReal x, PetscReal y) {
    if ((0.1 < x) && (x < 0.6) && (-0.25 < y) && (y < 0.25))
        return 1.0;
    else
        return 0.0;
}

typedef PetscReal (*PointwiseFcn)(PetscReal,PetscReal);
static PointwiseFcn initialptr[] = {&stump, &smooth, &cone, &box};


static PetscReal centered(PetscReal theta) { return 0.5;}
static PetscReal vanleer(PetscReal theta) { const PetscReal abstheta = PetscAbsReal(theta); return 0.5 * (theta + abstheta) / (1.0 + abstheta);}
static PetscReal koren(PetscReal theta) { const PetscReal z = (1.0/3.0) + (1.0/6.0) * theta; return PetscMax(0.0, PetscMin(1.0, PetscMin(z, theta)));}

typedef PetscReal (*LimiterFcn)(PetscReal);
static LimiterFcn limiterptr[] = {NULL, &centered, &vanleer, &koren};


static PetscReal a_wind(PetscReal x, PetscReal y, PetscInt dir, AppCtx* user) {
    switch (user->problem) {
        case STRAIGHT:
            return (dir == 0) ? user->windx : user->windy;
        case ROTATION:
            return (dir == 0) ? y : - x;
        default:
            return 0.0;
    }
}

static PetscReal g_source(PetscReal x, PetscReal y, PetscReal u, AppCtx* user) {
    return 0.0;
}


extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, AppCtx*);


int main(int argc, char **argv)
{
    /*petsc objects*/
    AppCtx              user;
    TS                  ts;
    DM                  da;
    Vec                 u;
    DMDALocalInfo       info;
    PetscReal           hx, hy;


    /*initialize*/
    PetscErrorCode ierr; 
    ierr = PetscInitialize(&argc, &argv, NULL, help); CHKERRQ(ierr);

    /*create grid*/
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, 5, 5, PETSC_DECIDE, PETSC_DECIDE, 1, 2, NULL, NULL, &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da, &user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    /*compute cell size*/
    hx = 2.0/info.mx; hy = 2.0/info.my;
    ierr = DMDASetUniformCoordinates(da, -1.0 + 0.5*hx, 1.0 - 0.5*hx, -1.0 + 0.5*hy, 1.0 - 0.5*hy, 0, 0); CHKERRQ(ierr);

    /*set ts object*/
    ierr = TSCreate(PETSC_COMM_WORLD, &ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts, da); CHKERRQ(ierr);



    /*destroy*/
    DMDestroy(&da);

    return PetscFinalize();
}

PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, AppCtx* user){
    PetscErrorCode  ierr;
    PetscInt        i, j;
    PetscReal       x, y, hx, hy, **au;

    ierr = VecSet(u, 0.0); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(info->da, u, &au); CHKERRQ(ierr);
}

