#include "plot.h"


int main() {

  // finite-volume operators:

  // laplacian:
  auto laplacian = [](const Array<double, 2> &u, int i, int j, double h) {
    return (1.0 / (h * h)) * (u(i + 1, j) + u(i - 1, j) + u(i, j + 1) +
                              u(i, j - 1) - 4.0 * u(i, j));
  };

  // u advection:
  auto adv_u = [](const Array<double, 2> &u, const Array<double, 2> &v, int i,
                  int j, double h) {
    auto uu_e = (u(i + 1, j) + u(i, j)) * (u(i + 1, j) + u(i, j));
    auto uu_w = (u(i - 1, j) + u(i, j)) * (u(i - 1, j) + u(i, j));
    auto uv_n = (u(i, j + 1) + u(i, j)) * (v(i + 1, j) + v(i, j));
    auto uv_s = (u(i, j) + u(i, j - 1)) * (v(i + 1, j - 1) + v(i, j - 1));

    return (0.25 / h) * (uu_e - uu_w + uv_n - uv_s);
  };

  // v advection:
  auto adv_v = [](const Array<double, 2> &u, const Array<double, 2> &v, int i,
                  int j, double h) {
    auto vu_e = (u(i, j + 1) + u(i, j)) * (v(i + 1, j) + v(i, j));
    auto vu_w = (u(i - 1, j + 1) + u(i - 1, j)) * (v(i, j) + v(i - 1, j));
    auto vv_n = (v(i, j + 1) + v(i, j)) * (v(i, j + 1) + v(i, j));
    auto vv_s = (v(i, j) + v(i, j - 1)) * (v(i, j) + v(i, j - 1));

    return (0.25 / h) * (vu_e - vu_w + vv_n - vv_s);
  };

	// divergence;
	auto div = [](const Array<double, 2> &u, const Array<double, 2> &v, int i,
                  int j, double h)
	{
		return (1.0/h)*(u(i,j) - u(i-1,j)) + (1.0/h)*(v(i,j) - v(i,j-1));
	};

  // velocity magnitude:
  auto magnitude = [](const Array<double,2>& u, const Array<double,2>& v, int i, int j, double h)
  {
    return sqrt( u(i,j)*u(i,j) + v(i,j)*v(i,j) );
  };
		 

  // fluid property:
  constexpr double nu = 0.01; // kinematic viscosity

  // 2d square cavity:
  constexpr int dim = 2;
  constexpr double lx = 1.0;    // domain size along x
  constexpr double ly = 1.0;    // domain size along y
  constexpr int nx = 100;       // no of cells along x
  constexpr int ny = 100;       // no of cells along y
  constexpr double h = lx / nx; // cell size

  // x and y coordinates of cell-centers:
  Array<double, 1> x(nx), y(ny);
  for (int i = 0; i < nx; ++i)
    x(i) = (i + 0.5) * h;
  for (int j = 0; j < ny; ++j)
    y(j) = (j + 0.5) * h;

  // allocate memory (interior + ghost):
  Array<double, dim> u(nx + 1, ny + 2), un(nx + 1, ny + 2), uu(nx, ny); // u
  Array<double, dim> v(nx + 2, ny + 1), vn(nx + 2, ny + 1), vv(nx, ny); // v
  Array<double, dim> p(nx + 2, ny + 2), div_u(nx + 2, ny +2), c(nx+2,ny+2), mag_u(nx+2,ny+2);


  /*Array indexing guide:
    u interior cells: i = 1:nx-1, j = 1:ny
    v interior cells: i = 1:nx  , j = 1:ny-1
    p interior cells: i = 1:nx  , j = 1:ny
  */

  //initialize c:
  c = 0.25;
  c(1,Range(2,ny-1)) = 1.0/3.0;
  c(nx,Range(2,ny-1)) = 1.0/3.0;
  c(Range(2,nx-1),1) = 1.0/3.0;
  c(Range(2,nx-1),ny) = 1.0/3.0;
  c(1,1) = 0.5;
  c(1,ny) = 0.5;
  c(nx,1) = 0.5;
  c(nx,ny) = 0.5;

  // initial condition:
  u = 0.0;
  un = 0.0;
  uu = 0.0;
  v = 0.0;
  vn = 0.0;
  vv = 0.0;
  p = 0.0;

  // u boundary condition:
  double ut = 10.0;

  // time step:
  double dt = .0005;
  double t = 0.0;
  int nt = 10000;
  int maxit = 100;

  // stabilty criteria:
  double alpha = (nu * dt) / (h * h);
  double cfl   = (ut * dt)/h;

  //Re no
  double Re = (ut * lx)/nu;

  //SOR
  double beta = 1.2;


  // evolve in time:
  for (int n = 0; n < nt; ++n) {

    // set u = 1 at lid top:
    Range I(0, nx);
    u(I, ny + 1) = 2.0 * ut - u(I, ny);

    //update u: du/dt = -adv(u) + nu*diff(u)
    for (int i = 1; i <= nx - 1; ++i)
      for (int j = 1; j <= ny; ++j)
        un(i, j) = u(i, j) + dt * (-adv_u(u, v, i, j, h) + nu * laplacian(u, i, j, h));
    
    //update v: dv/dt = -adv(v) + nu*diff(v)
    for (int i = 1; i <= nx; ++i)
      for (int j = 1; j <= ny - 1; ++j)
        vn(i, j) = v(i, j) + dt * (-adv_v(u, v, i, j, h) + nu * laplacian(v, i, j, h));

    // laplacian(p)  = 1/dt*div(u)
    for (int it = 0; it < maxit; ++it){
      for (int i = 1; i <= nx; ++i){
        for (int j = 1; j <= ny; ++j){
         p(i,j)=beta*c(i,j)*(p(i+1,j)+p(i-1,j)+p(i,j+1)+p(i,j-1)-
         (h/dt)*(un(i,j)-un(i-1,j)+vn(i,j)-vn(i,j-1)))+(1-beta)*p(i,j);

        }
      }
    }

    //correct u velocity:
    u(Range(1,nx-1),Range(1,ny)) = un(Range(1,nx-1),Range(1,ny)) - (dt/h)*( p(Range(2,nx),Range(1,ny)) - p(Range(1,nx-1),Range(1,ny))  );
    v(Range(1,nx),Range(1,ny-1)) = vn(Range(1,nx),Range(1,ny-1)) - (dt/h)*( p(Range(1,nx),Range(2,ny)) - p(Range(1,nx), Range(1,ny-1))  );          

    //compute divergence of u: check for incompressibility:
    for (int i = 1; i <= nx; ++i)
      for (int j = 1; j <= ny; ++j)
        div_u(i,j) = div(u,v,i,j,h);
    
    //compute velocity magnitude:
    for (int i = 1; i <= nx; ++i)
      for (int j = 1; j <= ny; ++j)
        mag_u(i,j) = magnitude(u,v,i,j,h);
    


    // compute average of u and v to store at cell centers:
    uu = 0.5 * (u(Range(1, nx), Range(1, ny)) + u(Range(0, nx - 1), Range(1, ny)));
    vv = 0.5 * (v(Range(1, nx), Range(1, ny)) + v(Range(1, nx), Range(0, ny - 1)));

    t += dt;
		
    // write data:
	if ((n % 500) == 0)
	{
		std::cout<<"Re = "<<Re<<"\tcfl = "<<cfl<<"\talpha = "<<alpha<<"\tt = "<<t<<std::endl;
    	write_tecplot(n, t, uu, vv, p(Range(1, nx), Range(1, ny)), div_u(Range(1, nx), Range(1, ny)), mag_u(Range(1, nx), Range(1, ny)), x, y);
	}

  }
  return 0;
}
