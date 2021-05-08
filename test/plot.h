#pragma once

#include <blitz/array.h>
#include <fstream>
#include <iostream>
#include <string>

using namespace blitz;

std::string int_to_string(unsigned int value, const unsigned int digits)
{
  std::string lc_string = std::to_string(value);
  if (lc_string.size() < digits)
  {
    // We have to add the padding zeroes in front of the number
    const unsigned int padding_position = (lc_string[0] == '-') ? 1 : 0;

    const std::string padding(digits - lc_string.size(), '0');
    lc_string.insert(padding_position, padding);
  }

  return lc_string;
}

void write_vtk(int i, double t, const Array<double, 2> &u,
               const Array<double, 2> &v, const Array<double, 2> &p,
               const Array<double, 2> &div_u,
               const Array<double, 1> &x, const Array<double, 1> &y)
{

  int nx = x.numElements();
  int ny = y.numElements();

  std::ofstream vtk;
  const std::string filename = "plot/plot_" + int_to_string(i, 4) + ".vtk";
  vtk.open(filename);
  vtk.flags(std::ios::dec | std::ios::scientific);
  vtk.precision(6);
  vtk << "# vtk DataFile Version 3.0"
      << "\n";
  vtk << "2D lid driven"
      << "\n";
  vtk << "ASCII"
      << "\n";
  vtk << "\nDATASET RECTILINEAR_GRID"
      << "\n";
  vtk << "\nFIELD FieldData 2"
      << "\n";
  vtk << "TIME 1 1 double"
      << "\n";
  vtk << t << "\n";
  vtk << "\nDIMENSIONS " << ny << " " << nx << " " << 1 << "\n";
  vtk << "POINTS " << nx * ny << " double"
      << "\n";
  vtk << "\n";

  for (unsigned int i = 0; i < nx; i++)
    for (unsigned int j = 0; j < ny; j++)
      vtk << x(i) << " " << y(j) << " " << 0.0 << "\n";

  vtk << "\nPOINT_DATA " << nx * ny << "\n";
  vtk << "\nSCALARS P double 1"
      << "\n";
  vtk << "LOOKUP_TABLE default"
      << "\n";
  vtk << "\n";

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vtk << p(i, j) << "\n";

  vtk << "\nVECTORS "
      << "V double"
      << "\n";
  vtk << "\n";

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vtk << u(i, j) << " " << v(i, j)
          << " 0.0"
             "\n";

  vtk << "\nSCALARS DIV_U double 1"
      << "\n";
  vtk << "LOOKUP_TABLE default"
      << "\n";
  vtk << "\n";
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      vtk << div_u(i, j) << "\n";
}

void write_tecplot(int i, double t, const Array<double, 2> &u,
                   const Array<double, 2> &v, const Array<double, 2> &p,
                   const Array<double, 2> &div_u,
                   const Array<double, 2> &mag_u,
                   const Array<double, 1> &x, const Array<double, 1> &y)
{
  int nx = x.numElements();
  int ny = y.numElements();

  std::ofstream tecplot;
  const std::string filename = "plot/plot_" + int_to_string(i, 4) + ".dat";
  tecplot.open(filename);
  tecplot.flags(std::ios::dec | std::ios::scientific);
  tecplot.precision(6);

  tecplot << "TITLE = \"lid driven\" " << std::endl
          << "VARIABLES = \"x\", \"y\", \"u\", \"v\", \"p\", \"div_u\",\"mag_u\"" << std::endl;
  tecplot << "Zone I = " << ny << " J = " << nx << std::endl;
  tecplot << "SOLUTIONTIME = " << t << std::endl;

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
      tecplot << x(i) << "\t" << y(j) << "\t" << u(i, j) << "\t" << v(i, j) << "\t" << p(i, j) << "\t" << div_u(i, j)<<"\t"<< mag_u(i,j)<<"\t" << std::endl;
  tecplot.close();
}
