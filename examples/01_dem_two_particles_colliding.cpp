/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>
#include <math.h>

#include <iostream>

#define DIM 2


/*
  Start by declaring the types the particles will store. The first element
  will represent the coordinates, the second will be the particle's ID, the
  third velocity, and the fourth the radius of the particle.
*/

using DataTypes = Cabana::MemberTypes<double[3], // position(0)
				      int, // ids(1)
				      double[3], // velocity(2)
				      double[3], // force (3)
				      double, // mass(4)
				      double, // density(5)
				      double, // radius of the dem particles (6)
				      double, // Youngs mod (7)
				      double, // Poisson ratio (8)
				      double // Shear mod (9)
				      >;

/*
  Next declare the data layout of the AoSoA. We use the host space here
  for the purposes of this example but all memory spaces, vector lengths,
  and member type configurations are compatible.
*/
const int VectorLength = 8;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
// using ExecutionSpace = Kokkos::Cuda;
// using MemorySpace = ExecutionSpace::memory_space;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
using AoSoAType = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

// auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
// auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
// auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
// auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
// auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
// auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
// auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");
// auto aosoa_E = Cabana::slice<7>     ( aosoa,    "E");
// auto aosoa_nu = Cabana::slice<8>     ( aosoa,    "nu");
// auto aosoa_G = Cabana::slice<9>     ( aosoa,    "G");

using ListAlgorithm = Cabana::FullNeighborTag;
using ListType =
  Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;

typedef Kokkos::View<double*>   ViewVectorType;
typedef Kokkos::View<double**>  ViewMatrixType;

std::tuple<std::vector<double>,
	   std::vector<double>,
	   std::vector<double>> create_2d_block(double length, double height, double spacing){
  /*
    TODO: What is the minimum point of the created block
  */
  std::vector<double> x_vec;
  std::vector<double> y_vec;
  std::vector<double> z_vec;
  int x_no_points = length / spacing;
  int y_no_points = height / spacing;
  for(int i=0; i<=x_no_points; i++)
    {
      double x = i * spacing;
      for(int j=0; j<=y_no_points; j++)
	{
	  double y = j * spacing;
	  x_vec.push_back(x);
	  y_vec.push_back(y);
	  z_vec.push_back(0.);
	}
    }
  return std::make_tuple(x_vec, y_vec, z_vec);
}



void dem_stage_1(AoSoAType & aosoa, double dt, int * limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
  auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");

  auto half_dt = dt * 0.5;
  auto dem_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      double inv_mass_i = 1. / aosoa_mass( i );
      double tmp = inv_mass_i * half_dt;
      aosoa_velocity(i, 0) += aosoa_force( i, 0 ) * tmp;
      aosoa_velocity(i, 1) += aosoa_force( i, 1 ) * tmp;
      aosoa_velocity(i, 2) += aosoa_force( i, 2 ) * tmp;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:DEMStage1", policy,
			dem_stage_1_lambda_func );
}


void dem_stage_2(AoSoAType & aosoa, double dt, int * limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
  auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");

  auto half_dt = dt * 0.5;
  auto dem_stage_2_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      aosoa_position(i, 0) += aosoa_velocity( i, 0 ) * dt;
      aosoa_position(i, 1) += aosoa_velocity( i, 1 ) * dt;
      aosoa_position(i, 2) += aosoa_velocity( i, 2 ) * dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:DEMStage2", policy,
			dem_stage_2_lambda_func );
}


void dem_stage_3(AoSoAType & aosoa, double dt, int * limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
  auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");

  auto half_dt = dt * 0.5;
  auto dem_stage_3_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      double inv_mass_i = 1. / aosoa_mass( i );
      double tmp = inv_mass_i * half_dt;
      aosoa_velocity(i, 0) += aosoa_force( i, 0 ) * tmp;
      aosoa_velocity(i, 1) += aosoa_force( i, 1 ) * tmp;
      aosoa_velocity(i, 2) += aosoa_force( i, 2 ) * tmp;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:DEMStage1", policy,
			dem_stage_3_lambda_func );
}


void compute_force(AoSoAType &aosoa,
		   ViewVectorType & kn,
		   double dt,
		   ListType * verlet_list,
		   int * limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
  auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");
  auto aosoa_E = Cabana::slice<7>     ( aosoa,    "E");
  auto aosoa_nu = Cabana::slice<8>     ( aosoa,    "nu");
  auto aosoa_G = Cabana::slice<9>     ( aosoa,    "G");

  Cabana::deep_copy( aosoa_force, 0. );

  auto force_torque_dem_particles_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
    {
      /*
	Common to all equations in SPH.

	We compute:
	1.the vector passing from j to i
	2. Distance between the points i and j
	3. Distance square between the points i and j
	4. Velocity vector difference between i and j
	5. Kernel value
	6. Derivative of kernel value
      */
      double pos_i[3] = {aosoa_position( i, 0 ),
	aosoa_position( i, 1 ),
	aosoa_position( i, 2 )};

      double pos_j[3] = {aosoa_position( j, 0 ),
	aosoa_position( j, 1 ),
	aosoa_position( j, 2 )};

      double pos_ij[3] = {aosoa_position( i, 0 ) - aosoa_position( j, 0 ),
	aosoa_position( i, 1 ) - aosoa_position( j, 1 ),
	aosoa_position( i, 2 ) - aosoa_position( j, 2 )};

      double vel_ij[3] = {aosoa_velocity( i, 0 ) - aosoa_velocity( j, 0 ),
	aosoa_velocity( i, 1 ) - aosoa_velocity( j, 1 ),
	aosoa_velocity( i, 2 ) - aosoa_velocity( j, 2 )};

      // wij and dwij
      // double wij = 0.;
      double dwij[3] = {0., 0., 0.};

      // squared distance
      double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
      // distance between i and j
      double rij = sqrt(r2ij);
      // // compute the kernel wij
      // // compute_quintic_wij(rij, h_i, &wij);
      // // compute the gradient of kernel dwij
      // // compute_quintic_gradient_wij(pos_ij, rij, h_i, dwij);

      // normal vector passing from j to i
      double nij_x = pos_ij[0] / rij;
      double nij_y = pos_ij[1] / rij;
      double nij_z = pos_ij[2] / rij;
      /*
	====================================
	End: common to all equations in SPH.
	====================================
      */

      // const double mass_i = aosoa_mass( i );
      const double mass_j = aosoa_mass( j );

      // Find the overlap amount
      double overlap =  aosoa_radius( i ) + aosoa_radius( j ) - rij;

      // find the force if the particles are overlapping
      if (overlap > 0.) {

	// Compute stiffness
	// effective Young's modulus
	double tmp_1 = (1. - (aosoa_nu( i )*aosoa_nu( i ))) / aosoa_E( i );
	double tmp_2 = (1. - (aosoa_nu( j )*aosoa_nu( j ))) / aosoa_E( j );
	double E_eff = 1. / (tmp_1 + tmp_2);
	double tmp_3 = 1. / aosoa_radius( i );
	double tmp_4 = 1. / aosoa_radius( j );
	double R_eff = 1. / (tmp_3 + tmp_4);
	// # Eq 4 [1]
	double kn = 4. / 3. * E_eff * sqrt(R_eff);

	double tmp =  kn * overlap;
	double frc_x = tmp * nij_x;
	double frc_y = tmp * nij_y;
	double frc_z = tmp * nij_z;

	aosoa_force( i, 0 ) += frc_x;
	aosoa_force( i, 1 ) += frc_y;
	aosoa_force( i, 2 ) += frc_z;
      }

    };

  Kokkos::RangePolicy<ExecutionSpace> policy(limits[0], limits[1]);


  Cabana::neighbor_parallel_for( policy,
				 force_torque_dem_particles_lambda_func,
				 *verlet_list,
				 Cabana::FirstNeighborsTag(),
				 Cabana::SerialOpTag(),
				 "CabanaDEM:Equations:ForceTorqueComputation" );
  Kokkos::fence();
}


void output_data(AoSoAType & aosoa, int num_particles, int step, double time)
{
  // This is for setting HDF5 options
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
  auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");


  Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  Cabana::Experimental::HDF5ParticleOutput::
    writeTimeStep(
		  h5_config, "particles", MPI_COMM_WORLD,
		  step, time, num_particles,
		  aosoa_position,
		  aosoa_ids ,
		  aosoa_velocity,
		  aosoa_force,
		  aosoa_mass,
		  aosoa_density,
		  aosoa_radius
		  );
}


//---------------------------------------------------------------------------//
// TODO: explain this function in short
//---------------------------------------------------------------------------//
void run()
{
  /* This simulation is setup in three parts

     1. Create the particles (Both fluid and boundary)
     2. Assign the particle properties to the aosoa
     2a. This includes setting the limits of the particles
     3. Set the time step, time and pfreq
     4. Execute the run
     4a. Set up the neighbour list
  */

  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;

  /*
    ================================================
    Step 1
    ================================================
    1. Create the particles (Both fluid and boundary)
  */
  int no_dem_particles = 2;
  int no_bdry_particles = 0;

  int total_no_particles = no_dem_particles + no_bdry_particles;
  int num_particles = total_no_particles;
  double spacing = 0.1;
  int dem_limits[2] = {0, 2};
  /*
    ================================================
    End: Step 1
    ================================================
  */

  /*
    ================================================
    Step 2: Set the time step, time and pfreq
    ================================================
  */
  // Material properties of the particles
  auto rho = 1000.;
  ViewVectorType kn( "kn", 1 );
  ViewVectorType gravity( "gravity", 3 );
  // Create host mirrors of device views.
  ViewVectorType::HostMirror h_kn = Kokkos::create_mirror_view( kn );
  ViewVectorType::HostMirror h_gravity = Kokkos::create_mirror_view( gravity );

  h_kn[0] = 1e5;
  h_gravity[0] = 0.;
  h_gravity[1] = 0.;
  h_gravity[2] = 0.;
  double E = 1e7;
  double nu = 0.33;
  double G = 1e5;

  // Deep copy host views to device views.
  Kokkos::deep_copy( kn, h_kn );
  Kokkos::deep_copy( gravity, h_gravity );

  // Numerical parameters of the SPH scheme for fluid

  // integration related variables
  double dt = 1e-4;
  auto final_time = 1.;
  auto time = 0.;
  int steps = final_time / dt;
  // int steps = 100;
  // int steps = 1;
  // int print_freq = 1;
  int print_freq = 100;
  /*
    ================================================
    End: Step 2
    ================================================
  */


  /*
    ================================================
    Step 3: Assign the particle properties to the aosoa
    ================================================
  */
  AoSoAType aosoa( "particles", total_no_particles );
  // setup_aosoa(aosoa, &x, &y, &z, &u);
  auto aosoa_position = Cabana::slice<0>( aosoa,    "position");

  // create a mirror in the host space
  auto aosoa_host =
    Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );

  auto aosoa_host_position = Cabana::slice<0>     ( aosoa_host,   "position");
  auto aosoa_host_ids = Cabana::slice<1>          ( aosoa_host,   "ids");
  auto aosoa_host_velocity = Cabana::slice<2>     ( aosoa_host,   "velocity");
  auto aosoa_host_force = Cabana::slice<3>          ( aosoa_host,   "force");
  auto aosoa_host_mass = Cabana::slice<4>        ( aosoa_host,   "mass");
  auto aosoa_host_density = Cabana::slice<5>        ( aosoa_host,   "density");
  auto aosoa_host_radius = Cabana::slice<6>     ( aosoa_host,   "radius");
  auto aosoa_host_E = Cabana::slice<7>     ( aosoa_host,    "E");
  auto aosoa_host_nu = Cabana::slice<8>     ( aosoa_host,    "nu");
  auto aosoa_host_G = Cabana::slice<9>     ( aosoa_host,    "G");

  for ( std::size_t i = 0; i < aosoa_host_position.size(); ++i )
    {
      aosoa_host_position ( i, 0 ) = i * 2. * spacing;
      aosoa_host_position ( i, 1 ) = 0.;
      aosoa_host_position ( i, 2 ) = 0.;

      aosoa_host_ids ( i ) = i;

      aosoa_host_velocity ( i, 0 ) = 0.;
      aosoa_host_velocity ( i, 1 ) = 0.;
      aosoa_host_velocity ( i, 2 ) = 0.;

      aosoa_host_force ( i, 0 ) = 0.;
      aosoa_host_force ( i, 1 ) = 0.;
      aosoa_host_force ( i, 2 ) = 0.;

      aosoa_host_mass ( i ) = rho * pow(spacing, DIM);
      aosoa_host_density ( i ) = rho;

      aosoa_host_radius( i ) = spacing;

      aosoa_host_E( i ) = E;
      aosoa_host_nu( i ) = nu;
      aosoa_host_G( i ) = G;
    }
  aosoa_host_velocity ( 0, 0 ) = 1.;
  aosoa_host_velocity ( 1, 0 ) = -1.;
  // copy it back to aosoa
  Cabana::deep_copy( aosoa, aosoa_host );

  // int fluid_limits[2] = {0, no_fluid_particles};
  // int bdry_limits[2] = {no_fluid_particles, no_fluid_particles + no_bdry_particles};
  output_data(aosoa, num_particles, 0, time);
  // output_data(aosoa, num_particles, 100, time);
  /*
    ================================================
    End: Step 3
    ================================================
  */

  // ================================================
  // ================================================
  // create the neighbor list
  // ================================================
  // ================================================
  double neighborhood_radius = 5. * spacing;
  double grid_min[3] = { -2.0, -1.5, -neighborhood_radius };
  double grid_max[3] = { 3.0, 1.5, 0. * neighborhood_radius };
  double cell_ratio = 1.0;

  ListType verlet_list( aosoa_position, 0,
			aosoa_position.size(), neighborhood_radius,
			cell_ratio, grid_min, grid_max );

  // Main timestep loop
  for ( int step = 0; step < steps; step++ )
    {
      // move the velocity to t + dt / 2.
      dem_stage_1(aosoa, dt, dem_limits);

      // continuity_equation(aosoa, dt,
      // 			  &verlet_list,
      // 			  fluid_limits);

      // move positions to t + dt
      dem_stage_2(aosoa, dt, dem_limits);

      // // compute the forces using positions at t+dt and velocity at t+dt/2.
      // compute_force(aosoa, kn, dt,
      // 		    &verlet_list,
      // 		    dem_limits);
      dem_stage_3(aosoa, dt, dem_limits);

      // output
      if ( step % print_freq == 0 )
	{

	  std::cout << "Time is:" << time << std::endl;
	  output_data(aosoa, num_particles, step, time);
	}

      time += dt;

    }
}

int main( int argc, char* argv[] )
{

  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );

  run();

  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
