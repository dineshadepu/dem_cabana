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
				      double[3], // acceleration(3)
				      double, // mass(4)
				      double, // density(5)
				      double, // h (smoothing length) (6)
				      double, // pressure (7)
				      int, // is_fluid(8)
				      int, // is_boundary(9)
				      double, // rate of density change (10)
				      double, // rate of pressure change (11)
				      double, // sum_wij (12)
				      double[3], // velocity of wall (13)
				      double[3], // dummy velocity for comutation (14)
				      double[3], // force on dem particles (15)
				      double, // radius of the dem particles (16)
				      int, // Total no of current contacts a dem particle is in (17)
				      int[10], // Indices of the contacts particle has (18)
				      double[10][3], // tangential force of dem particles (19)
				      double[10][3], // tangential disp of dem particles (20)
				      double[10][3] // normal force of the contact (21)
				      >;

/*
  Next declare the data layout of the AoSoA. We use the host space here
  for the purposes of this example but all memory spaces, vector lengths,
  and member type configurations are compatible.
*/
const int VectorLength = 8;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
using AoSoAType = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

// auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
// auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
// auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
// auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "acc");
// auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
// auto aosoa_density = Cabana::slice<5>     ( aosoa,    "density");
// auto aosoa_h = Cabana::slice<6>           ( aosoa,    "h");
// auto aosoa_p = Cabana::slice<7>           ( aosoa,    "p");
// auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
// auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
// auto aosoa_density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
// auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
// auto aosoa_sum_wij = Cabana::slice<12>           ( aosoa,    "sum_wij");
// auto aosoa_velocity_g = Cabana::slice<13>     ( aosoa,    "velocity_g");
// auto aosoa_velocity_f = Cabana::slice<14>     ( aosoa,    "velocity_f");
// auto aosoa_force = Cabana::slice<15>     ( aosoa,    "force");
// auto aosoa_radius = Cabana::slice<16>     ( aosoa,    "radius");
// auto aosoa_tot_cnts = Cabana::slice<17>     ( aosoa,    "tot_no_cnts");
// auto aosoa_cnts_idx = Cabana::slice<18>     ( aosoa,    "cnts_idx");
// auto aosoa_tng_frc = Cabana::slice<19>     ( aosoa,    "tng_frc");
// auto aosoa_tng_disp = Cabana::slice<20>     ( aosoa,    "tng_disp");
// auto aosoa_normal_frc = Cabana::slice<21>     ( aosoa,    "normal_frc");

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


void dem_stage_1(AoSoAType & aosoa,
		 // ViewVectorType & kn_kt_fric_coeff,
		 double dt, int * limits){
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_force = Cabana::slice<15>     ( aosoa,    "force");

  auto half_dt = dt * 0.5;
  auto dem_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      // double tmp1 = kn_kt_fric_coeff[0];
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
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "density");
  auto aosoa_h = Cabana::slice<6>           ( aosoa,    "h");
  auto aosoa_p = Cabana::slice<7>           ( aosoa,    "p");
  auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
  auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
  auto aosoa_density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
  auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
  auto aosoa_sum_wij = Cabana::slice<12>           ( aosoa,    "sum_wij");
  auto aosoa_velocity_g = Cabana::slice<13>     ( aosoa,    "velocity_g");
  auto aosoa_velocity_f = Cabana::slice<14>     ( aosoa,    "velocity_f");
  auto aosoa_force = Cabana::slice<15>     ( aosoa,    "force");

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
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_force = Cabana::slice<15>     ( aosoa,    "force");

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

void update_tangential_contacts(AoSoAType & aosoa, double dt, int * limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "density");
  auto aosoa_h = Cabana::slice<6>           ( aosoa,    "h");
  auto aosoa_force = Cabana::slice<15>     ( aosoa,    "force");
  auto aosoa_radius = Cabana::slice<16>     ( aosoa,    "radius");
  auto aosoa_tot_no_cnts = Cabana::slice<17>     ( aosoa,    "tot_no_cnts");
  auto aosoa_cnts_idx = Cabana::slice<18>     ( aosoa,    "cnts_idx");
  auto aosoa_tng_frc = Cabana::slice<19>     ( aosoa,    "tng_frc");
  auto aosoa_tng_disp = Cabana::slice<20>     ( aosoa,    "tng_disp");
  auto aosoa_normal_frc = Cabana::slice<21>     ( aosoa,    "normal_frc");

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
		   ViewVectorType & kn_kt_fric_coeff,
		   double dt,
		   ListType * verlet_list,
		   int * limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "density");
  auto aosoa_h = Cabana::slice<6>           ( aosoa,    "h");
  auto aosoa_force = Cabana::slice<15>     ( aosoa,    "force");
  auto aosoa_radius = Cabana::slice<16>     ( aosoa,    "radius");
  auto aosoa_tot_no_cnts = Cabana::slice<17>     ( aosoa,    "tot_no_cnts");
  auto aosoa_cnts_idx = Cabana::slice<18>     ( aosoa,    "cnts_idx");
  auto aosoa_tng_frc = Cabana::slice<19>     ( aosoa,    "tng_frc");
  auto aosoa_tng_disp = Cabana::slice<20>     ( aosoa,    "tng_disp");
  auto aosoa_normal_frc = Cabana::slice<21>     ( aosoa,    "normal_frc");

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

      // h value of particle i
      double h_i = aosoa_h( i );

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
	// Check if the particle is being tracked

	// if the particle is being tracked then update the
	// tracked tangential index
	auto found = 0;
	auto found_at = -1;
	for ( std::size_t k = 0; k < aosoa_tot_no_cnts( i ); ++k )
	  {
	    if (j == aosoa_cnts_idx( i, k )) {
	      found = 1;
	      found_at = k;
	    }
	  }

	if (found == 1)
	  {
	    // don't do anything
	  }
	else{
	  // implies this is a new contact
	  // so add it to the contacting indices and increase the total
	  // count of the contacting indices

	  // create a new index at the end of the tracking
	  aosoa_tot_no_cnts ( i ) += 1;
	  found_at = aosoa_tot_no_cnts ( i ) - 1;
	  aosoa_cnts_idx ( i, found_at ) = j;
	  aosoa_tng_frc ( i, found_at, 0 ) = 0.;
	  aosoa_tng_frc ( i, found_at, 1 ) = 0.;
	  aosoa_tng_frc ( i, found_at, 2 ) = 0.;
	}

	// normal relative velocity
	double vel_ij_dot_nij = vel_ij[0] * nij_x + vel_ij[1] * nij_y + vel_ij[2] * nij_z;

	// d_overlap[t2] = overlap;
	// double tmp = kn[0] * overlap;
	double tmp = kn_kt_fric_coeff[0] * overlap;
	// double tmp = 1e5 * overlap;

	double fn_x = tmp * nij_x;
	double fn_y = tmp * nij_y;
	double fn_z = tmp * nij_z;

	// Incremenet the tangential force and add it to the total force
	// check if there is relative motion
	double vij_magn = pow(vel_ij[0]*vel_ij[0] + vel_ij[1]*vel_ij[1] + vel_ij[2]*vel_ij[2], 0.5);
	if (vij_magn < 1e-12) {
	  // make the tangential displacement
	  aosoa_tng_disp ( i, found_at, 0 ) = 0.;
	  aosoa_tng_disp ( i, found_at, 1 ) = 0.;
	  aosoa_tng_disp ( i, found_at, 2 ) = 0.;

	  // make the tangential tangential force
	  aosoa_tng_frc ( i, found_at, 0 ) = 0.;
	  aosoa_tng_frc ( i, found_at, 1 ) = 0.;
	  aosoa_tng_frc ( i, found_at, 2 ) = 0.;

	  // // # repulsive force
	  // d_fn_x[t2] = fn_x;
	  // d_fn_y[t2] = fn_y;
	  // d_fn_z[t2] = fn_z;

	}
	else{
	  // the tangential vector is
	  double tx_tmp = vel_ij[0] - nij_x * vel_ij_dot_nij;
	  double ty_tmp = vel_ij[1] - nij_y * vel_ij_dot_nij;
	  double tz_tmp = vel_ij[2] - nij_z * vel_ij_dot_nij;

	  double ti_magn = pow(tx_tmp*tx_tmp + ty_tmp*ty_tmp + tz_tmp*tz_tmp, 0.5);

	  double ti_x = 0.;
	  double ti_y = 0.;
	  double ti_z = 0.;

	  if (ti_magn > 1e-12){
	    ti_x = tx_tmp / ti_magn;
	    ti_y = ty_tmp / ti_magn;
	    ti_z = tz_tmp / ti_magn;
	  }

	  // // save the normals to output and view in viewer
	  // d_ti_x[t2] = ti_x;
	  // d_ti_y[t2] = ti_y;
	  // d_ti_z[t2] = ti_z;

	  // this is correct
	  double delta_lt_x_star = aosoa_tng_disp ( i, found_at, 0 ) + vel_ij[0] * dt;
	  double delta_lt_y_star = aosoa_tng_disp ( i, found_at, 1 ) + vel_ij[1] * dt;
	  double delta_lt_z_star = aosoa_tng_disp ( i, found_at, 2 ) + vel_ij[2] * dt;

	  double delta_lt_dot_ti = (delta_lt_x_star * ti_x +
				    delta_lt_y_star * ti_y +
				    delta_lt_z_star * ti_z);

	  aosoa_tng_disp ( i, found_at, 0 ) = delta_lt_dot_ti * ti_x;
	  aosoa_tng_disp ( i, found_at, 1 ) = delta_lt_dot_ti * ti_y;
	  aosoa_tng_disp ( i, found_at, 2 ) = delta_lt_dot_ti * ti_z;

	  // double ft_x_star = -kn_kt_fric_coeff[1] * aosoa_tng_disp ( i, found_at, 0 );
	  // double ft_y_star = -kn_kt_fric_coeff[1] * aosoa_tng_disp ( i, found_at, 1 );
	  // double ft_z_star = -kn_kt_fric_coeff[1] * aosoa_tng_disp ( i, found_at, 2 );
	  double ft_x_star = -1e3 * aosoa_tng_disp ( i, found_at, 0 );
	  double ft_y_star = -1e3 * aosoa_tng_disp ( i, found_at, 1 );
	  double ft_z_star = -1e3 * aosoa_tng_disp ( i, found_at, 2 );

	  double ft_magn = pow(ft_x_star*ft_x_star + ft_y_star*ft_y_star + ft_z_star*ft_z_star, 0.5);
	  double fn_magn = pow(fn_x*fn_x + fn_y*fn_y + fn_z*fn_z, 0.5);

	  double ft_magn_star = ft_magn;
	  // if (ft_magn_star > kn_kt_fric_coeff[2] * fn_magn){
	  if (ft_magn_star > 0.3 * fn_magn){
	    ft_magn_star = 0.3 * fn_magn;
	  }
	  // compute the tangential force, by equation 27
	  aosoa_tng_frc ( i, found_at, 0 ) = -ft_magn_star * ti_x;
	  aosoa_tng_frc ( i, found_at, 1 ) = -ft_magn_star * ti_y;
	  aosoa_tng_frc ( i, found_at, 2 ) = -ft_magn_star * ti_z;

	  // reset the spring length
	  // double modified_delta_lt_x = -aosoa_tng_frc ( i, found_at, 0 ) / kn_kt_fric_coeff[1];
	  // double modified_delta_lt_y = -aosoa_tng_frc ( i, found_at, 1 ) / kn_kt_fric_coeff[1];
	  // double modified_delta_lt_z = -aosoa_tng_frc ( i, found_at, 2 ) / kn_kt_fric_coeff[1];
	  double modified_delta_lt_x = -aosoa_tng_frc ( i, found_at, 0 ) / 1e3;
	  double modified_delta_lt_y = -aosoa_tng_frc ( i, found_at, 1 ) / 1e3;
	  double modified_delta_lt_z = -aosoa_tng_frc ( i, found_at, 2 ) / 1e3;

	  double lt_magn = pow(modified_delta_lt_x*modified_delta_lt_x + modified_delta_lt_y*modified_delta_lt_y +
			       modified_delta_lt_z*modified_delta_lt_z, 0.5);

	  aosoa_tng_disp ( i, found_at, 0 ) = modified_delta_lt_x / lt_magn;
	  aosoa_tng_disp ( i, found_at, 1 ) = modified_delta_lt_y / lt_magn;
	  aosoa_tng_disp ( i, found_at, 2 ) = modified_delta_lt_z / lt_magn;

	  // repulsive force
	  aosoa_normal_frc ( i, found_at, 0 ) = fn_x;
	  aosoa_normal_frc ( i, found_at, 1 ) = fn_y;
	  aosoa_normal_frc ( i, found_at, 2 ) = fn_z;
	}
	aosoa_force( i, 0 ) += fn_x + aosoa_tng_frc ( i, found_at, 0 );
	aosoa_force( i, 1 ) += fn_y + aosoa_tng_frc ( i, found_at, 1 );
	aosoa_force( i, 2 ) += fn_z + aosoa_tng_frc ( i, found_at, 2 );
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
  auto position = Cabana::slice<0>     ( aosoa,    "position");
  auto ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto acc = Cabana::slice<3>          ( aosoa,    "acc");
  auto mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto density = Cabana::slice<5>     ( aosoa,    "density");
  auto h = Cabana::slice<6>           ( aosoa,    "h");
  auto p = Cabana::slice<7>           ( aosoa,    "p");
  auto is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
  auto is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
  auto density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
  auto p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
  auto sum_wij = Cabana::slice<12>           ( aosoa,    "sum_wij");


  Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  Cabana::Experimental::HDF5ParticleOutput::
    writeTimeStep(
		  h5_config, "particles", MPI_COMM_WORLD,
		  step, time, num_particles,
		  position,
		  ids ,
		  velocity,
		  acc,
		  mass,
		  density,
		  h,
		  p,
		  is_fluid,
		  is_boundary,
		  density_acc,
		  p_acc,
		  sum_wij
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
  ViewVectorType kn_kt_fric_coeff( "kn_kt_fric_coeff", 3 );
  ViewVectorType gravity( "gravity", 3 );
  // Create host mirrors of device views.
  ViewVectorType::HostMirror h_kn_kt_fric_coeff = Kokkos::create_mirror_view( kn_kt_fric_coeff );
  ViewVectorType::HostMirror h_gravity = Kokkos::create_mirror_view( gravity );

  h_kn_kt_fric_coeff[0] = 1e5;
  h_kn_kt_fric_coeff[1] = 1e3;
  h_kn_kt_fric_coeff[2] = 0.1;

  h_gravity[0] = 0.;
  h_gravity[1] = 0.;
  h_gravity[2] = 0.;

  // Deep copy host views to device views.
  Kokkos::deep_copy( kn_kt_fric_coeff, h_kn_kt_fric_coeff );
  Kokkos::deep_copy( gravity, h_gravity );

  // Numerical parameters of the SPH scheme for fluid

  // integration related variables
  double dt = 1e-4;
  auto final_time = 10.;
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
  auto aosoa_host_position = Cabana::slice<0>( aosoa_host,    "position");
  auto aosoa_host_ids = Cabana::slice<1>( aosoa_host,    "ids");
  auto aosoa_host_velocity = Cabana::slice<2>( aosoa_host,    "velocity");
  auto aosoa_host_acc = Cabana::slice<3>( aosoa_host,    "acc");
  auto aosoa_host_mass = Cabana::slice<4>( aosoa_host,    "mass");
  auto aosoa_host_density = Cabana::slice<5>( aosoa_host,    "density");
  auto aosoa_host_h = Cabana::slice<6>( aosoa_host,    "h");
  auto aosoa_host_p = Cabana::slice<7>( aosoa_host,    "p");
  auto aosoa_host_is_fluid = Cabana::slice<8>( aosoa_host,    "is_fluid");
  auto aosoa_host_is_boundary = Cabana::slice<9>( aosoa_host,    "is_boundary");
  auto aosoa_host_density_acc = Cabana::slice<10>( aosoa_host,    "density_acc");
  auto aosoa_host_p_acc = Cabana::slice<11>( aosoa_host,    "p_acc");
  auto aosoa_host_sum_wij = Cabana::slice<12>( aosoa_host,    "wij");
  auto aosoa_host_force = Cabana::slice<15>     ( aosoa_host,    "force");
  auto aosoa_host_radius = Cabana::slice<16>     ( aosoa_host,    "radius");

  for ( std::size_t i = 0; i < aosoa_host_position.size(); ++i )
    {
      aosoa_host_position ( i, 0 ) = i * 2. * spacing;
      aosoa_host_position ( i, 1 ) = 0.;
      aosoa_host_position ( i, 2 ) = 0.;

      aosoa_host_ids ( i ) = i;

      aosoa_host_velocity ( i, 0 ) = 0.;
      aosoa_host_velocity ( i, 1 ) = 0.;
      aosoa_host_velocity ( i, 2 ) = 0.;

      aosoa_host_acc ( i, 0 ) = 0.;
      aosoa_host_acc ( i, 1 ) = 0.;
      aosoa_host_acc ( i, 2 ) = 0.;

      aosoa_host_mass ( i ) = rho * pow(spacing, DIM);
      aosoa_host_density ( i ) = rho;
      aosoa_host_h ( i ) = 1. * spacing;

      aosoa_host_p ( i ) = 0.;
      aosoa_host_is_fluid ( i ) = 0;
      aosoa_host_is_boundary ( i ) = 0;
      aosoa_host_density_acc ( i ) = 0.;
      aosoa_host_p_acc ( i ) = 0.;
      aosoa_host_sum_wij ( i ) = 0.;

      aosoa_host_force( i, 0 ) = 0.;
      aosoa_host_force( i, 1 ) = 0.;
      aosoa_host_force( i, 2 ) = 0.;

      aosoa_host_radius( i ) = spacing;
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
  // double grid_min[3] = { -2.0, -1.5, -neighborhood_radius };
  // double grid_max[3] = { 3.0, 1.5, 0. * neighborhood_radius };
  double grid_min[3] = { -20.0, -10.5, -neighborhood_radius };
  double grid_max[3] = { 30.0, 10.5, 0. * neighborhood_radius };
  double cell_ratio = 1.0;

  ListType verlet_list( aosoa_position, 0,
			aosoa_position.size(), neighborhood_radius,
			cell_ratio, grid_min, grid_max );

  // Main timestep loop
  for ( int step = 0; step < steps; step++ )
    {
      // move the velocity to t + dt / 2.
      dem_stage_1(aosoa,

		  // kn_kt_fric_coeff,
		  dt, dem_limits);

      // continuity_equation(aosoa, dt,
      // 			  &verlet_list,
      // 			  fluid_limits);

      // move positions to t + dt
      dem_stage_2(aosoa, dt, dem_limits);

      // compute the forces using positions at t+dt and velocity at t+dt/2.
      compute_force(aosoa,
		    kn_kt_fric_coeff,
		    dt,
		    &verlet_list,
		    dem_limits);
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
