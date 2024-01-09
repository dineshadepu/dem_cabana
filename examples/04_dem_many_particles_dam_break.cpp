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
#include <tuple>

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
				      double, // Shear mod (9)
				      double[3], // angular velocity (10)
				      double[3] // torque (11)
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
// auto aosoa_omega = Cabana::slice<10>     ( aosoa,    "omega");
// auto aosoa_torque = Cabana::slice<11>          ( aosoa,    "torque");

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


// void update_tangential_contacts(AoSoAType & aosoa, double dt, int * limits){
//   auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
//   auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
//   auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
//   auto aosoa_force = Cabana::slice<3>          ( aosoa,    "force");
//   auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
//   auto aosoa_density = Cabana::slice<5>        ( aosoa,    "density");
//   auto aosoa_radius = Cabana::slice<6>     ( aosoa,    "radius");

//   auto half_dt = dt * 0.5;
//   auto update_tangential_contacts_lambda_func = KOKKOS_LAMBDA( const int i )
//     {
//       int count = 0;
//       int k = 0;
//       int idx_total_ctcs = aosoa_total_no_tng_contacts( i );
//       int last_idx_tmp = aosoa_total_no_tng_contacts( i ) - ;
//       int sidx = -1;
//       // loop over all the contacts of particle d_idx
//       while (count < idx_total_ctcs){
// 	// The index of the particle with which
// 	// d_idx in contact is
// 	sidx = aosoa_tng_idx( i, k );
// 	if (sidx == -1){
// 	  break;
// 	}
// 	else {
// 	  double pos_i[3] = {aosoa_position( i, 0 ),
// 	    aosoa_position( i, 1 ),
// 	    aosoa_position( i, 2 )};

// 	  double pos_j[3] = {aosoa_position( j, 0 ),
// 	    aosoa_position( j, 1 ),
// 	    aosoa_position( j, 2 )};

// 	  double pos_ij[3] = {aosoa_position( i, 0 ) - aosoa_position( j, 0 ),
// 	    aosoa_position( i, 1 ) - aosoa_position( j, 1 ),
// 	    aosoa_position( i, 2 ) - aosoa_position( j, 2 )};

// 	  // squared distance
// 	  double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
// 	  // distance between i and j
// 	  double rij = sqrt(r2ij);
// 	  // Find the overlap amount
// 	  double overlap =  aosoa_radius( i ) + aosoa_radius( j ) - rij;

// 	  if (overlap <= 0.) {
// 	    // if the swap index is the current index then
// 	    // simply make it to null contact.
// 	    if (k == last_idx_tmp){
// 	      aosoa_tng_idx( i, k ) = -1;
// 	      aosoa_tng_ss_x( i, k ) = 0.;
// 	      aosoa_tng_ss_y( i, k ) = 0.;
// 	      aosoa_tng_ss_z( i, k ) = 0.;
// 	    }
// 	    else {
// 	      // swap the current tracking index with the final
// 	      // contact index
// 	      aosoa_tng_idx( i, k ) = aosoa_tng_idx( i, last_idx_tmp );
// 	      aosoa_tng_idx( i, last_idx_tmp ) = -1;

// 	      // swap tangential x displacement
// 	      aosoa_tng_ss_x( i, k ) = aosoa_tng_ss_x( i, last_idx_tmp );
// 	      aosoa_tng_ss_x( i, last_idx_tmp ) = 0.;

// 	      // swap tangential y displacement
// 	      aosoa_tng_ss_y( i, k ) = aosoa_tng_ss_y( i, last_idx_tmp );
// 	      aosoa_tng_ss_y( i, last_idx_tmp ) = 0.;

// 	      // swap tangential z displacement
// 	      aosoa_tng_ss_z( i, k ) = aosoa_tng_ss_z( i, last_idx_tmp );
// 	      aosoa_tng_ss_z( i, last_idx_tmp ) = 0.;

// 	      // decrease the last_idx_tmp, since we swapped it to
// 	      // -1
// 	      last_idx_tmp -= 1;
// 	    }

// 	    // decrement the total contacts of the particle
// 	    aosoa_total_no_tng_contacts( i ) -= 1;
// 	  }
// 	  else
// 	    {
// 	      k = k + 1;
// 	    }
// 	}
// 	else{
// 	  k = k + 1;
// 	}
// 	count += 1;
//       }
//     };
//   Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
//   Kokkos::parallel_for( "CabanaSPH:Integrator:UpdateTngCnts", policy,
// 			update_tangential_contacts_lambda_func );
// }


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
  auto aosoa_omega = Cabana::slice<10>     ( aosoa,    "omega");
  auto aosoa_torque = Cabana::slice<11>          ( aosoa,    "torque");

  Cabana::deep_copy( aosoa_force, 0. );
  Cabana::deep_copy( aosoa_torque, 0. );

  // auto force_torque_dem_particles_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
  //   {
  //     /*
  // 	Common to all equations in SPH.

  // 	We compute:
  // 	1.the vector passing from j to i
  // 	2. Distance between the points i and j
  // 	3. Distance square between the points i and j
  // 	4. Velocity vector difference between i and j
  // 	5. Kernel value
  // 	6. Derivative of kernel value
  //     */
  //     double pos_i[3] = {aosoa_position( i, 0 ),
  // 	aosoa_position( i, 1 ),
  // 	aosoa_position( i, 2 )};

  //     double pos_j[3] = {aosoa_position( j, 0 ),
  // 	aosoa_position( j, 1 ),
  // 	aosoa_position( j, 2 )};

  //     double pos_ij[3] = {aosoa_position( i, 0 ) - aosoa_position( j, 0 ),
  // 	aosoa_position( i, 1 ) - aosoa_position( j, 1 ),
  // 	aosoa_position( i, 2 ) - aosoa_position( j, 2 )};

  //     // squared distance
  //     double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
  //     // distance between i and j
  //     double rij = sqrt(r2ij);

  //     // const double mass_i = aosoa_mass( i );
  //     const double mass_j = aosoa_mass( j );

  //     // Find the overlap amount
  //     double overlap =  aosoa_radius( i ) + aosoa_radius( j ) - rij;

  //     double a_i = aosoa_radius( i ) - overlap / 2.;
  //     double a_j = aosoa_radius( j ) - overlap / 2.;

  //     // normal vector passing from j to i
  //     double nij_x = pos_ij[0] / rij;
  //     double nij_y = pos_ij[1] / rij;
  //     double nij_z = pos_ij[2] / rij;

  //     double vel_i[3] = {0., 0., 0.};

  //     vel_i[0] = aosoa_velocity( i, 0 ) +
  //     (aosoa_omega( i, 1 ) * nij_z - aosoa_omega( i, 2 ) * nij_y) * a_i;

  //     vel_i[1] = aosoa_velocity( i, 1 ) +
  //     (aosoa_omega( i, 2 ) * nij_x - aosoa_omega( i, 0 ) * nij_z) * a_i;

  //     vel_i[2] = aosoa_velocity( i, 2 ) +
  //     (aosoa_omega( i, 0 ) * nij_y - aosoa_omega( i, 1 ) * nij_x) * a_i;

  //     double vel_j[3] = {0., 0., 0.};

  //     vel_j[0] = aosoa_velocity( j, 0 ) +
  //     (-aosoa_omega( j, 1 ) * nij_z + aosoa_omega( j, 2 ) * nij_y) * a_j;

  //     vel_j[1] = aosoa_velocity( i, 1 ) +
  //     (-aosoa_omega( j, 2 ) * nij_x + aosoa_omega( j, 0 ) * nij_z) * a_j;

  //     vel_j[2] = aosoa_velocity( i, 2 ) +
  //     (-aosoa_omega( j, 0 ) * nij_y + aosoa_omega( j, 1 ) * nij_x) * a_j;

  //     // Now the relative velocity of particle i w.r.t j at the contact
  //     // point is
  //     double vel_ij[3] = {vel_i[0] - vel_j[0],
  // 	vel_i[1] - vel_j[1],
  // 	vel_i[2] - vel_j[2]};

  //     // normal velocity magnitude
  //     double vij_dot_nij = vel_ij[0] * nij_x + vel_ij[1] * nij_y + vel_ij[2] * nij_z;
  //     double vn_x = vij_dot_nij * nij_x;
  //     double vn_y = vij_dot_nij * nij_y;
  //     double vn_z = vij_dot_nij * nij_z;

  //     // tangential velocity
  //     double vt_x = vel_ij[0] - vn_x;
  //     double vt_y = vel_ij[1] - vn_y;
  //     double vt_z = vel_ij[2] - vn_z;

  //     /*
  // 	====================================
  // 	End: common to all equations in SPH.
  // 	====================================
  //     */
  //     // find the force if the particles are overlapping
  //     if (overlap > 0.) {

  // 	// Compute stiffness
  // 	// effective Young's modulus
  // 	double tmp_1 = (1. - (aosoa_nu( i )*aosoa_nu( i ))) / aosoa_E( i );
  // 	double tmp_2 = (1. - (aosoa_nu( j )*aosoa_nu( j ))) / aosoa_E( j );
  // 	double E_eff = 1. / (tmp_1 + tmp_2);
  // 	double tmp_3 = 1. / aosoa_radius( i );
  // 	double tmp_4 = 1. / aosoa_radius( j );
  // 	double R_eff = 1. / (tmp_3 + tmp_4);
  // 	// # Eq 4 [1]
  // 	double kn = 4. / 3. * E_eff * sqrt(R_eff);

  // 	// normal force
  // 	double fn =  kn * pow(overlap, 1.5);
  // 	double fn_x = fn * nij_x;
  // 	double fn_y = fn * nij_y;
  // 	double fn_z = fn * nij_z;

  // 	// /****************************
  // 	//  // tangential force computation
  // 	//  *****************************/
  // 	// // if the particle is not been tracked then assign an index in
  // 	// // tracking history.
  // 	// int tot_ctcs = aosoa_total_no_tng_contacts( i );

  // 	// // # check if the particle is in the tracking list
  // 	// // # if so, then save the location at found_at
  // 	// int found = 0;
  // 	// int found_at = -1;
  // 	// for(int k=0; k<tot_ctcs; k++)
  // 	//   {
  // 	//     if (j == aosoa_tng_idx( i, k )) {
  // 	//       found_at = k;
  // 	//       found = 1;
  // 	//       break;
  // 	//     }
  // 	//   }

  // 	double ft_x = 0.;
  // 	double ft_y = 0.;
  // 	double ft_z = 0.;

  // 	// if (found == 0) {
  // 	//   found_at = tot_ctcs;
  // 	//   aosoa_tng_idx( i, found_at ) = j;
  // 	//   aosoa_total_no_tng_contacts( i ) += 1;

  // 	//   aosoa_tng_ss_x( i, found_at ) = 0.;
  // 	//   aosoa_tng_ss_y( i, found_at ) = 0.;
  // 	//   aosoa_tng_ss_z( i, found_at ) = 0.;
  // 	// }

  // 	// // We are tracking the particle history at found_at
  // 	// // tangential velocity
  // 	// double vij_magn = sqrt(vel_ij[0] * vel_ij[0] + vel_ij[1] * vel_ij[1] +
  // 	// 		       vel_ij[2] * vel_ij[2]);

  // 	// if (vij_magn < 1e-12) {
  // 	//   aosoa_tng_ss_x( i, found_at ) = 0.;
  // 	//   aosoa_tng_ss_y( i, found_at ) = 0.;
  // 	//   aosoa_tng_ss_z( i, found_at ) = 0.;
  // 	// }
  // 	// else {
  // 	//   // magnitude of the tangential velocity
  // 	//   double ti_magn = sqrt(vt_x * vt_x + vt_y * vt_y + vt_z * vt_z);

  // 	//   double ti_x = 0.;
  // 	//   double ti_y = 0.;
  // 	//   double ti_z = 0.;

  // 	//   if (ti_magn < 1e-12) {
  // 	//     ti_x = vt_x / ti_magn;
  // 	//     ti_y = vt_y / ti_magn;
  // 	//     ti_z = vt_z / ti_magn;
  // 	//   }

  // 	//   double delta_lt_x_star = d_tng_ss_x[found_at] + vij_x * dt;
  // 	//   double delta_lt_y_star = d_tng_ss_y[found_at] + vij_y * dt;
  // 	//   double delta_lt_z_star = d_tng_ss_z[found_at] + vij_z * dt;

  // 	//   double delta_lt_dot_ti = (delta_lt_x_star * ti_x +
  // 	// 			    delta_lt_y_star * ti_y +
  // 	// 			    delta_lt_z_star * ti_z);
  // 	//   aosoa_tng_ss_x( i, found_at ) = delta_lt_dot_ti * ti_x;
  // 	//   aosoa_tng_ss_y( i, found_at ) = delta_lt_dot_ti * ti_y;
  // 	//   aosoa_tng_ss_z( i, found_at ) = delta_lt_dot_ti * ti_z;

  // 	//   // Compute the tangential stiffness
  // 	//   double tmp_1 = (2. - aosoa_nu( i )) / aosoa_G( i );
  // 	//   double tmp_2 = (2. - aosoa_nu( j )) / aosoa_G( j );
  // 	//   double G_eff = 1. / (tmp_1 + tmp_2);
  // 	//   // Eq 12 [1]
  // 	//   double kt = 8. * G_eff * sqrt(R_eff * overlap);
  // 	//   double S_t = kt;
  // 	//   double eta_t = -2. * sqrt(5./6) * beta * sqrt(S_t * m_eff);

  // 	//   double ft_x_star = -kt * aosoa_tng_ss_x( i, found_at ) - eta_t * vt_x;
  // 	//   double ft_y_star = -kt * aosoa_tng_ss_y( i, found_at ) - eta_t * vt_y;
  // 	//   double ft_z_star = -kt * aosoa_tng_ss_z( i, found_at ) - eta_t * vt_z;

  // 	//   double ft_magn = sqrt(ft_x_star*ft_x_star + ft_y_star*ft_y_star + ft_z_star*ft_z_star);

  // 	//   ti_x = 0.;
  // 	//   ti_y = 0.;
  // 	//   ti_z = 0.;

  // 	//   if (ft_magn > 1e-12) {
  // 	//     ti_x = ft_x_star / ft_magn;
  // 	//     ti_y = ft_y_star / ft_magn;
  // 	//     ti_z = ft_z_star / ft_magn;
  // 	//   }

  // 	//   double fn_magn = sqrt(fn_x*fn_x + fn_y*fn_y + fn_z*fn_z);

  // 	//   double ft_magn_star = min(fric_coeff * fn_magn, ft_magn);

  // 	//   // compute the tangential force, by equation 17 (Lethe)
  // 	//   ft_x = ft_magn_star * ti_x;
  // 	//   ft_y = ft_magn_star * ti_y;
  // 	//   ft_z = ft_magn_star * ti_z;

  // 	//   // Add damping to the limited force
  // 	//   ft_x += eta_t * vt_x;
  // 	//   ft_y += eta_t * vt_y;
  // 	//   ft_z += eta_t * vt_z;

  // 	//   // reset the spring length
  // 	//   aosoa_tng_ss_x( i, found_at ) = -ft_x / kt;
  // 	//   aosoa_tng_ss_y( i, found_at ) = -ft_y / kt;
  // 	//   aosoa_tng_ss_z( i, found_at ) = -ft_z / kt;
  // 	// }

  // 	// // Add force to the particle i due to contact with particle j
  // 	// aosoa_force( i, 0 ) += fn_x;
  // 	// aosoa_force( i, 1 ) += fn_y;
  // 	// aosoa_force( i, 2 ) += fn_z;

  // 	// // Add torque to the particle i due to contact with particle j
  // 	// aosoa_torque( i, 0 ) += (nij_y * ft_z - nij_z * ft_y) * a_i;
  // 	// aosoa_torque( i, 1 ) += (nij_z * ft_x - nij_x * ft_z) * a_i;
  // 	// aosoa_torque( i, 2 ) += (nij_x * ft_y - nij_y * ft_x) * a_i;
  //     }

  //   };

  // Kokkos::RangePolicy<ExecutionSpace> policy(limits[0], limits[1]);


  // Cabana::neighbor_parallel_for( policy,
  // 				 force_torque_dem_particles_lambda_func,
  // 				 *verlet_list,
  // 				 Cabana::FirstNeighborsTag(),
  // 				 Cabana::SerialOpTag(),
  // 				 "CabanaDEM:Equations:ForceTorqueComputation" );
  // Kokkos::fence();
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
    Step 1: Set the time step, time and pfreq
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
    End: Step 1
    ================================================
  */

  /*
    ================================================
    Step 2
    ================================================
    1. Create the particles (Both fluid and boundary)
  */

  // 1a. Create the particles for fluid
  double length_dem = 1.;
  double height_dem = 1.;
  double spacing_dem = 0.1;
  std::vector<double> x_host_sand;
  std::vector<double> y_host_sand;
  std::vector<double> z_host_sand;
  tie (x_host_sand, y_host_sand, z_host_sand) = create_2d_block(length_dem,
								height_dem,
								spacing_dem);
  for(double& d : y_host_sand)
    d += 3. * spacing_dem;

  int no_dem_particles = x_host_sand.size();

  // 1b. Create the particles for boundary
  std::vector<double> x_host_boundary;
  std::vector<double> y_host_boundary;
  std::vector<double> z_host_boundary;

  double length_boundary = 5.;
  double height_boundary = 0.1;
  double spacing_boundary = 0.1;
  std::vector<double> x_host_boundary_1;
  std::vector<double> y_host_boundary_1;
  std::vector<double> z_host_boundary_1;
  tie (x_host_boundary, y_host_boundary, z_host_boundary) =
    create_2d_block(length_boundary, height_boundary, spacing_boundary);

  // double length_boundary_2 = 0.1;
  // double height_boundary_2 = 5.;
  // double spacing_boundary_2 = 0.1;
  // std::vector<double> x_host_boundary_2;
  // std::vector<double> y_host_boundary_2;
  // std::vector<double> z_host_boundary_2;
  // tie (x_host_boundary_2, y_host_boundary_2, z_host_boundary_2) =
  //   create_2d_block(length_boundary_2, height_boundary_2, spacing_boundary_2);

  // x_host_boundary.insert( x_host_boundary.end(), x_host_boundary_2.begin(), x_host_boundary_2.end() );
  // y_host_boundary.insert( y_host_boundary.end(), y_host_boundary_2.begin(), y_host_boundary_2.end() );
  // z_host_boundary.insert( z_host_boundary.end(), z_host_boundary_2.begin(), z_host_boundary_2.end() );

  int no_bdry_particles = x_host_boundary.size();

  int total_no_particles = no_dem_particles + no_bdry_particles;
  int num_particles = total_no_particles;
  int dem_limits[2] = {0, no_dem_particles};
  /*
    ================================================
    End: Step 1
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
  auto aosoa_host_omega = Cabana::slice<10>     ( aosoa_host,    "omega");
  auto aosoa_host_torque = Cabana::slice<11>          ( aosoa_host,    "torque");

  // First assign DEM particles data to the aosoa host
  for ( std::size_t i = 0; i < no_dem_particles; ++i )
    {
      aosoa_host_position ( i, 0 ) = x_host_sand [ i ];
      aosoa_host_position ( i, 1 ) = y_host_sand [ i ];
      aosoa_host_position ( i, 2 ) = z_host_sand [ i ];

      aosoa_host_ids ( i ) = i;

      aosoa_host_velocity ( i, 0 ) = 5.;
      aosoa_host_velocity ( i, 1 ) = 0.;
      aosoa_host_velocity ( i, 2 ) = 0.;

      aosoa_host_force ( i, 0 ) = 0.;
      aosoa_host_force ( i, 1 ) = 0.;
      aosoa_host_force ( i, 2 ) = 0.;

      aosoa_host_mass ( i ) = rho * pow(spacing_dem, DIM);
      aosoa_host_density ( i ) = rho;

      aosoa_host_radius( i ) = spacing_dem / 2.;

      aosoa_host_E( i ) = E;
      aosoa_host_nu( i ) = nu;
      aosoa_host_G( i ) = G;

      aosoa_host_omega ( i, 0 ) = 0.;
      aosoa_host_omega ( i, 1 ) = 0.;
      aosoa_host_omega ( i, 2 ) = 0.;

      aosoa_host_torque ( i, 0 ) = 0.;
      aosoa_host_torque ( i, 1 ) = 0.;
      aosoa_host_torque ( i, 2 ) = 0.;
    }

  // Second assign Boundary particles data to the aosoa host
  for ( std::size_t i = no_dem_particles; i < total_no_particles; ++i )
    {
      aosoa_host_position ( i, 0 ) = x_host_boundary [ i - no_dem_particles ];
      aosoa_host_position ( i, 1 ) = y_host_boundary [ i - no_dem_particles ];
      aosoa_host_position ( i, 2 ) = z_host_boundary [ i - no_dem_particles ];

      aosoa_host_ids ( i ) = i;

      aosoa_host_velocity ( i, 0 ) = 0.;
      aosoa_host_velocity ( i, 1 ) = 0.;
      aosoa_host_velocity ( i, 2 ) = 0.;

      aosoa_host_force ( i, 0 ) = 0.;
      aosoa_host_force ( i, 1 ) = 0.;
      aosoa_host_force ( i, 2 ) = 0.;

      aosoa_host_mass ( i ) = rho * pow(spacing_boundary, DIM);
      aosoa_host_density ( i ) = rho;

      aosoa_host_radius( i ) = spacing_boundary / 2.;

      aosoa_host_E( i ) = E;
      aosoa_host_nu( i ) = nu;
      aosoa_host_G( i ) = G;
    }
  // aosoa_host_velocity ( 0, 0 ) = 1.;
  // aosoa_host_velocity ( 1, 0 ) = -1.;
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
  double neighborhood_radius = 5. * spacing_dem;
  double grid_min[3] = { -10.0, -10., -neighborhood_radius };
  double grid_max[3] = { -10., 10., neighborhood_radius };
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

      // compute the forces using positions at t+dt and velocity at t+dt/2.
      compute_force(aosoa, kn, dt,
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
