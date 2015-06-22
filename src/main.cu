/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "cc.h"
#include "mesh.h"
#include "particles.h"
#include "diagnostic.h"

/************************ FUNCTION PROTOTIPES *************************/




/*************************** MAIN FUNCTION ****************************/

int main (int argc, const char* argv[])
{
  /*--------------------------- function variables -----------------------*/
  
  // host variables definition
  double t;                             // time of simulation
  const double dt = init_dt();          // time step
  const int n_ini = init_n_ini();       // number of first iteration
  const int n_prev = init_n_prev();     // number of iterations before start analizing
  const int n_save = init_n_save();     // number of iterations between diagnostics
  const int n_fin = init_n_fin();       // number of last iteration
  int num_i;                            // number of particles (electrons and ions)
  int nn = init_nn();                   // number of nodes
  double U_i;                           // system energy for electrons and ions
  double mi = init_mi();                // ion mass
  double dtin_i = init_dtin_i();        // time between ion insertions
  double q_pi = 0;                      // probe's positive acumulated charge (ions)
  double vd_i = init_vd_i();            // ion's drift velocity
  char filename[50];                    // filename for saved data

  ifstream ifile;
  ofstream ofile;

  // device variables definition
  double *d_rho, *d_phi, *d_E;              // mesh properties
  double *d_avg_rho, *d_avg_phi, *d_avg_E;  // mesh averaged properties
  double *d_avg_ddf_i, *d_avg_vdf_i;        // density and velocity distribution function for ions
  double v_max_i = init_v_max_i();          // maximun velocity of ions (for histograms)
  double v_min_i = init_v_min_i();          // minimun velocity of ions (for histograms)
  int count_df_i = 0;                       // |
  int count_rho = 0;                        // |-> counters for avg data
  int count_phi = 0;                        // |
  int count_E = 0;                          // |
  particle *d_i;                            // particles vectors
  curandStatePhilox4_32_10_t *state;        // philox state for __device__ random number generation 

  /*----------------------------- function body -------------------------*/

  //---- INITIALITATION OF SIMULATION

  // initialize device and simulation variables
  init_dev();
  init_sim(&d_rho, &d_phi, &d_E, &d_avg_rho, &d_avg_phi, &d_avg_E, &d_i, &num_i, &d_avg_ddf_i, &d_avg_vdf_i, &t, &state);

  // save initial state
  sprintf(filename, "../output/particles/ions_t_%d", n_ini);
  particles_snapshot(d_i, num_i, filename);
  sprintf(filename, "../output/charge/avg_charge_t_%d", n_ini);
  save_mesh(d_avg_rho, filename);
  sprintf(filename, "../output/potential/avg_potential_t_%d", n_ini);
  save_mesh(d_avg_phi, filename);
  sprintf(filename, "../output/field/avg_field_t_%d", n_ini);
  save_mesh(d_avg_E, filename);
  t += dt;

  //---- SIMULATION BODY
  
  for (int i = n_ini+1; i <= n_fin; i++, t += dt) {
    // simulate one time step
    charge_deposition(d_rho, d_phi, d_i, num_i);
    poisson_solver(1.0e-4, d_rho, d_phi);
    field_solver(d_phi, d_E);
    particle_mover(d_i, num_i, d_E);
    cc(t, &num_i, &d_i, &dtin_i, &vd_i, &q_pi, d_phi, d_E, state);

    // average mesh variables and distribution functions
    avg_mesh(d_rho, d_avg_rho, &count_rho);
    avg_mesh(d_phi, d_avg_phi, &count_phi);
    avg_mesh(d_E, d_avg_E, &count_E);
    eval_df(d_avg_ddf_i, d_avg_vdf_i, v_max_i, v_min_i, d_i, num_i, &count_df_i);

    // store data
    if (i>=n_prev && i%n_save==0) {
      // save particles (snapshot)
      sprintf(filename, "../output/particles/ions_t_%d", i);
      particles_snapshot(d_i, num_i, filename);

      // save mesh properties
      sprintf(filename, "../output/charge/avg_charge_t_%d", i);
      save_mesh(d_avg_rho, filename);
      sprintf(filename, "../output/potential/avg_potential_t_%d", i);
      save_mesh(d_avg_phi, filename);
      sprintf(filename, "../output/field/avg_field_t_%d", i);
      save_mesh(d_avg_E, filename);

      // save distribution functions
      sprintf(filename, "../output/particles/ions_ddf_t_%d", i);
      save_ddf(d_avg_ddf_i, filename);
      sprintf(filename, "../output/particles/ions_vdf_t_%d", i);
      save_vdf(d_avg_vdf_i, v_max_i, v_min_i, filename);

      // save log
      U_i = eval_particle_energy(d_phi,  d_i, mi, 1.0, num_i);
      save_log(t, num_i, U_i, &q_pi, vd_i, d_phi);

      cout << "iteration = " << i << endl;
    }
  }

  //---- END OF SIMULATION

  cout << "Simulation finished!" << endl;
  return 0;
}
