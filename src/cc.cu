/****************************************************************************
 *                                                                          *
 *    CUPIC1D is a code that simulates the interaction between plasma and   *
 *    a langmuir probe in 1D using PIC techniques accelerated with the use  * 
 *    of GPU hardware (CUDA, extension of C/C++)                            *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "cc.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void cc (double t, int *num_e, particle **d_e, double *dtin_e, int *num_i, particle **d_i, double *dtin_i, 
         double *vd_i, double *q_pe, double *q_pi, double *d_phi, double *d_E, curandStatePhilox4_32_10_t *state) 
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double me = init_me();                       //
  static const double mi = init_mi();                       //
  static const double kte = init_kte();                     // particle
  static const double kti = init_kti();                     // properties
  static const double vd_e = init_vd_e();                   //
  
  static const double r_p = init_r_p();                     // probe radius
  static const double theta = init_theta_p();               // angular amplitude of the simulation
  static const bool fp_is_on = floating_potential_is_on();  // probe is floating or not
  static const bool flux_cal_on = flux_calibration_is_on(); // ion flux calibration is activated or not
  static const int nc = init_nc();                          // number of cells
  static const double ds = init_ds();                       // spatial step
  static const double epsilon0 = init_epsilon0();           // epsilon0 in simulation units
  
  static double tin_e = t+(*dtin_e);                        // time for next electron insertion
  static double tin_i = t+(*dtin_i);                        // time for next ion insertion
  
  static double q_p = 0.0;                                  // net charge acumulated by the probe (not reseted)
  double phi_s = -0.5*mi*(*vd_i)*(*vd_i);                   // potential at sheath edge
  double dummy_phi_p;                                       // dummy probe potential

  cudaError cuError;                                        // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  //---- electrons contour conditions
  
  abs_emi_cc(t, &tin_e, *dtin_e, kte, me, vd_e, -1.0, q_pe, num_e, d_e, d_E, state);

  //---- ions contour conditions

  abs_emi_cc(t, &tin_i, *dtin_i, kti, mi, *vd_i, +1.0, q_pi, num_i, d_i, d_E, state);

  //---- actualize probe potential because of the change in charge collected by the probe
  if (fp_is_on) {
    q_p += *q_pe;
    q_p += *q_pi;
    dummy_phi_p = q_p/(2.0*theta*epsilon0*r_p);
    if (dummy_phi_p > phi_s) dummy_phi_p = phi_s;
    cuError = cudaMemcpy (&d_phi[0], &dummy_phi_p, sizeof(double), cudaMemcpyHostToDevice);
    cu_check(cuError, __FILE__, __LINE__);
  }
  
  //---- actulize ion drift velocity if calibration is on
  if (flux_cal_on) {
    calibrate_ion_flux(vd_i, dtin_i, dtin_e, d_E, d_phi);
  }

  return;
}

/**********************************************************/

void abs_emi_cc(double t, double *tin, double dtin, double kt, double m, double vd, double q, double *q_p, 
                int *h_num_p, particle **d_p, double *d_E, curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double L = init_L();       //
  static const double r_p = init_r_p();   // geometric properties
  static const double ds = init_ds();     // of simulation
  static const int nn = init_nn();        // 
  
  static const double dt = init_dt();     //
  double fpt = t+dt;                      // timing variables
  double fvt = t+0.5*dt;                  //
  
  int in = 0;                             // number of particles added at plasma frontier
  int h_num_abs_p;                        // host number of particles absorved at the probe
  
  double dv;                              //
  int i;                                  // variables for 
  double xmax, ymax, y1, y2;              // rejection method
  double vth = sqrt(kt/m);                //
  
  cudaError cuError;                      // cuda error variable
  dim3 griddim, blockdim;                 // kernel execution configurations 

  // device memory
  int *d_num_p;                           // device number of particles
  int *d_num_abs_p;                       // device number of particles absorved at the probe
  particle *d_dummy_p;                    // device dummy vector for particle storage
  
  /*----------------------------- function body -------------------------*/
  
  // calculate number of particles that flow into the simulation
  if((*tin) < fpt) in = 1 + int((fpt-(*tin))/dtin);
  
  // copy number of particles from host to device 
  cuError = cudaMalloc((void **) &d_num_p, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (d_num_p, h_num_p, sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // initialize number of particles absorbed at the probe 
  cuError = cudaMalloc((void **) &d_num_abs_p, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemset((void *) d_num_abs_p, 0, sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  
  // execution configuration for particle remover kernel
  griddim = 1;
  blockdim = P_RMV_BLK_SZ;

  // execute particle remover kernel
  cudaGetLastError();
  pRemover<<<griddim, blockdim>>>(*d_p, d_num_p, L, d_num_abs_p);
  cu_sync_check(__FILE__, __LINE__);

  // copy number of particles absorbed at the probe from device to host (and free device memory)
  cuError = cudaMemcpy (&h_num_abs_p, d_num_abs_p, sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_num_abs_p);
  cu_check(cuError, __FILE__, __LINE__);

  // actualize probe acumulated charge
  *q_p += q*h_num_abs_p;

  // copy new number of particles from device to host (and free device memory)
  cuError = cudaMemcpy (h_num_p, d_num_p, sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_num_p);
  cu_check(cuError, __FILE__, __LINE__);

  // resize of particle vector with new number of particles
  cuError = cudaMalloc((void **) &d_dummy_p, ((*h_num_p)+in)*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(d_dummy_p, *d_p, (*h_num_p)*sizeof(particle), cudaMemcpyDeviceToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(*d_p);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc((void **) d_p, ((*h_num_p)+in)*sizeof(particle));   
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy(*d_p, d_dummy_p, (*h_num_p)*sizeof(particle), cudaMemcpyDeviceToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_dummy_p);
  cu_check(cuError, __FILE__, __LINE__);
  
  // add particles
  if (in != 0) {
    // prepare rejection algorithm for particle velocity generation in case it's needed
    if (vd != 0.0) {
      dv = (vth>fabs(vd)) ? vth/100.0 : fabs(vd)/100.0; 
      i = 0;
      y1 = host_vdf(double(i)*dv, vth, fabs(vd));
      do {
        y2 = host_vdf(double(i+1)*dv, vth, fabs(vd));
        ymax = (y1>y2) ? y1 : y2;
        i++;
        y1 = y2;
      } while (ymax==y2);
      do {
        y2 = host_vdf(double(i+1)*dv, vth, fabs(vd));
        i++;
      } while (y2>0.001*ymax);
      ymax *= 1.05;
      xmax = double(i)*dv;
    }

    // execution configuration for pEmi kernel
    griddim = 1;
    blockdim = CURAND_BLOCK_DIM;

    // launch kernel to add particles
    cudaGetLastError();
    pEmi<<<griddim, blockdim>>>(*d_p, *h_num_p, in, d_E, vth, vd, q/m, nn, L, r_p, fpt, fvt, *tin, dtin, xmax, ymax, state);
    cu_sync_check(__FILE__, __LINE__);

    // actualize time for next particle insertion
    (*tin) += double(in)*dtin;

    // actualize number of particles
    *h_num_p += in;
  }

  return;
}

/**********************************************************/

void calibrate_ion_flux(double *vd_i, double *dtin_i, double *dtin_e, double *d_E, double *d_phi)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double n = init_n();
  static const double l_p = init_l_p();
  static const double r_p = init_r_p();
  static const double theta = init_theta_p();
  static const double L = init_L();
  static const double mi = init_mi();
  static const double kti = init_kti();
  static const double me = init_me();
  static const double kte = init_kte();
  static const double vd_e = init_vd_e();
  static const int nn = init_nn();

  double phi_s;
  double E_mean;
  double *h_E;
  static const double increment = init_increment();
  static const int window_size = init_avg_nodes();
  static const double tol = init_field_tol();
 
  cudaError cuError;                            // cuda error variable

  // device memory
  
  /*----------------------------- function body -------------------------*/
 
 //---- Actualize ion drift velocity acording to the value of electric field at plasma frontier

 // allocate host memory for field
 h_E = (double*) malloc(window_size*sizeof(double));
  
 // copy field from device to host memory
 cuError = cudaMemcpy (h_E, &d_E[nn-1-window_size], window_size*sizeof(double), cudaMemcpyDeviceToHost);
 cu_check(cuError, __FILE__, __LINE__);

 // check mean value of electric field at plasma frontier
 E_mean = 0.0;
 for (int i=0; i<window_size; i++) {
   E_mean += h_E[i];
 }
 E_mean /= double(window_size);
 
 // free host memory for field
 free(h_E);

 // actualize ion drift velocity
 if (E_mean<tol && *vd_i>-1.0/sqrt(mi)) {
   *vd_i -= increment;
 } else if (E_mean>tol && *vd_i<0.0) {
   *vd_i += increment;
 }

 // actualize sheath edge potential
 phi_s = -0.5*mi*(*vd_i)*(*vd_i);
 cuError = cudaMemcpy (&d_phi[nn-1], &phi_s, sizeof(double), cudaMemcpyHostToDevice);
 cu_check(cuError, __FILE__, __LINE__);

 //---- Actualize time between ion/electron insertions

 *dtin_e = n*sqrt(kte/(2.0*PI*me))*exp(-0.5*me*vd_e*vd_e/kte);        // thermal component of input flux
 *dtin_e += 0.5*n*(-vd_e)*(1.0+erf(sqrt(0.5*me/kte)*(-vd_e)));        // drift component of input flux
 *dtin_e *= exp(phi_s);                                               // correction on density at sheath edge
 *dtin_e *= (r_p+L)*theta*l_p;      // number of particles that enter the simulation per unit of time
 *dtin_e = 1.0/(*dtin_e);           // time between consecutive particles injection

 *dtin_i = n*sqrt(kti/(2.0*PI*mi))*exp(-0.5*mi*(*vd_i)*(*vd_i)/kti);  // thermal component of input flux
 *dtin_i += 0.5*n*(-(*vd_i))*(1.0+erf(sqrt(0.5*mi/kti)*(-(*vd_i))));  // drift component of input flux
 *dtin_i *= exp(phi_s);                                               // correction on density at sheath edge
 *dtin_i *= (r_p+L)*theta*l_p;      // number of particles that enter the simulation per unit of time
 *dtin_i = 1.0/(*dtin_i);           // time between consecutive particles injection

 return;
}

/**********************************************************/

inline double host_vdf(double v, double vth, double vd)
{
  /*--------------------------- function variables -----------------------*/
  
  // host variables definition
  
  // device variables definition
  
  /*----------------------------- function body -------------------------*/
  
  return v*exp(-(v-vd)*(v-vd)/(2.0*vth*vth));
}


/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void pEmi(particle *g_p, int num_p, int n_in, double *g_E, double vth, double vd, double qm, int nn, 
                     double L, double r_p, double fpt, double fvt, double tin, double dtin, double xmax, double ymax, 
                     curandStatePhilox4_32_10_t *state)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ double sh_E;

  // kernel registers
  particle reg_p;
  int tid = (int) threadIdx.x + (int) blockIdx.x * (int) blockDim.x;
  int tpb = (int) blockDim.x;
  curandStatePhilox4_32_10_t local_state;
  double2 rnd;
  double dummy_r;
 
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  if (tid == 0) sh_E = g_E[nn-1];
  __syncthreads();

  //---- initialize registers
  local_state = state[tid];
  __syncthreads();

  //---- generate particles
  for (int i = tid; i < n_in; i+=tpb) {
    // generate register particles position
    reg_p.r = L;
    // generate register particles radial velocity
    if (vd == 0.0) {
      rnd = curand_normal2_double(&local_state);
      reg_p.vr = -sqrt(rnd.x*rnd.x+rnd.y*rnd.y)*vth;
    } else {
      do {
        rnd = curand_uniform2_double(&local_state);
        rnd.x *= xmax;
        rnd.y *= ymax;
      } while (rnd.y > device_vdf(rnd.x, vth, fabs(vd)));
      reg_p.vr = copysignf(rnd.x, vd);
    }
    // generate register particles tangential velocity
    rnd = curand_normal2_double(&local_state);
    reg_p.vt = rnd.x*vth;
    
    // simple push
    dummy_r = reg_p.r + (fpt-(tin+double(i)*dtin))*reg_p.vr;
    reg_p.vt *= reg_p.r/dummy_r;
    reg_p.r = dummy_r;
    reg_p.vr += (fvt-(tin+double(i)*dtin))*(sh_E*qm+reg_p.vt*reg_p.vt/(L+r_p));

    // store new particles in global memory
    g_p[num_p+i] = reg_p;
  }
  __syncthreads();

  //---- store local state in global memory
  state[tid] = local_state;

  return;
}

/**********************************************************/

__global__ void pRemover (particle *g_p, int *g_num_p, double L, int *g_num_abs_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_tail;
  __shared__ int sh_num_abs_p;
  
  // kernel registers
  int tid = (int) threadIdx.x;
  int bdim = (int) blockDim.x;
  int N = *g_num_p;
  int ite = (N/bdim)*bdim;
  int reg_tail;
  particle reg_p;
 
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  if (tid == 0) {
    sh_tail = 0;
    sh_num_abs_p = 0;
  }
  __syncthreads();

  //---- analize full batches of particles
  for (int i = tid; i<ite; i+=bdim) {
    // load particles from global memory to registers
    reg_p = g_p[i];

    // analize particle
    if (reg_p.r >= 0 && reg_p.r <= L) {
      reg_tail = atomicAdd(&sh_tail, 1);
    } else {
      reg_tail = -1;
      if (reg_p.r < 0.0) atomicAdd(&sh_num_abs_p, 1);
    }
    __syncthreads();

    // store accepted particles in global memory
    if (reg_tail >= 0) g_p[reg_tail] = reg_p;

    __syncthreads();
  }
  __syncthreads();

  //---- analize last batch of particles
  if (ite+tid < N) {
    // loag particles from global memory to registers
    reg_p = g_p[ite+tid];

    // analize particle
    if (reg_p.r >= 0 && reg_p.r <= L) {
      reg_tail = atomicAdd(&sh_tail, 1);
    } else {
      reg_tail = -1;
      if (reg_p.r < 0.0) atomicAdd(&sh_num_abs_p, 1);
    }
  }
  __syncthreads();

  // store accepted particles of last batch in global memory
  if (ite+tid < N && reg_tail >= 0) g_p[reg_tail] = reg_p;
  
  // store new number of particles in global memory
  if (tid == 0) {
    *g_num_p = sh_tail;
    *g_num_abs_p = sh_num_abs_p;
  }
  
  return; 
}

/**********************************************************/

__device__ inline double device_vdf(double v, double vth, double vd)
{
  /*--------------------------- function variables -----------------------*/
  
  /*----------------------------- function body -------------------------*/
  
  return v*exp(-(v-vd)*(v-vd)/(2.0*vth*vth));
}

/**********************************************************/

