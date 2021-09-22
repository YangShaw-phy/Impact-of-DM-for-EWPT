// ====================================================================
// This file is part of PhaseTracer

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// ====================================================================

#ifndef POTENTIAL_XSM_OS_LIKE_HPP_INCLUDED
#define POTENTIAL_XSM_OS_LIKE_HPP_INCLUDED

/**
   Z2 symmetric real scalar singlet extension of the Standard Model
*/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/math/tools/roots.hpp>
#include <Eigen/Eigenvalues>

#include "one_loop_potential.hpp"
#include "pow.hpp"

namespace EffectivePotential {

struct TerminationCondition  {
  bool operator() (double min, double max)  {
    return abs(min - max) <= 1E-3;
  }
};

class xSM_OSlike : public OneLoopPotential {
 public:
  
  xSM_OSlike(double lambda_hs_, double lambda_s_, double ms_):
    lambda_hs(lambda_hs_), lambda_s(lambda_s_), ms(ms_){
    mus_sq = square(ms) - 0.5*lambda_hs * square(v);
    
    Eigen::VectorXd EW_VEV(2);
    EW_VEV <<  v, 0.;
    scalar_masses_sq_EW = get_scalar_debye_sq(EW_VEV, 0, 0);
    
//    std::cout << "lambda_s = " << lambda_s << std::endl;
//    std::cout << "lambda_h = " << lambda_h << std::endl;
//    std::cout << "lambda_hs = " << lambda_hs << std::endl;
//    std::cout << "mu_h_sq = " << muh_sq << std::endl;
//    std::cout << "mu_s_sq = " << mus_sq << std::endl;
//    std::cout << "g = " << g << std::endl;
//    std::cout << "gp = " << gp << std::endl;
    
  }

  double V0(Eigen::VectorXd phi) const override {
    return 0.5 * muh_sq * square(phi[0]) +
           0.25 * lambda_h * pow_4(phi[0]) +
           0.25 * lambda_hs * square(phi[0]) * square(phi[1]) +
           0.5 * mus_sq * square(phi[1]) +
           0.25 * lambda_s * pow_4(phi[1]);
  }

  double V1(std::vector<double> scalar_masses_sq,
            std::vector<double> fermion_masses_sq,
            std::vector<double> vector_masses_sq) const override {
    double correction = 0;

    static const auto scalar_dofs = get_scalar_dofs();
    static const auto fermion_dofs = get_fermion_dofs();
    static const auto vector_dofs = get_vector_dofs();
    
    // scalar correction
    for (size_t i = 0; i < scalar_masses_sq.size(); ++i) {
      const double x = scalar_masses_sq[i] / scalar_masses_sq_EW[i];
      correction += scalar_dofs[i] * scalar_masses_sq[i] *
                    (scalar_masses_sq_EW[i] * xlogx(x) - scalar_masses_sq[i] * 3. / 2.);
      correction += scalar_dofs[i] * 2. * scalar_masses_sq[i] * scalar_masses_sq_EW[i];
    }

    // fermion correction
    for (size_t i = 0; i < fermion_masses_sq.size(); ++i) {
      const double x = fermion_masses_sq[i] / fermion_masses_sq_EW[i];
      correction -= fermion_dofs[i] * fermion_masses_sq[i] *
                    (fermion_masses_sq_EW[i] * xlogx(x) - fermion_masses_sq[i] * 3. / 2.);
      correction -= fermion_dofs[i] * 2. * fermion_masses_sq[i] * fermion_masses_sq_EW[i];
    }

    // vector correction
    for (size_t i = 0; i < vector_masses_sq.size(); ++i) {
      const double x = vector_masses_sq[i] / vector_masses_sq_EW[i];
      correction += vector_dofs[i] * vector_masses_sq[i] *
                    (vector_masses_sq_EW[i] * xlogx(x) - vector_masses_sq[i] * 3. / 2.);
      correction += vector_dofs[i] * 2. * vector_masses_sq[i] * vector_masses_sq_EW[i];
    }
    return correction / (64. * M_PI * M_PI);
  }

  /**
   * Thermal scalar masses of form c * T^2 etc for high-temperature expansion of potential
   */
  std::vector<double> get_scalar_thermal_sq(double T) {
    const double c_h = (9. * g_sq +
                        3. * gp_sq +
                        2. * (6. * yt_sq + 12. * lambda_h + lambda_hs)) / 48.;
    const double c_s = (2. * lambda_hs + 3. * lambda_s) / 12.;
    return {c_h * square(T), c_s * square(T)};
  }

  // Higgs
  std::vector<double> get_scalar_debye_sq(Eigen::VectorXd phi, double xi, double T) const override{
    const double h = phi[0];
    const double s = phi[1];
    const double mhh2 = muh_sq + 3. * lambda_h * square(h) + 0.5 * lambda_hs * square(s);
    const double mss2 = mus_sq + 3. * lambda_s * square(s) + 0.5 * lambda_hs * square(h);
    const double c_h = (9. * g_sq +
                        3. * gp_sq +
                        2. * (6. * yt_sq + 12. * lambda_h + lambda_hs)) / 48.;
    const double c_s = (2. * lambda_hs + 3. * lambda_s) / 12.;

    // CP even Higgs thermal temperature masses
    const double a = mhh2 + c_h * square(T);
    const double b = mss2 + c_s * square(T);
    const double c = lambda_hs * h * s;
    const double A = 0.5 * (a+b);
    const double B = sqrt(0.25*square(a-b) + square(c));  
    
    return {A+B, A-B};
  }
  std::vector<double> get_scalar_masses_sq(Eigen::VectorXd phi, double xi) const override {
    return get_scalar_debye_sq(phi, xi, 0.);
  }
  std::vector<double> get_scalar_dofs() const override { return {1., 1.}; }

  // W, Z
  std::vector<double> get_vector_debye_sq(Eigen::VectorXd phi, double T) const override{
    const double h_sq = square(phi[0]);


    const double MW_L_sq = g_sq * (0.25 * h_sq + 11./6. * square(T));
    const double MW_T_sq = g_sq * (0.25 * h_sq);
    
    // Z, gamma matrix
    const double a_T = 0.25 * g_sq * h_sq;
    const double b_T = 0.25 * gp_sq * h_sq;
    const double c = -0.25 * g * gp * h_sq;
    const double A_T = 0.5 * (a_T+b_T);
    const double B_T = sqrt(0.25*square(a_T-b_T) + square(c));      
    const double MZ_T_sq = A_T + B_T;
    
    const double a_L = g_sq * (0.25 * h_sq + 11./6. * square(T));
    const double b_L = gp_sq * (0.25 * h_sq + 11./6. * square(T));
    const double A_L = 0.5 * (a_L+b_L);
    const double B_L = sqrt(0.25*square(a_L-b_L) + square(c));      
    const double MZ_L_sq = A_L + B_L;
    
    return {MW_L_sq, MW_T_sq, MZ_L_sq, MZ_T_sq};
  }
  std::vector<double> get_vector_masses_sq(Eigen::VectorXd phi) const override{
    return get_vector_debye_sq(phi, 0.);
  }
  std::vector<double> get_vector_dofs() const override { return {2., 4., 1., 2.}; }
  
  // top
  std::vector<double> get_fermion_masses_sq(Eigen::VectorXd phi) const override{
    return {0.5 * yt_sq * square(phi[0])};
  }
  std::vector<double> get_fermion_dofs() const override {
    return {12.};
  }

  size_t get_n_scalars() const override {return 2;}

  std::vector<Eigen::VectorXd> apply_symmetry(Eigen::VectorXd phi) const override {
    auto phi1 = phi;
    phi1[0] = - phi[0];
    auto phi2 = phi;
    phi2[1] = - phi[1];
    return {phi1,phi2};
  };

 private:
  
  const double v = 246.;
  const double mh = 125.;
  const double mtop = 172.4;
  const double mZ = 91.2;
  const double mW = 80.4;
  const double g = 2.*mW/v;
  const double g_sq = g * g;
  const double gp = 2.* sqrt(square(mZ) - square(mW))/v;
  const double gp_sq = gp * gp;
  const double yt_sq = 2. * square(mtop/v);

  const double muh_sq = -0.5 * mh * mh;
  const double lambda_h = -muh_sq / square(v);
  double lambda_hs = 0.25;
  double ms = 65;
  double lambda_s = 0.1;
  double mus_sq = square(ms) - lambda_hs * square(v) / 2.;

          
  std::vector<double> scalar_masses_sq_EW = {ms*ms, mh*mh};
  const std::vector<double> vector_masses_sq_EW = {mW*mW, mW*mW, mZ*mZ, mZ*mZ};
  const std::vector<double> fermion_masses_sq_EW = {mtop*mtop};
  
};

}  // namespace EffectivePotential

#endif
