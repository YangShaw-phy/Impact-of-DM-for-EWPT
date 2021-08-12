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

#ifndef POTENTIAL_TEST_MODEL_HPP_INCLUDED
#define POTENTIAL_TEST_MODEL_HPP_INCLUDED

#include <vector>

#include "one_loop_potential.hpp"
#include "pow.hpp"


namespace EffectivePotential {

class EFTModel : public OneLoopPotential {
 public:

  double V0(Eigen::VectorXd phi) const override {
    return - 0.5 * lamda * renormScaleSq * square(phi[0]) \
           + 0.25 * lamda * pow_4(phi[0]) \
           + alpha * (0.5 * renormScaleSq * square(phi[0]) \
                      - sqrt(renormScaleSq) * cube(phi[0])/3.);
  }

  std::vector<double> get_scalar_masses_sq(Eigen::VectorXd phi, double xi) const override {
    return {square(h_boson * phi[0])};
  }
  std::vector<double> get_scalar_dofs() const override {
    return {num_boson_dof};
  }
  
  std::vector<double> get_fermion_masses_sq(Eigen::VectorXd phi) const override{
    return {square(h_ferimon * phi[0])};
  }
  // top, bottom and tau
  std::vector<double> get_fermion_dofs() const override {
    return {num_fermion_dof};
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

    return correction / (64. * M_PI * M_PI);
  }
  
  double counter_term(Eigen::VectorXd phi, double T) const override  {

    const auto scalar_masses_sq = get_scalar_masses_sq(phi,0);
    static const auto scalar_dofs = get_scalar_dofs();
    double y = scalar_dofs[0]*( pow(scalar_masses_sq[0],1.5)
                            - pow(scalar_masses_sq[0]+1./3.*square(h_boson*T) ,1.5) ) * T / (12. * M_PI);
    
    y += -1 * M_PI* M_PI * num_stanard_dof * pow_4(T)/90.;
    return y;
  }
  
  size_t get_n_scalars() const override { return 1; }

  std::vector<Eigen::VectorXd> apply_symmetry(Eigen::VectorXd phi) const override {
    return {-phi};
  };

  void set_dof(double dof){
    num_boson_dof = dof;
    num_fermion_dof = dof;
  }
 private:
  const double v = 246.;
  const double m1 = 120.;
  const double m2 = 50.;
  const double mu = 25.;
  const double l1 = 0.5 * square(m1 / v);
  const double l2 = 0.5 * square(m2 / v);
  const double y1 = 0.1;
  const double y2 = 0.15;
  
  double lamda = 0.1855;
  const double alpha = 0 * lamda;
  const double renormScaleSq = 246.22*246.22;
  const double num_stanard_dof = 100;
  double num_boson_dof = 1000;
  double num_fermion_dof = 1000;
  const double h_boson = 1;
  const double h_ferimon = 1;
  
  std::vector<double> scalar_masses_sq_EW = {renormScaleSq*h_boson*h_boson};
  std::vector<double> fermion_masses_sq_EW = {renormScaleSq*h_ferimon*h_ferimon};
};

}  // namespace EffectivePotential

#endif
