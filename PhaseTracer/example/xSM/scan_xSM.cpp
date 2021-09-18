/**
  Z2 real scalar singlet extension of
  the Standard Model 
  
  OS-like
  
*/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <random>

#include "xSM.hpp"
#include "phase_finder.hpp"
#include "transition_finder.hpp"
#include "logger.hpp"
#include "phase_plotter.hpp"
#include "thermal_function.hpp"

std::string run(double ms, double lambda_s, double lambda_hs, bool debug_mode = false) {
  std::stringstream data_str;

  // Construct our model
  EffectivePotential::xSM_OSlike model(lambda_hs, lambda_s, ms);
  model.set_daisy_method(EffectivePotential::DaisyMethod::Parwani);
  
  if (debug_mode) {
      Eigen::VectorXd x(2);
      x <<  246., 100;
      
      auto scalar_masses_sq = model.get_scalar_debye_sq(x,0,100);
      auto fermion_masses_sq = model.get_fermion_masses_sq(x);
      auto vector_masses_sq = model.get_vector_debye_sq(x, 100.);

      std::cout << "scalar_masses =" << sqrt(scalar_masses_sq[0]) << ", " << sqrt(scalar_masses_sq[1]) << std::endl;
      std::cout << "fermion_masses =" << sqrt(fermion_masses_sq[0]) << std::endl;
      std::cout << "vector_masses =" << sqrt(vector_masses_sq[0]) << ", " << sqrt(vector_masses_sq[1]) << ", " << sqrt(vector_masses_sq[2]) << ", " << sqrt(vector_masses_sq[3]) << std::endl;
      
      std::cout << "V0=" << model.V0(x) << std::endl;
      std::cout << "V1=" << model.V1(scalar_masses_sq, fermion_masses_sq, vector_masses_sq) << std::endl;
      std::cout << "V1T=" << model.V1T(x,100) << std::endl;
      std::cout << "VT=" << model.V(x,100) << std::endl;
      
      std::cout << "Numerically derivatives of the full potential at EW VEV:" << std::endl;
      auto d2Vdh2 = model.d2V_dx2(x,0);
      std::cout << "Sqrt[d^2V/dh^2] = "<< std::sqrt(abs(d2Vdh2(0,0))) << std::endl;
      std::cout << "Sqrt[d^2V/ds^2] = "<< std::sqrt(abs(d2Vdh2(1,1))) << std::endl;

  }  
  
  // Make PhaseFinder object and find the phases
  PhaseTracer::PhaseFinder pf(model);
      
  pf.set_check_vacuum_at_high(false);
  pf.set_seed(0);
//  pf.set_check_hessian_singular(false);
    
  try {
    pf.find_phases();
  } catch (...) {
    return data_str.str();
  }
  if (debug_mode) std::cout << pf;

  // Make TransitionFinder object and find the transitions
  PhaseTracer::TransitionFinder tf(pf);
  tf.find_transitions();
  if (debug_mode) std::cout << tf;
    
  auto t = tf.get_transitions();
  if (t.size()==0){
    return data_str.str();
  }
  
  // Find the transition with largest gamma from (0,vs) -> (vh,0)
  int jj = -1;
  double gamme_max = 0.;
  for (int i=0; i<t.size(); i++) {
    double gamma = t[i].gamma;
    if (gamme_max < gamma and abs(t[i].false_vacuum[0])<1. and abs(t[i].true_vacuum[1])<1.){
      jj = i;
      gamme_max = gamma;
    }
  }
  
  if (jj<0) {
    return data_str.str();
  }
  
  std::vector<double> out = {(float)t.size(), t[jj].TC, t[jj].true_vacuum[0], t[jj].true_vacuum[1], t[jj].false_vacuum[0], t[jj].false_vacuum[1]};
  
  data_str << ms << "\t" << lambda_s << "\t" << lambda_hs << "\t";
  for (auto i : out   ) data_str << i << "\t"; 
  data_str << std::endl;
  return data_str.str();
}


int main(int argc, char* argv[]) {

    if ( argc == 1 ) {
      LOGGER(debug);
      double lambda_hs = 0.25;
      double ms = 65;
      double lambda_s =  0.1;
      std::cout << "ms = " << ms << std::endl
                << "lambda_s = " << lambda_s << std::endl
                << "lambda_hs = " << lambda_hs << std::endl;
      run(ms, lambda_s, lambda_hs, true);
    } else {
      int num = atoi(argv[1]);
      LOGGER(fatal);
      std::ofstream output_file;  
      output_file.open("output.txt");

      std::random_device rd;
      std::default_random_engine eng(rd());
      std::uniform_real_distribution<double> lhs(0.1, 10);
      std::uniform_real_distribution<double> ls(0.01, 1);
      std::uniform_real_distribution<double> ms(10, 600);
      
      double RMAX = (double)RAND_MAX;
      for (int ii=0; ii<num; ii++) {
        double lambda_hs = lhs(eng);
        double m_s = ms(eng);
        double lambda_s =  ls(eng);
        std::cout << ii 
                  << ", ms = " << m_s
                  << ", lambda_s = " << lambda_s
                  << ", lambda_hs = " << lambda_hs << std::endl;
        output_file << run(m_s, lambda_s, lambda_hs, false);
      }
      output_file.close();  
    }


}



