/**
  2D example program for PhaseTracer.
*/

#include <iostream>

#include "EFT.hpp" // Located in effective-potential/include/models
#include "transition_finder.hpp"
#include "phase_finder.hpp"
#include "phase_plotter.hpp"
#include "potential_plotter.hpp"
#include "potential_line_plotter.hpp"
#include "logger.hpp"


int main(int argc, char* argv[]) {

  const bool debug_mode = false;
  double dof = 100;
  if ( argc > 1 ) {
    dof = atof(argv[1]);
  }
  // Set level of screen  output
  if (debug_mode) {
      LOGGER(debug);
  } else {
      LOGGER(fatal);
  }

  // Construct our model
  EffectivePotential::EFTModel model;

  model.set_dof(dof);
  
  Eigen::VectorXd test(1);
  test << 246.22;
  double testT = 40.;
  
//  std::cout << "V0    = "<< model.V0(test) << std::endl;
//  std::cout << "V1 = "<< model.V1(test,0) << std::endl;
//  std::cout << "Daisy = "<< model.counter_term(test,testT) << std::endl;
//  std::cout << "V     = "<< model.V(test,testT) << std::endl;
  
  // Make PhaseFinder object and find the phases
  PhaseTracer::PhaseFinder pf(model);
  pf.find_phases();
//  std::cout << pf;

  // Make TransitionFinder object and find the transitions
  PhaseTracer::TransitionFinder tf(pf);
  tf.find_transitions();
//  std::cout << std::setprecision (15) << tf;
  
  auto t = tf.get_transitions();
  if (t.size()==0){
    std::cout << 0. << std::endl;
  } else {
    std::cout << std::setprecision (15)<< t[0].TC << std::endl;
  }
  
  if (debug_mode) {
    PhaseTracer::potential_plotter(model, tf.get_transitions().front().TC, "EFTModel", 0., 2., 0.01, -2., 0., 0.01);
    PhaseTracer::potential_line_plotter(model, tf.get_transitions(), "EFTModel");
    PhaseTracer::phase_plotter(tf, "EFTModel");
  }
}
