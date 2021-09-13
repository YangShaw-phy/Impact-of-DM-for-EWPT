from EWPT.test_xSM import model as potential_model
from cosmoTransitions import transitionFinder
import numpy as np
import matplotlib.pyplot as plt



def plot_action(start_phase, phases):
    Tmax = start_phase.T[-1]
    Tmin = start_phase.T[0]
    T_highest_other = Tmin
    for phase in phases.values():
        T_highest_other = max(T_highest_other, phase.T[-1])
    Tmax = min(Tmax, T_highest_other)
    assert Tmax >= Tmin
    T_scan = np.linspace(Tmin, Tmax, 50)
    T_scan = T_scan[1:len(T_scan[1:]) - 15]
    print(Tmin)
    print(Tmax)
#para
    Ttol=1e-3
    maxiter=100
    phitol=1e-8
    overlapAngle=45.0
    nuclCriterion = lambda S, T: S/(T+1e-100) - 140
    verbose = True
    fullTunneling_params={}
    outdict = {}
#####
    action = []
    for i in T_scan:
        action.append(transitionFinder._tunnelFromPhaseAtT(i, phases, start_phase, model.Vtot, model.gradV,
                        phitol, overlapAngle, nuclCriterion,
                        fullTunneling_params, verbose, outdict)+140)
    print(action)
    plt.plot(T_scan, action)
    plt.plot(T_scan, 140 * np.ones(len(T_scan)))
    plt.xlabel('Tscan')
    plt.ylabel('action')
    plt.show()








lamda_hs_scan_var = 0.625
lamda_s_scan_var = 0.682
ms_scan_var = 34.44
model =potential_model(lamda_hs_scan_var, lamda_s_scan_var, ms_scan_var, True)
model.phases = model.getPhases()
phases = model.phases.copy()
#######################################################
start_phase = phases[transitionFinder.getStartPhase(model.phases, model.Vtot)]
start_phase = phases[1]
#del phases[2]
# del phases[4]
########################################################

Tmax = start_phase.T[-1]
transitions = []
tunnelFromPhase_args={}
print(phases)
print(start_phase)
print('start_phase.low_trans',start_phase.low_trans)
#start_phase.low_trans = [3]


while start_phase is not None:
    del phases[start_phase.key]
    trans =transitionFinder.tunnelFromPhase(phases, start_phase, model.Vtot, model.gradV, Tmax,**tunnelFromPhase_args)
    plot_action(start_phase, phases)
    if trans is None and not start_phase.low_trans:
        start_phase = None
    elif trans is None:
        low_key = None
        for key in start_phase.low_trans:
            if key in phases:
                low_key = key
                break
        if low_key is not None:
            low_phase = phases[low_key]
            transitions.append(transitionFinder.secondOrderTrans(start_phase, low_phase))
            start_phase = low_phase
            Tmax = low_phase.T[-1]
        else:
            start_phase = None
            phases1 = phases
            start_phase1 = start_phase
    else:
        transitions.append(trans)
        start_phase = phases[trans['low_phase']]
        Tmax = trans['Tnuc']

model.TnTrans = transitions
model.TcTrans = model.calcTcTrans()
transitionFinder.addCritTempsForFullTransitions(
    model.phases, model.TcTrans, model.TnTrans)
# Add in Delta_rho, Delta_p
for trans in model.TnTrans:
    T = trans['Tnuc']
    xlow = trans['low_vev']
    xhigh = trans['high_vev']
    trans['Delta_rho'] = model.energyDensity(xhigh, T, False) \
                         - model.energyDensity(xlow, T, False)
    trans['Delta_p'] = model.Vtot(xhigh, T, False) \
                       - model.Vtot(xlow, T, False)

print(model.dilution_factor(1e-6, 1e-6))
# print(model.TcTrans)
# print(model.TnTrans)
print(model.entropy(1e-6,1e-6))
