from test_xSM import model as potential_model
import numpy as np
import pandas as pd

def select_Tc(TcTrans):
    N = len(TcTrans)
    for i in range(N):
        if abs(TcTrans[i]['high_vev'][0]) < 0.1:
            a = i
            break
    return a

#
datasavepath = '/home/labor/research porject/EWPT/data'

lamda_hs_scan = np.linspace(1, 4, 2)
lamda_s_scan = np.linspace(5, 7, 2)
ms_scan = np.linspace(70, 101, 2)


length_lamda_hs = len(lamda_hs_scan)
length_lamda_s = len(lamda_s_scan)
length_ms = len(ms_scan)

total_index = length_ms * length_lamda_s * length_lamda_hs
run_index = 0

Tc = []
Tn = []
Tc_highvev_higgs = []
Tc_highvev_scalar = []
Tc_lowvev_higgs = []
Tc_lowvev_scalar = []
Tn_highvev_higgs = []
Tn_highvev_scalar = []
dilusion = []
for lamda_hs_index in range(length_lamda_hs):
    for lamda_s_index in range(length_lamda_s):
        for ms_index in range(length_ms):
            print("Having finish " + str(run_index) + ' times', 'complete ratio: ' + str(run_index / total_index))
            run_index = run_index + 1
            model = potential_model(lamda_hs_scan[lamda_hs_index], lamda_s_scan[lamda_s_index], ms_scan[ms_index], True)
            model.findAllTransitions()
            TcTrans = model.TcTrans
            TnTrans = model.TnTrans
            if TcTrans == [] or TnTrans == []:
                print('parameter is not good, No transition!')
                Tc.append(9999)
                Tc_highvev_scalar.append(9999)
                Tc_highvev_higgs.append(9999)
                Tc_lowvev_higgs.append(9999)
                Tc_lowvev_scalar.append(9999)
                Tn.append(9999)
                Tn_highvev_scalar.append(9999)
                Tn_highvev_higgs.append(9999)
                dilusion.append(9999)
                continue
            k = select_Tc(TcTrans)
            phyqun = [TcTrans[k]['Tcrit'], TcTrans[k]['high_vev'], TcTrans[k]['low_vev'], TnTrans[0]['Tnuc'], TnTrans[0]['high_vev']]
            if phyqun[0] <= phyqun[3]:
                print('Tc must be lager than Tn !')
                Tc.append(9999)
                Tc_highvev_scalar.append(9999)
                Tc_highvev_higgs.append(9999)
                Tc_lowvev_higgs.append(9999)
                Tc_lowvev_scalar.append(9999)
                Tn.append(9999)
                Tn_highvev_scalar.append(9999)
                Tn_highvev_higgs.append(9999)
                dilusion.append(9999)
                continue
            phyqun.append(model.dilution_factor(1e-6, 1e-6))
            Tc.append(phyqun[0])
            Tn.append(phyqun[3])
            Tc_highvev_higgs.append(phyqun[1][0])
            Tc_highvev_scalar.append(phyqun[1][1])
            Tc_lowvev_higgs.append(phyqun[2][0])
            Tc_lowvev_scalar.append(phyqun[2][1])
            Tn_highvev_higgs.append(phyqun[4][0])
            Tn_highvev_scalar.append(phyqun[4][1])
            dilusion.append(phyqun[5])


print('Saving data, DO NOT close!!!')
transdata = {
                'Tc': Tc,
                'Tn': Tn,
                'Tc_highvev_higgs': Tc_highvev_higgs,
                'Tc_highvev_scalar': Tc_highvev_scalar,
                'Tn_highvev_higgs': Tn_highvev_higgs,
                'Tn_highvev_scalar': Tn_highvev_scalar,
                'Tc_lowvev_higgs': Tc_lowvev_higgs,
                'Tc_lowvev_scalar': Tc_lowvev_scalar,
                'dilusion': dilusion
            }
transdata = pd.DataFrame(transdata)
transdata.to_csv(datasavepath + '/transdata.csv')



