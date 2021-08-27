from test_xSM import model as potential_model
import numpy as np
import pandas as pd


def select_Tc(TcTrans):
    N = len(TcTrans)
    for i in range(N):
        if abs(TcTrans[i]['high_vev'][0]) < 0.1 and TcTrans[i]['trantype'] == 1 and abs(TcTrans[i]['low_vev'][0]) > 0.1:
            a = i
            break
    return a

def select_Tn(TnTrans):
        N = len(TnTrans)
        for i in range(N):
            if TnTrans[i]['trantype'] == 1:
                a = i
                break
        return a

def insert_data():
    lamda_hs.append(lamda_hs_scan[lamda_hs_index])
    lamda_s.append(lamda_s_scan[lamda_s_index])
    ms.append(ms_scan[ms_index])
    Tc.append(666)
    Tc_highvev_scalar.append(666)
    Tc_highvev_higgs.append(666)
    Tc_lowvev_higgs.append(666)
    Tc_lowvev_scalar.append(666)
    Tn.append(666)
    Tn_highvev_scalar.append(666)
    Tn_highvev_higgs.append(666)
    dilusion.append(666)





def change_var(var):
    var_group = np.array([0, -2*var, -1*var, var, 2*var])
    for i in var_group:
        lamda_hs_scan_var = lamda_hs_scan[lamda_hs_index] + i
        print(lamda_hs_scan_var)
        lamda_s_scan_var = lamda_s_scan[lamda_s_index]
        ms_scan_var = ms_scan[ms_index] + i * 10
        model = potential_model(lamda_hs_scan_var, lamda_s_scan_var, ms_scan_var, True)
        try:
            model.findAllTransitions()
        except:
            if i == 2*var:
                insert_data()
                print("fail to change var ! ")
                break
            else:
                print("try to change var ! ")
        else:
            print("suscess change the var! ")
            TcTrans = model.TcTrans
            TnTrans = model.TnTrans
            if TcTrans == [] or TnTrans == []:
                print('parameter is not good, No transition!')
                if i == 2 * var:
                    insert_data()
                    print("fail to find Tc or Tn !")
                    break
                continue
            try:
                k = select_Tc(TcTrans)
                m = select_Tn(TnTrans)
            except UnboundLocalError:
                print("No transition that we needed ! ")
                if i == 2 * var:
                    insert_data()
                    print('fail to find trasition')
                    break
                continue
            phyqun = [TcTrans[k]['Tcrit'], TcTrans[k]['high_vev'], TcTrans[k]['low_vev'], TnTrans[m]['Tnuc'],
                      TnTrans[m]['high_vev']]
            if phyqun[0] <= phyqun[3]:
                print('Tc must be lager than Tn !')
                if i == 2 * var:
                    insert_data()
                    print('transtion is not right')
                    break
                continue
            phyqun.append(model.dilution_factor(1e-6, 1e-6))
            lamda_hs.append(lamda_hs_scan_var)
            lamda_s.append(lamda_s_scan_var)
            ms.append(ms_scan_var)
            Tc.append(phyqun[0])
            Tn.append(phyqun[3])
            Tc_highvev_higgs.append(phyqun[1][0])
            Tc_highvev_scalar.append(phyqun[1][1])
            Tc_lowvev_higgs.append(phyqun[2][0])
            Tc_lowvev_scalar.append(phyqun[2][1])
            Tn_highvev_higgs.append(phyqun[4][0])
            Tn_highvev_scalar.append(phyqun[4][1])
            dilusion.append(phyqun[5])
            break







datasavepath = '/home/labor/research porject/EWPT/data'

lamda_hs_scan = np.linspace(0.457, 0.457, 1)
lamda_s_scan = np.linspace(0.3, 0.3, 1)
ms_scan = np.linspace(50, 50, 1)

length_lamda_hs = len(lamda_hs_scan)
length_lamda_s = len(lamda_s_scan)
length_ms = len(ms_scan)

total_index = length_ms * length_lamda_s * length_lamda_hs
run_index = 0




lamda_s = []
lamda_hs = []
ms = []
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
            print("   ")
            print("Having finish " + str(run_index) + ' times', 'complete ratio: ' + str(run_index / total_index))
            print("   ")
            run_index = run_index + 1
            change_var(0.0001)

print('Saving data, DO NOT close!!!')
transdata = {
                'lamda_hs': lamda_hs,
                'lamda_s': lamda_s,
                'ms': ms,
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



