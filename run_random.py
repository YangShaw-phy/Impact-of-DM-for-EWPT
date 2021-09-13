from EWPT.test_xSM import model as potential_model
import numpy as np
import pandas as pd
import random

def select_Tc(TcTrans):
    N = len(TcTrans)
    for i in range(N):
        if abs(TcTrans[i]['high_vev'][0]) < 0.1 and  abs(TcTrans[i]['high_vev'][1]) > 5 and TcTrans[i]['trantype'] == 1 and abs(TcTrans[i]['low_vev'][0]) > 5 and abs(TcTrans[i]['low_vev'][1]) < 0.1:
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


#datasavepath = '/home/labor/research porject/EWPT/data'
datasavepath = r'E:\pycharm_project\EWPT\EWPT\data'

wanted_num = 5
find_num = 0

lamda_hs_scan = []
lamda_s_scan = []
ms_scan = []
Tc = []
Tn = []
Tc_highvev_higgs = []
Tc_highvev_scalar = []
Tc_lowvev_higgs = []
Tc_lowvev_scalar = []
Tn_highvev_higgs = []
Tn_highvev_scalar = []
dilusion = []


while len(lamda_s_scan) < wanted_num:
    print(" ")
    print("have find " + str(find_num) + " points !!")
    print(" ")
    lamda_hs = random.uniform(0, 1)
    #lamda_s = random.uniform(0, 1)
    lamda_s = 0.3
    #ms = 500
    ms = random.uniform(40, 200)
    model = potential_model(lamda_hs, lamda_s, ms, True)
    try:
            model.findAllTransitions()
    except:
            print(" ")
            print("point is not good !")
            print(" ")
    else:
            print(" ")
            print("suscess find the possible point ")
            print(" ")
            TcTrans = model.TcTrans
            TnTrans = model.TnTrans
            if TcTrans == [] or TnTrans == []:
                print('the point is not good, No transition!')
                continue
            try:
                k = select_Tc(TcTrans)
                m = select_Tn(TnTrans)
            except UnboundLocalError:
                print("No transition that we needed ! ")
                continue
            phyqun = [TcTrans[k]['Tcrit'], TcTrans[k]['high_vev'], TcTrans[k]['low_vev'], TnTrans[m]['Tnuc'],
                      TnTrans[m]['high_vev']]
            if phyqun[0] <= phyqun[3]:
                print('Tc must be lager than Tn !')
                continue
            phyqun.append(model.dilution_factor(1e-6, 1e-6))
            if phyqun[-1] > 700:
                print('f must less than 1')
                continue
            lamda_hs_scan.append(lamda_hs)
            lamda_s_scan.append(lamda_s)
            ms_scan.append(ms)
            Tc.append(phyqun[0])
            Tn.append(phyqun[3])
            Tc_highvev_higgs.append(phyqun[1][0])
            Tc_highvev_scalar.append(phyqun[1][1])
            Tc_lowvev_higgs.append(phyqun[2][0])
            Tc_lowvev_scalar.append(phyqun[2][1])
            Tn_highvev_higgs.append(phyqun[4][0])
            Tn_highvev_scalar.append(phyqun[4][1])
            dilusion.append(phyqun[5])
            print(" ")
            print("suscess find the point")
            print(" ")
            find_num = find_num + 1
            if find_num % 5 == 0:
                print('Saving data, DO NOT close!!!')
                transdata = {
                    'lamda_hs': lamda_hs_scan,
                    'lamda_s': lamda_s_scan,
                    'ms': ms_scan,
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






print('Saving data, DO NOT close!!!')
transdata = {
                'lamda_hs': lamda_hs_scan,
                'lamda_s': lamda_s_scan,
                'ms': ms_scan,
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





