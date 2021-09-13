import numpy as np
from cosmoTransitions import generic_potential
from cosmoTransitions import transitionFinder


class model(generic_potential.generic_potential):
    def init(self, lamda_hs, lamda_s, ms, RG_flag):
        self.Ndim = 2
        self.RG_onshell_method = RG_flag
        self.renormScaleSq = 1000
        self.vev_higgs_sq = 246 ** 2
        self.mh_sq = 125 ** 2
        self.ms_sq = ms ** 2
        self.lamda_hs = lamda_hs
        self.lamda_s = lamda_s
        ####calculated by the above parameters
        self.mu_h_sq = self.mh_sq / 2
        self.mu_s_sq = -1 * self.ms_sq + self.lamda_hs * self.vev_higgs_sq / 2
        self.lamda_h = self.mh_sq / (2 * self.vev_higgs_sq)
        ####degree of particle
        self.dof_wt = 4
        self.dof_wl = 2
        self.dof_zt = 2
        self.dof_zl = 1
        self.dof_t = 12
        self.dof_goldstone = 3
        ######coupling constant
        self.Y1 = 2 * 80.4 / 246  # g
        self.Y2 = 2 * (91.2 ** 2 - 80.4 ** 2) ** 0.5 / 246  # g'
        self.Yt = 2 ** 0.5 * 172.4 / 246  # just curious about 1/6, yukawa top
        self.Mass_boson_vev = self.boson_massSq([246, 0], 0)
        self.Mass_fermi_vev = self.fermion_massSq([246, 0])

    def forbidPhaseCrit(self, X):
        return any([np.array([X])[..., 0] < -5.0, np.array([X])[..., 1] < -5.0])

    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return [np.array([246., 0])]

    def V0(self, X):
        X = np.asanyarray(X)
        h, s = X[..., 0], X[..., 1]
        r = -0.5 * self.mu_h_sq * h ** 2 + 0.25 * self.lamda_h * h ** 4 + 0.25 * self.lamda_hs * h ** 2 * s ** 2 - 0.5 * self.mu_s_sq * s ** 2 + 0.25 * self.lamda_s * s ** 4
        return r

    def boson_massSq(self, X, T):
        X = np.array(X)
        h, s = X[..., 0], X[..., 1]
        ringh = (3. * self.Y1 ** 2. / 16. + self.Y2 ** 2. / 16. + self.lamda_h / 2 + self.Yt ** 2. / 4. + self.lamda_hs / 12.) * T ** 2. * h ** 0.
        rings = (self.lamda_s / 4. + self.lamda_hs / 3.) * T ** 2. * h ** 0.
        ringwl = 11. * self.Y1 ** 2. * T ** 2. * h ** 0. / 6.
        ringbl = 11. * self.Y2 ** 2. * T ** 2. * h ** 0. / 6.
        ringchi = (3. * self.Y1 ** 2. / 16. + self.Y2 ** 2. / 16. + self.lamda_h / 2. + self.Yt ** 2. / 4. + self.lamda_hs / 12.) * T ** 2. * h ** 0.

        mh = 2 * h **2 * self.lamda_h + ringh  # note, this is mh_sq, same for below
        ms = -1 * self.mu_s_sq + self.lamda_hs * h**2 / 2 + rings

        mwl = 0.25 * self.Y1 ** 2. * h ** 2. + ringwl
        mwt = 0.25 * self.Y1 ** 2. * h ** 2.

        mzgla = 0.25 * self.Y1 ** 2. * h ** 2. + ringwl  # this is the themal mass of Z boson, see arXiv:1507.06912v2
        mzglb = 0.25 * self.Y2 ** 2. * h ** 2. + ringbl
        mzgc = - 0.25 * self.Y1 * self.Y2 * h ** 2.
        mzglA = .5 * (mzgla + mzglb)
        mzglB = np.sqrt(.25 * (mzgla - mzglb) ** 2. + mzgc ** 2.)

        mzgta = 0.25 * self.Y1 ** 2. * h ** 2.
        mzgtb = 0.25 * self.Y2 ** 2. * h ** 2.
        mzgtA = .5 * (mzgta + mzgtb)
        mzgtB = np.sqrt(.25 * (mzgta - mzgtb) ** 2. + mzgc ** 2.)

        mzl = mzglA + mzglB
        mzt = mzgtA + mzgtB
        mgl = mzglA - mzglB  # this is the photon
        mgt = mzgtA - mzgtB

        mx = -self.mu_h_sq + self.lamda_h * h ** 2. + 0.5 * self.lamda_hs * s ** 2. + ringchi
        if self.RG_onshell_method == True:
            #M = np.array([mh, ms, mwl, mwt, mzl, mzt, mgl + 1e-10, mgt + 1e-10, mx + 1e-10])
            M = np.array([mh, ms, mwl, mwt, mzl, mzt])
            M = np.rollaxis(M, 0, len(M.shape))
            dof = np.array([1, 1, self.dof_wl, self.dof_wt, self.dof_zl, self.dof_zt])
            c = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])  # check Goldstones
        else:
            M = np.array([mh, ms, mwl, mwt, mzl, mzt, mgl, mgt, mx])
            M = np.rollaxis(M, 0, len(M.shape))
            dof = np.array([1, 1, self.dof_wl, self.dof_wt, self.dof_zl, self.dof_zt, self.dof_zl, self.dof_zt,
                            self.dof_goldstone])
            c = np.array([1.5, 1.5, 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 5. / 6., 1.5])  # check Goldstones

        return M, dof, c

    def fermion_massSq(self, X):
        X = np.array(X)
        h = X[..., 0]
        mt = 0.5 * self.Yt ** 2 * h ** 2
        M = np.array([mt])
        M = np.rollaxis(M, 0, len(M.shape))
        dof = np.array(self.dof_t)
        return M, dof

    def V1_MS(self, bosons, fermions):  # MS bar
        m2, n, c = bosons
        y = np.sum(n * m2 * m2 * (np.log(np.abs(m2 / self.renormScaleSq) + 1e-100)
                                  - c), axis=-1)
        m2, n = fermions
        c = 1.5
        y -= np.sum(n * m2 * m2 * (np.log(np.abs(m2 / self.renormScaleSq) + 1e-100)
                                   - c), axis=-1)
        return y / (64. * np.pi * np.pi)

    def V1_onshell(self, bosons, fermions):  # on shell
        m2, n, c = bosons
        y = np.sum(n * (m2 * m2 * (np.log(np.abs(m2 / self.Mass_boson_vev[0]) + 1e-100)
                                   - c) + 2 * m2 * self.Mass_boson_vev[0]), axis=-1)
        m2, n = fermions
        y -= np.sum(n * (m2 * m2 * (np.log(np.abs(m2 / self.Mass_fermi_vev[0]) + 1e-100)
                                    - 1.5) + 2 * m2 * self.Mass_fermi_vev[0]), axis=-1)
        return y / (64. * np.pi * np.pi)

    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        if self.RG_onshell_method == True:
            y += self.V1_onshell(bosons, fermions)
        else:
            y += self.V1_MS(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y

    def select_Tc(self):
        N = len(self.TcTrans)
        for i in range(N):
            if abs(self.TcTrans[i]['high_vev'][0]) < 0.1 and self.TcTrans[i]['trantype'] == 1 and abs(
                    self.TcTrans[i]['low_vev'][0]) > 0.1:
                a = i
                break
        return a

    def select_Tn(self):
        N = len(self.TnTrans)
        for i in range(N):
            if self.TnTrans[i]['trantype'] == 1:
                a = i
                break
        return a

    def entropy(self, dt, dphi):  # find the high T phase entropy
        if self.TcTrans == [] or self.TnTrans == []:
            print("NO transition!! entropy is ill defined")
        else:
            k = self.select_Tc()
            Tcrit = self.TcTrans[k]['Tcrit']
            Tcrit_highvev = self.TcTrans[k]['high_vev']
            Tcrit_lowvev = self.TcTrans[k]['low_vev']
            m = self.select_Tn()
            Tnuc = self.TnTrans[m]['Tnuc']
            Tnuc_highvev = self.TnTrans[m]['high_vev']
            ###### Tcrit highvev
            dvdt = (self.Vtot(Tcrit_highvev, Tcrit + dt) - self.Vtot(Tcrit_highvev, Tcrit - dt)) / (2 * dt)
            dvdphi = self.gradV(Tcrit_highvev, Tcrit)
            try:
                dphidt = np.matmul(np.linalg.inv(self.d2V(Tcrit_highvev, Tcrit)),
                                   -1 * self.dgradV_dT(Tcrit_highvev, Tcrit))  # self.d2V equal 0
            except:
                S_crit_highvev = 1
            else:
                S_crit_highvev = -(dvdt + np.sum(dphidt * dvdphi))

            ###### Tcrit lowvev
            dvdt = (self.Vtot(Tcrit_lowvev, Tcrit + dt) - self.Vtot(Tcrit_lowvev, Tcrit - dt)) / (2 * dt)
            dvdphi = self.gradV(Tcrit_lowvev, Tcrit)
            try:
                dphidt = np.matmul(np.linalg.inv(self.d2V(Tcrit_lowvev, Tcrit)),
                                   -1 * self.dgradV_dT(Tcrit_lowvev, Tcrit))
            except:
                S_crit_lowvev = 1
            else:
                S_crit_lowvev = -(dvdt + np.sum(dphidt * dvdphi))

            #####Tnul high vev
            dvdt = (self.Vtot(Tnuc_highvev, Tnuc + dt) - self.Vtot(Tnuc_highvev, Tnuc - dt)) / (2 * dt)
            dvdphi = self.gradV(Tnuc_highvev, Tnuc)
            try:
                dphidt = np.matmul(np.linalg.inv(self.d2V(Tnuc_highvev, Tnuc)), -1 * self.dgradV_dT(Tnuc_highvev, Tnuc))
            except:
                S_nuc_highvev = S_crit_highvev
            else:
                S_nuc_highvev = -(dvdt + np.sum(dphidt * dvdphi))

            entropy = [S_crit_highvev, S_crit_lowvev, S_nuc_highvev]
            return entropy

    def dilution_factor(self, dt, dphi):
        if self.TcTrans == [] or self.TnTrans == []:
            print("NO transition!! entropy is ill defined")
        else:
            k = self.select_Tc()
            Tcrit = self.TcTrans[k]['Tcrit']
            Tcrit_highvev = self.TcTrans[k]['high_vev']
            Tcrit_lowvev = self.TcTrans[k]['low_vev']
            m = self.select_Tn()
            Tnuc = self.TnTrans[m]['Tnuc']
            Tnuc_highvev = self.TnTrans[m]['high_vev']
            entropy = self.entropy(dt, dphi)
            if (Tcrit - Tnuc) < 2:
                # print('condition one')
                dilution_factor = entropy[0] / entropy[1]
            else:
                L = self.energyDensity(Tcrit_highvev, Tcrit) - self.energyDensity(Tcrit_lowvev, Tcrit)
                diffpho = self.energyDensity(Tcrit_highvev, Tcrit) - self.energyDensity(Tnuc_highvev, Tnuc)
                f = diffpho / L
                print(diffpho)
                print(L)
                if f <=1 :
                    dilution_factor = (1 - f * (entropy[0] - entropy[1]) / entropy[0]) * entropy[0] / (
                        (1 - (entropy[0] - entropy[1]) / entropy[0]) * entropy[2])
                else:
                    dilution_factor=999
                    print('f must be less than 1')
        return dilution_factor

    def getPhases(self,tracingArgs={}):
        tstop = self.Tmax
        points = []
        for x0 in self.approxZeroTMin():
            points.append([x0,0.0])
        tracingArgs_ = dict(forbidCrit=self.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        phases = transitionFinder.traceMultiMin(
            self.Vtot, self.dgradV_dT, self.d2V, points,
            tLow=0.0, tHigh=tstop, deltaX_target=100*self.x_eps,
            **tracingArgs_)
        self.phases = phases
        transitionFinder.removeRedundantPhases(
            self.Vtot, phases, self.x_eps*1e-2, self.x_eps*10)  #可以考虑注释掉
        return self.phases



#model = model(0.479, 0.3, 60, True)

# model = model(8.96, 1, 530, True)
# model.findAllTransitions()
# print(model.TcTrans)
# print(model.TnTrans)
# # # # print(model.TnTrans[1]['action'])
# # print('Tc', model.TcTrans[model.select_Tc()]['Tcrit'])
# # print('Tn', model.TnTrans[model.select_Tn()]['Tnuc'])
# # print('Tc_highvev', model.TcTrans[model.select_Tc()]['high_vev'])
# # print('Tc_lowvev', model.TcTrans[model.select_Tc()]['low_vev'])
# # print('Tn_highvev', model.TnTrans[model.select_Tn()]['high_vev'])
#print(model.dilution_factor(1e-6, 1e-6))
