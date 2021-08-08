import matplotlib.pyplot as plt
import numpy as np
from cosmoTransitions import generic_potential


class model(generic_potential.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.

    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self):
        self.lamda = 0.1855   #para lamda
        self.alpha = 0 * self.lamda
        self.Ndim = 1   #para field dimension
        self.renormScaleSq = 246.22 **2 #246.22**2
        self.num_stanard_dof = 100
        self.num_boson_dof = 0
        self.boson_particle = 1000
        self.num_fermion_dof = 1000
        self.h_boson = 1
        self.h_ferimon = 1
        #self.deriv_order = 4
    def approxZeroTMin(self):
        """
        Returns approximate values of the zero-temperature minima.

        This should be overridden by subclasses, although it is not strictly
        necessary if there is only one minimum at tree level. The precise values
        of the minima will later be found using :func:`scipy.optimize.fmin`.

        Returns
        -------
        minima : list
            A list of points of the approximate minima.
        """
        # This should be overridden.
        return [-1 * np.ones(self.Ndim)*self.renormScaleSq**.5]

    def V0(self, X):
        """
        This method defines the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        phi = X[...,0]
        r = -0.5 * self.lamda * self.renormScaleSq * phi * phi + 0.25 * self.lamda * phi**4 + self.alpha * (0.5 * self.renormScaleSq * phi * phi - self.renormScaleSq**0.5 * phi ** 3 / 3)
 #       r = np.zeros(phi.shape)
        return r

    def boson_massSq(self, X, T):
        X = np.array(X)
        phi = X[...,0]
        mass_boson = (self.h_boson * phi) ** 2
        # We need to define the field-dependnet boson masses. This is obviously
        # model-dependent.
        # Note that these can also include temperature-dependent corrections.
        M = np.array([mass_boson])
        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([self.boson_particle])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        c = np.array([3/2])

        return M, dof, c


    def fermion_massSq(self, X):
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `X`. The
            shape should be such that  ``massSq.shape == (X[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.

        Notes
        -----
        Unlike :func:`boson_massSq`, no constant `c` is needed since it is
        assumed to be `c = 3/2` for all fermions. Also, no thermal mass
        corrections are needed.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        # Nfermions = self.num_fermion_dof
        # phi = X[..., 0]
        # #phi2 = X[...,1] # Comment out so that the placeholder doesn't
        #                  # raise an exception for Ndim < 2.
        # mass_fermion = (self.h_ferimon * phi)**2  # First fermion mass
        # massSq = np.empty(mass_fermion.shape + (Nfermions,))
        # massSq[..., 0] = mass_fermion
        # dof = np.array([self.num_fermion_dof])
        X = np.array(X)
        phi = X[...,0]
        mass_fermion = (self.h_ferimon * phi) ** 2
        massSq = np.array([mass_fermion])
        massSq = np.rollaxis(massSq, 0, len(massSq.shape))
        dof = np.array([self.num_fermion_dof])
        return massSq, dof


    def V1(self, bosons, fermions):
        """
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.
        """
        # This does not need to be overridden.
        m2, n, c = bosons
        y = np.sum(n*(m2*m2 * (np.log(np.abs(m2/(self.renormScaleSq * self.h_boson**2)) + 1e-100)
                              - c) + 2 * m2 * self.renormScaleSq * self.h_boson**2), axis=-1)
        m2, n = fermions
        c = 1.5
        y -= np.sum(n*(m2*m2 * (np.log(np.abs(m2/(self.renormScaleSq * self.h_ferimon**2)) + 1e-100)
                               - c) + 2 * m2 * self.renormScaleSq * self.h_ferimon**2), axis=-1)
        return y/(64*np.pi*np.pi)

    def Daisy(self, X, T):
        X = np.array(X)
        T = np.array(T)
        #T = T.reshape(len(X), 1)
        m2, n, c = self.boson_massSq(X, T)
        y = np.sum((m2**(3/2) - (m2+1/3*self.h_boson**2 * T**2)**(3/2))*n, axis=-1)
    # (m2**2+1/3*self.h_boson**2 * T**2)**(3/2)
        y = T * y/(np.pi*12)
        y += np.sum(-1 * np.pi**2 * self.num_stanard_dof * T**4/90)
        return y

    def Vtot(self, X, T, include_radiation=False):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        #print('V1T', self.V1T(bosons, fermions, T, include_radiation))
        y += self.Daisy(X, T)
        return y
    def select_Tc(self):
        N = len(self.TcTrans)
        for i in range(N):
            if abs(self.TcTrans[i]['high_vev']) < 0.1:
                a = i
                break
        return a


    def entropy(self, dt, dphi): #find the high T phase entropy
        if self.TcTrans == [] or self.TcTrans == None:
            print("NO transition!! entropy is ill defined")
        else:
            k =self.select_Tc()
            Tcrit = float(self.TcTrans[k]['Tcrit'])
            Tcrit_highvev = float(self.TcTrans[k]['high_vev'])
            Tcrit_lowvev = float(self.TcTrans[k]['low_vev'])
            Tnuc = float(self.TnTrans[0]['Tnuc'])
            Tnuc_highvev = float(self.TnTrans[0]['high_vev'])
            Tnuc_lowvev = float(self.TnTrans[0]['low_vev'])
            ###### Tcrit highvev
            dvdt = (self.Vtot([Tcrit_highvev], [Tcrit + dt]) - self.Vtot([Tcrit_highvev], [Tcrit - dt])) / (2 * dt)
            dvdphi = (self.Vtot([Tcrit_highvev + dphi], [Tcrit]) - self.Vtot([Tcrit_highvev - dphi], [Tcrit])) / (2 * dphi)
            dphidt = (self.phases[0].valAt(Tcrit + dt) - self.phases[0].valAt(Tcrit - dt)) / (2 * dt)
            #dphidt = 0
            S_crit_highvev = -(dvdt + dphidt * dvdphi)
            #print('S1', S_crit_highvev)
            # dvdphi = self.gradV([Tcrit_highvev], Tcrit)
            # dphidt = -1 * self.dgradV_dT([Tcrit_highvev], Tcrit)/self.d2V([Tcrit_highvev], Tcrit)
            # S_crit_highvev = -(dvdt + dphidt * dvdphi)
            # print('S1', S_crit_highvev)

            ###### Tcrit lowvev
            dvdt = (self.Vtot([Tcrit_lowvev], [Tcrit + dt]) - self.Vtot([Tcrit_lowvev], [Tcrit - dt])) / (2 * dt)
            dvdphi = (self.Vtot([Tcrit_lowvev + dphi], [Tcrit]) - self.Vtot([Tcrit_lowvev - dphi], [Tcrit])) / (2 * dphi)
            dphidt = (self.phases[0].valAt(Tcrit + dt) - self.phases[0].valAt(Tcrit - dt)) / (2 * dt)
            #dphidt = 0
            S_crit_lowvev = -(dvdt + dphidt * dvdphi)
            #print('S2', S_crit_lowvev)
            # dvdphi = self.gradV([Tcrit_lowvev], Tcrit)
            # dphidt = -1 * self.dgradV_dT([Tcrit_lowvev], Tcrit) / self.d2V([Tcrit_lowvev], Tcrit)
            # S_crit_lowvev = -(dvdt + dphidt * dvdphi)
            # print('S2', S_crit_lowvev)

            #####Tnul high vev
            dvdt = (self.Vtot([Tnuc_highvev], [Tnuc + dt]) - self.Vtot([Tnuc_highvev], [Tnuc - dt])) / (2 * dt)
            dvdphi = (self.Vtot([Tnuc_highvev + dphi], [Tnuc]) - self.Vtot([Tnuc_highvev - dphi], [Tnuc])) / (2 * dphi)
            dphidt = (self.phases[0].valAt(Tnuc + dt) - self.phases[0].valAt(Tnuc - dt)) / (2 * dt)
            #dphidt = 0
            #print('dvdt', dvdt)
            #print('dvdphi', dvdphi)
            #print('dphidt', dphidt)
            S_nuc_highvev = -(dvdt + dphidt * dvdphi)
            #print('S3', S_nuc_highvev)
            # dvdphi = self.gradV([Tnuc_highvev], Tnuc)
            # dphidt = -1 * self.dgradV_dT([Tnuc_highvev], Tnuc) / self.d2V([Tnuc_highvev], Tnuc)
            # S_nuc_highvev = -(dvdt + dphidt * dvdphi)
            # print('S3', S_nuc_highvev)

            #####Tnul low vev
            dvdt = (self.Vtot([Tnuc_lowvev], [Tnuc + dt]) - self.Vtot([Tnuc_lowvev], [Tnuc - dt])) / (2 * dt)
            dvdphi = (self.Vtot([Tnuc_lowvev + dphi], [Tnuc]) - self.Vtot([Tnuc_lowvev - dphi], [Tnuc])) / (2 * dphi)
            dphidt = (self.phases[0].valAt(Tnuc + dt) - self.phases[0].valAt(Tnuc - dt)) / (2 * dt)
            # dvdphi = self.gradV([Tnuc_highvev], Tnuc)
            # dphidt = -1 * self.dgradV_dT([Tnuc_highvev], Tnuc) / self.d2V([Tnuc_highvev], Tnuc)
            S_nuc_lowvev = -(dvdt + dphidt * dvdphi)
            entropy = [S_crit_highvev, S_crit_lowvev, S_nuc_highvev]
            return entropy

    def dilution_factor(self, dt, dphi):
        if self.TcTrans == [] or self.TcTrans == None:
            print("NO transition!! entropy is ill defined")
        else:
            k = self.select_Tc()
            Tcrit = float(self.TcTrans[k]['Tcrit'])
            Tcrit_highvev = float(self.TcTrans[k]['high_vev'])
            Tcrit_lowvev = float(self.TcTrans[k]['low_vev'])
            Tnuc = float(self.TnTrans[0]['Tnuc'])
            Tnuc_highvev = float(self.TnTrans[0]['high_vev'])
            entropy = self.entropy(dt, dphi)
            #print('Tcrit_highvev', Tcrit_highvev)
            if (Tcrit - Tnuc) < 2:
                print('condition one')
                dilution_factor = entropy[0]/entropy[1]
            else:
                L = self.energyDensity(np.array([Tcrit_highvev]), np.array([Tcrit])) - self.energyDensity(np.array([Tcrit_lowvev]), np.array([Tcrit]))
                # print('he', self.energyDensity(np.array([Tcrit_highvev]), np.array([Tcrit])))
                # print('le', self.energyDensity(np.array([Tcrit_lowvev]), np.array([Tcrit])))
                diffpho = self.energyDensity(np.array([Tcrit_highvev]), np.array([Tcrit])) - self.energyDensity(np.array([Tnuc_highvev]), np.array([Tnuc]))
                f = diffpho/L
                # print('L', L)
                # print('diffpho', diffpho)
                # print('f', f)
                dilution_factor = (1 - f * (entropy[0] - entropy[1])/entropy[0]) * entropy[0] / ((1 - (entropy[0] - entropy[1])/entropy[0]) * entropy[2])
        return dilution_factor





# x = [246]
# T = [40]
# bosons = model.boson_massSq(x, T)
# fermions = model.fermion_massSq(x)
# print('bosons', bosons)
# print('fermions', fermions)
# print('V0: ', model.V0(x))
# print('V1: ', model.V1(bosons, fermions))
# print('Daisy', model.Daisy(x, T))
# print('Vtot: ', model.Vtot(x,T))

model = model()
model.findAllTransitions()
# print('                             ')
# print(model.TnTrans)
# print('                             ')
# print(model.TcTrans)
print(model.dilution_factor(1e-8, 1e-8))
print('         ')


# model.plotPhasesPhi()
# plt.axis([0,300,-50,550])
# plt.title("Minima as a function of temperature")
# plt.show()

# print(model.Daisy([30],[40]))
# print(model.Vtot([30], [40]))
# model.plot1d(-300, 300, 0, treelevel=False, subtract=False)
# plt.show()

# T = np.linspace(0, 50, 100)
# phi1 = []
# phi2 = []
# for i in range(len(T)):
#     phi1.append(model.Vtot([0.0004483], [T[i]]))
#     phi2.append(model.Vtot([-266.4914325], [T[i]]))
# plt.plot(T, phi1,'b')
# plt.plot(T, phi2, 'r')
# plt.show()




#
#
# X=np.linspace(-600,600, 1000)
# V = []
# for i in range(len(X)):
#     V.append(model.Vtot([X[i]], [36.83]))
#
# plt.plot(X,V)
# plt.show()





#print(model.Vtot([29.89], [-0.0042]))
#print(model.energyDensity(np.array([20]), np.array([20])))
# for key, val in model.TcTrans[0].items():
#     if key != 'instanton':
#         print(key,  ":  ",  val)


# if len(model.TnTrans) != 0:
#     for key, val in model.TnTrans[0].items():
#         if key != 'instanton':
#             print(key,  ":  ",  val)
# else:
#     print('NO transition!!')

#
# model.plot1d(0, 600, 124.54, treelevel=False, subtract=True)
# model.plot1d(0, 600, 126.14, treelevel=False, subtract=True)
# model.plot1d(0, 600, 30, treelevel=False, subtract=True)
# model.plot1d(0, 600, 42, treelevel=False, subtract=True)
# model.plot1d(0, 600, 50, treelevel=False, subtract=True)
#plt.show()


