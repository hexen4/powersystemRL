from numpy import complex128, float64, int32
from numpy.core.multiarray import zeros, empty, array
from scipy.sparse import csr_matrix as sparse, vstack, hstack
from pandapower.pypower.dSbus_dV import dSbus_dV



def _create_J_without_numba2(Ybus, V, pvpq, pq):
    # create Jacobian with standard pypower implementation.
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V) #Computes partial derivatives of power injection w.r.t. voltage -> sparse and dense

    ## evaluate Jacobian
    J11 = dS_dVa[array([pvpq]).T, pvpq].real
    J12 = dS_dVm[array([pvpq]).T, pq].real
    if len(pq) > 0:
        J21 = dS_dVa[array([pq]).T, pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag
        J = vstack([
            hstack([J11, J12]),
            hstack([J21, J22])
        ], format="csr")
    else:
        J = vstack([
            hstack([J11, J12])
        ], format="csr")
    return J


def create_jacobian_matrix(Ybus, V, pvpq, pq):
    J = _create_J_without_numba2(Ybus, V, pvpq, pq)
    return J

