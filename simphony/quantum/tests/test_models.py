# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

from simphony.quantum import QuantumBeamSplitter, CoherentState, SqueezedState, compose_qstate
import numpy as np


class TestQBS:
    bs_theta = np.pi / 4  # np.pi/4 is normal
    bs_phi = np.pi / 2  # np.pi/2 is normal
    T0, T1, T2, T3 = (0.4, 0.3, 0.2, 0.1)

    bs = QuantumBeamSplitter(theta=bs_theta, phi=bs_phi, T=(T0, T1, T2, T3))

    alpha = 4
    phi = np.pi / 4  # 0 is normal
    compl = alpha * np.exp(1j * phi)
    compl2 = alpha * 2 * np.exp(1j * phi * 4)
    compl_vac = 0+0j

    dsqueezed1 = SqueezedState(r=1.01, phi=7, alpha=compl, pin=bs.pins["in1"])
    dsqueezed2 = SqueezedState(r=1.01, phi=np.pi / 2, alpha=compl, pin=bs.pins["in2"])
    coherent1 = CoherentState(alpha=compl, pin=bs.pins["in1"])
    coherent2 = CoherentState(alpha=compl2, pin=bs.pins["in2"])
    squeezed = SqueezedState(r=2, phi=np.pi/4, alpha=compl_vac, pin=bs.pins["in2"])
    vacuum = CoherentState(alpha=compl_vac, pin=bs.pins["in2"])

    def test_coherent(self):
        input_state = compose_qstate(self.coherent1, self.coherent2)

        transform, qstate = self.bs.quantum_transform(input=input_state, freqs=np.array([0]))

        transform_expected = np.array(
            [
                [2.000000000000000389e-01,-2.785350134042221088e-17,-0.000000000000000000e+00,-1.732050807568877304e-01],
                [8.659560562354935323e-18,1.224744871391589135e-01,-1.414213562373095312e-01,-0.000000000000000000e+00],
                [0.000000000000000000e+00,1.732050807568877304e-01,2.000000000000000389e-01,-2.785350134042221088e-17],
                [1.414213562373095312e-01,0.000000000000000000e+00,8.659560562354935323e-18,1.224744871391589135e-01]
            ]
        )
        output_means_expected = np.array(
            [
                5.656854249492381248e-01,
                -1.379795897113271330e+00,
                -8.199552211058637186e-01,
                4.000000000000001887e-01
            ]
        )
        output_covs_expected = np.array(
            [
                [2.500000000000000000e-01,-4.198577948067745477e-19,7.040499976020820914e-35,3.252606517456513302e-18],
                [-4.198577948067745477e-19,2.500000000000000000e-01,-3.469446951953614189e-18,1.341196071044629494e-36],
                [7.040499976020820914e-35,-3.469446951953614189e-18,2.500000000000000000e-01,-4.198577948067745477e-19],
                [3.252606517456513302e-18,1.341196071044629494e-36,-4.198577948067745477e-19,2.500000000000000000e-01]
            ]
        )

        assert np.allclose(transform, transform_expected)
        assert np.allclose(qstate[0].means, output_means_expected)
        assert np.allclose(qstate[0].cov, output_covs_expected)


    def test_displaced_squeezed(self):
        input_state = compose_qstate(self.dsqueezed1, self.dsqueezed2)

        transform, qstate = self.bs.quantum_transform(input=input_state, freqs=np.array([0]))

        transform_expected = np.array(
            [
                [2.000000000000000389e-01,-2.785350134042221088e-17,-0.000000000000000000e+00,-1.732050807568877304e-01],
                [8.659560562354935323e-18,1.224744871391589135e-01,-1.414213562373095312e-01,-0.000000000000000000e+00],
                [0.000000000000000000e+00,1.732050807568877304e-01,2.000000000000000389e-01,-2.785350134042221088e-17],
                [1.414213562373095312e-01,0.000000000000000000e+00,8.659560562354935323e-18,1.224744871391589135e-01]

            ]
        )
        output_means_expected = np.array(
            [
                7.578747639260241531e-02,
                -5.358983848622456136e-02,
                1.055583373505873723e+00,
                7.464101615137755941e-01
            ]
        )
        output_covs_expected = np.array(
            [
                [2.717053239525668573e-01,3.683912378605301574e-02,3.444132530186051235e-03,-1.472693346473784304e-02],
                [3.683912378605301574e-02,2.887684165127094582e-01,-2.475190520455408660e-02,-1.722066265093030171e-03],
                [3.444132530186053403e-03,-2.475190520455408660e-02,3.275368330254189164e-01,-3.683912378605301574e-02],
                [-1.472693346473784477e-02,-1.722066265093030388e-03,-3.683912378605301574e-02,2.608526619762834287e-01]
            ]
        )
        assert np.allclose(transform, transform_expected)
        assert np.allclose(qstate[0].means, output_means_expected)
        assert np.allclose(qstate[0].cov, output_covs_expected)


