package it.univr.fima.correction;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import it.univr.fima.correction.BarrierOptions.BarrierType;

public class BarrierOptionsSoftBarrierCollapseTest {

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.01;
    private static final double SIGMA = 0.20;
    private static final double T = 1.0;

    private static final double K = 100.0;
    private static final double H = 90.0;
    private static final double REBATE = 0.0;

    private static final double TOL = 1E-12;

    @Test
    public void testDownOutSoftBarrierCollapsesToStandardBarrier() {
        final double standard = BarrierOptions.blackScholesBarrierOptionValue(
                S0, R, Q, SIGMA, T, K, true, REBATE, H, BarrierType.DOWN_OUT);

        final double soft = BarrierOptions.blackScholesSoftBarrierOptionValue(
                S0, R, Q, SIGMA, T, K, true, H, H, BarrierType.DOWN_OUT);

        assertEquals(standard, soft, TOL);
    }

    @Test
    public void testUpInSoftBarrierCollapsesToStandardBarrier() {
        final double standard = BarrierOptions.blackScholesBarrierOptionValue(
                S0, R, Q, SIGMA, T, K, false, REBATE, 110.0, BarrierType.UP_IN);

        final double soft = BarrierOptions.blackScholesSoftBarrierOptionValue(
                S0, R, Q, SIGMA, T, K, false, 110.0, 110.0, BarrierType.UP_IN);

        assertEquals(standard, soft, TOL);
    }
}