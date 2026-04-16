package it.univr.fima.correction;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import net.finmath.functions.AnalyticFormulas;
import it.univr.fima.correction.BarrierOptions.DoubleBarrierType;

public class BarrierOptionsDoubleBarrierParityTest {

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.01;
    private static final double SIGMA = 0.20;
    private static final double T = 1.0;

    private static final double K = 100.0;
    private static final double L = 80.0;
    private static final double U = 120.0;

    private static final int SERIES = 12;
    private static final double TOL = 1E-8;

    @Test
    public void testCallDoubleBarrierInOutParity() {
        assertDoubleBarrierInOutParity(true);
    }

    @Test
    public void testPutDoubleBarrierInOutParity() {
        assertDoubleBarrierInOutParity(false);
    }

    private static void assertDoubleBarrierInOutParity(final boolean isCall) {
        final double knockOut = BarrierOptions.blackScholesDoubleBarrierOptionValue(
                S0, R, Q, SIGMA, T, K, isCall, L, U, DoubleBarrierType.KNOCK_OUT, SERIES);

        final double knockIn = BarrierOptions.blackScholesDoubleBarrierOptionValue(
                S0, R, Q, SIGMA, T, K, isCall, L, U, DoubleBarrierType.KNOCK_IN, SERIES);

        final double forward = S0 * Math.exp((R - Q) * T);
        final double payoffUnit = Math.exp(-R * T);
        final double vanillaCall = AnalyticFormulas.blackScholesGeneralizedOptionValue(
                forward, SIGMA, T, K, payoffUnit);
        final double vanilla = isCall
                ? vanillaCall
                : vanillaCall - payoffUnit * (forward - K);

        assertEquals(vanilla, knockIn + knockOut, TOL);
    }
}