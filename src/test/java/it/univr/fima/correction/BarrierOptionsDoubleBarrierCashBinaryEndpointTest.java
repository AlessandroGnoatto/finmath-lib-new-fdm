package it.univr.fima.correction;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import it.univr.fima.correction.BarrierOptions.DoubleBarrierType;

public class BarrierOptionsDoubleBarrierCashBinaryEndpointTest {

    private static final double R = 0.05;
    private static final double Q = 0.01;
    private static final double SIGMA = 0.20;
    private static final double T = 1.0;

    private static final double CASH = 10.0;
    private static final double L = 80.0;
    private static final double U = 120.0;

    private static final int MAX_ITER = 200;
    private static final double CONV = 1E-10;
    private static final double TOL = 1E-12;

    @Test
    public void testKnockOutOutsideBandIsZero() {
        assertEquals(0.0, value(75.0, DoubleBarrierType.KNOCK_OUT), TOL);
        assertEquals(0.0, value(125.0, DoubleBarrierType.KNOCK_OUT), TOL);
    }

    @Test
    public void testKnockInOutsideBandPaysCash() {
        assertEquals(CASH, value(75.0, DoubleBarrierType.KNOCK_IN), TOL);
        assertEquals(CASH, value(125.0, DoubleBarrierType.KNOCK_IN), TOL);
    }

    @Test
    public void testKikoEndpointBehavior() {
        assertEquals(CASH, value(75.0, DoubleBarrierType.KIKO), TOL);
        assertEquals(0.0, value(125.0, DoubleBarrierType.KIKO), TOL);
    }

    @Test
    public void testKokiEndpointBehavior() {
        assertEquals(0.0, value(75.0, DoubleBarrierType.KOKI), TOL);
        assertEquals(CASH, value(125.0, DoubleBarrierType.KOKI), TOL);
    }

    private static double value(final double s0, final DoubleBarrierType type) {
        return BarrierOptions.blackScholesDoubleBarrierCashBinaryValue(
                s0, R, Q, SIGMA, T, CASH, L, U, type, MAX_ITER, CONV);
    }
}