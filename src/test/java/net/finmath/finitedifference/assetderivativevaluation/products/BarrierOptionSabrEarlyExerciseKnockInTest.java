package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for direct 2D SABR knock-ins with early exercise.
 *
 * <p>
 * The test suite verifies:
 * </p>
 * <ul>
 *   <li>one-date Bermudan knock-in equals European knock-in,</li>
 *   <li>European <= Bermudan <= American for the knock-in itself,</li>
 *   <li>Bermudan knock-in <= Bermudan vanilla,</li>
 *   <li>in the already-hit region, American knock-in = American vanilla.</li>
 * </ul>
 *
 * <p>
 * This mirrors the Heston early-exercise structural test, but for
 * {@link FDMSabrModel}.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionSabrEarlyExerciseKnockInTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double REBATE = 0.0;

    private static final double SPOT = 100.0;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.02;

    private static final double INITIAL_VOLATILITY = 0.20;
    private static final double BETA = 1.0;
    private static final double NU = 0.30;
    private static final double RHO = -0.50;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 120;
    private static final int NUMBER_OF_SPACE_STEPS_S = 180;
    private static final int NUMBER_OF_SPACE_STEPS_VOL = 100;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

    private static final double EQUALITY_TOLERANCE = 7.5E-2;
    private static final double ORDERING_TOLERANCE = 5.0E-3;
    private static final double DOMINANCE_TOLERANCE = 7.5E-2;
    private static final double ALREADY_HIT_TOLERANCE = 1.0E-10;

    private static final double DOWN_ALREADY_HIT_SPOT = 78.0;
    private static final double UP_ALREADY_HIT_SPOT = 122.0;

    @Test
    public void testOneDateBermudanEqualsEuropeanDownInCall() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOneDateBermudanEqualsEuropeanDownInPut() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOneDateBermudanEqualsEuropeanUpInCall() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testOneDateBermudanEqualsEuropeanUpInPut() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testOrderingDownInCall() {
        runOrderingTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOrderingDownInPut() {
        runOrderingTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOrderingUpInCall() {
        runOrderingTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testOrderingUpInPut() {
        runOrderingTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testBermudanDownInCallLessOrEqualBermudanVanillaCall() {
        runBermudanDominanceTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testBermudanDownInPutLessOrEqualBermudanVanillaPut() {
        runBermudanDominanceTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testBermudanUpInCallLessOrEqualBermudanVanillaCall() {
        runBermudanDominanceTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testBermudanUpInPutLessOrEqualBermudanVanillaPut() {
        runBermudanDominanceTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testAmericanDownInCallMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testAmericanDownInPutMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testAmericanUpInCallMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.CALL, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testAmericanUpInPutMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.PUT, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    private void runOneDateBermudanEqualsEuropeanTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier) {

        final TestSetup setup = createSetup(barrier, barrierType);

        final BarrierOption european = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType
        );

        final BarrierOption bermudan = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType,
                new BermudanExercise(new double[] { MATURITY })
        );

        final double europeanPrice = interpolateAt(
                european.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        final double bermudanPrice = interpolateAt(
                bermudan.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        System.out.println("One-date Bermudan vs European");
        System.out.println("Type            = " + barrierType + " " + callOrPut);
        System.out.println("European        = " + europeanPrice);
        System.out.println("One-date Berm   = " + bermudanPrice);
        System.out.println("Abs error       = " + Math.abs(europeanPrice - bermudanPrice));

        assertEquals(europeanPrice, bermudanPrice, EQUALITY_TOLERANCE);
    }

    private void runOrderingTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier) {

        final TestSetup setup = createSetup(barrier, barrierType);

        final double[] bermudanExerciseTimes = new double[] { 0.25, 0.50, 0.75, 1.00 };

        final BarrierOption european = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType
        );

        final BarrierOption bermudan = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType,
                new BermudanExercise(bermudanExerciseTimes)
        );

        final BarrierOption american = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType,
                new AmericanExercise(MATURITY)
        );

        final double europeanPrice = interpolateAt(
                european.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        final double bermudanPrice = interpolateAt(
                bermudan.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        final double americanPrice = interpolateAt(
                american.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        System.out.println("Knock-in ordering");
        System.out.println("Type            = " + barrierType + " " + callOrPut);
        System.out.println("European        = " + europeanPrice);
        System.out.println("Bermudan        = " + bermudanPrice);
        System.out.println("American        = " + americanPrice);

        assertTrue(bermudanPrice + ORDERING_TOLERANCE >= europeanPrice);
        assertTrue(americanPrice + ORDERING_TOLERANCE >= bermudanPrice);
    }

    private void runBermudanDominanceTest(
            final CallOrPut callOrPut,
            final BarrierType knockInType,
            final double barrier) {

        final TestSetup setup = createSetup(barrier, knockInType);

        final double[] bermudanExerciseTimes = new double[] { 0.25, 0.50, 0.75, 1.00 };

        final BarrierOption knockIn = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                knockInType,
                new BermudanExercise(bermudanExerciseTimes)
        );

        final BermudanOption vanilla = new BermudanOption(
                bermudanExerciseTimes,
                STRIKE,
                callOrPut
        );

        final double knockInPrice = interpolateAt(
                knockIn.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        final double vanillaPrice = interpolateAt(
                vanilla.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                SPOT,
                INITIAL_VOLATILITY
        );

        System.out.println("Bermudan knock-in dominance");
        System.out.println("Type            = " + knockInType + " " + callOrPut);
        System.out.println("Knock-in        = " + knockInPrice);
        System.out.println("Vanilla         = " + vanillaPrice);

        assertTrue(knockInPrice <= vanillaPrice + DOMINANCE_TOLERANCE);
    }

    private void runAmericanAlreadyHitConsistencyTest(
            final CallOrPut callOrPut,
            final BarrierType knockInType,
            final double barrier,
            final double alreadyHitSpot) {

        final TestSetup setup = createSetup(barrier, knockInType);
        final Exercise americanExercise = new AmericanExercise(MATURITY);

        final BarrierOption knockIn = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                knockInType,
                americanExercise
        );

        final AmericanOption vanilla = new AmericanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final double knockInPrice = interpolateAt(
                knockIn.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                alreadyHitSpot,
                INITIAL_VOLATILITY
        );

        final double vanillaPrice = interpolateAt(
                vanilla.getValue(0.0, setup.model),
                setup.sNodes,
                setup.volNodes,
                alreadyHitSpot,
                INITIAL_VOLATILITY
        );

        System.out.println("American already-hit consistency");
        System.out.println("Type            = " + knockInType + " " + callOrPut);
        System.out.println("Already-hit spot= " + alreadyHitSpot);
        System.out.println("Knock-in        = " + knockInPrice);
        System.out.println("Vanilla         = " + vanillaPrice);
        System.out.println("Abs error       = " + Math.abs(knockInPrice - vanillaPrice));

        assertEquals(vanillaPrice, knockInPrice, ALREADY_HIT_TOLERANCE);
    }

    private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

        final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
                0.0,
                NUMBER_OF_TIME_STEPS,
                MATURITY / NUMBER_OF_TIME_STEPS
        );

        final Grid sGrid = createSpotGrid(barrier, barrierType);

        final double volMax = Math.max(4.0 * INITIAL_VOLATILITY, 1.0);
        final Grid volGrid = new UniformGrid(
                NUMBER_OF_SPACE_STEPS_VOL,
                0.0,
                volMax
        );

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, volGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, INITIAL_VOLATILITY }
        );

        final FDMSabrModel model = new FDMSabrModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                INITIAL_VOLATILITY,
                BETA,
                NU,
                RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), volGrid.getGrid());
    }

    private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

        final double deltaS = Math.abs(barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;

        final boolean isKnockIn =
                barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN;

        if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {

            final double sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
            final double sMax = Math.max(3.0 * SPOT, SPOT + 12.0 * deltaS);

            final int numberOfSteps = Math.max(
                    NUMBER_OF_SPACE_STEPS_S,
                    (int)Math.round((sMax - sMin) / deltaS)
            );

            if(isKnockIn) {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier,
                        BARRIER_CLUSTERING_EXPONENT
                );
            }
            else {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier
                );
            }
        }
        else {
            final double sMin = 0.0;
            final double sMax = barrier + 8.0 * deltaS;

            final int numberOfSteps = Math.max(
                    NUMBER_OF_SPACE_STEPS_S,
                    (int)Math.round((sMax - sMin) / deltaS)
            );

            if(isKnockIn) {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier,
                        BARRIER_CLUSTERING_EXPONENT
                );
            }
            else {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier
                );
            }
        }
    }

    private double interpolateAt(
            final double[] values,
            final double[] sNodes,
            final double[] volNodes,
            final double spot,
            final double volatility) {

        assertTrue("Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12);
        assertTrue("Volatility must lie inside the grid domain.",
                volatility >= volNodes[0] - 1E-12 && volatility <= volNodes[volNodes.length - 1] + 1E-12);

        final int nS = sNodes.length;
        final int nVol = volNodes.length;

        final double[][] valueSurface = new double[nS][nVol];
        for(int j = 0; j < nVol; j++) {
            for(int i = 0; i < nS; i++) {
                valueSurface[i][j] = values[flatten(i, j, nS)];
            }
        }

        final BiLinearInterpolation interpolation = new BiLinearInterpolation(sNodes, volNodes, valueSurface);
        return interpolation.apply(spot, volatility);
    }

    private int flatten(final int iS, final int iVol, final int numberOfSNodes) {
        return iS + iVol * numberOfSNodes;
    }

    private static class TestSetup {

        private final FDMSabrModel model;
        private final double[] sNodes;
        private final double[] volNodes;

        private TestSetup(
                final FDMSabrModel model,
                final double[] sNodes,
                final double[] volNodes) {
            this.model = model;
            this.sNodes = sNodes;
            this.volNodes = volNodes;
        }
    }
}