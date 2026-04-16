package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import it.univr.fima.correction.BarrierOptions;
import it.univr.fima.correction.BarrierOptions.BinaryPayoffType;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.TouchSettlementTiming;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Combined regression / sanity test for one-touch AT_HIT under Black-Scholes.
 *
 * Covered in one class:
 * <ul>
 *   <li>analytic regression for DOWN_IN and UP_IN one-touch AT_HIT,</li>
 *   <li>degenerate and invalid-usage cases for AT_HIT,</li>
 *   <li>timing consistency: AT_HIT >= AT_EXPIRY for positive rates.</li>
 * </ul>
 */
public class TouchOptionAtHitBlackScholesCombinedTest {

    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS = 300;

    /**
     * Desired number of space intervals between barrier and spot.
     */
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    /**
     * Default interior extension for knock-in grids.
     */
    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;

    private static final double ANALYTIC_TOLERANCE = 0.40;
    private static final double VALUE_TOLERANCE = 5E-2;
    private static final double ZERO_TOLERANCE = 1E-8;
    private static final double TIMING_TOLERANCE = 1E-8;

    @Test
    public void testDownAndInOneTouchAtHitBlackScholesFiniteDifferenceVsAnalytic() {
        runAnalyticRegressionTest(BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testUpAndInOneTouchAtHitBlackScholesFiniteDifferenceVsAnalytic() {
        runAnalyticRegressionTest(BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testAlreadyHitAtValuationPaysImmediateCash() {
        final FDMBlackScholesModel model = createBlackScholesModel(
                new UniformGrid(NUMBER_OF_SPACE_STEPS, 0.0, 200.0),
                1.0
        );

        final TouchOption upAndInAlreadyHit = TouchOption.oneTouchAtHit(
                1.0,
                90.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption downAndInAlreadyHit = TouchOption.oneTouchAtHit(
                1.0,
                110.0,
                BarrierType.DOWN_IN,
                CASH_PAYOFF
        );

        final double upAndInValue = valueAtSpot(upAndInAlreadyHit, model, S0);
        final double downAndInValue = valueAtSpot(downAndInAlreadyHit, model, S0);

        assertEquals(CASH_PAYOFF, upAndInValue, VALUE_TOLERANCE);
        assertEquals(CASH_PAYOFF, downAndInValue, VALUE_TOLERANCE);
    }

    @Test
    public void testMaturityZeroAtHitEndpointValues() {
        final FDMBlackScholesModel model = createBlackScholesModel(
                new UniformGrid(NUMBER_OF_SPACE_STEPS, 0.0, 200.0),
                1.0
        );

        final TouchOption oneTouchHitAtMaturity = TouchOption.oneTouchAtHit(
                0.0,
                90.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption oneTouchNotHitAtMaturity = TouchOption.oneTouchAtHit(
                0.0,
                110.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final double oneTouchHitValue = valueAtSpot(oneTouchHitAtMaturity, model, S0);
        final double oneTouchNotHitValue = valueAtSpot(oneTouchNotHitAtMaturity, model, S0);

        assertEquals(CASH_PAYOFF, oneTouchHitValue, VALUE_TOLERANCE);
        assertEquals(0.0, oneTouchNotHitValue, ZERO_TOLERANCE);
    }

    @Test
    public void testAtHitRejectsOutBarrierTypesThroughConstructor() {
        expectIllegalArgument(() ->
            new TouchOption(
                    MATURITY,
                    80.0,
                    BarrierType.DOWN_OUT,
                    CASH_PAYOFF,
                    TouchSettlementTiming.AT_HIT
            )
        );

        expectIllegalArgument(() ->
            new TouchOption(
                    MATURITY,
                    120.0,
                    BarrierType.UP_OUT,
                    CASH_PAYOFF,
                    TouchSettlementTiming.AT_HIT
            )
        );
    }

    @Test
    public void testBarrierOutsideGridThrows() {
        final FDMBlackScholesModel model = createBlackScholesModel(
                new UniformGrid(NUMBER_OF_SPACE_STEPS, 0.0, 150.0),
                1.0
        );

        final TouchOption option = TouchOption.oneTouchAtHit(
                1.0,
                180.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        expectIllegalArgument(() -> option.getValue(0.0, model));
    }

    @Test
    public void testAtHitDominatesAtExpiryDownIn() {
        runTimingConsistencyTest(BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testAtHitDominatesAtExpiryUpIn() {
        runTimingConsistencyTest(BarrierType.UP_IN, 120.0);
    }

    private void runAnalyticRegressionTest(
            final BarrierType barrierType,
            final double barrier) {

        final Grid sGrid = createKnockInInteriorGrid(barrier, barrierType);
        final FDMBlackScholesModel model = createBlackScholesModel(sGrid, MATURITY);

        final TouchOption product = TouchOption.oneTouchAtHit(
                MATURITY,
                barrier,
                barrierType,
                CASH_PAYOFF
        );

        final double[] fdValuesOnGrid = product.getValue(0.0, model);
        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        final double fdPrice = interpolateAtSpot(fdValuesOnGrid, sNodes, S0);

        final double analyticPrice = BarrierOptions.blackScholesBinaryBarrierAtHitValue(
                S0,
                R,
                Q,
                SIGMA,
                MATURITY,
                barrier,
                mapBarrierType(barrierType),
                BinaryPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF
        );

        System.out.println("AT_HIT analytic regression");
        System.out.println("Barrier type    = " + barrierType);
        System.out.println("Barrier         = " + barrier);
        System.out.println("FD price        = " + fdPrice);
        System.out.println("Analytic price  = " + analyticPrice);

        assertTrue(fdPrice >= -1E-10);
        assertTrue(analyticPrice >= -1E-10);

        assertEquals(
                "FD vs analytic one-touch AT_HIT for " + barrierType,
                analyticPrice,
                fdPrice,
                ANALYTIC_TOLERANCE
        );
    }

    private void runTimingConsistencyTest(
            final BarrierType barrierType,
            final double barrier) {

        final Grid sGrid = createKnockInInteriorGrid(barrier, barrierType);
        final FDMBlackScholesModel model = createBlackScholesModel(sGrid, MATURITY);

        final TouchOption atHit = TouchOption.oneTouchAtHit(
                MATURITY,
                barrier,
                barrierType,
                CASH_PAYOFF
        );

        final TouchOption atExpiry = TouchOption.oneTouchAtExpiry(
                MATURITY,
                barrier,
                barrierType,
                CASH_PAYOFF
        );

        final double atHitValue = valueAtSpot(atHit, model, S0);
        final double atExpiryValue = valueAtSpot(atExpiry, model, S0);

        System.out.println("Timing consistency");
        System.out.println("Barrier type    = " + barrierType);
        System.out.println("Barrier         = " + barrier);
        System.out.println("AT_HIT          = " + atHitValue);
        System.out.println("AT_EXPIRY       = " + atExpiryValue);

        assertTrue(
                "For positive rates, AT_HIT should dominate AT_EXPIRY for " + barrierType,
                atHitValue + TIMING_TOLERANCE >= atExpiryValue
        );
    }

    private FDMBlackScholesModel createBlackScholesModel(
            final Grid sGrid,
            final double finalTime) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        finalTime / NUMBER_OF_TIME_STEPS
                );

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { S0 }
        );

        return new FDMBlackScholesModel(
                S0,
                riskFreeCurve,
                dividendCurve,
                SIGMA,
                spaceTime
        );
    }

    private Grid createKnockInInteriorGrid(
            final double barrier,
            final BarrierType barrierType) {

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_IN) {
            final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;

            final int maxExtraStepsBelowBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsBelowBarrier =
                    Math.max(1, Math.min(DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS, maxExtraStepsBelowBarrier));

            final double sMin = barrier - extraStepsBelowBarrier * deltaS;
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;

            validateInteriorBarrierPlacement(sMin, barrier, deltaS);

            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.UP_IN) {
            final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;

            final int maxExtraStepsAboveBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsAboveBarrier =
                    Math.max(1, Math.min(DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS, maxExtraStepsAboveBarrier));

            final double sMax = barrier + extraStepsAboveBarrier * deltaS;
            final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;

            validateInteriorBarrierPlacement(sMin, barrier, deltaS);

            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else {
            throw new IllegalArgumentException("AT_HIT one-touch requires DOWN_IN or UP_IN.");
        }
    }

    private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
        final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(S0 - barrier)));
        final int cappedByGrid = NUMBER_OF_SPACE_STEPS - 2;
        return Math.max(1, Math.min(Math.min(STEPS_BETWEEN_BARRIER_AND_SPOT, cappedByGrid), naturalSteps));
    }

    private void validateInteriorBarrierPlacement(
            final double sMin,
            final double barrier,
            final double deltaS) {

        final double barrierIndex = (barrier - sMin) / deltaS;
        final long roundedBarrierIndex = Math.round(barrierIndex);

        if(Math.abs(barrierIndex - roundedBarrierIndex) > 1E-10) {
            throw new IllegalStateException("Barrier is not located on a grid node.");
        }
        if(roundedBarrierIndex <= 0 || roundedBarrierIndex >= NUMBER_OF_SPACE_STEPS) {
            throw new IllegalStateException("Barrier is not located on an interior grid node.");
        }
    }

    private double valueAtSpot(
            final TouchOption option,
            final FDMBlackScholesModel model,
            final double spot) {

        final double[] values = option.getValue(0.0, model);
        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        return interpolateAtSpot(values, sNodes, spot);
    }

    private double interpolateAtSpot(
            final double[] values,
            final double[] sNodes,
            final double spot) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );

        final int spotIndex = getGridIndex(sNodes, spot);
        if(spotIndex >= 0) {
            return values[spotIndex];
        }

        final PolynomialSplineFunction interpolation =
                new LinearInterpolator().interpolate(sNodes, values);
        return interpolation.value(spot);
    }

    private int getGridIndex(final double[] grid, final double value) {
        for(int i = 0; i < grid.length; i++) {
            if(Math.abs(grid[i] - value) < 1E-12) {
                return i;
            }
        }
        return -1;
    }

    private static BarrierOptions.BarrierType mapBarrierType(final BarrierType barrierType) {
        switch(barrierType) {
        case DOWN_IN:
            return BarrierOptions.BarrierType.DOWN_IN;
        case UP_IN:
            return BarrierOptions.BarrierType.UP_IN;
        default:
            throw new IllegalArgumentException("Unsupported one-touch AT_HIT barrier type: " + barrierType);
        }
    }

    private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
        final double[] times = new double[] { 0.0, 1.0 };
        final double[] zeroRates = new double[] { rate, rate };

        return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
                name,
                LocalDate.of(2010, 8, 1),
                times,
                zeroRates,
                InterpolationMethod.LINEAR,
                ExtrapolationMethod.CONSTANT,
                InterpolationEntity.VALUE
        );
    }

    private void expectIllegalArgument(final ThrowingRunnable runnable) {
        try {
            runnable.run();
            fail("Expected IllegalArgumentException.");
        }
        catch(final IllegalArgumentException expected) {
            // expected
        }
        catch(final Exception other) {
            fail("Expected IllegalArgumentException but got: " + other.getClass().getName());
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}