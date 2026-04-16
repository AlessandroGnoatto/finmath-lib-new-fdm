package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.BarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression test for one-touch AT_HIT under the Bachelier model.
 *
 * <p>
 * This class checks:
 * </p>
 * <ul>
 *   <li>non-negativity,</li>
 *   <li>AT_HIT bounded above by the immediate cash payoff,</li>
 *   <li>AT_HIT dominates AT_EXPIRY for positive rates,</li>
 *   <li>already-hit-at-valuation returns immediate cash,</li>
 *   <li>maturity-zero endpoint behavior.</li>
 * </ul>
 */
public class TouchOptionAtHitBachelierStructuralTest {

    private static final double SPOT = 100.0;
    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double VOLATILITY = 20.0;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.02;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 200;
    private static final int NUMBER_OF_SPACE_STEPS = 400;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;

    private static final double NON_NEGATIVITY_TOLERANCE = 1E-10;
    private static final double UPPER_BOUND_TOLERANCE = 1E-6;
    private static final double VALUE_TOLERANCE = 5E-2;
    private static final double TIMING_TOLERANCE = 1E-8;
    private static final double ZERO_TOLERANCE = 1E-8;

    @Test
    public void testDownBarrierStructure() {
        runStructuralTest(80.0, BarrierType.DOWN_IN);
    }

    @Test
    public void testUpBarrierStructure() {
        runStructuralTest(120.0, BarrierType.UP_IN);
    }

    @Test
    public void testAlreadyHitAtValuationPaysImmediateCash() {
        final FDMBachelierModel model = createModel(new UniformGrid(NUMBER_OF_SPACE_STEPS, 0.0, 200.0), MATURITY);

        final TouchOption upAndInAlreadyHit = TouchOption.oneTouchAtHit(
                MATURITY,
                90.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption downAndInAlreadyHit = TouchOption.oneTouchAtHit(
                MATURITY,
                110.0,
                BarrierType.DOWN_IN,
                CASH_PAYOFF
        );

        final double upAndInValue = valueAtSpot(upAndInAlreadyHit, model, SPOT);
        final double downAndInValue = valueAtSpot(downAndInAlreadyHit, model, SPOT);

        assertEquals(CASH_PAYOFF, upAndInValue, VALUE_TOLERANCE);
        assertEquals(CASH_PAYOFF, downAndInValue, VALUE_TOLERANCE);
    }

    @Test
    public void testMaturityZeroEndpointValues() {
        final FDMBachelierModel model = createModel(new UniformGrid(NUMBER_OF_SPACE_STEPS, 0.0, 200.0), MATURITY);

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

        final double oneTouchHitValue = valueAtSpot(oneTouchHitAtMaturity, model, SPOT);
        final double oneTouchNotHitValue = valueAtSpot(oneTouchNotHitAtMaturity, model, SPOT);

        assertEquals(CASH_PAYOFF, oneTouchHitValue, VALUE_TOLERANCE);
        assertEquals(0.0, oneTouchNotHitValue, ZERO_TOLERANCE);
    }

    private void runStructuralTest(final double barrier, final BarrierType barrierType) {
        final TestSetup setup = createBarrierSetup(barrier, barrierType);

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

        final double atHitValue = interpolateAtSpot(
                atHit.getValue(0.0, setup.model),
                setup.sNodes,
                SPOT
        );

        final double atExpiryValue = interpolateAtSpot(
                atExpiry.getValue(0.0, setup.model),
                setup.sNodes,
                SPOT
        );

        System.out.println("Bachelier one-touch AT_HIT structural regression");
        System.out.println("Barrier         = " + barrier);
        System.out.println("Barrier type    = " + barrierType);
        System.out.println("AT_HIT          = " + atHitValue);
        System.out.println("AT_EXPIRY       = " + atExpiryValue);
        System.out.println("Cash payoff     = " + CASH_PAYOFF);

        assertTrue(
                "AT_HIT must be non-negative for " + barrierType,
                atHitValue >= -NON_NEGATIVITY_TOLERANCE
        );

        assertTrue(
                "AT_HIT must not exceed immediate cash payoff for " + barrierType,
                atHitValue <= CASH_PAYOFF + UPPER_BOUND_TOLERANCE
        );

        assertTrue(
                "For positive rates, AT_HIT should dominate AT_EXPIRY for " + barrierType,
                atHitValue + TIMING_TOLERANCE >= atExpiryValue
        );
    }

    private TestSetup createBarrierSetup(final double barrier, final BarrierType barrierType) {
        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createBarrierGrid(barrier, barrierType);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { SPOT }
        );

        final FDMBachelierModel model = new FDMBachelierModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                VOLATILITY,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid());
    }

    private Grid createBarrierGrid(final double barrier, final BarrierType barrierType) {
        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_IN) {
            final double deltaS = (SPOT - barrier) / effectiveStepsBetweenBarrierAndSpot;

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
            final double deltaS = (barrier - SPOT) / effectiveStepsBetweenBarrierAndSpot;

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
        final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(SPOT - barrier)));
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

    private FDMBachelierModel createModel(final Grid sGrid, final double finalTime) {
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
                new double[] { SPOT }
        );

        return new FDMBachelierModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                VOLATILITY,
                spaceTime
        );
    }

    private double valueAtSpot(
            final TouchOption option,
            final FDMBachelierModel model,
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

    private int getGridIndex(final double[] grid, final double x) {
        for(int i = 0; i < grid.length; i++) {
            if(Math.abs(grid[i] - x) < 1E-12) {
                return i;
            }
        }
        return -1;
    }

    private static class TestSetup {

        private final FDMBachelierModel model;
        private final double[] sNodes;

        private TestSetup(final FDMBachelierModel model, final double[] sNodes) {
            this.model = model;
            this.sNodes = sNodes;
        }
    }
}