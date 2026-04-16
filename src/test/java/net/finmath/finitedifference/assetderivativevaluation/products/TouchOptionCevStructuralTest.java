package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.BarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression test for expiry-settled cash {@link TouchOption}
 * under the CEV model.
 *
 * <p>
 * For each barrier direction, the test checks:
 * </p>
 * <ul>
 *   <li>one-touch >= 0,</li>
 *   <li>no-touch >= 0,</li>
 *   <li>no-touch <= discounted cash payoff,</li>
 *   <li>one-touch + no-touch ~= discounted cash payoff.</li>
 * </ul>
 */
public class TouchOptionCevStructuralTest {

    private static final double SPOT = 100.0;
    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double VOLATILITY = 0.20;
    private static final double EXPONENT = 0.8;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.02;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 200;
    private static final int NUMBER_OF_SPACE_STEPS = 400;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;

    private static final double NON_NEGATIVITY_TOLERANCE = 1E-10;
    private static final double UPPER_BOUND_TOLERANCE = 1E-6;

    /**
     * Partition tolerance for one-touch + no-touch ~= discounted cash.
     *
     * <p>
     * This is a structural finite-difference test, not a closed-form regression.
     * </p>
     */
    private static final double PARTITION_TOLERANCE = 5E-2;

    @Test
    public void testDownBarrierStructure() {
        runStructuralTest(80.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT);
    }

    @Test
    public void testUpBarrierStructure() {
        runStructuralTest(120.0, BarrierType.UP_IN, BarrierType.UP_OUT);
    }

    private void runStructuralTest(
            final double barrier,
            final BarrierType oneTouchBarrierType,
            final BarrierType noTouchBarrierType) {

        final TestSetup oneTouchSetup = createBarrierSetup(barrier, oneTouchBarrierType);
        final TestSetup noTouchSetup = createBarrierSetup(barrier, noTouchBarrierType);

        final TouchOption oneTouch = TouchOption.oneTouchAtExpiry(
                MATURITY,
                barrier,
                oneTouchBarrierType,
                CASH_PAYOFF
        );

        final TouchOption noTouch = TouchOption.noTouchAtExpiry(
                MATURITY,
                barrier,
                noTouchBarrierType,
                CASH_PAYOFF
        );

        final double oneTouchValue = interpolateAtSpot(
                oneTouch.getValue(0.0, oneTouchSetup.model),
                oneTouchSetup.sNodes,
                SPOT
        );

        final double noTouchValue = interpolateAtSpot(
                noTouch.getValue(0.0, noTouchSetup.model),
                noTouchSetup.sNodes,
                SPOT
        );

        final double discountedCash = CASH_PAYOFF * Math.exp(-RISK_FREE_RATE * MATURITY);

        System.out.println("CEV touch structural regression");
        System.out.println("Barrier              = " + barrier);
        System.out.println("One-touch type       = " + oneTouchBarrierType);
        System.out.println("No-touch type        = " + noTouchBarrierType);
        System.out.println("One-touch value      = " + oneTouchValue);
        System.out.println("No-touch value       = " + noTouchValue);
        System.out.println("Discounted cash      = " + discountedCash);
        System.out.println("Partition sum        = " + (oneTouchValue + noTouchValue));

        assertTrue(
                "One-touch must be non-negative for " + oneTouchBarrierType,
                oneTouchValue >= -NON_NEGATIVITY_TOLERANCE
        );

        assertTrue(
                "No-touch must be non-negative for " + noTouchBarrierType,
                noTouchValue >= -NON_NEGATIVITY_TOLERANCE
        );

        assertTrue(
                "No-touch must be bounded by discounted cash for " + noTouchBarrierType,
                noTouchValue <= discountedCash + UPPER_BOUND_TOLERANCE
        );

        assertTrue(
                "One-touch + no-touch should approximately equal discounted cash for barrier "
                        + barrier + ". Difference = "
                        + Math.abs(oneTouchValue + noTouchValue - discountedCash),
                Math.abs(oneTouchValue + noTouchValue - discountedCash) <= PARTITION_TOLERANCE
        );
    }

    private TestSetup createBarrierSetup(
            final double barrier,
            final BarrierType barrierType) {

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

        final FDMCevModel model = new FDMCevModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                VOLATILITY,
                EXPONENT,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid());
    }

    private Grid createBarrierGrid(
            final double barrier,
            final BarrierType barrierType) {

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_OUT) {
            final double deltaS = (SPOT - barrier) / effectiveStepsBetweenBarrierAndSpot;
            final double sMin = barrier;
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.UP_OUT) {
            final double deltaS = (barrier - SPOT) / effectiveStepsBetweenBarrierAndSpot;
            final double sMax = barrier;
            final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.DOWN_IN) {
            final double deltaS = (SPOT - barrier) / effectiveStepsBetweenBarrierAndSpot;

            final int maxExtraStepsBelowBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsBelowBarrier =
                    Math.max(1, Math.min(DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS, maxExtraStepsBelowBarrier));

            final double sMin = Math.max(barrier - extraStepsBelowBarrier * deltaS, 0.0);
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
            final double sMin = Math.max(sMax - NUMBER_OF_SPACE_STEPS * deltaS, 0.0);

            validateInteriorBarrierPlacement(sMin, barrier, deltaS);

            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else {
            throw new IllegalArgumentException("Unsupported barrier type.");
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

    private double interpolateAtSpot(
            final double[] values,
            final double[] sNodes,
            final double spot) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );

        if(isGridNode(sNodes, spot)) {
            return values[getGridIndex(sNodes, spot)];
        }

        final PolynomialSplineFunction interpolation =
                new LinearInterpolator().interpolate(sNodes, values);
        return interpolation.value(spot);
    }

    private boolean isGridNode(final double[] grid, final double x) {
        for(final double node : grid) {
            if(Math.abs(node - x) < 1E-12) {
                return true;
            }
        }
        return false;
    }

    private int getGridIndex(final double[] grid, final double x) {
        for(int i = 0; i < grid.length; i++) {
            if(Math.abs(grid[i] - x) < 1E-12) {
                return i;
            }
        }
        throw new IllegalArgumentException("Point is not a grid node.");
    }

    private static class TestSetup {

        private final FDMCevModel model;
        private final double[] sNodes;

        private TestSetup(final FDMCevModel model, final double[] sNodes) {
            this.model = model;
            this.sNodes = sNodes;
        }
    }
}