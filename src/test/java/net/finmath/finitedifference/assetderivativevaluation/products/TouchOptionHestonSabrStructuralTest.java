package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression test for expiry-settled cash {@link TouchOption}
 * under Heston and SABR.
 *
 * <p>
 * For each barrier direction, the test checks:
 * </p>
 * <ul>
 *   <li>one-touch >= 0,</li>
 *   <li>no-touch >= 0,</li>
 *   <li>no-touch <= discounted cash payoff,</li>
 *   <li>one-touch + no-touch ~= discounted cash payoff,</li>
 *   <li>already-hit one-touch at valuation pays discounted cash,</li>
 *   <li>already-knocked-out no-touch at valuation is zero.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class TouchOptionHestonSabrStructuralTest {

    private enum ModelType {
        HESTON,
        SABR
    }

    private static final double SPOT = 100.0;
    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.02;

    private static final double HESTON_VOLATILITY = 0.25;
    private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
    private static final double HESTON_KAPPA = 1.5;
    private static final double HESTON_THETA_V = HESTON_INITIAL_VARIANCE;
    private static final double HESTON_XI = 0.30;
    private static final double HESTON_RHO = -0.70;

    private static final double SABR_INITIAL_ALPHA = 0.20;
    private static final double SABR_BETA = 1.0;
    private static final double SABR_NU = 0.30;
    private static final double SABR_RHO = -0.50;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 160;
    private static final int NUMBER_OF_SPACE_STEPS = 260;
    private static final int NUMBER_OF_SECOND_STEPS = 100;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;

    private static final int WIDE_NUMBER_OF_SPACE_STEPS = 400;

    private static final double NON_NEGATIVITY_TOLERANCE = 1E-10;
    private static final double UPPER_BOUND_TOLERANCE = 1E-6;
    private static final double PARTITION_TOLERANCE = 1.0E-1;
    private static final double VALUE_TOLERANCE = 1.0E-1;
    private static final double ZERO_TOLERANCE = 1E-8;

    @Test
    public void testHestonDownBarrierStructure() {
        runStructuralPartitionTest(ModelType.HESTON, 80.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT);
    }

    @Test
    public void testHestonUpBarrierStructure() {
        runStructuralPartitionTest(ModelType.HESTON, 120.0, BarrierType.UP_IN, BarrierType.UP_OUT);
    }

    @Test
    public void testSabrDownBarrierStructure() {
        runStructuralPartitionTest(ModelType.SABR, 80.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT);
    }

    @Test
    public void testSabrUpBarrierStructure() {
        runStructuralPartitionTest(ModelType.SABR, 120.0, BarrierType.UP_IN, BarrierType.UP_OUT);
    }

    @Test
    public void testHestonAlreadyHitOneTouchAtValuationPaysDiscountedCash() {
        runAlreadyHitOneTouchTest(ModelType.HESTON);
    }

    @Test
    public void testSabrAlreadyHitOneTouchAtValuationPaysDiscountedCash() {
        runAlreadyHitOneTouchTest(ModelType.SABR);
    }

    @Test
    public void testHestonAlreadyKnockedOutNoTouchAtValuationIsZero() {
        runAlreadyKnockedOutNoTouchTest(ModelType.HESTON);
    }

    @Test
    public void testSabrAlreadyKnockedOutNoTouchAtValuationIsZero() {
        runAlreadyKnockedOutNoTouchTest(ModelType.SABR);
    }

    private void runStructuralPartitionTest(
            final ModelType modelType,
            final double barrier,
            final BarrierType oneTouchBarrierType,
            final BarrierType noTouchBarrierType) {

        final TestSetup oneTouchSetup = createBarrierSetup(modelType, barrier, oneTouchBarrierType);
        final TestSetup noTouchSetup = createBarrierSetup(modelType, barrier, noTouchBarrierType);

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

        final double oneTouchValue = interpolateAtSpotAndSecondState(
                oneTouch.getValue(0.0, oneTouchSetup.model),
                oneTouchSetup.sNodes,
                oneTouchSetup.secondNodes,
                SPOT,
                oneTouchSetup.initialSecondState
        );

        final double noTouchValue = interpolateAtSpotAndSecondState(
                noTouch.getValue(0.0, noTouchSetup.model),
                noTouchSetup.sNodes,
                noTouchSetup.secondNodes,
                SPOT,
                noTouchSetup.initialSecondState
        );

        final double discountedCash = CASH_PAYOFF * Math.exp(-RISK_FREE_RATE * MATURITY);

        System.out.println(modelType + " touch structural regression");
        System.out.println("Barrier              = " + barrier);
        System.out.println("One-touch type       = " + oneTouchBarrierType);
        System.out.println("No-touch type        = " + noTouchBarrierType);
        System.out.println("One-touch value      = " + oneTouchValue);
        System.out.println("No-touch value       = " + noTouchValue);
        System.out.println("Discounted cash      = " + discountedCash);
        System.out.println("Partition sum        = " + (oneTouchValue + noTouchValue));

        assertTrue(
                "One-touch must be non-negative for " + oneTouchBarrierType + " under " + modelType,
                oneTouchValue >= -NON_NEGATIVITY_TOLERANCE
        );

        assertTrue(
                "No-touch must be non-negative for " + noTouchBarrierType + " under " + modelType,
                noTouchValue >= -NON_NEGATIVITY_TOLERANCE
        );

        assertTrue(
                "No-touch must be bounded by discounted cash for " + noTouchBarrierType + " under " + modelType,
                noTouchValue <= discountedCash + UPPER_BOUND_TOLERANCE
        );

        assertTrue(
                "One-touch + no-touch should approximately equal discounted cash for barrier "
                        + barrier + " under " + modelType + ". Difference = "
                        + Math.abs(oneTouchValue + noTouchValue - discountedCash),
                Math.abs(oneTouchValue + noTouchValue - discountedCash) <= PARTITION_TOLERANCE
        );
    }

    private void runAlreadyHitOneTouchTest(final ModelType modelType) {
        final TestSetup setup = createWideSetup(modelType, 0.0, 200.0);

        final TouchOption upAndInAlreadyHit = TouchOption.oneTouchAtExpiry(
                MATURITY,
                90.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption downAndInAlreadyHit = TouchOption.oneTouchAtExpiry(
                MATURITY,
                110.0,
                BarrierType.DOWN_IN,
                CASH_PAYOFF
        );

        final double discountedCash = CASH_PAYOFF * Math.exp(-RISK_FREE_RATE * MATURITY);

        final double upAndInValue = interpolateAtSpotAndSecondState(
                upAndInAlreadyHit.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        final double downAndInValue = interpolateAtSpotAndSecondState(
                downAndInAlreadyHit.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        assertEquals(discountedCash, upAndInValue, VALUE_TOLERANCE);
        assertEquals(discountedCash, downAndInValue, VALUE_TOLERANCE);
    }

    private void runAlreadyKnockedOutNoTouchTest(final ModelType modelType) {
        final TestSetup setup = createWideSetup(modelType, 0.0, 200.0);

        final TouchOption upAndOutAlreadyOut = TouchOption.noTouchAtExpiry(
                MATURITY,
                90.0,
                BarrierType.UP_OUT,
                CASH_PAYOFF
        );

        final TouchOption downAndOutAlreadyOut = TouchOption.noTouchAtExpiry(
                MATURITY,
                110.0,
                BarrierType.DOWN_OUT,
                CASH_PAYOFF
        );

        final double upAndOutValue = interpolateAtSpotAndSecondState(
                upAndOutAlreadyOut.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        final double downAndOutValue = interpolateAtSpotAndSecondState(
                downAndOutAlreadyOut.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        assertEquals(0.0, upAndOutValue, ZERO_TOLERANCE);
        assertEquals(0.0, downAndOutValue, ZERO_TOLERANCE);
    }

    private TestSetup createBarrierSetup(
            final ModelType modelType,
            final double barrier,
            final BarrierType barrierType) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createBarrierGrid(barrier, barrierType);
        final Grid secondGrid = createSecondGrid(modelType);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, secondGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, getInitialSecondState(modelType) }
        );

        return new TestSetup(
                createModel(modelType, spaceTime),
                sGrid.getGrid(),
                secondGrid.getGrid(),
                getInitialSecondState(modelType)
        );
    }

    private TestSetup createWideSetup(
            final ModelType modelType,
            final double sMin,
            final double sMax) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = new UniformGrid(WIDE_NUMBER_OF_SPACE_STEPS, sMin, sMax);
        final Grid secondGrid = createSecondGrid(modelType);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, secondGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, getInitialSecondState(modelType) }
        );

        return new TestSetup(
                createModel(modelType, spaceTime),
                sGrid.getGrid(),
                secondGrid.getGrid(),
                getInitialSecondState(modelType)
        );
    }

    private Grid createBarrierGrid(
            final double barrier,
            final BarrierType barrierType) {

        final int effectiveStepsBetweenBarrierAndSpot = getEffectiveStepsBetweenBarrierAndSpot(barrier);

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

    private Grid createSecondGrid(final ModelType modelType) {
        if(modelType == ModelType.HESTON) {
            final double vMax = Math.max(
                    4.0 * HESTON_THETA_V,
                    HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
            );
            return new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, vMax);
        }
        else {
            final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);
            return new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, alphaMax);
        }
    }

    private FiniteDifferenceEquityModel createModel(
            final ModelType modelType,
            final SpaceTimeDiscretization spaceTime) {

        if(modelType == ModelType.HESTON) {
            return new FDMHestonModel(
                    SPOT,
                    HESTON_INITIAL_VARIANCE,
                    RISK_FREE_RATE,
                    DIVIDEND_YIELD,
                    HESTON_KAPPA,
                    HESTON_THETA_V,
                    HESTON_XI,
                    HESTON_RHO,
                    spaceTime
            );
        }
        else {
            return new FDMSabrModel(
                    SPOT,
                    SABR_INITIAL_ALPHA,
                    RISK_FREE_RATE,
                    DIVIDEND_YIELD,
                    SABR_BETA,
                    SABR_NU,
                    SABR_RHO,
                    spaceTime
            );
        }
    }

    private double getInitialSecondState(final ModelType modelType) {
        return modelType == ModelType.HESTON ? HESTON_INITIAL_VARIANCE : SABR_INITIAL_ALPHA;
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

    private double interpolateAtSpotAndSecondState(
            final double[] values,
            final double[] sNodes,
            final double[] secondNodes,
            final double spot,
            final double secondState) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );
        assertTrue(
                "Second state must lie inside the grid domain.",
                secondState >= secondNodes[0] - 1E-12 && secondState <= secondNodes[secondNodes.length - 1] + 1E-12
        );

        final int nS = sNodes.length;
        final int n2 = secondNodes.length;

        final double[][] valueSurface = new double[nS][n2];
        for(int j = 0; j < n2; j++) {
            for(int i = 0; i < nS; i++) {
                valueSurface[i][j] = values[flatten(i, j, nS)];
            }
        }

        final BiLinearInterpolation interpolation =
                new BiLinearInterpolation(sNodes, secondNodes, valueSurface);

        return interpolation.apply(spot, secondState);
    }

    private int flatten(final int iS, final int i2, final int numberOfSNodes) {
        return iS + i2 * numberOfSNodes;
    }

    private static final class TestSetup {
        private final FiniteDifferenceEquityModel model;
        private final double[] sNodes;
        private final double[] secondNodes;
        private final double initialSecondState;

        private TestSetup(
                final FiniteDifferenceEquityModel model,
                final double[] sNodes,
                final double[] secondNodes,
                final double initialSecondState) {
            this.model = model;
            this.sNodes = sNodes;
            this.secondNodes = secondNodes;
            this.initialSecondState = initialSecondState;
        }
    }
}