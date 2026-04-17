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
 * Structural regression test for 2D one-touch AT_HIT under Heston and SABR.
 *
 * <p>
 * Checks:
 * </p>
 * <ul>
 *   <li>non-negativity,</li>
 *   <li>AT_HIT bounded above by immediate cash payoff,</li>
 *   <li>AT_HIT dominates AT_EXPIRY for positive rates,</li>
 *   <li>already-hit-at-valuation returns immediate cash,</li>
 *   <li>maturity-zero endpoint behavior.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class TouchOptionAtHitHestonSabrStructuralTest {

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

    private static final double NON_NEGATIVITY_TOLERANCE = 1E-10;
    private static final double UPPER_BOUND_TOLERANCE = 1E-6;
    private static final double VALUE_TOLERANCE = 7.5E-2;
    private static final double TIMING_TOLERANCE = 1E-8;
    private static final double ZERO_TOLERANCE = 1E-8;

    @Test
    public void testHestonDownBarrierStructure() {
        runStructuralTest(ModelType.HESTON, 80.0, BarrierType.DOWN_IN);
    }

    @Test
    public void testHestonUpBarrierStructure() {
        runStructuralTest(ModelType.HESTON, 120.0, BarrierType.UP_IN);
    }

    @Test
    public void testSabrDownBarrierStructure() {
        runStructuralTest(ModelType.SABR, 80.0, BarrierType.DOWN_IN);
    }

    @Test
    public void testSabrUpBarrierStructure() {
        runStructuralTest(ModelType.SABR, 120.0, BarrierType.UP_IN);
    }

    @Test
    public void testHestonAlreadyHitAtValuationPaysImmediateCash() {
        runAlreadyHitTest(ModelType.HESTON);
    }

    @Test
    public void testSabrAlreadyHitAtValuationPaysImmediateCash() {
        runAlreadyHitTest(ModelType.SABR);
    }

    @Test
    public void testHestonMaturityZeroEndpointValues() {
        runMaturityZeroEndpointTest(ModelType.HESTON);
    }

    @Test
    public void testSabrMaturityZeroEndpointValues() {
        runMaturityZeroEndpointTest(ModelType.SABR);
    }

    private void runStructuralTest(
            final ModelType modelType,
            final double barrier,
            final BarrierType barrierType) {

        final TestSetup setup = createBarrierSetup(modelType, barrier, barrierType, MATURITY);

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

        final double atHitValue = interpolateAtSpotAndSecondState(
                atHit.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        final double atExpiryValue = interpolateAtSpotAndSecondState(
                atExpiry.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        System.out.println(modelType + " one-touch AT_HIT structural regression");
        System.out.println("Barrier         = " + barrier);
        System.out.println("Barrier type    = " + barrierType);
        System.out.println("AT_HIT          = " + atHitValue);
        System.out.println("AT_EXPIRY       = " + atExpiryValue);
        System.out.println("Cash payoff     = " + CASH_PAYOFF);

        assertTrue(
                "AT_HIT must be non-negative for " + barrierType + " under " + modelType,
                atHitValue >= -NON_NEGATIVITY_TOLERANCE
        );

        assertTrue(
                "AT_HIT must not exceed immediate cash payoff for " + barrierType + " under " + modelType,
                atHitValue <= CASH_PAYOFF + UPPER_BOUND_TOLERANCE
        );

        assertTrue(
                "For positive rates, AT_HIT should dominate AT_EXPIRY for " + barrierType + " under " + modelType,
                atHitValue + TIMING_TOLERANCE >= atExpiryValue
        );
    }

    private void runAlreadyHitTest(final ModelType modelType) {

        final TestSetup setup = createWideSetup(modelType, MATURITY);

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

        assertEquals(CASH_PAYOFF, upAndInValue, VALUE_TOLERANCE);
        assertEquals(CASH_PAYOFF, downAndInValue, VALUE_TOLERANCE);
    }

    private void runMaturityZeroEndpointTest(final ModelType modelType) {

        final TestSetup setup = createWideSetup(modelType, MATURITY);

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

        final double oneTouchHitValue = interpolateAtSpotAndSecondState(
                oneTouchHitAtMaturity.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        final double oneTouchNotHitValue = interpolateAtSpotAndSecondState(
                oneTouchNotHitAtMaturity.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.initialSecondState
        );

        assertEquals(CASH_PAYOFF, oneTouchHitValue, VALUE_TOLERANCE);
        assertEquals(0.0, oneTouchNotHitValue, ZERO_TOLERANCE);
    }

    private TestSetup createBarrierSetup(
            final ModelType modelType,
            final double barrier,
            final BarrierType barrierType,
            final double finalTime) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        finalTime / NUMBER_OF_TIME_STEPS
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
            final double finalTime) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        finalTime / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, 0.0, 200.0);
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

    private Grid createSecondGrid(final ModelType modelType) {
        if(modelType == ModelType.HESTON) {
            final double vMax = Math.max(
                    4.0 * HESTON_THETA_V,
                    HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
            );
            return new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, vMax);
        }

        final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);
        return new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, alphaMax);
    }

    private FiniteDifferenceEquityModel createModel(
            final ModelType modelType,
            final SpaceTimeDiscretization spaceTime) {

        if(modelType == ModelType.HESTON) {
            return new FDMHestonModel(
                    SPOT,
                    RISK_FREE_RATE,
                    DIVIDEND_YIELD,
                    HESTON_INITIAL_VARIANCE,
                    HESTON_KAPPA,
                    HESTON_THETA_V,
                    HESTON_XI,
                    HESTON_RHO,
                    spaceTime
            );
        }

        return new FDMSabrModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                SABR_INITIAL_ALPHA,
                SABR_BETA,
                SABR_NU,
                SABR_RHO,
                spaceTime
        );
    }

    private double getInitialSecondState(final ModelType modelType) {
        return modelType == ModelType.HESTON
                ? HESTON_INITIAL_VARIANCE
                : SABR_INITIAL_ALPHA;
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
                secondState >= secondNodes[0] - 1E-12
                        && secondState <= secondNodes[secondNodes.length - 1] + 1E-12
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