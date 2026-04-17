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
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for {@link DoubleBarrierBinaryOption}
 * under Heston and SABR.
 *
 * <p>
 * The tests validate:
 * </p>
 * <ul>
 *   <<li>KNOCK_OUT + KNOCK_IN ~= cash payoff,</li>
 *  <li>KIKO + KOKI ~= KNOCK_IN,</li>
 *  <li>all values are non-negative and bounded by the cash payoff
 *   when starting inside the alive band,</li>
 *   <li>outside-band endpoint behavior:
 *       KNOCK_OUT = 0,
 *       KNOCK_IN = cash,
 *       KIKO = cash at lower side / 0 at upper side,
 *       KOKI = 0 at lower side / cash at upper side.</li>
 * </ul>
 * 
 * 
 *
 * @author Alessandro Gnoatto
 */
public class DoubleBarrierBinaryHestonSabrStructuralTest {

    private enum ModelType {
        HESTON,
        SABR
    }

    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double LOWER_BARRIER = 80.0;
    private static final double UPPER_BARRIER = 120.0;

    private static final double SPOT = 100.0;
    private static final double LOWER_OUTSIDE_SPOT = 75.0;
    private static final double UPPER_OUTSIDE_SPOT = 125.0;

    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.00;

    private static final double HESTON_VOLATILITY = 0.25;
    private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
    private static final double HESTON_KAPPA = 1.5;
    private static final double HESTON_THETA_V = HESTON_INITIAL_VARIANCE;
    private static final double HESTON_XI = 0.30;
    private static final double HESTON_RHO = -0.70;

    private static final double SABR_INITIAL_VOLATILITY = 0.20;
    private static final double SABR_BETA = 1.0;
    private static final double SABR_NU = 0.30;
    private static final double SABR_RHO = -0.50;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 120;
    private static final int NUMBER_OF_SECOND_STEPS = 100;

    /*
     * Centered grid:
     * step size = (160 - 40) / 240 = 0.5
     * so 80, 100, 120 are exact nodes.
     */
    private static final int CENTERED_NUMBER_OF_SPACE_STEPS = 240;
    private static final double CENTERED_GRID_MIN = 40.0;
    private static final double CENTERED_GRID_MAX = 160.0;

    /*
     * Wide grid:
     * step size = (200 - 0) / 200 = 1.0
     * so 75, 80, 120, 125 are exact nodes.
     */
    private static final int WIDE_NUMBER_OF_SPACE_STEPS = 200;
    private static final double WIDE_GRID_MIN = 0.0;
    private static final double WIDE_GRID_MAX = 200.0;

    private static final double NON_NEGATIVITY_TOLERANCE = 1E-10;
    private static final double UPPER_BOUND_TOLERANCE = 1E-6;
    private static final double PARTITION_TOLERANCE = 2.5E-1;
    private static final double ENDPOINT_TOLERANCE = 1E-6;

    @Test
    public void testHestonCenteredPartition() {
        runCenteredPartitionTest(ModelType.HESTON);
    }

    @Test
    public void testSabrCenteredPartition() {
        runCenteredPartitionTest(ModelType.SABR);
    }

    @Test
    public void testHestonEndpointBehavior() {
        runEndpointBehaviorTest(ModelType.HESTON);
    }

    @Test
    public void testSabrEndpointBehavior() {
        runEndpointBehaviorTest(ModelType.SABR);
    }

    private void runCenteredPartitionTest(final ModelType modelType) {

        final TestSetup setup = createCenteredSetup(modelType, SPOT);

        final double knockOutValue = valueAtSpot(setup, SPOT, DoubleBarrierType.KNOCK_OUT);
        final double knockInValue = valueAtSpot(setup, SPOT, DoubleBarrierType.KNOCK_IN);
        final double kikoValue = valueAtSpot(setup, SPOT, DoubleBarrierType.KIKO);
        final double kokiValue = valueAtSpot(setup, SPOT, DoubleBarrierType.KOKI);

        System.out.println("====================================================");
        System.out.println(modelType + " double-barrier binary structural regression");
        System.out.println("Lower barrier        = " + LOWER_BARRIER);
        System.out.println("Upper barrier        = " + UPPER_BARRIER);
        System.out.println("Spot                 = " + SPOT);
        System.out.println("Cash payoff          = " + CASH_PAYOFF);
        System.out.println("Knock-out            = " + knockOutValue);
        System.out.println("Knock-in             = " + knockInValue);
        System.out.println("KO + KI              = " + (knockOutValue + knockInValue));
        System.out.println("KIKO                 = " + kikoValue);
        System.out.println("KOKI                 = " + kokiValue);
        System.out.println("KIKO + KOKI          = " + (kikoValue + kokiValue));
        System.out.println("====================================================");

        assertTrue(knockOutValue >= -NON_NEGATIVITY_TOLERANCE);
        assertTrue(knockInValue >= -NON_NEGATIVITY_TOLERANCE);
        assertTrue(kikoValue >= -NON_NEGATIVITY_TOLERANCE);
        assertTrue(kokiValue >= -NON_NEGATIVITY_TOLERANCE);

        assertTrue(knockOutValue <= CASH_PAYOFF + UPPER_BOUND_TOLERANCE);
        assertTrue(knockInValue <= CASH_PAYOFF + UPPER_BOUND_TOLERANCE);
        assertTrue(kikoValue <= CASH_PAYOFF + UPPER_BOUND_TOLERANCE);
        assertTrue(kokiValue <= CASH_PAYOFF + UPPER_BOUND_TOLERANCE);

        assertEquals(
                "KNOCK_OUT + KNOCK_IN partition failed under " + modelType,
                CASH_PAYOFF,
                knockOutValue + knockInValue,
                PARTITION_TOLERANCE
        );

        assertEquals(
                "KIKO + KOKI partition failed under " + modelType,
                knockInValue,
                kikoValue + kokiValue,
                PARTITION_TOLERANCE
        );
    }

    private void runEndpointBehaviorTest(final ModelType modelType) {

        final TestSetup lowerSetup = createWideSetup(modelType, LOWER_OUTSIDE_SPOT);
        final TestSetup upperSetup = createWideSetup(modelType, UPPER_OUTSIDE_SPOT);

        final double knockOutLower = valueAtSpot(lowerSetup, LOWER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_OUT);
        final double knockOutUpper = valueAtSpot(upperSetup, UPPER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_OUT);

        final double knockInLower = valueAtSpot(lowerSetup, LOWER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_IN);
        final double knockInUpper = valueAtSpot(upperSetup, UPPER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_IN);

        final double kikoLower = valueAtSpot(lowerSetup, LOWER_OUTSIDE_SPOT, DoubleBarrierType.KIKO);
        final double kikoUpper = valueAtSpot(upperSetup, UPPER_OUTSIDE_SPOT, DoubleBarrierType.KIKO);

        final double kokiLower = valueAtSpot(lowerSetup, LOWER_OUTSIDE_SPOT, DoubleBarrierType.KOKI);
        final double kokiUpper = valueAtSpot(upperSetup, UPPER_OUTSIDE_SPOT, DoubleBarrierType.KOKI);

        System.out.println("====================================================");
        System.out.println(modelType + " double-barrier binary endpoint regression");
        System.out.println("Lower outside spot   = " + LOWER_OUTSIDE_SPOT);
        System.out.println("Upper outside spot   = " + UPPER_OUTSIDE_SPOT);
        System.out.println("KO lower / upper     = " + knockOutLower + " / " + knockOutUpper);
        System.out.println("KI lower / upper     = " + knockInLower + " / " + knockInUpper);
        System.out.println("KIKO lower / upper   = " + kikoLower + " / " + kikoUpper);
        System.out.println("KOKI lower / upper   = " + kokiLower + " / " + kokiUpper);
        System.out.println("====================================================");

        assertEquals(0.0, knockOutLower, ENDPOINT_TOLERANCE);
        assertEquals(0.0, knockOutUpper, ENDPOINT_TOLERANCE);

        assertEquals(CASH_PAYOFF, knockInLower, ENDPOINT_TOLERANCE);
        assertEquals(CASH_PAYOFF, knockInUpper, ENDPOINT_TOLERANCE);

        assertEquals(CASH_PAYOFF, kikoLower, ENDPOINT_TOLERANCE);
        assertEquals(0.0, kikoUpper, ENDPOINT_TOLERANCE);

        assertEquals(0.0, kokiLower, ENDPOINT_TOLERANCE);
        assertEquals(CASH_PAYOFF, kokiUpper, ENDPOINT_TOLERANCE);
    }

    private double valueAtSpot(
            final TestSetup setup,
            final double spot,
            final DoubleBarrierType doubleBarrierType) {

        final DoubleBarrierBinaryOption product = new DoubleBarrierBinaryOption(
                MATURITY,
                CASH_PAYOFF,
                LOWER_BARRIER,
                UPPER_BARRIER,
                doubleBarrierType
        );

        return interpolateAtSpotAndSecondState(
                product.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                spot,
                setup.initialSecondState
        );
    }

    private TestSetup createCenteredSetup(
            final ModelType modelType,
            final double spot) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = new UniformGrid(
                CENTERED_NUMBER_OF_SPACE_STEPS,
                CENTERED_GRID_MIN,
                CENTERED_GRID_MAX
        );

        final Grid secondGrid = createSecondGrid(modelType);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, secondGrid },
                timeDiscretization,
                THETA,
                new double[] { spot, getInitialSecondState(modelType) }
        );

        return new TestSetup(
                createModel(modelType, spot, spaceTime),
                sGrid.getGrid(),
                secondGrid.getGrid(),
                getInitialSecondState(modelType)
        );
    }

    private TestSetup createWideSetup(
            final ModelType modelType,
            final double spot) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = new UniformGrid(
                WIDE_NUMBER_OF_SPACE_STEPS,
                WIDE_GRID_MIN,
                WIDE_GRID_MAX
        );

        final Grid secondGrid = createSecondGrid(modelType);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, secondGrid },
                timeDiscretization,
                THETA,
                new double[] { spot, getInitialSecondState(modelType) }
        );

        return new TestSetup(
                createModel(modelType, spot, spaceTime),
                sGrid.getGrid(),
                secondGrid.getGrid(),
                getInitialSecondState(modelType)
        );
    }

    private Grid createSecondGrid(final ModelType modelType) {
        if(modelType == ModelType.HESTON) {
            final double vMax = Math.max(
                    4.0 * HESTON_THETA_V,
                    HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
            );
            return new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, vMax);
        }

        final double alphaMax = Math.max(4.0 * SABR_INITIAL_VOLATILITY, 1.0);
        return new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, alphaMax);
    }

    private FiniteDifferenceEquityModel createModel(
            final ModelType modelType,
            final double spot,
            final SpaceTimeDiscretization spaceTime) {

        if(modelType == ModelType.HESTON) {
            return new FDMHestonModel(
                    spot,
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
                spot,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                SABR_INITIAL_VOLATILITY,
                SABR_BETA,
                SABR_NU,
                SABR_RHO,
                spaceTime
        );
    }

    private double getInitialSecondState(final ModelType modelType) {
        return modelType == ModelType.HESTON
                ? HESTON_INITIAL_VARIANCE
                : SABR_INITIAL_VOLATILITY;
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