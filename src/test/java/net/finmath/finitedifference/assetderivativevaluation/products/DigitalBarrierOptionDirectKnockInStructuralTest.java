package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for the direct 2D knock-in implementation of
 * {@link DigitalBarrierOption} under Heston and SABR.
 *
 * <p>
 * These tests intentionally avoid Monte Carlo and instead validate:
 * </p>
 * <ul>
 *   <li>European in-out parity at (S0, secondState0):
 *       vanilla digital ~= knock-in + knock-out,</li>
 *   <li>knock-in dominance: knock-in <= vanilla digital,</li>
 *   <li>already-hit consistency: in the already-activated region, the knock-in
 *       price matches the corresponding vanilla digital.</li>
 * </ul>
 *
 * <p>
 * This is the right first regression layer for the new direct 2D digital
 * knock-in formulation.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalBarrierOptionDirectKnockInStructuralTest {

    private enum ModelType {
        HESTON,
        SABR
    }

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double SPOT = 100.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.00;

    private static final double HESTON_VOLATILITY = 0.25;
    private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
    private static final double HESTON_LONG_RUN_VARIANCE = HESTON_INITIAL_VARIANCE;
    private static final double HESTON_KAPPA = 1.5;
    private static final double HESTON_XI = 0.30;
    private static final double HESTON_RHO = -0.70;

    private static final double SABR_INITIAL_VOLATILITY = 0.20;
    private static final double SABR_BETA = 1.0;
    private static final double SABR_NU = 0.30;
    private static final double SABR_RHO = -0.50;

    private static final double THETA = 0.5;

    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS_S = 180;
    private static final int NUMBER_OF_SPACE_STEPS_2 = 100;

    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

    private static final double PARITY_TOLERANCE = 5.0E-1;
    private static final double ACTIVATED_REGION_TOLERANCE = 5.0E-1;
    private static final double DOMINANCE_TOLERANCE = 1E-8;

    private static final double DOWN_ALREADY_HIT_SPOT = 78.0;
    private static final double UP_ALREADY_HIT_SPOT = 122.0;

    private static final EuropeanExercise EUROPEAN_EXERCISE = new EuropeanExercise(MATURITY);

    /*
     * HESTON
     */

    @Test
    public void testHestonDownInCashCallParity() {
        assertDirectKnockInMatchesParity(
                ModelType.HESTON,
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN
        );
    }

    @Test
    public void testHestonDownInCashPutParity() {
        assertDirectKnockInMatchesParity(
                ModelType.HESTON,
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN
        );
    }

    @Test
    public void testHestonUpInCashCallParity() {
        assertDirectKnockInMatchesParity(
                ModelType.HESTON,
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN
        );
    }

    @Test
    public void testHestonUpInCashPutParity() {
        assertDirectKnockInMatchesParity(
                ModelType.HESTON,
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN
        );
    }

    @Test
    public void testHestonDownInCashCallMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.HESTON,
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testHestonDownInCashPutMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.HESTON,
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testHestonUpInCashCallMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.HESTON,
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testHestonUpInCashPutMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.HESTON,
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    /*
     * SABR
     */

    @Test
    public void testSabrDownInCashCallParity() {
        assertDirectKnockInMatchesParity(
                ModelType.SABR,
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN
        );
    }

    @Test
    public void testSabrDownInCashPutParity() {
        assertDirectKnockInMatchesParity(
                ModelType.SABR,
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN
        );
    }

    @Test
    public void testSabrUpInCashCallParity() {
        assertDirectKnockInMatchesParity(
                ModelType.SABR,
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN
        );
    }

    @Test
    public void testSabrUpInCashPutParity() {
        assertDirectKnockInMatchesParity(
                ModelType.SABR,
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN
        );
    }

    @Test
    public void testSabrDownInCashCallMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.SABR,
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testSabrDownInCashPutMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.SABR,
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testSabrUpInCashCallMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.SABR,
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testSabrUpInCashPutMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                ModelType.SABR,
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    private void assertDirectKnockInMatchesParity(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final double barrier,
            final BarrierType outType,
            final BarrierType inType) {

        final TestSetup setup = createSetup(modelType, barrier, outType);

        final DigitalBarrierOption outOption = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                outType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                EUROPEAN_EXERCISE
        );

        final DigitalBarrierOption inOption = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                inType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                EUROPEAN_EXERCISE
        );

        final DigitalOption vanillaOption = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                EUROPEAN_EXERCISE
        );

        final double outPrice = interpolateAtSecondState0(
                outOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        final double inPrice = interpolateAtSecondState0(
                inOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        final double vanillaPrice = interpolateAtSecondState0(
                vanillaOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        System.out.println("====================================================");
        System.out.println(modelType + " direct digital knock-in parity check");
        System.out.println("Call/Put         = " + callOrPut);
        System.out.println("Barrier          = " + barrier);
        System.out.println("Out type         = " + outType);
        System.out.println("In type          = " + inType);
        System.out.println("Out              = " + outPrice);
        System.out.println("In (direct)      = " + inPrice);
        System.out.println("Vanilla          = " + vanillaPrice);
        System.out.println("In + Out         = " + (inPrice + outPrice));
        System.out.println("Parity error     = " + Math.abs(vanillaPrice - (inPrice + outPrice)));
        System.out.println("====================================================");

        assertTrue("Knock-out digital price must be non-negative.", outPrice >= -1E-10);
        assertTrue("Knock-in digital price must be non-negative.", inPrice >= -1E-10);
        assertTrue("Vanilla digital price must be non-negative.", vanillaPrice >= -1E-10);

        assertEquals(
                modelType + " direct digital knock-in parity failed for " + inType + " " + callOrPut,
                vanillaPrice,
                inPrice + outPrice,
                PARITY_TOLERANCE
        );

        assertTrue(
                "Knock-in digital price should not exceed the vanilla digital price for "
                        + inType + " " + callOrPut + " under " + modelType,
                inPrice <= vanillaPrice + DOMINANCE_TOLERANCE
        );
    }

    private void assertAlreadyActivatedRegionMatchesVanilla(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final double barrier,
            final BarrierType knockInType,
            final double alreadyHitSpot) {

        final TestSetup setup = createSetup(modelType, barrier, knockInType);

        final DigitalBarrierOption inOption = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                knockInType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                EUROPEAN_EXERCISE
        );

        final DigitalOption vanillaOption = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                EUROPEAN_EXERCISE
        );

        final double inPrice = interpolateAtSecondState0(
                inOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                alreadyHitSpot,
                setup.secondState0
        );

        final double vanillaPrice = interpolateAtSecondState0(
                vanillaOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                alreadyHitSpot,
                setup.secondState0
        );

        System.out.println("====================================================");
        System.out.println(modelType + " direct digital knock-in already-hit consistency");
        System.out.println("Call/Put         = " + callOrPut);
        System.out.println("Barrier type     = " + knockInType);
        System.out.println("Barrier          = " + barrier);
        System.out.println("Already-hit spot = " + alreadyHitSpot);
        System.out.println("In (direct)      = " + inPrice);
        System.out.println("Vanilla          = " + vanillaPrice);
        System.out.println("Abs error        = " + Math.abs(vanillaPrice - inPrice));
        System.out.println("====================================================");

        assertTrue("Knock-in digital price must be non-negative.", inPrice >= -1E-10);
        assertTrue("Vanilla digital price must be non-negative.", vanillaPrice >= -1E-10);

        assertEquals(
                "Already-hit region should match vanilla digital for "
                        + knockInType + " " + callOrPut + " under " + modelType,
                vanillaPrice,
                inPrice,
                ACTIVATED_REGION_TOLERANCE
        );
    }

    private TestSetup createSetup(
            final ModelType modelType,
            final double barrier,
            final BarrierType barrierType) {

        if(modelType == ModelType.HESTON) {
            return createHestonSetup(barrier, barrierType);
        }
        return createSabrSetup(barrier, barrierType);
    }

    private TestSetup createHestonSetup(final double barrier, final BarrierType barrierType) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

        final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
                0.0,
                NUMBER_OF_TIME_STEPS,
                MATURITY / NUMBER_OF_TIME_STEPS
        );

        final Grid sGrid = createSpotGrid(barrier, barrierType);

        final double vMin = 0.0;
        final double vMax = Math.max(
                4.0 * HESTON_LONG_RUN_VARIANCE,
                HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
        );
        final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_2, vMin, vMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, vGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, HESTON_INITIAL_VARIANCE }
        );

        final FDMHestonModel model = new FDMHestonModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                HESTON_INITIAL_VARIANCE,
                HESTON_KAPPA,
                HESTON_LONG_RUN_VARIANCE,
                HESTON_XI,
                HESTON_RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), vGrid.getGrid(), HESTON_INITIAL_VARIANCE);
    }

    private TestSetup createSabrSetup(final double barrier, final BarrierType barrierType) {

        final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
                0.0,
                NUMBER_OF_TIME_STEPS,
                MATURITY / NUMBER_OF_TIME_STEPS
        );

        final Grid sGrid = createSpotGrid(barrier, barrierType);

        final double volMax = Math.max(4.0 * SABR_INITIAL_VOLATILITY, 1.0);
        final Grid volGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_2, 0.0, volMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, volGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, SABR_INITIAL_VOLATILITY }
        );

        final FDMSabrModel model = new FDMSabrModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                SABR_INITIAL_VOLATILITY,
                SABR_BETA,
                SABR_NU,
                SABR_RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), volGrid.getGrid(), SABR_INITIAL_VOLATILITY);
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

    private double interpolateAtSecondState0(
            final double[] flattenedValues,
            final double[] sNodes,
            final double[] secondNodes,
            final double spot,
            final double secondState) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );
        assertTrue(
                "Second state variable must lie inside the grid domain.",
                secondState >= secondNodes[0] - 1E-12 && secondState <= secondNodes[secondNodes.length - 1] + 1E-12
        );

        final int nS = sNodes.length;
        final int n2 = secondNodes.length;

        final double[][] valueSurface = new double[nS][n2];
        for(int j = 0; j < n2; j++) {
            for(int i = 0; i < nS; i++) {
                valueSurface[i][j] = flattenedValues[flatten(i, j, nS)];
            }
        }

        final BiLinearInterpolation interpolation =
                new BiLinearInterpolation(sNodes, secondNodes, valueSurface);

        return interpolation.apply(spot, secondState);
    }

    private int flatten(final int iS, final int i2, final int numberOfSNodes) {
        return iS + i2 * numberOfSNodes;
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

    private static final class TestSetup {

        private final FiniteDifferenceEquityModel model;
        private final double[] sNodes;
        private final double[] secondNodes;
        private final double secondState0;

        private TestSetup(
                final FiniteDifferenceEquityModel model,
                final double[] sNodes,
                final double[] secondNodes,
                final double secondState0) {
            this.model = model;
            this.sNodes = sNodes;
            this.secondNodes = secondNodes;
            this.secondState0 = secondState0;
        }
    }
}