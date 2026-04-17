package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
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
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for the direct 2D knock-in implementation of
 * {@link BarrierOption} under {@link FDMHestonModel}.
 *
 * <p>
 * These tests intentionally avoid an external benchmark and instead validate:
 * </p>
 * <ul>
 *   <li>European in-out parity at (S0, v0): vanilla ~= knock-in + knock-out,</li>
 *   <li>knock-in dominance: knock-in <= vanilla,</li>
 *   <li>already-hit consistency: in the already-activated region, the knock-in
 *       price should match the corresponding vanilla value.</li>
 * </ul>
 *
 * <p>
 * This is the right first regression layer for the new direct 2D formulation:
 * it validates the activated-vanilla + pre-hit assembly end to end before
 * moving on to Bermudan/American cases.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonDirectKnockInStructuralTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double REBATE = 0.0;

    private static final double SPOT = 100.0;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.00;

    private static final double INITIAL_VOLATILITY = 0.25;
    private static final double INITIAL_VARIANCE = INITIAL_VOLATILITY * INITIAL_VOLATILITY;

    private static final double KAPPA = 1.5;
    private static final double LONG_RUN_VARIANCE = INITIAL_VARIANCE;
    private static final double XI = 0.30;
    private static final double RHO = -0.70;

    private static final double THETA = 0.5;

    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS_S = 160;
    private static final int NUMBER_OF_SPACE_STEPS_V = 100;

    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

    /*
     * Tolerances:
     *
     * - PARITY_TOLERANCE validates the new direct knock-in price against the
     *   old decomposition vanilla - knock-out on the same Heston grid.
     *
     * - ACTIVATED_REGION_TOLERANCE validates that, once already beyond the
     *   barrier at evaluation time, the knock-in behaves like the activated
     *   vanilla product.
     *
     * These are structural regression tolerances, not asymptotic convergence
     * tolerances.
     */
    private static final double PARITY_TOLERANCE = 3.0E-1;
    private static final double ACTIVATED_REGION_TOLERANCE = 3.0E-1;
    private static final double DOMINANCE_TOLERANCE = 1E-8;

    /*
     * Points inside the already-hit region, but still comfortably inside the
     * numerical grid produced by createSpotGrid(...).
     */
    private static final double DOWN_ALREADY_HIT_SPOT = 78.0;
    private static final double UP_ALREADY_HIT_SPOT = 122.0;

    @Test
    public void testDownInCallMatchesVanillaMinusDownOutAtSpotAndInitialVariance() {
        assertDirectKnockInMatchesParity(
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN
        );
    }

    @Test
    public void testDownInPutMatchesVanillaMinusDownOutAtSpotAndInitialVariance() {
        assertDirectKnockInMatchesParity(
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN
        );
    }

    @Test
    public void testUpInCallMatchesVanillaMinusUpOutAtSpotAndInitialVariance() {
        assertDirectKnockInMatchesParity(
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN
        );
    }

    @Test
    public void testUpInPutMatchesVanillaMinusUpOutAtSpotAndInitialVariance() {
        assertDirectKnockInMatchesParity(
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN
        );
    }

    @Test
    public void testDownInCallMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testDownInPutMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testUpInCallMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testUpInPutMatchesVanillaInAlreadyHitRegion() {
        assertAlreadyActivatedRegionMatchesVanilla(
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    private void assertDirectKnockInMatchesParity(
            final CallOrPut callOrPut,
            final double barrier,
            final BarrierType outType,
            final BarrierType inType) {

        final TestSetup setup = createSetup(barrier, outType);

        final BarrierOption outOption = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                outType
        );

        final BarrierOption inOption = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                inType
        );

        final EuropeanOption vanillaOption = new EuropeanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final double outPrice = interpolateAtSpotAndInitialVariance(
                outOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        final double inPrice = interpolateAtSpotAndInitialVariance(
                inOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        final double vanillaPrice = interpolateAtSpotAndInitialVariance(
                vanillaOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        System.out.println("====================================================");
        System.out.println("Heston direct knock-in parity check");
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

        assertTrue("Knock-out price must be non-negative.", outPrice >= -1E-10);
        assertTrue("Knock-in price must be non-negative.", inPrice >= -1E-10);
        assertTrue("Vanilla price must be non-negative.", vanillaPrice >= -1E-10);

        assertEquals(
                "Heston direct knock-in parity failed for " + inType + " " + callOrPut,
                vanillaPrice,
                inPrice + outPrice,
                PARITY_TOLERANCE
        );

        assertTrue(
                "Knock-in price should not exceed the vanilla price for " + inType + " " + callOrPut,
                inPrice <= vanillaPrice + DOMINANCE_TOLERANCE
        );
    }

    private void assertAlreadyActivatedRegionMatchesVanilla(
            final CallOrPut callOrPut,
            final double barrier,
            final BarrierType knockInType,
            final double alreadyHitSpot) {

        final TestSetup setup = createSetup(barrier, knockInType);

        final BarrierOption inOption = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                knockInType
        );

        final EuropeanOption vanillaOption = new EuropeanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final double inPrice = interpolateAtSpotAndInitialVariance(
                inOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                alreadyHitSpot,
                INITIAL_VARIANCE
        );

        final double vanillaPrice = interpolateAtSpotAndInitialVariance(
                vanillaOption.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                alreadyHitSpot,
                INITIAL_VARIANCE
        );

        System.out.println("====================================================");
        System.out.println("Heston direct knock-in already-hit consistency");
        System.out.println("Call/Put         = " + callOrPut);
        System.out.println("Barrier type     = " + knockInType);
        System.out.println("Barrier          = " + barrier);
        System.out.println("Already-hit spot = " + alreadyHitSpot);
        System.out.println("In (direct)      = " + inPrice);
        System.out.println("Vanilla          = " + vanillaPrice);
        System.out.println("Abs error        = " + Math.abs(vanillaPrice - inPrice));
        System.out.println("====================================================");

        assertTrue("Knock-in price must be non-negative.", inPrice >= -1E-10);
        assertTrue("Vanilla price must be non-negative.", vanillaPrice >= -1E-10);

        assertEquals(
                "Already-hit region should match vanilla for " + knockInType + " " + callOrPut,
                vanillaPrice,
                inPrice,
                ACTIVATED_REGION_TOLERANCE
        );
    }

    private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

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
                4.0 * LONG_RUN_VARIANCE,
                INITIAL_VARIANCE + 4.0 * XI * Math.sqrt(MATURITY)
        );

        final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_V, vMin, vMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, vGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, INITIAL_VARIANCE }
        );

        final FDMHestonModel model = new FDMHestonModel(
                SPOT,
                INITIAL_VARIANCE,
                riskFreeCurve,
                dividendCurve,
                KAPPA,
                LONG_RUN_VARIANCE,
                XI,
                RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), vGrid.getGrid());
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

    private double interpolateAtSpotAndInitialVariance(
            final double[] flattenedValues,
            final double[] sNodes,
            final double[] vNodes,
            final double spot,
            final double variance) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );
        assertTrue(
                "Variance must lie inside the grid domain.",
                variance >= vNodes[0] - 1E-12 && variance <= vNodes[vNodes.length - 1] + 1E-12
        );

        final int nS = sNodes.length;
        final int nV = vNodes.length;

        final double[][] valueSurface = new double[nS][nV];
        for(int j = 0; j < nV; j++) {
            for(int i = 0; i < nS; i++) {
                valueSurface[i][j] = flattenedValues[flatten(i, j, nS)];
            }
        }

        final BiLinearInterpolation interpolation =
                new BiLinearInterpolation(sNodes, vNodes, valueSurface);

        return interpolation.apply(spot, variance);
    }

    private int flatten(final int iS, final int iV, final int numberOfSNodes) {
        return iS + iV * numberOfSNodes;
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

        private final FDMHestonModel model;
        private final double[] sNodes;
        private final double[] vNodes;

        private TestSetup(
                final FDMHestonModel model,
                final double[] sNodes,
                final double[] vNodes) {
            this.model = model;
            this.sNodes = sNodes;
            this.vNodes = vNodes;
        }
    }
}