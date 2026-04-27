package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural parity / monotonicity tests for European vanilla double-barrier options
 * under Heston and SABR.
 *
 * <p>
 * This class checks:
 * </p>
 * <ul>
 *   <li>double knock-in + double knock-out ~= vanilla,</li>
 *   <li>a very wide double knock-out band approaches the vanilla price,</li>
 *   <li>for knock-out options, widening the alive band increases the price.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class DoubleBarrierOptionHestonSabrStructuralTest {

    private enum ModelType {
        HESTON,
        SABR
    }

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;

    /*
     * Heston parameters.
     */
    private static final double HESTON_VOLATILITY = 0.25;
    private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
    private static final double HESTON_KAPPA = 1.5;
    private static final double HESTON_THETA_V = HESTON_INITIAL_VARIANCE;
    private static final double HESTON_XI = 0.30;
    private static final double HESTON_RHO = -0.70;

    /*
     * SABR parameters.
     */
    private static final double SABR_INITIAL_VOLATILITY = 0.20;
    private static final double SABR_BETA = 1.0;
    private static final double SABR_NU = 0.30;
    private static final double SABR_RHO = -0.50;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SECOND_STEPS = 100;

    /**
     * Fixed centered spot grid chosen so that 70, 80, 100, 120, 130 are all nodes.
     * Step size = (160 - 40) / 240 = 0.5.
     */
    private static final int NUMBER_OF_SPACE_STEPS = 240;
    private static final double GRID_MIN = 40.0;
    private static final double GRID_MAX = 160.0;

    private static final double PARITY_TOLERANCE = 2.5E-1;
    private static final double WIDE_BAND_TOLERANCE = 5E-1;
    private static final double MONOTONICITY_TOLERANCE = 1E-10;

    @Test
    public void testHestonInOutParityCall() {
        runInOutParityTest(ModelType.HESTON, CallOrPut.CALL, 80.0, 120.0);
    }

    @Test
    public void testHestonInOutParityPut() {
        runInOutParityTest(ModelType.HESTON, CallOrPut.PUT, 80.0, 120.0);
    }

    @Test
    public void testSabrInOutParityCall() {
        runInOutParityTest(ModelType.SABR, CallOrPut.CALL, 80.0, 120.0);
    }

    @Test
    public void testSabrInOutParityPut() {
        runInOutParityTest(ModelType.SABR, CallOrPut.PUT, 80.0, 120.0);
    }

    @Test
    public void testHestonWideBandKnockOutApproachesVanillaCall() {
        runWideBandLimitTest(ModelType.HESTON, CallOrPut.CALL, 10.0, 250.0, createWideGrid(), 6E-1);
    }

    @Test
    public void testHestonWideBandKnockOutApproachesVanillaPut() {
        runWideBandLimitTest(ModelType.HESTON, CallOrPut.PUT, 10.0, 250.0, createWideGrid(), 6E-1);
    }

    @Test
    public void testSabrWideBandKnockOutApproachesVanillaCall() {
        runWideBandLimitTest(ModelType.SABR, CallOrPut.CALL, 10.0, 250.0, createWideGrid(), 6E-1);
    }

    @Test
    public void testSabrWideBandKnockOutApproachesVanillaPut() {
        runWideBandLimitTest(ModelType.SABR, CallOrPut.PUT, 10.0, 250.0, createWideGrid(), 6E-1);
    }

    @Test
    public void testHestonKnockOutBandMonotonicityCall() {
        runKnockOutBandMonotonicityTest(ModelType.HESTON, CallOrPut.CALL, 80.0, 120.0, 70.0, 130.0);
    }

    @Test
    public void testHestonKnockOutBandMonotonicityPut() {
        runKnockOutBandMonotonicityTest(ModelType.HESTON, CallOrPut.PUT, 80.0, 120.0, 70.0, 130.0);
    }

    @Test
    public void testSabrKnockOutBandMonotonicityCall() {
        runKnockOutBandMonotonicityTest(ModelType.SABR, CallOrPut.CALL, 80.0, 120.0, 70.0, 130.0);
    }

    @Test
    public void testSabrKnockOutBandMonotonicityPut() {
        runKnockOutBandMonotonicityTest(ModelType.SABR, CallOrPut.PUT, 80.0, 120.0, 70.0, 130.0);
    }

    private void runInOutParityTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final double lowerBarrier,
            final double upperBarrier) {

        final FiniteDifferenceEquityModel model = createModel(modelType, createGrid());

        final DoubleBarrierOption knockOut = new DoubleBarrierOption(
                MATURITY,
                STRIKE,
                lowerBarrier,
                upperBarrier,
                callOrPut,
                DoubleBarrierType.KNOCK_OUT
        );

        final DoubleBarrierOption knockIn = new DoubleBarrierOption(
                MATURITY,
                STRIKE,
                lowerBarrier,
                upperBarrier,
                callOrPut,
                DoubleBarrierType.KNOCK_IN
        );

        final EuropeanOption vanilla = new EuropeanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final double knockOutValue = valueAtSpotAndSecondState0(knockOut, model, S0);
        final double knockInValue = valueAtSpotAndSecondState0(knockIn, model, S0);
        final double vanillaValue = valueAtSpotAndSecondState0(vanilla, model, S0);

        System.out.println("Double-barrier in-out parity");
        System.out.println("Model          = " + modelType);
        System.out.println("Option type    = " + callOrPut);
        System.out.println("Lower barrier  = " + lowerBarrier);
        System.out.println("Upper barrier  = " + upperBarrier);
        System.out.println("Knock-out      = " + knockOutValue);
        System.out.println("Knock-in       = " + knockInValue);
        System.out.println("Vanilla        = " + vanillaValue);
        System.out.println("Sum            = " + (knockInValue + knockOutValue));

        assertTrue(knockOutValue >= -1E-10);
        assertTrue(knockInValue >= -1E-10);
        assertTrue(vanillaValue >= -1E-10);

        assertEquals(
                "Double-barrier in-out parity failed for " + modelType + " " + callOrPut,
                vanillaValue,
                knockInValue + knockOutValue,
                PARITY_TOLERANCE
        );
    }

    private void runWideBandLimitTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final double lowerBarrier,
            final double upperBarrier,
            final Grid grid,
            final double tolerance) {

        final FiniteDifferenceEquityModel model = createModel(modelType, grid);

        final DoubleBarrierOption wideBandKnockOut = new DoubleBarrierOption(
                MATURITY,
                STRIKE,
                lowerBarrier,
                upperBarrier,
                callOrPut,
                DoubleBarrierType.KNOCK_OUT
        );

        final EuropeanOption vanilla = new EuropeanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final double knockOutValue = valueAtSpotAndSecondState0(wideBandKnockOut, model, S0);
        final double vanillaValue = valueAtSpotAndSecondState0(vanilla, model, S0);

        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        System.out.println("Double-barrier wide-band limit");
        System.out.println("Model          = " + modelType);
        System.out.println("Option type    = " + callOrPut);
        System.out.println("Lower barrier  = " + lowerBarrier);
        System.out.println("Upper barrier  = " + upperBarrier);
        System.out.println("Grid min       = " + sNodes[0]);
        System.out.println("Grid max       = " + sNodes[sNodes.length - 1]);
        System.out.println("Knock-out      = " + knockOutValue);
        System.out.println("Vanilla        = " + vanillaValue);
        System.out.println("Difference     = " + Math.abs(vanillaValue - knockOutValue));

        assertTrue(knockOutValue >= -1E-10);
        assertTrue(vanillaValue >= -1E-10);

        assertEquals(
                "Wide double-barrier knock-out should approach vanilla for "
                        + modelType + " " + callOrPut,
                vanillaValue,
                knockOutValue,
                tolerance
        );
    }

    private void runKnockOutBandMonotonicityTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final double narrowLowerBarrier,
            final double narrowUpperBarrier,
            final double wideLowerBarrier,
            final double wideUpperBarrier) {

        final FiniteDifferenceEquityModel model = createModel(modelType, createGrid());

        final DoubleBarrierOption narrowBandKnockOut = new DoubleBarrierOption(
                MATURITY,
                STRIKE,
                narrowLowerBarrier,
                narrowUpperBarrier,
                callOrPut,
                DoubleBarrierType.KNOCK_OUT
        );

        final DoubleBarrierOption wideBandKnockOut = new DoubleBarrierOption(
                MATURITY,
                STRIKE,
                wideLowerBarrier,
                wideUpperBarrier,
                callOrPut,
                DoubleBarrierType.KNOCK_OUT
        );

        final double narrowValue = valueAtSpotAndSecondState0(narrowBandKnockOut, model, S0);
        final double wideValue = valueAtSpotAndSecondState0(wideBandKnockOut, model, S0);

        System.out.println("Double-barrier knock-out monotonicity");
        System.out.println("Model          = " + modelType);
        System.out.println("Option type    = " + callOrPut);
        System.out.println("Narrow band    = [" + narrowLowerBarrier + ", " + narrowUpperBarrier + "]");
        System.out.println("Wide band      = [" + wideLowerBarrier + ", " + wideUpperBarrier + "]");
        System.out.println("Narrow value   = " + narrowValue);
        System.out.println("Wide value     = " + wideValue);

        assertTrue(
                "Wider knock-out band should not reduce value for "
                        + modelType + " " + callOrPut,
                wideValue + MONOTONICITY_TOLERANCE >= narrowValue
        );
    }

    private FiniteDifferenceEquityModel createModel(
            final ModelType modelType,
            final Grid sGrid) {

        if(modelType == ModelType.HESTON) {
            return createHestonModel(sGrid);
        }
        return createSabrModel(sGrid);
    }

    private FDMHestonModel createHestonModel(final Grid sGrid) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final double vMin = 0.0;
        final double vMax = Math.max(
                4.0 * HESTON_THETA_V,
                HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
        );

        final Grid vGrid = new UniformGrid(NUMBER_OF_SECOND_STEPS, vMin, vMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, vGrid },
                timeDiscretization,
                THETA,
                new double[] { S0, HESTON_INITIAL_VARIANCE }
        );

        return new FDMHestonModel(
                S0,
                HESTON_INITIAL_VARIANCE,
                riskFreeCurve,
                dividendCurve,
                HESTON_KAPPA,
                HESTON_THETA_V,
                HESTON_XI,
                HESTON_RHO,
                spaceTime
        );
    }

    private FDMSabrModel createSabrModel(final Grid sGrid) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final double volMax = Math.max(4.0 * SABR_INITIAL_VOLATILITY, 1.0);
        final Grid volGrid = new UniformGrid(NUMBER_OF_SECOND_STEPS, 0.0, volMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, volGrid },
                timeDiscretization,
                THETA,
                new double[] { S0, SABR_INITIAL_VOLATILITY }
        );

        return new FDMSabrModel(
                S0,
                SABR_INITIAL_VOLATILITY,
                riskFreeCurve,
                dividendCurve,
                SABR_BETA,
                SABR_NU,
                SABR_RHO,
                spaceTime
        );
    }

    private Grid createGrid() {
        return new UniformGrid(NUMBER_OF_SPACE_STEPS, GRID_MIN, GRID_MAX);
    }

    private Grid createWideGrid() {
        return new UniformGrid(600, 0.0, 300.0);
    }

    private double valueAtSpotAndSecondState0(
            final FiniteDifferenceEquityProduct product,
            final FiniteDifferenceEquityModel model,
            final double spot) {

        final double[] values = product.getValue(0.0, model);
        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
        final double[] secondNodes = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

        final double secondState0 = model instanceof FDMHestonModel
                ? HESTON_INITIAL_VARIANCE
                : SABR_INITIAL_VOLATILITY;

        return interpolateAtSpotAndSecondState(values, sNodes, secondNodes, spot, secondState0);
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
}