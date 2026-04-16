package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
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
 * under the Black-Scholes model.
 *
 * <p>
 * This class checks:
 * </p>
 * <ul>
 *   <li>double knock-in + double knock-out ~= vanilla,</li>
 *   <li>a very wide double knock-out band approaches the vanilla price,</li>
 *   <li>for knock-out options, widening the alive band increases the price.</li>
 * </ul>
 */
public class DoubleBarrierOptionParityTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;

    /**
     * Fixed grid chosen so that all barriers below are nodes.
     * Step size = (160 - 40) / 240 = 0.5.
     */
    private static final int NUMBER_OF_SPACE_STEPS = 240;
    private static final double GRID_MIN = 40.0;
    private static final double GRID_MAX = 160.0;

    private static final double PARITY_TOLERANCE = 1E-8;
    private static final double WIDE_BAND_TOLERANCE = 3E-1;
    private static final double MONOTONICITY_TOLERANCE = 1E-10;

    @Test
    public void testInOutParityCall() {
        runInOutParityTest(CallOrPut.CALL, 80.0, 120.0);
    }

    @Test
    public void testInOutParityPut() {
        runInOutParityTest(CallOrPut.PUT, 80.0, 120.0);
    }

    @Test
    public void testWideBandKnockOutApproachesVanillaCall() {
        runWideBandLimitTest(CallOrPut.CALL, 10.0, 250.0, createWideGrid(), 4E-1);
    }

    @Test
    public void testWideBandKnockOutApproachesVanillaPut() {
        runWideBandLimitTest(CallOrPut.PUT, 10.0, 250.0, createWideGrid(), 4E-1);
    }

    @Test
    public void testKnockOutBandMonotonicityCall() {
        runKnockOutBandMonotonicityTest(CallOrPut.CALL, 80.0, 120.0, 70.0, 130.0);
    }

    @Test
    public void testKnockOutBandMonotonicityPut() {
        runKnockOutBandMonotonicityTest(CallOrPut.PUT, 80.0, 120.0, 70.0, 130.0);
    }

    private void runInOutParityTest(
            final CallOrPut callOrPut,
            final double lowerBarrier,
            final double upperBarrier) {

        final FDMBlackScholesModel model = createBlackScholesModel(createGrid());

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

        final double knockOutValue = valueAtSpot(knockOut, model, S0);
        final double knockInValue = valueAtSpot(knockIn, model, S0);
        final double vanillaValue = valueAtSpot(vanilla, model, S0);

        System.out.println("Double-barrier in-out parity");
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
                "Double-barrier in-out parity failed for " + callOrPut,
                vanillaValue,
                knockInValue + knockOutValue,
                PARITY_TOLERANCE
        );
    }

    private Grid createWideGrid() {
        return new UniformGrid(600, 0.0, 300.0);
    }

    private void runWideBandLimitTest(
            final CallOrPut callOrPut,
            final double lowerBarrier,
            final double upperBarrier,
            final Grid grid,
            final double tolerance) {

        final FDMBlackScholesModel model = createBlackScholesModel(grid);

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

        final double knockOutValue = valueAtSpot(wideBandKnockOut, model, S0);
        final double vanillaValue = valueAtSpot(vanilla, model, S0);

        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        System.out.println("Double-barrier wide-band limit");
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
                "Wide double-barrier knock-out should approach vanilla for " + callOrPut,
                vanillaValue,
                knockOutValue,
                tolerance
        );
    }

    private void runKnockOutBandMonotonicityTest(
            final CallOrPut callOrPut,
            final double narrowLowerBarrier,
            final double narrowUpperBarrier,
            final double wideLowerBarrier,
            final double wideUpperBarrier) {

        final FDMBlackScholesModel model = createBlackScholesModel(createGrid());

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

        final double narrowValue = valueAtSpot(narrowBandKnockOut, model, S0);
        final double wideValue = valueAtSpot(wideBandKnockOut, model, S0);

        System.out.println("Double-barrier knock-out monotonicity");
        System.out.println("Option type    = " + callOrPut);
        System.out.println("Narrow band    = [" + narrowLowerBarrier + ", " + narrowUpperBarrier + "]");
        System.out.println("Wide band      = [" + wideLowerBarrier + ", " + wideUpperBarrier + "]");
        System.out.println("Narrow value   = " + narrowValue);
        System.out.println("Wide value     = " + wideValue);

        assertTrue(
                "Wider knock-out band should not reduce value for " + callOrPut,
                wideValue + MONOTONICITY_TOLERANCE >= narrowValue
        );
    }

    private FDMBlackScholesModel createBlackScholesModel(final Grid sGrid) {
        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
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

    private Grid createGrid() {
        return new UniformGrid(NUMBER_OF_SPACE_STEPS, GRID_MIN, GRID_MAX);
    }

    private double valueAtSpot(
            final FiniteDifferenceProduct product,
            final FDMBlackScholesModel model,
            final double spot) {

        final double[] values = product.getValue(0.0, model);
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

    private static int getGridIndex(final double[] grid, final double value) {
        final double tolerance = 1E-12;
        for(int i = 0; i < grid.length; i++) {
            if(Math.abs(grid[i] - value) < tolerance) {
                return i;
            }
        }
        return -1;
    }

    private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
        final double[] times = new double[] { 0.0, 1.0 };
        final double[] zeroRates = new double[] { rate, rate };
        final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
        final InterpolationEntity interpolationEntity = InterpolationEntity.VALUE;
        final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;

        return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
                name,
                LocalDate.of(2010, 8, 1),
                times,
                zeroRates,
                interpolationMethod,
                extrapolationMethod,
                interpolationEntity
        );
    }
}