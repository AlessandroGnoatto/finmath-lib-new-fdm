package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import it.univr.fima.correction.BarrierOptions;
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
 * Unified Black-Scholes finite-difference vs analytic regression test for
 * European vanilla double-barrier options.
 *
 * <p>
 * Covered products:
 * </p>
 * <ul>
 *   <li>call / put,</li>
 *   <li>knock-out and knock-in.</li>
 * </ul>
 *
 * <p>
 * Grid policy:
 * </p>
 * <ul>
 *   <li>use one fixed uniform grid for all cases,</li>
 *   <li>ensure lower barrier, spot, and upper barrier are all grid nodes,</li>
 *   <li>use the same grid for knock-out and knock-in so parity-based in-pricing
 *       is tested on exactly the same discretization.</li>
 * </ul>
 */
public class DoubleBarrierOptionBlackScholesFdmVsAnalyticTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;

    private static final double LOWER_BARRIER = 80.0;
    private static final double UPPER_BARRIER = 120.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;

    /**
     * Fixed grid chosen so that 80, 100, and 120 are all nodes.
     * Step size = (160 - 40) / 240 = 0.5.
     */
    private static final int NUMBER_OF_SPACE_STEPS = 240;
    private static final double GRID_MIN = 40.0;
    private static final double GRID_MAX = 160.0;

    /**
     * Number of image terms retained in the analytic helper series.
     */
    private static final int SERIES = 12;

    @Test
    public void testKnockOutCallBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierTest(CallOrPut.CALL, DoubleBarrierType.KNOCK_OUT, 0.60);
    }

    @Test
    public void testKnockInCallBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierTest(CallOrPut.CALL, DoubleBarrierType.KNOCK_IN, 0.60);
    }

    @Test
    public void testKnockOutPutBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierTest(CallOrPut.PUT, DoubleBarrierType.KNOCK_OUT, 0.60);
    }

    @Test
    public void testKnockInPutBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierTest(CallOrPut.PUT, DoubleBarrierType.KNOCK_IN, 0.60);
    }

    private void runDoubleBarrierTest(
            final CallOrPut callOrPut,
            final DoubleBarrierType doubleBarrierType,
            final double tolerance) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createGrid();

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { S0 }
        );

        final FDMBlackScholesModel fdmModel = new FDMBlackScholesModel(
                S0,
                riskFreeCurve,
                dividendCurve,
                SIGMA,
                spaceTime
        );

        final DoubleBarrierOption fdmProduct = new DoubleBarrierOption(
                MATURITY,
                STRIKE,
                LOWER_BARRIER,
                UPPER_BARRIER,
                callOrPut,
                doubleBarrierType
        );

        final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
        final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

        assertTrue(
                "S0 must lie inside the grid domain.",
                S0 >= sNodes[0] - 1E-12 && S0 <= sNodes[sNodes.length - 1] + 1E-12
        );
        assertTrue("Lower barrier must be a grid node.", isGridNode(sNodes, LOWER_BARRIER));
        assertTrue("Spot must be a grid node.", isGridNode(sNodes, S0));
        assertTrue("Upper barrier must be a grid node.", isGridNode(sNodes, UPPER_BARRIER));

        final double fdPrice;
        if(isGridNode(sNodes, S0)) {
            fdPrice = fdValuesOnGrid[getGridIndex(sNodes, S0)];
        }
        else {
            final PolynomialSplineFunction interpolation =
                    new LinearInterpolator().interpolate(sNodes, fdValuesOnGrid);
            fdPrice = interpolation.value(S0);
        }

        final double analyticPrice = BarrierOptions.blackScholesDoubleBarrierOptionValue(
                S0,
                R,
                Q,
                SIGMA,
                MATURITY,
                STRIKE,
                callOrPut == CallOrPut.CALL,
                LOWER_BARRIER,
                UPPER_BARRIER,
                mapDoubleBarrierType(doubleBarrierType),
                SERIES
        );

        System.out.println("Double barrier type = " + doubleBarrierType);
        System.out.println("Option type         = " + callOrPut);
        System.out.println("Grid min            = " + sNodes[0]);
        System.out.println("Grid max            = " + sNodes[sNodes.length - 1]);
        System.out.println("Lower on grid       = " + isGridNode(sNodes, LOWER_BARRIER));
        System.out.println("S0 on grid          = " + isGridNode(sNodes, S0));
        System.out.println("Upper on grid       = " + isGridNode(sNodes, UPPER_BARRIER));
        System.out.println("FD price            = " + fdPrice);
        System.out.println("Analytic price      = " + analyticPrice);

        assertTrue(fdPrice >= -1E-10);
        assertTrue(analyticPrice >= -1E-10);

        assertEquals(
                "FD vs analytic double-barrier price for " + doubleBarrierType + " " + callOrPut,
                analyticPrice,
                fdPrice,
                tolerance
        );
    }

    private Grid createGrid() {
        return new UniformGrid(NUMBER_OF_SPACE_STEPS, GRID_MIN, GRID_MAX);
    }

    private static boolean isGridNode(final double[] grid, final double value) {
        return getGridIndex(grid, value) >= 0;
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

    private static BarrierOptions.DoubleBarrierType mapDoubleBarrierType(
            final DoubleBarrierType doubleBarrierType) {

        switch(doubleBarrierType) {
        case KNOCK_IN:
            return BarrierOptions.DoubleBarrierType.KNOCK_IN;
        case KNOCK_OUT:
            return BarrierOptions.DoubleBarrierType.KNOCK_OUT;
        default:
            throw new IllegalArgumentException(
                    "Unsupported vanilla double-barrier type in this test: " + doubleBarrierType);
        }
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