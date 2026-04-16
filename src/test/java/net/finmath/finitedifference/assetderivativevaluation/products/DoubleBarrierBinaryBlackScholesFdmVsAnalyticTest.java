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
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unified Black-Scholes finite-difference vs analytic regression test for
 * continuously monitored double-barrier cash binaries.
 *
 * <p>
 * Covered products:
 * </p>
 * <ul>
 *   <li>KNOCK_OUT,</li>
 *   <li>KNOCK_IN,</li>
 *   <li>KIKO,</li>
 *   <li>KOKI.</li>
 * </ul>
 */
public class DoubleBarrierBinaryBlackScholesFdmVsAnalyticTest {

    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

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

    private static final int MAX_ITER_STANDARD = 200;
    private static final int MAX_ITER_KIKO_KOKI = 5000;

    private static final double CONVERGENCE_TOLERANCE_STANDARD = 1E-10;
    private static final double CONVERGENCE_TOLERANCE_KIKO_KOKI = 1E-8;

    private static final double PRICE_TOLERANCE_STANDARD = 0.75;
    private static final double PRICE_TOLERANCE_KIKO_KOKI = 1.00;

    @Test
    public void testKnockOutBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierBinaryTest(DoubleBarrierType.KNOCK_OUT);
    }

    @Test
    public void testKnockInBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierBinaryTest(DoubleBarrierType.KNOCK_IN);
    }

    @Test
    public void testKikoBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierBinaryTest(DoubleBarrierType.KIKO);
    }

    @Test
    public void testKokiBlackScholesFiniteDifferenceVsAnalytic() {
        runDoubleBarrierBinaryTest(DoubleBarrierType.KOKI);
    }

    private void runDoubleBarrierBinaryTest(final DoubleBarrierType doubleBarrierType) {

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

        final DoubleBarrierBinaryOption fdmProduct = new DoubleBarrierBinaryOption(
                MATURITY,
                CASH_PAYOFF,
                LOWER_BARRIER,
                UPPER_BARRIER,
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

        final boolean isKikoOrKoki =
                doubleBarrierType == DoubleBarrierType.KIKO
                || doubleBarrierType == DoubleBarrierType.KOKI;

        final int maxIter =
                isKikoOrKoki ? MAX_ITER_KIKO_KOKI : MAX_ITER_STANDARD;

        final double convergenceTolerance =
                isKikoOrKoki ? CONVERGENCE_TOLERANCE_KIKO_KOKI : CONVERGENCE_TOLERANCE_STANDARD;

        final double priceTolerance =
                isKikoOrKoki ? PRICE_TOLERANCE_KIKO_KOKI : PRICE_TOLERANCE_STANDARD;

        final double analyticPrice = BarrierOptions.blackScholesDoubleBarrierCashBinaryValue(
                S0,
                R,
                Q,
                SIGMA,
                MATURITY,
                CASH_PAYOFF,
                LOWER_BARRIER,
                UPPER_BARRIER,
                mapDoubleBarrierType(doubleBarrierType),
                maxIter,
                convergenceTolerance
        );

        System.out.println("Double barrier binary type = " + doubleBarrierType);
        System.out.println("Grid min                   = " + sNodes[0]);
        System.out.println("Grid max                   = " + sNodes[sNodes.length - 1]);
        System.out.println("Lower on grid              = " + isGridNode(sNodes, LOWER_BARRIER));
        System.out.println("S0 on grid                 = " + isGridNode(sNodes, S0));
        System.out.println("Upper on grid              = " + isGridNode(sNodes, UPPER_BARRIER));
        System.out.println("FD price                   = " + fdPrice);
        System.out.println("Analytic price             = " + analyticPrice);
        System.out.println("Max iter                   = " + maxIter);
        System.out.println("Conv tol                   = " + convergenceTolerance);

        assertTrue(fdPrice >= -1E-10);
        assertTrue(analyticPrice >= -1E-10);

        assertEquals(
                "FD vs analytic double-barrier binary price for " + doubleBarrierType,
                analyticPrice,
                fdPrice,
                priceTolerance
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
        case KIKO:
            return BarrierOptions.DoubleBarrierType.KIKO;
        case KOKI:
            return BarrierOptions.DoubleBarrierType.KOKI;
        default:
            throw new IllegalArgumentException("Unsupported double barrier type.");
        }
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