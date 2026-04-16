package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import it.univr.fima.correction.BarrierOptions;
import it.univr.fima.correction.BarrierOptions.BinaryBarrierEventType;
import it.univr.fima.correction.BarrierOptions.BinaryPayoffType;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unified Black-Scholes finite-difference vs analytic regression test for
 * expiry-settled single-barrier cash touch products.
 *
 * <p>
 * Covered products:
 * </p>
 * <ul>
 *   <li>down-in one-touch at expiry,</li>
 *   <li>up-in one-touch at expiry,</li>
 *   <li>down-out no-touch at expiry,</li>
 *   <li>up-out no-touch at expiry.</li>
 * </ul>
 *
 * <p>
 * Grid policy mirrors the current barrier regression tests:
 * </p>
 * <ul>
 *   <li>knock-out / no-touch products use a barrier-on-boundary grid,</li>
 *   <li>knock-in / one-touch products use an interior-barrier grid,</li>
 *   <li>all grids are built so that S0 lies inside the domain and, whenever possible,
 *       exactly on a node.</li>
 * </ul>
 */
public class TouchOptionBlackScholesFdmVsAnalyticTest {

    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS = 300;

    /**
     * Desired number of space intervals between barrier and spot.
     */
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    /**
     * Default interior extension for knock-in grids.
     */
    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;

    @Test
    public void testDownAndInOneTouchAtExpiryBlackScholesFiniteDifferenceVsAnalytic() {
        runTouchTest(BarrierType.DOWN_IN, 80.0, 0.40);
    }

    @Test
    public void testUpAndInOneTouchAtExpiryBlackScholesFiniteDifferenceVsAnalytic() {
        runTouchTest(BarrierType.UP_IN, 120.0, 0.40);
    }

    @Test
    public void testDownAndOutNoTouchAtExpiryBlackScholesFiniteDifferenceVsAnalytic() {
        runTouchTest(BarrierType.DOWN_OUT, 80.0, 0.40);
    }

    @Test
    public void testUpAndOutNoTouchAtExpiryBlackScholesFiniteDifferenceVsAnalytic() {
        runTouchTest(BarrierType.UP_OUT, 120.0, 0.40);
    }

    private void runTouchTest(
            final BarrierType barrierType,
            final double barrier,
            final double tolerance) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createGrid(barrier, barrierType);

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

        final TouchOption fdmProduct = new TouchOption(
                MATURITY,
                barrier,
                barrierType,
                CASH_PAYOFF
        );

        final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
        final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

        assertTrue(
                "S0 must lie inside the grid domain.",
                S0 >= sNodes[0] - 1E-12 && S0 <= sNodes[sNodes.length - 1] + 1E-12
        );

        final double fdPrice;
        if(isGridNode(sNodes, S0)) {
            fdPrice = fdValuesOnGrid[getGridIndex(sNodes, S0)];
        }
        else {
            final PolynomialSplineFunction interpolation =
                    new LinearInterpolator().interpolate(sNodes, fdValuesOnGrid);
            fdPrice = interpolation.value(S0);
        }

        final double analyticPrice = BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                S0,
                R,
                Q,
                SIGMA,
                MATURITY,
                barrier,
                mapBarrierType(barrierType),
                mapEventType(barrierType),
                BinaryPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF
        );

        System.out.println("Barrier type    = " + barrierType);
        System.out.println("Barrier         = " + barrier);
        System.out.println("Grid min        = " + sNodes[0]);
        System.out.println("Grid max        = " + sNodes[sNodes.length - 1]);
        System.out.println("S0 on grid      = " + isGridNode(sNodes, S0));
        System.out.println("FD price        = " + fdPrice);
        System.out.println("Analytic price  = " + analyticPrice);

        assertTrue(fdPrice >= -1E-10);
        assertTrue(analyticPrice >= -1E-10);

        assertEquals(
                "FD vs analytic touch price for " + barrierType,
                analyticPrice,
                fdPrice,
                tolerance
        );
    }

    private Grid createGrid(
            final double barrier,
            final BarrierType barrierType) {

        if(barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.UP_OUT) {
            return createKnockOutGrid(barrier, barrierType);
        }
        else if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN) {
            return createKnockInInteriorGrid(barrier, barrierType);
        }
        else {
            throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
        }
    }

    private Grid createKnockOutGrid(final double barrier, final BarrierType barrierType) {

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_OUT) {
            final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;
            final double sMin = barrier;
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.UP_OUT) {
            final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;
            final double sMax = barrier;
            final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else {
            throw new IllegalArgumentException("Knock-out grid requested for non knock-out type: " + barrierType);
        }
    }

    private Grid createKnockInInteriorGrid(
            final double barrier,
            final BarrierType barrierType) {

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_IN) {
            final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;

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
            final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;

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
            throw new IllegalArgumentException("Knock-in interior grid requested for non knock-in type: " + barrierType);
        }
    }

    private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
        final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(S0 - barrier)));
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

    private static BarrierOptions.BarrierType mapBarrierType(final BarrierType barrierType) {
        switch(barrierType) {
        case DOWN_IN:
            return BarrierOptions.BarrierType.DOWN_IN;
        case DOWN_OUT:
            return BarrierOptions.BarrierType.DOWN_OUT;
        case UP_IN:
            return BarrierOptions.BarrierType.UP_IN;
        case UP_OUT:
            return BarrierOptions.BarrierType.UP_OUT;
        default:
            throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
        }
    }

    private static BinaryBarrierEventType mapEventType(final BarrierType barrierType) {
        switch(barrierType) {
        case DOWN_IN:
        case UP_IN:
            return BinaryBarrierEventType.HIT;
        case DOWN_OUT:
        case UP_OUT:
            return BinaryBarrierEventType.NO_HIT;
        default:
            throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
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