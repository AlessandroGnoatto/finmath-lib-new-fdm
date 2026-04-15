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
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for Bermudan knock-out DigitalBarrierOption under Black-Scholes.
 *
 * <p>Checks:
 * <ul>
 *   <li>one-date Bermudan is approximately equal to European,</li>
 *   <li>multi-date Bermudan >= one-date Bermudan,</li>
 *   <li>multi-date Bermudan >= European.</li>
 * </ul>
 *
 * <p>
 * A one-date Bermudan with sole exercise at maturity is mathematically equivalent
 * to the European product. However, in the current implementation the numerical
 * schemes differ:
 * </p>
 *
 * <ul>
 *   <li>European 1D knock-out uses cell-averaged terminal initialization,</li>
 *   <li>Bermudan knock-out uses pointwise terminal/exercise payoff handling.</li>
 * </ul>
 *
 * <p>
 * Hence we test approximate equality rather than machine-close equality.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalBarrierOptionBlackScholesBermudanKnockOutRegressionTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 120;
    private static final int NUMBER_OF_SPACE_STEPS = 320;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    private static final double ONE_DATE_EQUALITY_TOL_CASH = 0.08;
    private static final double ONE_DATE_EQUALITY_TOL_ASSET = 0.75;
    private static final double ORDERING_TOL = 1E-8;

    @Test
    public void testDownOutCashCallRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                80.0,
                ONE_DATE_EQUALITY_TOL_CASH
        );
    }

    @Test
    public void testDownOutCashPutRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                80.0,
                ONE_DATE_EQUALITY_TOL_CASH
        );
    }

    @Test
    public void testDownOutAssetCallRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                80.0,
                ONE_DATE_EQUALITY_TOL_ASSET
        );
    }

    @Test
    public void testDownOutAssetPutRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                80.0,
                ONE_DATE_EQUALITY_TOL_ASSET
        );
    }

    @Test
    public void testUpOutCashCallRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.UP_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                120.0,
                ONE_DATE_EQUALITY_TOL_CASH
        );
    }

    @Test
    public void testUpOutCashPutRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.UP_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                120.0,
                ONE_DATE_EQUALITY_TOL_CASH
        );
    }

    @Test
    public void testUpOutAssetCallRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.UP_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                120.0,
                ONE_DATE_EQUALITY_TOL_ASSET
        );
    }

    @Test
    public void testUpOutAssetPutRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.UP_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                120.0,
                ONE_DATE_EQUALITY_TOL_ASSET
        );
    }

    private void runRegressionTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final DigitalPayoffType digitalPayoffType,
            final double barrier,
            final double oneDateEqualityTolerance) {

        final FDMBlackScholesModel model = createBlackScholesModel(barrier, barrierType);

        final DigitalBarrierOption european = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                digitalPayoffType,
                CASH_PAYOFF,
                new EuropeanExercise(MATURITY)
        );

        final Exercise oneDateBermudanExercise = new BermudanExercise(new double[] { MATURITY });

        final DigitalBarrierOption bermudanOneDate = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                digitalPayoffType,
                CASH_PAYOFF,
                oneDateBermudanExercise
        );

        final Exercise multiDateBermudanExercise = new BermudanExercise(
                new double[] { 0.25, 0.50, 0.75, MATURITY }
        );

        final DigitalBarrierOption bermudanMultiDate = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                digitalPayoffType,
                CASH_PAYOFF,
                multiDateBermudanExercise
        );

        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        final double europeanValue = extractValueAtSpot(
                european.getValue(0.0, model),
                sNodes,
                S0
        );

        final double oneDateBermudanValue = extractValueAtSpot(
                bermudanOneDate.getValue(0.0, model),
                sNodes,
                S0
        );

        final double multiDateBermudanValue = extractValueAtSpot(
                bermudanMultiDate.getValue(0.0, model),
                sNodes,
                S0
        );

        System.out.println("Regression test");
        System.out.println("Barrier type       = " + barrierType);
        System.out.println("Option type        = " + callOrPut);
        System.out.println("Digital payoff     = " + digitalPayoffType);
        System.out.println("European           = " + europeanValue);
        System.out.println("One-date Bermudan  = " + oneDateBermudanValue);
        System.out.println("Multi-date Berm.   = " + multiDateBermudanValue);
        System.out.println("One-date diff      = " + Math.abs(oneDateBermudanValue - europeanValue));
        System.out.println("Multi - one-date   = " + (multiDateBermudanValue - oneDateBermudanValue));
        System.out.println("Multi - european   = " + (multiDateBermudanValue - europeanValue));

        assertTrue(europeanValue >= -1E-10);
        assertTrue(oneDateBermudanValue >= -1E-10);
        assertTrue(multiDateBermudanValue >= -1E-10);

        assertEquals(
                "One-date Bermudan should approximately match European for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                europeanValue,
                oneDateBermudanValue,
                oneDateEqualityTolerance
        );

        assertTrue(
                "Multi-date Bermudan should be >= one-date Bermudan for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                multiDateBermudanValue + ORDERING_TOL >= oneDateBermudanValue
        );

        assertTrue(
                "Multi-date Bermudan should be >= European for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                multiDateBermudanValue + ORDERING_TOL >= europeanValue
        );
    }

    private FDMBlackScholesModel createBlackScholesModel(
            final double barrier,
            final BarrierType barrierType) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createKnockOutGrid(barrier, barrierType);

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

    private Grid createKnockOutGrid(final double barrier, final BarrierType barrierType) {

        if(barrierType != BarrierType.DOWN_OUT && barrierType != BarrierType.UP_OUT) {
            throw new IllegalArgumentException("This regression test supports knock-out barriers only.");
        }

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_OUT) {
            final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;
            final double sMin = barrier;
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else {
            final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;
            final double sMax = barrier;
            final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
    }

    private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
        final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(S0 - barrier)));
        final int cappedByGrid = NUMBER_OF_SPACE_STEPS - 2;
        return Math.max(1, Math.min(Math.min(STEPS_BETWEEN_BARRIER_AND_SPOT, cappedByGrid), naturalSteps));
    }

    private double extractValueAtSpot(final double[] valuesOnGrid, final double[] sNodes, final double spot) {
        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );

        if(isGridNode(sNodes, spot)) {
            return valuesOnGrid[getGridIndex(sNodes, spot)];
        }

        final PolynomialSplineFunction interpolation =
                new LinearInterpolator().interpolate(sNodes, valuesOnGrid);
        return interpolation.value(spot);
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