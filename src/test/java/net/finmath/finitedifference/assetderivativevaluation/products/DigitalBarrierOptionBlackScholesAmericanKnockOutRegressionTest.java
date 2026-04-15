package net.finmath.finitedifference.assetderivativevaluation.products;

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
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for American knock-out DigitalBarrierOption under Black-Scholes.
 *
 * <p>Checks:
 * <ul>
 *   <li>American >= Bermudan >= European,</li>
 *   <li>American knock-out <= corresponding American vanilla digital,</li>
 *   <li>Bermudan knock-out <= corresponding Bermudan vanilla digital,</li>
 *   <li>European knock-out <= corresponding European vanilla digital.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalBarrierOptionBlackScholesAmericanKnockOutRegressionTest {

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

    private static final double ORDERING_TOL = 1E-8;
    private static final double VANILLA_COMPARISON_TOL = 1E-6;

    @Test
    public void testDownOutCashCallAmericanRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                80.0
        );
    }

    @Test
    public void testDownOutCashPutAmericanRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                80.0
        );
    }

    @Test
    public void testDownOutAssetCallAmericanRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                80.0
        );
    }

    @Test
    public void testDownOutAssetPutAmericanRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.DOWN_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                80.0
        );
    }

    @Test
    public void testUpOutCashCallAmericanRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.UP_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                120.0
        );
    }

    @Test
    public void testUpOutCashPutAmericanRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.UP_OUT,
                DigitalPayoffType.CASH_OR_NOTHING,
                120.0
        );
    }

    @Test
    public void testUpOutAssetCallAmericanRegression() {
        runRegressionTest(
                CallOrPut.CALL,
                BarrierType.UP_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                120.0
        );
    }

    @Test
    public void testUpOutAssetPutAmericanRegression() {
        runRegressionTest(
                CallOrPut.PUT,
                BarrierType.UP_OUT,
                DigitalPayoffType.ASSET_OR_NOTHING,
                120.0
        );
    }

    private void runRegressionTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final DigitalPayoffType digitalPayoffType,
            final double barrier) {

        final FDMBlackScholesModel model = createBlackScholesModel(barrier, barrierType);

        final Exercise europeanExercise = new EuropeanExercise(MATURITY);
        final Exercise bermudanExercise = new BermudanExercise(
                new double[] { 0.25, 0.50, 0.75, MATURITY }
        );
        final Exercise americanExercise = new AmericanExercise(0.0, MATURITY);

        final DigitalBarrierOption europeanKnockOut = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                digitalPayoffType,
                CASH_PAYOFF,
                europeanExercise
        );

        final DigitalBarrierOption bermudanKnockOut = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                digitalPayoffType,
                CASH_PAYOFF,
                bermudanExercise
        );

        final DigitalBarrierOption americanKnockOut = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                digitalPayoffType,
                CASH_PAYOFF,
                americanExercise
        );

        final DigitalOption europeanVanilla = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                digitalPayoffType,
                CASH_PAYOFF,
                europeanExercise
        );

        final DigitalOption bermudanVanilla = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                digitalPayoffType,
                CASH_PAYOFF,
                bermudanExercise
        );

        final DigitalOption americanVanilla = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                digitalPayoffType,
                CASH_PAYOFF,
                americanExercise
        );

        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        final double europeanKnockOutValue = extractValueAtSpot(
                europeanKnockOut.getValue(0.0, model),
                sNodes,
                S0
        );

        final double bermudanKnockOutValue = extractValueAtSpot(
                bermudanKnockOut.getValue(0.0, model),
                sNodes,
                S0
        );

        final double americanKnockOutValue = extractValueAtSpot(
                americanKnockOut.getValue(0.0, model),
                sNodes,
                S0
        );

        final double europeanVanillaValue = extractValueAtSpot(
                europeanVanilla.getValue(0.0, model),
                sNodes,
                S0
        );

        final double bermudanVanillaValue = extractValueAtSpot(
                bermudanVanilla.getValue(0.0, model),
                sNodes,
                S0
        );

        final double americanVanillaValue = extractValueAtSpot(
                americanVanilla.getValue(0.0, model),
                sNodes,
                S0
        );

        System.out.println("American knock-out regression test");
        System.out.println("Barrier type         = " + barrierType);
        System.out.println("Option type          = " + callOrPut);
        System.out.println("Digital payoff       = " + digitalPayoffType);
        System.out.println("European knock-out   = " + europeanKnockOutValue);
        System.out.println("Bermudan knock-out   = " + bermudanKnockOutValue);
        System.out.println("American knock-out   = " + americanKnockOutValue);
        System.out.println("European vanilla     = " + europeanVanillaValue);
        System.out.println("Bermudan vanilla     = " + bermudanVanillaValue);
        System.out.println("American vanilla     = " + americanVanillaValue);
        System.out.println("B - E                = " + (bermudanKnockOutValue - europeanKnockOutValue));
        System.out.println("A - B                = " + (americanKnockOutValue - bermudanKnockOutValue));
        System.out.println("A barrier - A van    = " + (americanKnockOutValue - americanVanillaValue));

        assertTrue(europeanKnockOutValue >= -1E-10);
        assertTrue(bermudanKnockOutValue >= -1E-10);
        assertTrue(americanKnockOutValue >= -1E-10);

        assertTrue(europeanVanillaValue >= -1E-10);
        assertTrue(bermudanVanillaValue >= -1E-10);
        assertTrue(americanVanillaValue >= -1E-10);

        assertTrue(
                "Bermudan knock-out should be >= European knock-out for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                bermudanKnockOutValue + ORDERING_TOL >= europeanKnockOutValue
        );

        assertTrue(
                "American knock-out should be >= Bermudan knock-out for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                americanKnockOutValue + ORDERING_TOL >= bermudanKnockOutValue
        );

        assertTrue(
                "European knock-out should be <= corresponding European vanilla digital for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                europeanKnockOutValue <= europeanVanillaValue + VANILLA_COMPARISON_TOL
        );

        assertTrue(
                "Bermudan knock-out should be <= corresponding Bermudan vanilla digital for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                bermudanKnockOutValue <= bermudanVanillaValue + VANILLA_COMPARISON_TOL
        );

        assertTrue(
                "American knock-out should be <= corresponding American vanilla digital for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                americanKnockOutValue <= americanVanillaValue + VANILLA_COMPARISON_TOL
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