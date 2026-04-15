package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.GridWithMandatoryPoint;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
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
 * Consolidated structural regression test for DigitalBarrierOption under the
 * CEV model.
 *
 * <p>For each product configuration, the test checks:
 * <ul>
 *   <li>Bermudan >= European,</li>
 *   <li>American >= Bermudan,</li>
 *   <li>barrier European <= corresponding vanilla European digital,</li>
 *   <li>barrier Bermudan <= corresponding vanilla Bermudan digital,</li>
 *   <li>barrier American <= corresponding vanilla American digital.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalBarrierOptionCevRegressionTest {

    private static final double SPOT = 100.0;
    private static final double STRIKE = 100.0;
    private static final double MATURITY = 1.0;

    private static final double VOLATILITY = 0.20;
    private static final double EXPONENT = 0.8;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.02;
    private static final double CASH_PAYOFF = 10.0;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 200;
    private static final int NUMBER_OF_SPACE_STEPS = 400;
    private static final int NUMBER_OF_STANDARD_DEVIATIONS = 6;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;
    private static final int DOWN_IN_PUT_EXTRA_STEPS = 160;
    private static final int UP_IN_CALL_EXTRA_STEPS = 160;

    private static final double ORDERING_TOLERANCE = 1E-10;
    private static final double DOMINANCE_TOLERANCE = 5E-2;

    @Test
    public void testDownOutCashCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.DOWN_OUT, DigitalPayoffType.CASH_OR_NOTHING, 80.0);
    }

    @Test
    public void testDownOutCashPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.DOWN_OUT, DigitalPayoffType.CASH_OR_NOTHING, 80.0);
    }

    @Test
    public void testDownOutAssetCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.DOWN_OUT, DigitalPayoffType.ASSET_OR_NOTHING, 80.0);
    }

    @Test
    public void testDownOutAssetPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.DOWN_OUT, DigitalPayoffType.ASSET_OR_NOTHING, 80.0);
    }

    @Test
    public void testUpOutCashCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.UP_OUT, DigitalPayoffType.CASH_OR_NOTHING, 120.0);
    }

    @Test
    public void testUpOutCashPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.UP_OUT, DigitalPayoffType.CASH_OR_NOTHING, 120.0);
    }

    @Test
    public void testUpOutAssetCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.UP_OUT, DigitalPayoffType.ASSET_OR_NOTHING, 120.0);
    }

    @Test
    public void testUpOutAssetPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.UP_OUT, DigitalPayoffType.ASSET_OR_NOTHING, 120.0);
    }

    @Test
    public void testDownInCashCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.DOWN_IN, DigitalPayoffType.CASH_OR_NOTHING, 80.0);
    }

    @Test
    public void testDownInCashPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.DOWN_IN, DigitalPayoffType.CASH_OR_NOTHING, 80.0);
    }

    @Test
    public void testDownInAssetCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.DOWN_IN, DigitalPayoffType.ASSET_OR_NOTHING, 80.0);
    }

    @Test
    public void testDownInAssetPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.DOWN_IN, DigitalPayoffType.ASSET_OR_NOTHING, 80.0);
    }

    @Test
    public void testUpInCashCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.UP_IN, DigitalPayoffType.CASH_OR_NOTHING, 120.0);
    }

    @Test
    public void testUpInCashPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.UP_IN, DigitalPayoffType.CASH_OR_NOTHING, 120.0);
    }

    @Test
    public void testUpInAssetCallRegression() {
        runRegression(CallOrPut.CALL, BarrierType.UP_IN, DigitalPayoffType.ASSET_OR_NOTHING, 120.0);
    }

    @Test
    public void testUpInAssetPutRegression() {
        runRegression(CallOrPut.PUT, BarrierType.UP_IN, DigitalPayoffType.ASSET_OR_NOTHING, 120.0);
    }

    private void runRegression(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final DigitalPayoffType digitalPayoffType,
            final double barrier) {

        final TestSetup barrierSetup = createBarrierSetup(barrier, barrierType, callOrPut);
        final TestSetup vanillaSetup = createVanillaSetup();

        final Exercise europeanExercise = new EuropeanExercise(MATURITY);
        final Exercise bermudanExercise = new BermudanExercise(new double[] { 0.25, 0.50, 0.75, MATURITY });
        final Exercise americanExercise = new AmericanExercise(0.0, MATURITY);

        final DigitalBarrierOption europeanBarrier = new DigitalBarrierOption(
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

        final DigitalBarrierOption bermudanBarrier = new DigitalBarrierOption(
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

        final DigitalBarrierOption americanBarrier = new DigitalBarrierOption(
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

        final double europeanBarrierValue =
                interpolateAtSpot(europeanBarrier.getValue(0.0, barrierSetup.model), barrierSetup.sNodes, SPOT);
        final double bermudanBarrierValue =
                interpolateAtSpot(bermudanBarrier.getValue(0.0, barrierSetup.model), barrierSetup.sNodes, SPOT);
        final double americanBarrierValue =
                interpolateAtSpot(americanBarrier.getValue(0.0, barrierSetup.model), barrierSetup.sNodes, SPOT);

        final double europeanVanillaValue =
                interpolateAtSpot(europeanVanilla.getValue(0.0, vanillaSetup.model), vanillaSetup.sNodes, SPOT);
        final double bermudanVanillaValue =
                interpolateAtSpot(bermudanVanilla.getValue(0.0, vanillaSetup.model), vanillaSetup.sNodes, SPOT);
        final double americanVanillaValue =
                interpolateAtSpot(americanVanilla.getValue(0.0, vanillaSetup.model), vanillaSetup.sNodes, SPOT);

        System.out.println("CEV regression");
        System.out.println("Barrier type       = " + barrierType);
        System.out.println("Option type        = " + callOrPut);
        System.out.println("Digital payoff     = " + digitalPayoffType);
        System.out.println("European barrier   = " + europeanBarrierValue);
        System.out.println("Bermudan barrier   = " + bermudanBarrierValue);
        System.out.println("American barrier   = " + americanBarrierValue);
        System.out.println("European vanilla   = " + europeanVanillaValue);
        System.out.println("Bermudan vanilla   = " + bermudanVanillaValue);
        System.out.println("American vanilla   = " + americanVanillaValue);

        assertTrue(europeanBarrierValue >= -1E-10);
        assertTrue(bermudanBarrierValue >= -1E-10);
        assertTrue(americanBarrierValue >= -1E-10);

        assertTrue(europeanVanillaValue >= -1E-10);
        assertTrue(bermudanVanillaValue >= -1E-10);
        assertTrue(americanVanillaValue >= -1E-10);

        assertTrue(
                "Bermudan barrier should be >= European barrier for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                bermudanBarrierValue + ORDERING_TOLERANCE >= europeanBarrierValue
        );

        assertTrue(
                "American barrier should be >= Bermudan barrier for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                americanBarrierValue + ORDERING_TOLERANCE >= bermudanBarrierValue
        );

        assertTrue(
                "European barrier should be <= European vanilla for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                europeanBarrierValue <= europeanVanillaValue + DOMINANCE_TOLERANCE
        );

        assertTrue(
                "Bermudan barrier should be <= Bermudan vanilla for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                bermudanBarrierValue <= bermudanVanillaValue + DOMINANCE_TOLERANCE
        );

        assertTrue(
                "American barrier should be <= American vanilla for "
                        + barrierType + " " + callOrPut + " " + digitalPayoffType,
                americanBarrierValue <= americanVanillaValue + DOMINANCE_TOLERANCE
        );
    }

    private TestSetup createBarrierSetup(
            final double barrier,
            final BarrierType barrierType,
            final CallOrPut callOrPut) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createBarrierGrid(barrier, barrierType, callOrPut);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { SPOT }
        );

        final FDMCevModel model = new FDMCevModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                VOLATILITY,
                EXPONENT,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid());
    }

    private TestSetup createVanillaSetup() {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final double forward =
                SPOT * Math.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY);
        final double varianceProxy =
                SPOT * SPOT
                * Math.exp(2.0 * (RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY)
                * (Math.exp(VOLATILITY * VOLATILITY * MATURITY) - 1.0);

        final double sMin =
                Math.max(forward - NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy), 0.0);
        final double sMax =
                forward + NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy);

        final Grid sGrid = new GridWithMandatoryPoint(
                NUMBER_OF_SPACE_STEPS,
                sMin,
                sMax,
                STRIKE
        );

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { SPOT }
        );

        final FDMCevModel model = new FDMCevModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                VOLATILITY,
                EXPONENT,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid());
    }

    private Grid createBarrierGrid(
            final double barrier,
            final BarrierType barrierType,
            final CallOrPut callOrPut) {

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_OUT) {
            final double deltaS = (SPOT - barrier) / effectiveStepsBetweenBarrierAndSpot;
            final double sMin = barrier;
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.UP_OUT) {
            final double deltaS = (barrier - SPOT) / effectiveStepsBetweenBarrierAndSpot;
            final double sMax = barrier;
            final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.DOWN_IN) {
            final double deltaS = (SPOT - barrier) / effectiveStepsBetweenBarrierAndSpot;

            final int desiredExtraStepsBelowBarrier =
                    callOrPut == CallOrPut.PUT
                    ? DOWN_IN_PUT_EXTRA_STEPS
                    : DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS;

            final int maxExtraStepsBelowBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsBelowBarrier =
                    Math.max(1, Math.min(desiredExtraStepsBelowBarrier, maxExtraStepsBelowBarrier));

            final double sMin = Math.max(barrier - extraStepsBelowBarrier * deltaS, 0.0);
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;

            validateInteriorBarrierPlacement(sMin, barrier, deltaS);

            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.UP_IN) {
            final double deltaS = (barrier - SPOT) / effectiveStepsBetweenBarrierAndSpot;

            final int desiredExtraStepsAboveBarrier =
                    callOrPut == CallOrPut.CALL
                    ? UP_IN_CALL_EXTRA_STEPS
                    : DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS;

            final int maxExtraStepsAboveBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsAboveBarrier =
                    Math.max(1, Math.min(desiredExtraStepsAboveBarrier, maxExtraStepsAboveBarrier));

            final double sMax = barrier + extraStepsAboveBarrier * deltaS;
            final double sMin = Math.max(sMax - NUMBER_OF_SPACE_STEPS * deltaS, 0.0);

            validateInteriorBarrierPlacement(sMin, barrier, deltaS);

            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else {
            throw new IllegalArgumentException("Unsupported barrier type.");
        }
    }

    private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
        final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(SPOT - barrier)));
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

    private double interpolateAtSpot(
            final double[] values,
            final double[] sNodes,
            final double spot) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );

        if(isGridNode(sNodes, spot)) {
            return values[getGridIndex(sNodes, spot)];
        }

        final PolynomialSplineFunction interpolation =
                new LinearInterpolator().interpolate(sNodes, values);
        return interpolation.value(spot);
    }

    private boolean isGridNode(final double[] grid, final double x) {
        for(final double node : grid) {
            if(Math.abs(node - x) < 1E-12) {
                return true;
            }
        }
        return false;
    }

    private int getGridIndex(final double[] grid, final double x) {
        for(int i = 0; i < grid.length; i++) {
            if(Math.abs(grid[i] - x) < 1E-12) {
                return i;
            }
        }
        throw new IllegalArgumentException("Point is not a grid node.");
    }

    private static class TestSetup {

        private final FDMCevModel model;
        private final double[] sNodes;

        private TestSetup(final FDMCevModel model, final double[] sNodes) {
            this.model = model;
            this.sNodes = sNodes;
        }
    }
}