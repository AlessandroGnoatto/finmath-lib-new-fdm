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
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * In-out parity regression tests for European DigitalBarrierOption under Black-Scholes.
 *
 * <p>
 * For European digital barriers, the identity
 * </p>
 *
 * <pre>
 * knock-in + knock-out = corresponding vanilla digital
 * </pre>
 *
 * <p>
 * should hold for the same strike, maturity, barrier, and payoff style.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalBarrierOptionBlackScholesInOutParityTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS = 300;

    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;
    private static final int DOWN_IN_PUT_EXTRA_STEPS = 160;
    private static final int UP_IN_CALL_EXTRA_STEPS = 160;

    @Test
    public void testDownBarrierCashCallParity() {
        runParityTest(CallOrPut.CALL, DigitalPayoffType.CASH_OR_NOTHING, 80.0, true, 0.12);
    }

    @Test
    public void testUpBarrierCashCallParity() {
        runParityTest(CallOrPut.CALL, DigitalPayoffType.CASH_OR_NOTHING, 120.0, false, 0.12);
    }

    @Test
    public void testDownBarrierCashPutParity() {
        runParityTest(CallOrPut.PUT, DigitalPayoffType.CASH_OR_NOTHING, 80.0, true, 0.12);
    }

    @Test
    public void testUpBarrierCashPutParity() {
        runParityTest(CallOrPut.PUT, DigitalPayoffType.CASH_OR_NOTHING, 120.0, false, 0.12);
    }

    @Test
    public void testDownBarrierAssetCallParity() {
        runParityTest(CallOrPut.CALL, DigitalPayoffType.ASSET_OR_NOTHING, 80.0, true, 0.35);
    }

    @Test
    public void testUpBarrierAssetCallParity() {
        runParityTest(CallOrPut.CALL, DigitalPayoffType.ASSET_OR_NOTHING, 120.0, false, 0.35);
    }

    @Test
    public void testDownBarrierAssetPutParity() {
        runParityTest(CallOrPut.PUT, DigitalPayoffType.ASSET_OR_NOTHING, 80.0, true, 0.35);
    }

    @Test
    public void testUpBarrierAssetPutParity() {
        runParityTest(CallOrPut.PUT, DigitalPayoffType.ASSET_OR_NOTHING, 120.0, false, 0.35);
    }

    private void runParityTest(
            final CallOrPut callOrPut,
            final DigitalPayoffType digitalPayoffType,
            final double barrier,
            final boolean isDownBarrier,
            final double tolerance) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final BarrierType knockInType = isDownBarrier ? BarrierType.DOWN_IN : BarrierType.UP_IN;
        final BarrierType knockOutType = isDownBarrier ? BarrierType.DOWN_OUT : BarrierType.UP_OUT;

        final Grid knockInGrid = createGrid(barrier, knockInType, callOrPut);
        final Grid knockOutGrid = createGrid(barrier, knockOutType, callOrPut);

        final SpaceTimeDiscretization knockInSpaceTime = new SpaceTimeDiscretization(
                knockInGrid,
                timeDiscretization,
                THETA,
                new double[] { S0 }
        );

        final SpaceTimeDiscretization knockOutSpaceTime = new SpaceTimeDiscretization(
                knockOutGrid,
                timeDiscretization,
                THETA,
                new double[] { S0 }
        );

        final FDMBlackScholesModel knockInModel = new FDMBlackScholesModel(
                S0,
                riskFreeCurve,
                dividendCurve,
                SIGMA,
                knockInSpaceTime
        );

        final FDMBlackScholesModel knockOutModel = new FDMBlackScholesModel(
                S0,
                riskFreeCurve,
                dividendCurve,
                SIGMA,
                knockOutSpaceTime
        );

        final DigitalBarrierOption knockIn = new DigitalBarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                knockInType,
                digitalPayoffType,
                CASH_PAYOFF
        );

        final DigitalBarrierOption knockOut = new DigitalBarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                knockOutType,
                digitalPayoffType,
                CASH_PAYOFF
        );

        final DigitalOption vanilla = new DigitalOption(
                MATURITY,
                STRIKE,
                callOrPut,
                digitalPayoffType,
                CASH_PAYOFF
        );

        final double knockInValue = extractValueAtSpot(knockIn.getValue(0.0, knockInModel), knockInSpaceTime.getSpaceGrid(0).getGrid(), S0);
        final double knockOutValue = extractValueAtSpot(knockOut.getValue(0.0, knockOutModel), knockOutSpaceTime.getSpaceGrid(0).getGrid(), S0);
        final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, knockInModel), knockInSpaceTime.getSpaceGrid(0).getGrid(), S0);

        System.out.println("Barrier side    = " + (isDownBarrier ? "DOWN" : "UP"));
        System.out.println("Option type     = " + callOrPut);
        System.out.println("Digital payoff  = " + digitalPayoffType);
        System.out.println("Knock-in value  = " + knockInValue);
        System.out.println("Knock-out value = " + knockOutValue);
        System.out.println("Vanilla value   = " + vanillaValue);
        System.out.println("Parity error    = " + ((knockInValue + knockOutValue) - vanillaValue));

        assertTrue(knockInValue >= -1E-10);
        assertTrue(knockOutValue >= -1E-10);
        assertTrue(vanillaValue >= -1E-10);

        assertEquals(
                "European in-out parity failed for "
                        + (isDownBarrier ? "DOWN" : "UP") + " barrier, "
                        + callOrPut + ", " + digitalPayoffType,
                vanillaValue,
                knockInValue + knockOutValue,
                tolerance
        );
    }

    private Grid createGrid(
            final double barrier,
            final BarrierType barrierType,
            final CallOrPut callOrPut) {

        if(barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.UP_OUT) {
            return createKnockOutGrid(barrier, barrierType);
        }
        else if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN) {
            return createKnockInInteriorGrid(barrier, barrierType, callOrPut);
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
            final BarrierType barrierType,
            final CallOrPut callOrPut) {

        final int effectiveStepsBetweenBarrierAndSpot =
                getEffectiveStepsBetweenBarrierAndSpot(barrier);

        if(barrierType == BarrierType.DOWN_IN) {
            final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;

            final int desiredExtraStepsBelowBarrier =
                    callOrPut == CallOrPut.PUT
                    ? DOWN_IN_PUT_EXTRA_STEPS
                    : DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS;

            final int maxExtraStepsBelowBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsBelowBarrier =
                    Math.max(1, Math.min(desiredExtraStepsBelowBarrier, maxExtraStepsBelowBarrier));

            final double sMin = barrier - extraStepsBelowBarrier * deltaS;
            final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;

            validateInteriorBarrierPlacement(sMin, barrier, deltaS);

            return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
        }
        else if(barrierType == BarrierType.UP_IN) {
            final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;

            final int desiredExtraStepsAboveBarrier =
                    callOrPut == CallOrPut.CALL
                    ? UP_IN_CALL_EXTRA_STEPS
                    : DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS;

            final int maxExtraStepsAboveBarrier =
                    NUMBER_OF_SPACE_STEPS - effectiveStepsBetweenBarrierAndSpot - 1;

            final int extraStepsAboveBarrier =
                    Math.max(1, Math.min(desiredExtraStepsAboveBarrier, maxExtraStepsAboveBarrier));

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