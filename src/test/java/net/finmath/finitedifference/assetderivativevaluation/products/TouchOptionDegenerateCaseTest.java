package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.BarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Degenerate-case and invalid-usage tests for expiry-settled cash {@link TouchOption}.
 */
public class TouchOptionDegenerateCaseTest {

    private static final double SPOT = 100.0;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.02;
    private static final double VOLATILITY = 20.0;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS = 200;

    private static final double CASH_PAYOFF = 10.0;

    private static final double VALUE_TOLERANCE = 5E-2;
    private static final double ZERO_TOLERANCE = 1E-8;

    @Test
    public void testAlreadyHitOneTouchAtValuationPaysDiscountedCash() {
        final FDMBachelierModel model = createModel(0.0, 200.0, 1.0);

        final TouchOption upAndInAlreadyHit = TouchOption.oneTouchAtExpiry(
                1.0,
                90.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption downAndInAlreadyHit = TouchOption.oneTouchAtExpiry(
                1.0,
                110.0,
                BarrierType.DOWN_IN,
                CASH_PAYOFF
        );

        final double discountedCash = CASH_PAYOFF * Math.exp(-RISK_FREE_RATE * 1.0);

        final double upAndInValue = valueAtSpot(upAndInAlreadyHit, model, SPOT);
        final double downAndInValue = valueAtSpot(downAndInAlreadyHit, model, SPOT);

        assertEquals(discountedCash, upAndInValue, VALUE_TOLERANCE);
        assertEquals(discountedCash, downAndInValue, VALUE_TOLERANCE);
    }

    @Test
    public void testAlreadyKnockedOutNoTouchAtValuationIsZero() {
        final FDMBachelierModel model = createModel(0.0, 200.0, 1.0);

        final TouchOption upAndOutAlreadyOut = TouchOption.noTouchAtExpiry(
                1.0,
                90.0,
                BarrierType.UP_OUT,
                CASH_PAYOFF
        );

        final TouchOption downAndOutAlreadyOut = TouchOption.noTouchAtExpiry(
                1.0,
                110.0,
                BarrierType.DOWN_OUT,
                CASH_PAYOFF
        );

        final double upAndOutValue = valueAtSpot(upAndOutAlreadyOut, model, SPOT);
        final double downAndOutValue = valueAtSpot(downAndOutAlreadyOut, model, SPOT);

        assertEquals(0.0, upAndOutValue, ZERO_TOLERANCE);
        assertEquals(0.0, downAndOutValue, ZERO_TOLERANCE);
    }

    @Test
    public void testMaturityZeroEndpointValues() {
        final FDMBachelierModel model = createModel(0.0, 200.0, 1.0);

        final TouchOption oneTouchHitAtMaturity = TouchOption.oneTouchAtExpiry(
                0.0,
                90.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption oneTouchNotHitAtMaturity = TouchOption.oneTouchAtExpiry(
                0.0,
                110.0,
                BarrierType.UP_IN,
                CASH_PAYOFF
        );

        final TouchOption noTouchAliveAtMaturity = TouchOption.noTouchAtExpiry(
                0.0,
                110.0,
                BarrierType.UP_OUT,
                CASH_PAYOFF
        );

        final TouchOption noTouchAlreadyOutAtMaturity = TouchOption.noTouchAtExpiry(
                0.0,
                90.0,
                BarrierType.UP_OUT,
                CASH_PAYOFF
        );

        final double oneTouchHitValue = valueAtSpot(oneTouchHitAtMaturity, model, SPOT);
        final double oneTouchNotHitValue = valueAtSpot(oneTouchNotHitAtMaturity, model, SPOT);
        final double noTouchAliveValue = valueAtSpot(noTouchAliveAtMaturity, model, SPOT);
        final double noTouchAlreadyOutValue = valueAtSpot(noTouchAlreadyOutAtMaturity, model, SPOT);

        assertEquals(CASH_PAYOFF, oneTouchHitValue, VALUE_TOLERANCE);
        assertEquals(0.0, oneTouchNotHitValue, ZERO_TOLERANCE);
        assertEquals(CASH_PAYOFF, noTouchAliveValue, VALUE_TOLERANCE);
        assertEquals(0.0, noTouchAlreadyOutValue, ZERO_TOLERANCE);
    }

    @Test
    public void testOneTouchFactoryRejectsOutBarrierTypes() {
        expectIllegalArgument(() ->
            TouchOption.oneTouchAtExpiry(
                    1.0,
                    80.0,
                    BarrierType.DOWN_OUT,
                    CASH_PAYOFF
            )
        );

        expectIllegalArgument(() ->
            TouchOption.oneTouchAtExpiry(
                    1.0,
                    120.0,
                    BarrierType.UP_OUT,
                    CASH_PAYOFF
            )
        );
    }

    @Test
    public void testNoTouchFactoryRejectsInBarrierTypes() {
        expectIllegalArgument(() ->
            TouchOption.noTouchAtExpiry(
                    1.0,
                    80.0,
                    BarrierType.DOWN_IN,
                    CASH_PAYOFF
            )
        );

        expectIllegalArgument(() ->
            TouchOption.noTouchAtExpiry(
                    1.0,
                    120.0,
                    BarrierType.UP_IN,
                    CASH_PAYOFF
            )
        );
    }

    @Test
    public void testBarrierOutsideGridThrows() {
        final FDMBachelierModel model = createModel(0.0, 150.0, 1.0);

        final TouchOption option = TouchOption.noTouchAtExpiry(
                1.0,
                180.0,
                BarrierType.UP_OUT,
                CASH_PAYOFF
        );

        expectIllegalArgument(() -> option.getValue(0.0, model));
    }

    private FDMBachelierModel createModel(
            final double sMin,
            final double sMax,
            final double finalTime) {

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        finalTime / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { SPOT }
        );

        return new FDMBachelierModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                VOLATILITY,
                spaceTime
        );
    }

    private double valueAtSpot(
            final TouchOption option,
            final FDMBachelierModel model,
            final double spot) {

        final double[] values = option.getValue(0.0, model);
        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        final int spotIndex = getGridIndex(sNodes, spot);
        return values[spotIndex];
    }

    private int getGridIndex(final double[] grid, final double value) {
        for(int i = 0; i < grid.length; i++) {
            if(Math.abs(grid[i] - value) < 1E-12) {
                return i;
            }
        }
        throw new IllegalArgumentException("Point is not a grid node.");
    }

    private void expectIllegalArgument(final ThrowingRunnable runnable) {
        try {
            runnable.run();
            fail("Expected IllegalArgumentException.");
        }
        catch(final IllegalArgumentException expected) {
            // expected
        }
        catch(final Exception other) {
            fail("Expected IllegalArgumentException but got: " + other.getClass().getName());
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}