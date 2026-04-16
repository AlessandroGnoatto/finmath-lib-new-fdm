package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

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
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Endpoint tests for double-barrier cash binaries.
 *
 * <p>
 * These tests lock down the immediate already-triggered behavior when the
 * evaluation spot starts outside the alive band.
 * </p>
 */
public class DoubleBarrierBinaryEndpointTest {

    private static final double MATURITY = 1.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double LOWER_BARRIER = 80.0;
    private static final double UPPER_BARRIER = 120.0;

    private static final double R = 0.05;
    private static final double Q = 0.00;
    private static final double SIGMA = 0.25;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS = 200;

    /**
     * Grid with step size 1 so 75, 80, 120, 125 are all nodes.
     */
    private static final double GRID_MIN = 0.0;
    private static final double GRID_MAX = 200.0;

    private static final double LOWER_OUTSIDE_SPOT = 75.0;
    private static final double UPPER_OUTSIDE_SPOT = 125.0;

    private static final double VALUE_TOLERANCE = 1E-6;

    @Test
    public void testKnockOutOutsideBandIsZero() {
        assertEquals(0.0, valueAtSpot(LOWER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_OUT), VALUE_TOLERANCE);
        assertEquals(0.0, valueAtSpot(UPPER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_OUT), VALUE_TOLERANCE);
    }

    @Test
    public void testKnockInOutsideBandPaysCash() {
        assertEquals(CASH_PAYOFF, valueAtSpot(LOWER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_IN), VALUE_TOLERANCE);
        assertEquals(CASH_PAYOFF, valueAtSpot(UPPER_OUTSIDE_SPOT, DoubleBarrierType.KNOCK_IN), VALUE_TOLERANCE);
    }

    @Test
    public void testKikoEndpointBehavior() {
        assertEquals(CASH_PAYOFF, valueAtSpot(LOWER_OUTSIDE_SPOT, DoubleBarrierType.KIKO), VALUE_TOLERANCE);
        assertEquals(0.0, valueAtSpot(UPPER_OUTSIDE_SPOT, DoubleBarrierType.KIKO), VALUE_TOLERANCE);
    }

    @Test
    public void testKokiEndpointBehavior() {
        assertEquals(0.0, valueAtSpot(LOWER_OUTSIDE_SPOT, DoubleBarrierType.KOKI), VALUE_TOLERANCE);
        assertEquals(CASH_PAYOFF, valueAtSpot(UPPER_OUTSIDE_SPOT, DoubleBarrierType.KOKI), VALUE_TOLERANCE);
    }

    private double valueAtSpot(
            final double spot,
            final DoubleBarrierType doubleBarrierType) {

        final FDMBlackScholesModel model = createBlackScholesModel(spot);

        final DoubleBarrierBinaryOption product = new DoubleBarrierBinaryOption(
                MATURITY,
                CASH_PAYOFF,
                LOWER_BARRIER,
                UPPER_BARRIER,
                doubleBarrierType
        );

        final double[] values = product.getValue(0.0, model);
        final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

        final int spotIndex = getGridIndex(sNodes, spot);
        assertTrue("Spot must be a grid node.", spotIndex >= 0);

        return values[spotIndex];
    }

    private FDMBlackScholesModel createBlackScholesModel(final double spot) {
        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, GRID_MIN, GRID_MAX);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                sGrid,
                timeDiscretization,
                THETA,
                new double[] { spot }
        );

        return new FDMBlackScholesModel(
                spot,
                riskFreeCurve,
                dividendCurve,
                SIGMA,
                spaceTime
        );
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