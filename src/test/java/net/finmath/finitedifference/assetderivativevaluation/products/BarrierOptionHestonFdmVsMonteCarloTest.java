package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.assetderivativevaluation.models.HestonModel.Scheme;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Compares finite-difference prices of European barrier options under a Heston model
 * with Monte Carlo benchmark prices.
 *
 * <p>
 * Important legacy note:
 * the Monte Carlo EuropeanOption product in this finmath version prices only calls.
 * Therefore, for knock-in options, the Monte Carlo benchmark is constructed via
 * parity against the corresponding knock-out:
 * </p>
 *
 * <p>
 * MC(knock-in) = MC(vanilla) - MC(corresponding knock-out).
 * </p>
 *
 * <p>
 * For puts, the vanilla Monte Carlo value is reconstructed from the Monte Carlo
 * call via put-call parity:
 * </p>
 *
 * <p>
 * P = C - S0 exp(-qT) + K exp(-rT).
 * </p>
 *
 * <p>
 * The finite-difference value is evaluated from the full 2D surface via bilinear
 * interpolation in (spot, variance).
 * </p>
 *
 * <p>
 * For knock-ins benchmarked via MC parity, a hybrid acceptance criterion is used:
 * the test passes if either the relative error is within tolerance or the absolute
 * error is within a fixed small-price tolerance. This avoids over-penalizing small
 * residual prices such as down-in calls or up-in puts.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonFdmVsMonteCarloTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double REBATE = 0.0;

    private static final double S0 = 100.0;
    private static final double R = 0.05;
    private static final double Q = 0.00;

    private static final double VOLATILITY = 0.25;
    private static final double VOLATILITY_SQUARED = VOLATILITY * VOLATILITY;
    private static final double KAPPA = 1.5;
    private static final double THETA_H = VOLATILITY_SQUARED;
    private static final double XI = 0.30;
    private static final double RHO = -0.70;

    private static final double THETA = 0.5;

    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS_S = 160;
    private static final int NUMBER_OF_SPACE_STEPS_V = 100;

    /**
     * Defines the target uniform spacing between barrier and spot.
     * With this choice, both the barrier and S0 are exact spot-grid nodes.
     */
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    /**
     * Absolute tolerance used for knock-ins benchmarked via Monte Carlo parity.
     * This is intentionally small in absolute terms, but avoids unstable relative
     * errors for residual prices.
     */
    private static final double ABSOLUTE_TOLERANCE_KNOCK_IN = 0.25;

    @Test
    public void testDownAndOutEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_OUT, 80.0, 0.20);
    }

    @Test
    public void testDownAndInEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, 0.25);
    }

    @Test
    public void testUpAndOutEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.CALL, BarrierType.UP_OUT, 120.0, 0.20);
    }

    @Test
    public void testUpAndInEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0, 0.25);
    }

    @Test
    public void testDownAndOutEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_OUT, 80.0, 0.20);
    }

    @Test
    public void testDownAndInEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, 0.25);
    }

    @Test
    public void testUpAndOutEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.PUT, BarrierType.UP_OUT, 120.0, 0.20);
    }

    @Test
    public void testUpAndInEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
        runBarrierTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0, 0.25);
    }

    private void runBarrierTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier,
            final double relativeTolerance) throws Exception {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS);

        final Grid sGrid = createSpotGrid(barrier, barrierType);

        final double vMin = 0.0;
        final double vMax = Math.max(
                4.0 * THETA_H,
                VOLATILITY_SQUARED + 4.0 * XI * Math.sqrt(MATURITY)
        );
        final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_V, vMin, vMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, vGrid },
                timeDiscretization,
                THETA,
                new double[] { S0, VOLATILITY_SQUARED }
        );

        final FDMHestonModel fdmModel = new FDMHestonModel(
                S0,
                VOLATILITY_SQUARED,
                riskFreeCurve,
                dividendCurve,
                KAPPA,
                THETA_H,
                XI,
                RHO,
                spaceTime
        );

        final double callOrPutSign = callOrPut == CallOrPut.CALL ? 1.0 : -1.0;

        final BarrierOption fdmProduct = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPutSign,
                barrierType
        );

        final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
        final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
        final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

        assertTrue(
                "S0 must lie inside the spot grid domain.",
                S0 >= sNodes[0] - 1E-12 && S0 <= sNodes[sNodes.length - 1] + 1E-12
        );
        assertTrue(
                "Initial variance must lie inside the variance grid domain.",
                VOLATILITY_SQUARED >= vNodes[0] - 1E-12 && VOLATILITY_SQUARED <= vNodes[vNodes.length - 1] + 1E-12
        );
        assertTrue(
                "Barrier must be an exact spot-grid node.",
                getGridIndex(sNodes, barrier) >= 0
        );
        assertTrue(
                "S0 must be an exact spot-grid node.",
                getGridIndex(sNodes, S0) >= 0
        );

        final double fdPrice = interpolateAtSpotAndVariance(
                fdValuesOnGrid,
                sNodes,
                vNodes,
                S0,
                VOLATILITY_SQUARED
        );

        final int numberOfPaths = 50_000;
        final int seed = 31415;
        final int mcNumberOfTimeSteps = 500;

        final TimeDiscretization mcTimes =
                new TimeDiscretizationFromArray(
                        0.0,
                        mcNumberOfTimeSteps,
                        MATURITY / mcNumberOfTimeSteps);

        final BrownianMotionFromMersenneRandomNumbers brownianMotion =
                new BrownianMotionFromMersenneRandomNumbers(mcTimes, 2, numberOfPaths, seed);

        final net.finmath.montecarlo.assetderivativevaluation.models.HestonModel mcModel =
                new net.finmath.montecarlo.assetderivativevaluation.models.HestonModel(
                        S0,
                        R - Q,
                        Math.sqrt(VOLATILITY_SQUARED),
                        R,
                        THETA_H,
                        KAPPA,
                        XI,
                        RHO,
                        Scheme.FULL_TRUNCATION
                );

        final EulerSchemeFromProcessModel process =
                new EulerSchemeFromProcessModel(mcModel, brownianMotion);

        final MonteCarloAssetModel mcSimulation = new MonteCarloAssetModel(process);

        final double mcPrice;
        final String mcBenchmarkLabel;

        if(isOutOption(barrierType)) {
            final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcOutProduct =
                    new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
                            MATURITY,
                            STRIKE,
                            barrier,
                            REBATE,
                            callOrPut,
                            barrierType
                    );

            mcPrice = mcOutProduct.getValue(mcSimulation);
            mcBenchmarkLabel = "direct MC barrier";
        }
        else {
            final net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption mcVanillaCallProduct =
                    new net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption(
                            MATURITY,
                            STRIKE
                    );

            final double mcVanillaCall = mcVanillaCallProduct.getValue(mcSimulation);
            final double mcVanilla = convertCallToRequestedOptionByParity(mcVanillaCall, callOrPut);

            final BarrierType correspondingOutType = getCorrespondingOutBarrierType(barrierType);

            final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcOutProduct =
                    new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
                            MATURITY,
                            STRIKE,
                            barrier,
                            REBATE,
                            callOrPut,
                            correspondingOutType
                    );

            final double mcOut = mcOutProduct.getValue(mcSimulation);

            mcPrice = mcVanilla - mcOut;
            mcBenchmarkLabel = "MC parity benchmark";
        }

        System.out.println("Type           = " + barrierType + " " + callOrPut);
        System.out.println("Grid min       = " + sNodes[0]);
        System.out.println("Grid max       = " + sNodes[sNodes.length - 1]);
        System.out.println("Barrier on grid= " + (getGridIndex(sNodes, barrier) >= 0));
        System.out.println("S0 on grid     = " + (getGridIndex(sNodes, S0) >= 0));
        System.out.println("v0 on grid     = " + (getGridIndex(vNodes, VOLATILITY_SQUARED) >= 0));
        System.out.println("FD price       = " + fdPrice);
        System.out.println("MC price       = " + mcPrice);
        System.out.println("Benchmark type = " + mcBenchmarkLabel);

        assertTrue(fdPrice >= -1E-10);
        assertTrue(mcPrice >= -1E-10);

        final double absoluteError = Math.abs(fdPrice - mcPrice);
        final double denominator = Math.max(Math.abs(mcPrice), 1E-8);
        final double relativeError = absoluteError / denominator;

        System.out.println("Absolute error = " + absoluteError);
        System.out.println("Relative error = " + relativeError);

        if(isOutOption(barrierType)) {
            assertEquals(
                    "Relative FD vs MC barrier price error for " + barrierType + " " + callOrPut,
                    0.0,
                    relativeError,
                    relativeTolerance
            );
        }
        else {
            final boolean withinHybridTolerance =
                    relativeError <= relativeTolerance
                    || absoluteError <= ABSOLUTE_TOLERANCE_KNOCK_IN;

            assertTrue(
                    "FD vs MC knock-in error too large for "
                            + barrierType + " " + callOrPut
                            + " (relative error = " + relativeError
                            + ", absolute error = " + absoluteError + ")",
                    withinHybridTolerance
            );
        }
    }

    private double convertCallToRequestedOptionByParity(
            final double callValue,
            final CallOrPut callOrPut) {

        if(callOrPut == CallOrPut.CALL) {
            return callValue;
        }

        final double discountedSpot = S0 * Math.exp(-Q * MATURITY);
        final double discountedStrike = STRIKE * Math.exp(-R * MATURITY);

        return callValue - discountedSpot + discountedStrike;
    }

    private static boolean isOutOption(final BarrierType barrierType) {
        return barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.UP_OUT;
    }

    private static BarrierType getCorrespondingOutBarrierType(final BarrierType barrierType) {
        if(barrierType == BarrierType.DOWN_IN) {
            return BarrierType.DOWN_OUT;
        }
        if(barrierType == BarrierType.UP_IN) {
            return BarrierType.UP_OUT;
        }
        throw new IllegalArgumentException("No corresponding out barrier type for " + barrierType);
    }

    /**
     * Uses a uniform barrier-aligned grid for all barrier types.
     *
     * <p>
     * Since deltaS is chosen as |barrier - S0| / STEPS_BETWEEN_BARRIER_AND_SPOT,
     * both the barrier and S0 are exact nodes on this grid.
     * </p>
     */
    private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

        final double deltaS = Math.abs(barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

        final double sMin;
        final double sMax;

        if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
            sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
            sMax = Math.max(3.0 * S0, S0 + 12.0 * deltaS);
        }
        else {
            sMin = 0.0;
            sMax = barrier + 8.0 * deltaS;
        }

        final int numberOfSteps = Math.max(
                NUMBER_OF_SPACE_STEPS_S,
                (int)Math.round((sMax - sMin) / deltaS)
        );

        return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
                numberOfSteps,
                sMin,
                sMax,
                barrier
        );
    }

    private static double interpolateAtSpotAndVariance(
            final double[] flattenedValues,
            final double[] sNodes,
            final double[] vNodes,
            final double spot,
            final double variance) {

        final int nS = sNodes.length;
        final int nV = vNodes.length;

        final double[][] valueSurface = new double[nS][nV];
        for(int j = 0; j < nV; j++) {
            for(int i = 0; i < nS; i++) {
                valueSurface[i][j] = flattenedValues[flatten(i, j, nS)];
            }
        }

        final BiLinearInterpolation interpolation =
                new BiLinearInterpolation(sNodes, vNodes, valueSurface);

        return interpolation.apply(spot, variance);
    }

    private static int flatten(final int iS, final int iV, final int numberOfSNodes) {
        return iS + iV * numberOfSNodes;
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