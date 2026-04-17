package net.finmath.finitedifference.assetderivativevaluation.products;

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
 * Diagnostic comparison of FD and MC parity components for European Heston barrier options.
 *
 * <p>
 * This test does not enforce price tolerances. Its purpose is to isolate whether
 * the remaining discrepancy in the direct knock-in cases comes from:
 * </p>
 * <ul>
 *   <li>the direct knock-in PDE/interface formulation, or</li>
 *   <li>the underlying vanilla / knock-out finite-difference numerics, whose
 *       difference defines the small knock-in residual.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonFdmVsMonteCarloParityDiagnosticTest {

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

    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    @Test
    public void diagnoseDownOutCall() throws Exception {
        runDiagnostic(CallOrPut.CALL, BarrierType.DOWN_OUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void diagnoseDownInCall() throws Exception {
        runDiagnostic(CallOrPut.CALL, BarrierType.DOWN_OUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void diagnoseUpOutCall() throws Exception {
        runDiagnostic(CallOrPut.CALL, BarrierType.UP_OUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void diagnoseUpInCall() throws Exception {
        runDiagnostic(CallOrPut.CALL, BarrierType.UP_OUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void diagnoseDownOutPut() throws Exception {
        runDiagnostic(CallOrPut.PUT, BarrierType.DOWN_OUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void diagnoseDownInPut() throws Exception {
        runDiagnostic(CallOrPut.PUT, BarrierType.DOWN_OUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void diagnoseUpOutPut() throws Exception {
        runDiagnostic(CallOrPut.PUT, BarrierType.UP_OUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void diagnoseUpInPut() throws Exception {
        runDiagnostic(CallOrPut.PUT, BarrierType.UP_OUT, BarrierType.UP_IN, 120.0);
    }

    private void runDiagnostic(
            final CallOrPut callOrPut,
            final BarrierType outType,
            final BarrierType inType,
            final double barrier) throws Exception {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
                        0.0,
                        NUMBER_OF_TIME_STEPS,
                        MATURITY / NUMBER_OF_TIME_STEPS
                );

        final Grid sGrid = createSpotGrid(barrier, outType);

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

        final EuropeanOption fdVanillaProduct = new EuropeanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final BarrierOption fdOutProduct = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPutSign,
                outType
        );

        final BarrierOption fdInProduct = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPutSign,
                inType
        );

        final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
        final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

        final double fdVanilla = interpolateAtSpotAndVariance(
                fdVanillaProduct.getValue(0.0, fdmModel),
                sNodes,
                vNodes,
                S0,
                VOLATILITY_SQUARED
        );

        final double fdOut = interpolateAtSpotAndVariance(
                fdOutProduct.getValue(0.0, fdmModel),
                sNodes,
                vNodes,
                S0,
                VOLATILITY_SQUARED
        );

        final double fdIn = interpolateAtSpotAndVariance(
                fdInProduct.getValue(0.0, fdmModel),
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
                        MATURITY / mcNumberOfTimeSteps
                );

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

        final net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption mcVanillaProduct =
                new net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption(
                        MATURITY,
                        STRIKE
                );

        final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcOutProduct =
                new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
                        MATURITY,
                        STRIKE,
                        barrier,
                        REBATE,
                        callOrPut,
                        outType
                );

        final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcInProduct =
                new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
                        MATURITY,
                        STRIKE,
                        barrier,
                        REBATE,
                        callOrPut,
                        inType
                );

        final double mcVanilla = mcVanillaProduct.getValue(mcSimulation);
        final double mcOut = mcOutProduct.getValue(mcSimulation);
        final double mcIn = mcInProduct.getValue(mcSimulation);

        final double fdParityResidual = fdVanilla - fdOut - fdIn;
        final double mcParityResidual = mcVanilla - mcOut - mcIn;

        final double fdParityImpliedIn = fdVanilla - fdOut;
        final double mcParityImpliedIn = mcVanilla - mcOut;

        System.out.println("====================================================");
        System.out.println("Heston FD vs MC parity diagnostic");
        System.out.println("Type                = " + inType + " " + callOrPut);
        System.out.println("Barrier             = " + barrier);
        System.out.println("Grid min            = " + sNodes[0]);
        System.out.println("Grid max            = " + sNodes[sNodes.length - 1]);
        System.out.println("Barrier on grid     = " + (getGridIndex(sNodes, barrier) >= 0));
        System.out.println("S0 on grid          = " + (getGridIndex(sNodes, S0) >= 0));
        System.out.println("v0 on grid          = " + (getGridIndex(vNodes, VOLATILITY_SQUARED) >= 0));
        System.out.println("----------------------------------------------------");
        System.out.println("FD vanilla          = " + fdVanilla);
        System.out.println("FD out              = " + fdOut);
        System.out.println("FD direct in        = " + fdIn);
        System.out.println("FD implied in       = " + fdParityImpliedIn);
        System.out.println("FD parity residual  = " + fdParityResidual);
        System.out.println("----------------------------------------------------");
        System.out.println("MC vanilla          = " + mcVanilla);
        System.out.println("MC out              = " + mcOut);
        System.out.println("MC direct in        = " + mcIn);
        System.out.println("MC implied in       = " + mcParityImpliedIn);
        System.out.println("MC parity residual  = " + mcParityResidual);
        System.out.println("----------------------------------------------------");
        System.out.println("FD vs MC vanilla abs diff     = " + Math.abs(fdVanilla - mcVanilla));
        System.out.println("FD vs MC out abs diff         = " + Math.abs(fdOut - mcOut));
        System.out.println("FD vs MC in abs diff          = " + Math.abs(fdIn - mcIn));
        System.out.println("FD direct-vs-implied in diff  = " + Math.abs(fdIn - fdParityImpliedIn));
        System.out.println("MC direct-vs-implied in diff  = " + Math.abs(mcIn - mcParityImpliedIn));
        System.out.println("====================================================");

        assertTrue(fdVanilla >= -1E-10);
        assertTrue(fdOut >= -1E-10);
        assertTrue(fdIn >= -1E-10);
        assertTrue(mcVanilla >= -1E-10);
        assertTrue(mcOut >= -1E-10);
        assertTrue(mcIn >= -1E-10);
    }

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