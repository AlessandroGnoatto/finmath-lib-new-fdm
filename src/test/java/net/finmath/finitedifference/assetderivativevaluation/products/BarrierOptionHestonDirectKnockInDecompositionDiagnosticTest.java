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
import net.finmath.interpolation.RationalFunctionInterpolation;
import net.finmath.interpolation.RationalFunctionInterpolation.ExtrapolationMethod;
import net.finmath.interpolation.RationalFunctionInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Diagnostic decomposition tests for the direct 2D Heston knock-in implementation.
 *
 * <p>
 * These tests isolate three components:
 * </p>
 * <ol>
 *   <li>Activated auxiliary vanilla vs original vanilla.</li>
 *   <li>Direct knock-in vs activated auxiliary vanilla in the already-hit region.</li>
 *   <li>Direct knock-in vs auxiliary parity:
 *       activated-aux-vanilla minus knock-out.</li>
 * </ol>
 *
 * <p>
 * Interpretation:
 * </p>
 * <ul>
 *   <li>If component (2) is tight, the final assembly is likely correct.</li>
 *   <li>If component (1) is not tight, the activated auxiliary branch is a likely
 *       source of bias.</li>
 *   <li>If component (3) is tighter than original parity using vanilla on the
 *       original grid, the pre-hit PDE is likely consistent with the auxiliary
 *       decomposition and the main discrepancy comes from the activated branch.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonDirectKnockInDecompositionDiagnosticTest {

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double REBATE = 0.0;

    private static final double SPOT = 100.0;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.00;

    private static final double INITIAL_VOLATILITY = 0.25;
    private static final double INITIAL_VARIANCE = INITIAL_VOLATILITY * INITIAL_VOLATILITY;

    private static final double KAPPA = 1.5;
    private static final double LONG_RUN_VARIANCE = INITIAL_VARIANCE;
    private static final double XI = 0.30;
    private static final double RHO = -0.70;

    private static final double THETA = 0.5;

    private static final int NUMBER_OF_TIME_STEPS = 100;
    private static final int NUMBER_OF_SPACE_STEPS_S = 160;
    private static final int NUMBER_OF_SPACE_STEPS_V = 100;

    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

    /*
     * This one should be very tight if the final direct knock-in surface really
     * copies the activated auxiliary branch in the already-hit region.
     */
    private static final double ASSEMBLY_MATCH_TOLERANCE = 1E-8;

    /*
     * Just sanity / numerical tolerance.
     */
    private static final double NON_NEG_TOLERANCE = 1E-10;

    private static final double DOWN_ALREADY_HIT_SPOT = 78.0;
    private static final double UP_ALREADY_HIT_SPOT = 122.0;

    @Test
    public void testDownInCallDecompositionDiagnostics() {
        runDiagnostics(
                CallOrPut.CALL,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testDownInPutDecompositionDiagnostics() {
        runDiagnostics(
                CallOrPut.PUT,
                80.0,
                BarrierType.DOWN_OUT,
                BarrierType.DOWN_IN,
                DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testUpInCallDecompositionDiagnostics() {
        runDiagnostics(
                CallOrPut.CALL,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testUpInPutDecompositionDiagnostics() {
        runDiagnostics(
                CallOrPut.PUT,
                120.0,
                BarrierType.UP_OUT,
                BarrierType.UP_IN,
                UP_ALREADY_HIT_SPOT
        );
    }

    private void runDiagnostics(
            final CallOrPut callOrPut,
            final double barrier,
            final BarrierType outType,
            final BarrierType inType,
            final double alreadyHitSpot) {

        final TestSetup setup = createSetup(barrier, inType);

        final EuropeanOption vanillaOption = new EuropeanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final BarrierOption knockOutOption = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                outType
        );

        final BarrierOption directKnockInOption = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                inType
        );

        final double[] originalVanillaSurface = vanillaOption.getValue(0.0, setup.originalModel);
        final double[] knockOutSurface = knockOutOption.getValue(0.0, setup.originalModel);
        final double[] directKnockInSurface = directKnockInOption.getValue(0.0, setup.originalModel);

        final FDMHestonModel activatedAuxiliaryModel =
                createActivatedAuxiliaryModel2D(setup.originalModel, barrier);

        final double[] activatedAuxiliaryVanillaSurfaceOnAuxGrid =
                vanillaOption.getValue(0.0, activatedAuxiliaryModel);

        final double[] activatedAuxiliaryVanillaSurfaceOnOriginalGrid =
                interpolateSurfaceToOriginalGrid2DAlongFirstState(
                        activatedAuxiliaryVanillaSurfaceOnAuxGrid,
                        activatedAuxiliaryModel.getSpaceTimeDiscretization(),
                        setup.originalModel.getSpaceTimeDiscretization()
                );

        final DiagnosticValues values = new DiagnosticValues(
                interpolateAt(
                        originalVanillaSurface,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        SPOT,
                        INITIAL_VARIANCE
                ),
                interpolateAt(
                        activatedAuxiliaryVanillaSurfaceOnOriginalGrid,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        SPOT,
                        INITIAL_VARIANCE
                ),
                interpolateAt(
                        knockOutSurface,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        SPOT,
                        INITIAL_VARIANCE
                ),
                interpolateAt(
                        directKnockInSurface,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        SPOT,
                        INITIAL_VARIANCE
                ),
                interpolateAt(
                        originalVanillaSurface,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        alreadyHitSpot,
                        INITIAL_VARIANCE
                ),
                interpolateAt(
                        activatedAuxiliaryVanillaSurfaceOnOriginalGrid,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        alreadyHitSpot,
                        INITIAL_VARIANCE
                ),
                interpolateAt(
                        directKnockInSurface,
                        setup.originalSNodes,
                        setup.originalVNodes,
                        alreadyHitSpot,
                        INITIAL_VARIANCE
                )
        );

        printDiagnostics(callOrPut, barrier, outType, inType, alreadyHitSpot, values);

        /*
         * Component 2: this is the strongest diagnostic assertion.
         * In the already-hit region, the direct knock-in surface should coincide
         * with the activated auxiliary vanilla surface after interpolation onto
         * the original grid.
         */
        assertEquals(
                "Direct knock-in does not match activated auxiliary vanilla in already-hit region for "
                        + inType + " " + callOrPut,
                values.activatedAuxiliaryVanillaAlreadyHit,
                values.directKnockInAlreadyHit,
                ASSEMBLY_MATCH_TOLERANCE
        );

        /*
         * Sanity checks.
         */
        assertTrue(values.originalVanillaAtSpot >= -NON_NEG_TOLERANCE);
        assertTrue(values.activatedAuxiliaryVanillaAtSpot >= -NON_NEG_TOLERANCE);
        assertTrue(values.knockOutAtSpot >= -NON_NEG_TOLERANCE);
        assertTrue(values.directKnockInAtSpot >= -NON_NEG_TOLERANCE);
        assertTrue(values.originalVanillaAlreadyHit >= -NON_NEG_TOLERANCE);
        assertTrue(values.activatedAuxiliaryVanillaAlreadyHit >= -NON_NEG_TOLERANCE);
        assertTrue(values.directKnockInAlreadyHit >= -NON_NEG_TOLERANCE);

        /*
         * Dominance sanity: knock-in should not exceed the activated vanilla.
         * At spot this compares against the activated auxiliary branch.
         */
        assertTrue(
                "Direct knock-in exceeds activated auxiliary vanilla at spot for "
                        + inType + " " + callOrPut,
                values.directKnockInAtSpot <= values.activatedAuxiliaryVanillaAtSpot + 1E-10
        );
    }

    private void printDiagnostics(
            final CallOrPut callOrPut,
            final double barrier,
            final BarrierType outType,
            final BarrierType inType,
            final double alreadyHitSpot,
            final DiagnosticValues values) {

        final double originalParityError =
                Math.abs(values.originalVanillaAtSpot
                        - (values.directKnockInAtSpot + values.knockOutAtSpot));

        final double auxiliaryParityError =
                Math.abs(values.activatedAuxiliaryVanillaAtSpot
                        - (values.directKnockInAtSpot + values.knockOutAtSpot));

        final double activatedBranchSpotError =
                Math.abs(values.originalVanillaAtSpot - values.activatedAuxiliaryVanillaAtSpot);

        final double activatedBranchAlreadyHitError =
                Math.abs(values.originalVanillaAlreadyHit - values.activatedAuxiliaryVanillaAlreadyHit);

        final double assemblyError =
                Math.abs(values.directKnockInAlreadyHit - values.activatedAuxiliaryVanillaAlreadyHit);

        System.out.println("====================================================");
        System.out.println("Heston direct knock-in decomposition diagnostics");
        System.out.println("Call/Put                = " + callOrPut);
        System.out.println("Barrier                 = " + barrier);
        System.out.println("Out type                = " + outType);
        System.out.println("In type                 = " + inType);
        System.out.println("Already-hit spot        = " + alreadyHitSpot);
        System.out.println("----------------------------------------------------");
        System.out.println("Component 1: activated auxiliary vanilla vs original vanilla");
        System.out.println("Original vanilla @ spot         = " + values.originalVanillaAtSpot);
        System.out.println("Activated aux vanilla @ spot    = " + values.activatedAuxiliaryVanillaAtSpot);
        System.out.println("Abs diff @ spot                 = " + activatedBranchSpotError);
        System.out.println("Original vanilla @ hit spot     = " + values.originalVanillaAlreadyHit);
        System.out.println("Activated aux vanilla @ hit spot= " + values.activatedAuxiliaryVanillaAlreadyHit);
        System.out.println("Abs diff @ hit spot             = " + activatedBranchAlreadyHitError);
        System.out.println("----------------------------------------------------");
        System.out.println("Component 2: direct knock-in vs activated auxiliary vanilla");
        System.out.println("Direct knock-in @ hit spot      = " + values.directKnockInAlreadyHit);
        System.out.println("Activated aux vanilla @ hit spot= " + values.activatedAuxiliaryVanillaAlreadyHit);
        System.out.println("Assembly abs diff               = " + assemblyError);
        System.out.println("----------------------------------------------------");
        System.out.println("Component 3: direct knock-in vs auxiliary parity");
        System.out.println("Knock-out @ spot                = " + values.knockOutAtSpot);
        System.out.println("Direct knock-in @ spot          = " + values.directKnockInAtSpot);
        System.out.println("Original parity error           = " + originalParityError);
        System.out.println("Auxiliary parity error          = " + auxiliaryParityError);
        System.out.println("====================================================");
    }

    /**
     * Mirrors the current BarrierOption auxiliary activated-model construction.
     */
    private FDMHestonModel createActivatedAuxiliaryModel2D(
            final FDMHestonModel barrierModel,
            final double barrier) {

        final SpaceTimeDiscretization base = barrierModel.getSpaceTimeDiscretization();
        final double[] baseSpotGrid = base.getSpaceGrid(0).getGrid();

        if(baseSpotGrid.length < 2) {
            throw new IllegalArgumentException("Barrier grid must contain at least two points.");
        }

        boolean barrierAlreadyOnSpotGrid = false;
        for(final double s : baseSpotGrid) {
            if(Math.abs(s - barrier) <= 1E-8) {
                barrierAlreadyOnSpotGrid = true;
                break;
            }
        }

        if(barrierAlreadyOnSpotGrid) {
            return barrierModel;
        }

        final double[] secondGrid = base.getSpaceGrid(1).getGrid();

        final Grid activatedSpotGrid = buildActivatedSpotGridAlignedAtBarrier(
                baseSpotGrid,
                barrierModel.getInitialValue()[0],
                barrier
        );

        final Grid preservedSecondGrid = new UniformGrid(
                secondGrid.length - 1,
                secondGrid[0],
                secondGrid[secondGrid.length - 1]
        );

        final SpaceTimeDiscretization activatedDiscretization =
                new SpaceTimeDiscretization(
                        new Grid[] { activatedSpotGrid, preservedSecondGrid },
                        base.getTimeDiscretization(),
                        base.getTheta(),
                        barrierModel.getInitialValue()
                );

        return (FDMHestonModel) barrierModel.getCloneWithModifiedSpaceTimeDiscretization(
                activatedDiscretization
        );
    }

    /**
     * Mirrors the current BarrierOption activated spot-grid construction.
     */
    private Grid buildActivatedSpotGridAlignedAtBarrier(
            final double[] baseSpotGrid,
            final double initialSpot,
            final double barrier) {

        final double deltaS = baseSpotGrid[1] - baseSpotGrid[0];
        final double currentMin = baseSpotGrid[0];
        final double currentMax = baseSpotGrid[baseSpotGrid.length - 1];
        final double currentHalfWidth = Math.max(initialSpot - currentMin, currentMax - initialSpot);

        final double targetMin = Math.max(1E-8, initialSpot - 2.0 * currentHalfWidth);
        final double targetMax = initialSpot + 2.0 * currentHalfWidth;

        final int leftSteps = Math.max(1, (int)Math.ceil((barrier - targetMin) / deltaS));
        final int rightSteps = Math.max(1, (int)Math.ceil((targetMax - barrier) / deltaS));

        final double sMin = barrier - leftSteps * deltaS;
        final double sMax = barrier + rightSteps * deltaS;

        return new UniformGrid(leftSteps + rightSteps, sMin, sMax);
    }

    /**
     * Mirrors the current BarrierOption interpolation from the activated auxiliary
     * grid back to the original grid along the first state variable only.
     */
    private double[] interpolateSurfaceToOriginalGrid2DAlongFirstState(
            final double[] valuesOnAuxiliaryGrid,
            final SpaceTimeDiscretization auxiliaryDiscretization,
            final SpaceTimeDiscretization originalDiscretization) {

        final double[] auxiliaryX0 = auxiliaryDiscretization.getSpaceGrid(0).getGrid();
        final double[] auxiliaryX1 = auxiliaryDiscretization.getSpaceGrid(1).getGrid();

        final double[] originalX0 = originalDiscretization.getSpaceGrid(0).getGrid();
        final double[] originalX1 = originalDiscretization.getSpaceGrid(1).getGrid();

        if(auxiliaryX1.length != originalX1.length) {
            throw new IllegalArgumentException(
                    "Second state-variable grids must coincide.");
        }

        for(int j = 0; j < originalX1.length; j++) {
            if(Math.abs(auxiliaryX1[j] - originalX1[j]) > 1E-12) {
                throw new IllegalArgumentException(
                        "Second state-variable grids must coincide.");
            }
        }

        final int auxiliaryN0 = auxiliaryX0.length;
        final int originalN0 = originalX0.length;
        final int originalN1 = originalX1.length;

        final double[] interpolatedValues = new double[originalN0 * originalN1];

        for(int j = 0; j < originalN1; j++) {

            final double[] auxiliarySlice = new double[auxiliaryN0];
            for(int i = 0; i < auxiliaryN0; i++) {
                auxiliarySlice[i] = valuesOnAuxiliaryGrid[flatten(i, j, auxiliaryN0)];
            }

            final RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(
                    auxiliaryX0,
                    auxiliarySlice,
                    InterpolationMethod.LINEAR,
                    ExtrapolationMethod.CONSTANT
            );

            for(int i = 0; i < originalN0; i++) {
                interpolatedValues[flatten(i, j, originalN0)] =
                        interpolator.getValue(originalX0[i]);
            }
        }

        return interpolatedValues;
    }

    private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

        final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
                0.0,
                NUMBER_OF_TIME_STEPS,
                MATURITY / NUMBER_OF_TIME_STEPS
        );

        final Grid sGrid = createSpotGrid(barrier, barrierType);

        final double vMin = 0.0;
        final double vMax = Math.max(
                4.0 * LONG_RUN_VARIANCE,
                INITIAL_VARIANCE + 4.0 * XI * Math.sqrt(MATURITY)
        );

        final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_V, vMin, vMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, vGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, INITIAL_VARIANCE }
        );

        final FDMHestonModel model = new FDMHestonModel(
                SPOT,
                INITIAL_VARIANCE,
                riskFreeCurve,
                dividendCurve,
                KAPPA,
                LONG_RUN_VARIANCE,
                XI,
                RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), vGrid.getGrid());
    }

    private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

        final double deltaS = Math.abs(barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;

        final boolean isKnockIn =
                barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN;

        if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {

            final double sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
            final double sMax = Math.max(3.0 * SPOT, SPOT + 12.0 * deltaS);

            final int numberOfSteps = Math.max(
                    NUMBER_OF_SPACE_STEPS_S,
                    (int)Math.round((sMax - sMin) / deltaS)
            );

            if(isKnockIn) {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier,
                        BARRIER_CLUSTERING_EXPONENT
                );
            }
            else {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier
                );
            }
        }
        else {

            final double sMin = 0.0;
            final double sMax = barrier + 8.0 * deltaS;

            final int numberOfSteps = Math.max(
                    NUMBER_OF_SPACE_STEPS_S,
                    (int)Math.round((sMax - sMin) / deltaS)
            );

            if(isKnockIn) {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier,
                        BARRIER_CLUSTERING_EXPONENT
                );
            }
            else {
                return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
                        numberOfSteps,
                        sMin,
                        sMax,
                        barrier
                );
            }
        }
    }

    private double interpolateAt(
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

    private int flatten(final int iS, final int iV, final int numberOfSNodes) {
        return iS + iV * numberOfSNodes;
    }

    private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {

        final double[] times = new double[] { 0.0, 1.0 };
        final double[] zeroRates = new double[] { rate, rate };

        return DiscountCurveInterpolation.createDiscountCurveFromZeroRates(
                name,
                times,
                zeroRates
        );
    }
 
    private static final class TestSetup {

        private final FDMHestonModel originalModel;
        private final double[] originalSNodes;
        private final double[] originalVNodes;

        private TestSetup(
                final FDMHestonModel originalModel,
                final double[] originalSNodes,
                final double[] originalVNodes) {
            this.originalModel = originalModel;
            this.originalSNodes = originalSNodes;
            this.originalVNodes = originalVNodes;
        }
    }

    private static final class DiagnosticValues {

        private final double originalVanillaAtSpot;
        private final double activatedAuxiliaryVanillaAtSpot;
        private final double knockOutAtSpot;
        private final double directKnockInAtSpot;

        private final double originalVanillaAlreadyHit;
        private final double activatedAuxiliaryVanillaAlreadyHit;
        private final double directKnockInAlreadyHit;

        private DiagnosticValues(
                final double originalVanillaAtSpot,
                final double activatedAuxiliaryVanillaAtSpot,
                final double knockOutAtSpot,
                final double directKnockInAtSpot,
                final double originalVanillaAlreadyHit,
                final double activatedAuxiliaryVanillaAlreadyHit,
                final double directKnockInAlreadyHit) {
            this.originalVanillaAtSpot = originalVanillaAtSpot;
            this.activatedAuxiliaryVanillaAtSpot = activatedAuxiliaryVanillaAtSpot;
            this.knockOutAtSpot = knockOutAtSpot;
            this.directKnockInAtSpot = directKnockInAtSpot;
            this.originalVanillaAlreadyHit = originalVanillaAlreadyHit;
            this.activatedAuxiliaryVanillaAlreadyHit = activatedAuxiliaryVanillaAlreadyHit;
            this.directKnockInAlreadyHit = directKnockInAlreadyHit;
        }
    }
}