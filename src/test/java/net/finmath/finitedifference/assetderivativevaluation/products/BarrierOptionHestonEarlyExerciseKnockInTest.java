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
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for direct 2D Heston knock-ins with early exercise.
 *
 * <p>
 * The test suite verifies:
 * </p>
 * <ul>
 *   <li>one-date Bermudan knock-in equals European knock-in,</li>
 *   <li>European <= Bermudan <= American for the knock-in itself,</li>
 *   <li>Bermudan knock-in <= Bermudan vanilla,</li>
 *   <li>in the already-hit region, American knock-in = American vanilla.</li>
 * </ul>
 *
 * <p>
 * These are the right first validation properties for the direct
 * activated-vanilla + pre-hit architecture under Heston.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonEarlyExerciseKnockInTest {

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

    private static final int NUMBER_OF_TIME_STEPS = 160;
    private static final int NUMBER_OF_SPACE_STEPS_S = 200;
    private static final int NUMBER_OF_SPACE_STEPS_V = 120;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

    private static final double EQUALITY_TOLERANCE = 5.0E-2;
    private static final double ORDERING_TOLERANCE = 5.0E-3;
    private static final double DOMINANCE_TOLERANCE = 5.0E-2;
    private static final double ALREADY_HIT_TOLERANCE = 1.0E-10;

    private static final double DOWN_ALREADY_HIT_SPOT = 78.0;
    private static final double UP_ALREADY_HIT_SPOT = 122.0;

    @Test
    public void testOneDateBermudanEqualsEuropeanDownInCall() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOneDateBermudanEqualsEuropeanDownInPut() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOneDateBermudanEqualsEuropeanUpInCall() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testOneDateBermudanEqualsEuropeanUpInPut() {
        runOneDateBermudanEqualsEuropeanTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testOrderingDownInCall() {
        runOrderingTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOrderingDownInPut() {
        runOrderingTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testOrderingUpInCall() {
        runOrderingTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testOrderingUpInPut() {
        runOrderingTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testBermudanDownInCallLessOrEqualBermudanVanillaCall() {
        runBermudanDominanceTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testBermudanDownInPutLessOrEqualBermudanVanillaPut() {
        runBermudanDominanceTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testBermudanUpInCallLessOrEqualBermudanVanillaCall() {
        runBermudanDominanceTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testBermudanUpInPutLessOrEqualBermudanVanillaPut() {
        runBermudanDominanceTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testAmericanDownInCallMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testAmericanDownInPutMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testAmericanUpInCallMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.CALL, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testAmericanUpInPutMatchesAmericanVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                CallOrPut.PUT, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    private void runOneDateBermudanEqualsEuropeanTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier) {

        final TestSetup setup = createSetup(barrier, barrierType);

        final Exercise europeanExercise = new net.finmath.modelling.EuropeanExercise(MATURITY);
        final Exercise bermudanExercise = new BermudanExercise(new double[] { MATURITY });

        final FDMHestonModel referenceModel = getReferenceModelForExerciseConsistency(
                setup.model,
                callOrPut,
                barrierType,
                barrier,
                bermudanExercise
        );

        final BarrierOption european = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType
        );

        final BarrierOption bermudan = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType,
                bermudanExercise
        );

        final double europeanPrice = interpolateAt(
                european.getValue(0.0, referenceModel),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                SPOT,
                INITIAL_VARIANCE
        );

        final double bermudanPrice = interpolateAt(
                bermudan.getValue(0.0, referenceModel),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                SPOT,
                INITIAL_VARIANCE
        );

        System.out.println("One-date Bermudan vs European");
        System.out.println("Type            = " + barrierType + " " + callOrPut);
        System.out.println("European        = " + europeanPrice);
        System.out.println("One-date Berm   = " + bermudanPrice);
        System.out.println("Abs error       = " + Math.abs(europeanPrice - bermudanPrice));

        assertEquals(europeanPrice, bermudanPrice, EQUALITY_TOLERANCE);
    }

    private void runOrderingTest(
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier) {

        final TestSetup setup = createSetup(barrier, barrierType);

        final double[] bermudanExerciseTimes = new double[] { 0.25, 0.50, 0.75, 1.00 };

        final BarrierOption european = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType
        );

        final BarrierOption bermudan = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType,
                new BermudanExercise(bermudanExerciseTimes)
        );

        final BarrierOption american = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                barrierType,
                new AmericanExercise(MATURITY)
        );

        final double europeanPrice = interpolateAt(
                european.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        final double bermudanPrice = interpolateAt(
                bermudan.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        final double americanPrice = interpolateAt(
                american.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        System.out.println("Knock-in ordering");
        System.out.println("Type            = " + barrierType + " " + callOrPut);
        System.out.println("European        = " + europeanPrice);
        System.out.println("Bermudan        = " + bermudanPrice);
        System.out.println("American        = " + americanPrice);

        assertTrue(bermudanPrice + ORDERING_TOLERANCE >= europeanPrice);
        assertTrue(americanPrice + ORDERING_TOLERANCE >= bermudanPrice);
    }

    private void runBermudanDominanceTest(
            final CallOrPut callOrPut,
            final BarrierType knockInType,
            final double barrier) {

        final TestSetup setup = createSetup(barrier, knockInType);

        final double[] bermudanExerciseTimes = new double[] { 0.25, 0.50, 0.75, 1.00 };
        final Exercise bermudanExercise = new BermudanExercise(bermudanExerciseTimes);

        final BarrierOption knockIn = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                knockInType,
                bermudanExercise
        );

        final BermudanOption vanilla = new BermudanOption(
                bermudanExerciseTimes,
                STRIKE,
                callOrPut
        );

        final double knockInPrice = interpolateAt(
                knockIn.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        final double vanillaPrice = interpolateAt(
                vanilla.getValue(0.0, setup.model),
                setup.sNodes,
                setup.vNodes,
                SPOT,
                INITIAL_VARIANCE
        );

        System.out.println("Bermudan knock-in dominance");
        System.out.println("Type            = " + knockInType + " " + callOrPut);
        System.out.println("Knock-in        = " + knockInPrice);
        System.out.println("Vanilla         = " + vanillaPrice);

        assertTrue(knockInPrice <= vanillaPrice + DOMINANCE_TOLERANCE);
    }

    private void runAmericanAlreadyHitConsistencyTest(
            final CallOrPut callOrPut,
            final BarrierType knockInType,
            final double barrier,
            final double alreadyHitSpot) {

        final TestSetup setup = createSetup(barrier, knockInType);
        final Exercise americanExercise = new AmericanExercise(MATURITY);

        /*
         * The knock-in must be priced on the original model.
         * In the problematic DOWN_IN PUT case, the production code will internally
         * build a widened activated auxiliary model. Therefore the vanilla reference
         * must be priced on that activated model, not on the outer setup model.
         */
        final FDMHestonModel activatedReferenceModel = getActivatedReferenceModelUsedByKnockIn(
                setup.model,
                callOrPut,
                knockInType,
                barrier,
                americanExercise
        );

        final BarrierOption knockIn = new BarrierOption(
                MATURITY,
                STRIKE,
                barrier,
                REBATE,
                callOrPut,
                knockInType,
                americanExercise
        );

        final AmericanOption vanilla = new AmericanOption(
                MATURITY,
                STRIKE,
                callOrPut
        );

        final double knockInPrice = interpolateAt(
                knockIn.getValue(0.0, setup.model),
                setup.model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                setup.model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                alreadyHitSpot,
                INITIAL_VARIANCE
        );

        final double vanillaPrice = interpolateAt(
                vanilla.getValue(0.0, activatedReferenceModel),
                activatedReferenceModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                activatedReferenceModel.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                alreadyHitSpot,
                INITIAL_VARIANCE
        );

        System.out.println("American already-hit consistency");
        System.out.println("Type            = " + knockInType + " " + callOrPut);
        System.out.println("Already-hit spot= " + alreadyHitSpot);
        System.out.println("Knock-in        = " + knockInPrice);
        System.out.println("Vanilla         = " + vanillaPrice);
        System.out.println("Abs error       = " + Math.abs(knockInPrice - vanillaPrice));

        assertEquals(vanillaPrice, knockInPrice, ALREADY_HIT_TOLERANCE);
    }

    private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

        final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
        final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

        final TimeDiscretization timeDiscretization =
                new TimeDiscretizationFromArray(
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

        final double sMin;
        final double sMax;

        if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
            sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
            sMax = Math.max(3.0 * SPOT, SPOT + 12.0 * deltaS);
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

    private static final class TestSetup {

        private final FDMHestonModel model;
        private final double[] sNodes;
        private final double[] vNodes;

        private TestSetup(
                final FDMHestonModel model,
                final double[] sNodes,
                final double[] vNodes) {
            this.model = model;
            this.sNodes = sNodes;
            this.vNodes = vNodes;
        }
    }
    private FDMHestonModel getReferenceModelForExerciseConsistency(
            final FDMHestonModel baseModel,
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier,
            final Exercise exercise) {

        final boolean needsWidenedActivatedReference =
                baseModel instanceof FDMHestonModel
                && !exercise.isEuropean()
                && barrierType == BarrierType.DOWN_IN
                && callOrPut == CallOrPut.PUT;

        if(!needsWidenedActivatedReference) {
            return baseModel;
        }

        return createWidenedActivatedReferenceModel(baseModel, barrier);
    }

    private FDMHestonModel createWidenedActivatedReferenceModel(
            final FDMHestonModel barrierModel,
            final double barrier) {

        final SpaceTimeDiscretization base = barrierModel.getSpaceTimeDiscretization();
        final double[] baseSpotGrid = base.getSpaceGrid(0).getGrid();
        final double[] secondGrid = base.getSpaceGrid(1).getGrid();

        if(baseSpotGrid.length < 2) {
            throw new IllegalArgumentException("Barrier grid must contain at least two points.");
        }

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

    private FDMHestonModel getActivatedReferenceModelUsedByKnockIn(
            final FDMHestonModel baseModel,
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier,
            final Exercise exercise) {

        final SpaceTimeDiscretization base = baseModel.getSpaceTimeDiscretization();
        final double[] baseSpotGrid = base.getSpaceGrid(0).getGrid();

        if(baseSpotGrid.length < 2) {
            throw new IllegalArgumentException("Barrier grid must contain at least two points.");
        }

        final boolean forceWidenedActivatedGrid =
                !exercise.isEuropean()
                && barrierType == BarrierType.DOWN_IN
                && callOrPut == CallOrPut.PUT;

        boolean barrierAlreadyOnSpotGrid = false;
        for(final double s : baseSpotGrid) {
            if(Math.abs(s - barrier) <= 1E-8) {
                barrierAlreadyOnSpotGrid = true;
                break;
            }
        }

        if(barrierAlreadyOnSpotGrid && !forceWidenedActivatedGrid) {
            return baseModel;
        }

        final double[] secondGrid = base.getSpaceGrid(1).getGrid();

        final Grid activatedSpotGrid = buildActivatedSpotGridAlignedAtBarrier(
                baseSpotGrid,
                baseModel.getInitialValue()[0],
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
                        baseModel.getInitialValue()
                );

        return (FDMHestonModel) baseModel.getCloneWithModifiedSpaceTimeDiscretization(
                activatedDiscretization
        );
    }
}