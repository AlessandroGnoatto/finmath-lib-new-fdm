package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
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
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for direct 2D digital knock-ins with early exercise
 * under Heston and SABR.
 *
 * <p>
 * This first class focuses on cash-or-nothing digitals only.
 * </p>
 *
 * <p>
 * Verified properties:
 * </p>
 * <ul>
 *   <li>one-date Bermudan knock-in = European knock-in,</li>
 *   <li>European <= Bermudan <= American for the knock-in itself,</li>
 *   <li>Bermudan knock-in <= Bermudan vanilla digital,</li>
 *   <li>in the already-hit region, American knock-in = American vanilla digital.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalBarrierOptionEarlyExerciseKnockInTest {

    private enum ModelType {
        HESTON,
        SABR
    }

    private static final double MATURITY = 1.0;
    private static final double STRIKE = 100.0;
    private static final double CASH_PAYOFF = 10.0;

    private static final double SPOT = 100.0;
    private static final double RISK_FREE_RATE = 0.05;
    private static final double DIVIDEND_YIELD = 0.00;

    private static final double HESTON_VOLATILITY = 0.25;
    private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
    private static final double HESTON_LONG_RUN_VARIANCE = HESTON_INITIAL_VARIANCE;
    private static final double HESTON_KAPPA = 1.5;
    private static final double HESTON_XI = 0.30;
    private static final double HESTON_RHO = -0.70;

    private static final double SABR_INITIAL_VOLATILITY = 0.20;
    private static final double SABR_BETA = 1.0;
    private static final double SABR_NU = 0.30;
    private static final double SABR_RHO = -0.50;

    private static final double THETA = 0.5;
    private static final int NUMBER_OF_TIME_STEPS = 140;
    private static final int NUMBER_OF_SPACE_STEPS_S = 180;
    private static final int NUMBER_OF_SPACE_STEPS_2 = 100;
    private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
    private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

    private static final double GRID_TOLERANCE = 1E-8;
    private static final double EQUALITY_TOLERANCE = 1.0E-1;
    private static final double ORDERING_TOLERANCE = 5.0E-3;
    private static final double DOMINANCE_TOLERANCE = 1.0E-1;
    private static final double ALREADY_HIT_TOLERANCE = 1.0E-10;

    private static final double DOWN_ALREADY_HIT_SPOT = 78.0;
    private static final double UP_ALREADY_HIT_SPOT = 122.0;

    /*
     * HESTON
     */

    @Test
    public void testHestonOneDateBermudanEqualsEuropeanDownInCashCall() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.HESTON, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0
        );
    }

    @Test
    public void testHestonOneDateBermudanEqualsEuropeanDownInCashPut() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.HESTON, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0
        );
    }

    @Test
    public void testHestonOneDateBermudanEqualsEuropeanUpInCashCall() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.HESTON, CallOrPut.CALL, BarrierType.UP_IN, 120.0
        );
    }

    @Test
    public void testHestonOneDateBermudanEqualsEuropeanUpInCashPut() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.HESTON, CallOrPut.PUT, BarrierType.UP_IN, 120.0
        );
    }

    @Test
    public void testHestonOrderingDownInCashCall() {
        runOrderingTest(ModelType.HESTON, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testHestonOrderingDownInCashPut() {
        runOrderingTest(ModelType.HESTON, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testHestonOrderingUpInCashCall() {
        runOrderingTest(ModelType.HESTON, CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testHestonOrderingUpInCashPut() {
        runOrderingTest(ModelType.HESTON, CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testHestonBermudanDownInCashCallLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.HESTON, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testHestonBermudanDownInCashPutLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.HESTON, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testHestonBermudanUpInCashCallLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.HESTON, CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testHestonBermudanUpInCashPutLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.HESTON, CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testHestonAmericanDownInCashCallMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.HESTON, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testHestonAmericanDownInCashPutMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.HESTON, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testHestonAmericanUpInCashCallMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.HESTON, CallOrPut.CALL, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testHestonAmericanUpInCashPutMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.HESTON, CallOrPut.PUT, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    /*
     * SABR
     */

    @Test
    public void testSabrOneDateBermudanEqualsEuropeanDownInCashCall() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.SABR, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0
        );
    }

    @Test
    public void testSabrOneDateBermudanEqualsEuropeanDownInCashPut() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.SABR, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0
        );
    }

    @Test
    public void testSabrOneDateBermudanEqualsEuropeanUpInCashCall() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.SABR, CallOrPut.CALL, BarrierType.UP_IN, 120.0
        );
    }

    @Test
    public void testSabrOneDateBermudanEqualsEuropeanUpInCashPut() {
        runOneDateBermudanEqualsEuropeanTest(
                ModelType.SABR, CallOrPut.PUT, BarrierType.UP_IN, 120.0
        );
    }

    @Test
    public void testSabrOrderingDownInCashCall() {
        runOrderingTest(ModelType.SABR, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testSabrOrderingDownInCashPut() {
        runOrderingTest(ModelType.SABR, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testSabrOrderingUpInCashCall() {
        runOrderingTest(ModelType.SABR, CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testSabrOrderingUpInCashPut() {
        runOrderingTest(ModelType.SABR, CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testSabrBermudanDownInCashCallLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.SABR, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testSabrBermudanDownInCashPutLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.SABR, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
    }

    @Test
    public void testSabrBermudanUpInCashCallLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.SABR, CallOrPut.CALL, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testSabrBermudanUpInCashPutLessOrEqualVanilla() {
        runBermudanDominanceTest(ModelType.SABR, CallOrPut.PUT, BarrierType.UP_IN, 120.0);
    }

    @Test
    public void testSabrAmericanDownInCashCallMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.SABR, CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testSabrAmericanDownInCashPutMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.SABR, CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, DOWN_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testSabrAmericanUpInCashCallMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.SABR, CallOrPut.CALL, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    @Test
    public void testSabrAmericanUpInCashPutMatchesVanillaInAlreadyHitRegion() {
        runAmericanAlreadyHitConsistencyTest(
                ModelType.SABR, CallOrPut.PUT, BarrierType.UP_IN, 120.0, UP_ALREADY_HIT_SPOT
        );
    }

    private void runOneDateBermudanEqualsEuropeanTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier) {

        final TestSetup setup = createSetup(modelType, barrier, barrierType);

        final Exercise bermudanExercise = new BermudanExercise(new double[] { MATURITY });

        final FiniteDifferenceEquityModel referenceModel =
                getReferenceModelForExerciseConsistency(
                        setup.model,
                        callOrPut,
                        barrierType,
                        barrier,
                        bermudanExercise
                );

        final DigitalBarrierOption european = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                new EuropeanExercise(MATURITY)
        );

        final DigitalBarrierOption bermudan = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                bermudanExercise
        );

        final double europeanPrice = interpolateAt(
                european.getValue(0.0, referenceModel),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                SPOT,
                setup.secondState0
        );

        final double bermudanPrice = interpolateAt(
                bermudan.getValue(0.0, referenceModel),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                referenceModel.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                SPOT,
                setup.secondState0
        );

        System.out.println("One-date Bermudan vs European digital");
        System.out.println("Model           = " + modelType);
        System.out.println("Type            = " + barrierType + " " + callOrPut);
        System.out.println("European        = " + europeanPrice);
        System.out.println("One-date Berm   = " + bermudanPrice);
        System.out.println("Abs error       = " + Math.abs(europeanPrice - bermudanPrice));

        assertEquals(europeanPrice, bermudanPrice, EQUALITY_TOLERANCE);
    }

    private void runOrderingTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier) {

        final TestSetup setup = createSetup(modelType, barrier, barrierType);
        final double[] bermudanExerciseTimes = new double[] { 0.25, 0.50, 0.75, 1.00 };

        final DigitalBarrierOption european = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                new EuropeanExercise(MATURITY)
        );

        final DigitalBarrierOption bermudan = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                new BermudanExercise(bermudanExerciseTimes)
        );

        final DigitalBarrierOption american = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                barrierType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                new AmericanExercise(MATURITY)
        );

        final double europeanPrice = interpolateAt(
                european.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        final double bermudanPrice = interpolateAt(
                bermudan.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        final double americanPrice = interpolateAt(
                american.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        System.out.println("Digital knock-in ordering");
        System.out.println("Model           = " + modelType);
        System.out.println("Type            = " + barrierType + " " + callOrPut);
        System.out.println("European        = " + europeanPrice);
        System.out.println("Bermudan        = " + bermudanPrice);
        System.out.println("American        = " + americanPrice);

        assertTrue(bermudanPrice + ORDERING_TOLERANCE >= europeanPrice);
        assertTrue(americanPrice + ORDERING_TOLERANCE >= bermudanPrice);
    }

    private void runBermudanDominanceTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final BarrierType knockInType,
            final double barrier) {

        final TestSetup setup = createSetup(modelType, barrier, knockInType);
        final Exercise bermudanExercise = new BermudanExercise(new double[] { 0.25, 0.50, 0.75, 1.00 });

        final DigitalBarrierOption knockIn = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                knockInType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                bermudanExercise
        );

        final DigitalOption vanilla = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                bermudanExercise
        );

        final double knockInPrice = interpolateAt(
                knockIn.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        final double vanillaPrice = interpolateAt(
                vanilla.getValue(0.0, setup.model),
                setup.sNodes,
                setup.secondNodes,
                SPOT,
                setup.secondState0
        );

        System.out.println("Bermudan digital knock-in dominance");
        System.out.println("Model           = " + modelType);
        System.out.println("Type            = " + knockInType + " " + callOrPut);
        System.out.println("Knock-in        = " + knockInPrice);
        System.out.println("Vanilla         = " + vanillaPrice);

        assertTrue(knockInPrice <= vanillaPrice + DOMINANCE_TOLERANCE);
    }

    private void runAmericanAlreadyHitConsistencyTest(
            final ModelType modelType,
            final CallOrPut callOrPut,
            final BarrierType knockInType,
            final double barrier,
            final double alreadyHitSpot) {

        final TestSetup setup = createSetup(modelType, barrier, knockInType);
        final Exercise americanExercise = new AmericanExercise(MATURITY);

        final FiniteDifferenceEquityModel activatedReferenceModel =
                getActivatedReferenceModelUsedByKnockIn(
                        setup.model,
                        callOrPut,
                        knockInType,
                        barrier,
                        americanExercise
                );

        final DigitalBarrierOption knockIn = new DigitalBarrierOption(
                null,
                MATURITY,
                STRIKE,
                barrier,
                callOrPut,
                knockInType,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                americanExercise
        );

        final DigitalOption vanilla = new DigitalOption(
                null,
                MATURITY,
                STRIKE,
                callOrPut,
                DigitalPayoffType.CASH_OR_NOTHING,
                CASH_PAYOFF,
                americanExercise
        );

        final double knockInPrice = interpolateAt(
                knockIn.getValue(0.0, setup.model),
                setup.model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                setup.model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                alreadyHitSpot,
                setup.secondState0
        );

        final double vanillaPrice = interpolateAt(
                vanilla.getValue(0.0, activatedReferenceModel),
                activatedReferenceModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
                activatedReferenceModel.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid(),
                alreadyHitSpot,
                setup.secondState0
        );

        System.out.println("American already-hit digital consistency");
        System.out.println("Model           = " + modelType);
        System.out.println("Type            = " + knockInType + " " + callOrPut);
        System.out.println("Already-hit spot= " + alreadyHitSpot);
        System.out.println("Knock-in        = " + knockInPrice);
        System.out.println("Vanilla         = " + vanillaPrice);
        System.out.println("Abs error       = " + Math.abs(knockInPrice - vanillaPrice));

        assertEquals(vanillaPrice, knockInPrice, ALREADY_HIT_TOLERANCE);
    }

    private FiniteDifferenceEquityModel getReferenceModelForExerciseConsistency(
            final FiniteDifferenceEquityModel baseModel,
            final CallOrPut callOrPut,
            final BarrierType barrierType,
            final double barrier,
            final Exercise exercise) {

        final FiniteDifferenceEquityModel effectiveModel =
                getEffectiveModelForExerciseConsistency(baseModel, exercise);

        final boolean needsWidenedActivatedReference =
                !exercise.isEuropean()
                && barrierType == BarrierType.DOWN_IN
                && callOrPut == CallOrPut.PUT;

        if(!needsWidenedActivatedReference) {
            return effectiveModel;
        }

        return createWidenedActivatedReferenceModel(effectiveModel, barrier);
    }

    private FiniteDifferenceEquityModel getActivatedReferenceModelUsedByKnockIn(
            final FiniteDifferenceEquityModel baseModel,
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
            if(Math.abs(s - barrier) <= GRID_TOLERANCE) {
                barrierAlreadyOnSpotGrid = true;
                break;
            }
        }

        if(barrierAlreadyOnSpotGrid && !forceWidenedActivatedGrid) {
            return baseModel;
        }

        return createWidenedActivatedReferenceModel(baseModel, barrier);
    }

    private FiniteDifferenceEquityModel getEffectiveModelForExerciseConsistency(
            final FiniteDifferenceEquityModel baseModel,
            final Exercise exercise) {

        if(!exercise.isBermudan()) {
            return baseModel;
        }

        final SpaceTimeDiscretization base = baseModel.getSpaceTimeDiscretization();
        final TimeDiscretization refinedTimeDiscretization =
                FiniteDifferenceExerciseUtil.refineTimeDiscretization(
                        base.getTimeDiscretization(),
                        exercise
                );

        final Grid[] spaceGrids = new Grid[base.getNumberOfSpaceGrids()];
        for(int i = 0; i < base.getNumberOfSpaceGrids(); i++) {
            spaceGrids[i] = base.getSpaceGrid(i);
        }

        final SpaceTimeDiscretization refinedDiscretization =
                new SpaceTimeDiscretization(
                        spaceGrids,
                        refinedTimeDiscretization,
                        base.getTheta(),
                        baseModel.getInitialValue()
                );

        return baseModel.getCloneWithModifiedSpaceTimeDiscretization(refinedDiscretization);
    }

    private FiniteDifferenceEquityModel createWidenedActivatedReferenceModel(
            final FiniteDifferenceEquityModel barrierModel,
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

        return barrierModel.getCloneWithModifiedSpaceTimeDiscretization(
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

    private TestSetup createSetup(
            final ModelType modelType,
            final double barrier,
            final BarrierType barrierType) {

        if(modelType == ModelType.HESTON) {
            return createHestonSetup(barrier, barrierType);
        }
        return createSabrSetup(barrier, barrierType);
    }

    private TestSetup createHestonSetup(final double barrier, final BarrierType barrierType) {

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
                4.0 * HESTON_LONG_RUN_VARIANCE,
                HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
        );

        final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_2, vMin, vMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, vGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, HESTON_INITIAL_VARIANCE }
        );

        final FDMHestonModel model = new FDMHestonModel(
                SPOT,
                HESTON_INITIAL_VARIANCE,
                riskFreeCurve,
                dividendCurve,
                HESTON_KAPPA,
                HESTON_LONG_RUN_VARIANCE,
                HESTON_XI,
                HESTON_RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), vGrid.getGrid(), HESTON_INITIAL_VARIANCE);
    }

    private TestSetup createSabrSetup(final double barrier, final BarrierType barrierType) {

        final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
                0.0,
                NUMBER_OF_TIME_STEPS,
                MATURITY / NUMBER_OF_TIME_STEPS
        );

        final Grid sGrid = createSpotGrid(barrier, barrierType);

        final double volMax = Math.max(4.0 * SABR_INITIAL_VOLATILITY, 1.0);
        final Grid volGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_2, 0.0, volMax);

        final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
                new Grid[] { sGrid, volGrid },
                timeDiscretization,
                THETA,
                new double[] { SPOT, SABR_INITIAL_VOLATILITY }
        );

        final FDMSabrModel model = new FDMSabrModel(
                SPOT,
                RISK_FREE_RATE,
                DIVIDEND_YIELD,
                SABR_INITIAL_VOLATILITY,
                SABR_BETA,
                SABR_NU,
                SABR_RHO,
                spaceTime
        );

        return new TestSetup(model, sGrid.getGrid(), volGrid.getGrid(), SABR_INITIAL_VOLATILITY);
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
            final double[] values,
            final double[] sNodes,
            final double[] secondNodes,
            final double spot,
            final double secondState) {

        assertTrue(
                "Spot must lie inside the grid domain.",
                spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
        );
        assertTrue(
                "Second state must lie inside the grid domain.",
                secondState >= secondNodes[0] - 1E-12
                        && secondState <= secondNodes[secondNodes.length - 1] + 1E-12
        );

        final int nS = sNodes.length;
        final int n2 = secondNodes.length;

        final double[][] valueSurface = new double[nS][n2];
        for(int j = 0; j < n2; j++) {
            for(int i = 0; i < nS; i++) {
                valueSurface[i][j] = values[flatten(i, j, nS)];
            }
        }

        final BiLinearInterpolation interpolation =
                new BiLinearInterpolation(sNodes, secondNodes, valueSurface);

        return interpolation.apply(spot, secondState);
    }

    private int flatten(final int iS, final int i2, final int numberOfSNodes) {
        return iS + i2 * numberOfSNodes;
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

    private static final class TestSetup {

        private final FiniteDifferenceEquityModel model;
        private final double[] sNodes;
        private final double[] secondNodes;
        private final double secondState0;

        private TestSetup(
                final FiniteDifferenceEquityModel model,
                final double[] sNodes,
                final double[] secondNodes,
                final double secondState0) {
            this.model = model;
            this.sNodes = sNodes;
            this.secondNodes = secondNodes;
            this.secondState0 = secondState0;
        }
    }
}