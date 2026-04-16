package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DoubleBarrierType;

/**
 * Finite-difference valuation of a continuously monitored vanilla double-barrier option.
 *
 * <p>
 * Current implementation policy:
 * </p>
 * <ul>
 *   <li>supports only vanilla call / put,</li>
 *   <li>supports only {@link DoubleBarrierType#KNOCK_OUT} and {@link DoubleBarrierType#KNOCK_IN},</li>
 *   <li>knock-out is priced directly by the finite-difference solver using internal state constraints,</li>
 *   <li>knock-in is obtained by in-out parity against the corresponding vanilla European option,</li>
 *   <li>current first-pass implementation is European only.</li>
 * </ul>
 * 
 * @author Alessandro Gnoatto
 */
public class DoubleBarrierOption implements
        FiniteDifferenceProduct,
        FiniteDifferenceInternalStateConstraint {

    private enum PricingMode {
        DIRECT_OUT,
        PARITY_IN
    }

    private final String underlyingName;
    private final double maturity;
    private final double strike;
    private final double lowerBarrier;
    private final double upperBarrier;
    private final CallOrPut callOrPutSign;
    private final DoubleBarrierType doubleBarrierType;
    private final Exercise exercise;

    public DoubleBarrierOption(
            final String underlyingName,
            final double maturity,
            final double strike,
            final double lowerBarrier,
            final double upperBarrier,
            final CallOrPut callOrPutSign,
            final DoubleBarrierType doubleBarrierType,
            final Exercise exercise) {

        if(callOrPutSign == null) {
            throw new IllegalArgumentException("Option type must not be null.");
        }
        if(doubleBarrierType == null) {
            throw new IllegalArgumentException("Double barrier type must not be null.");
        }
        if(exercise == null) {
            throw new IllegalArgumentException("Exercise must not be null.");
        }
        if(!exercise.isEuropean()) {
            throw new IllegalArgumentException("DoubleBarrierOption currently supports only European exercise.");
        }
        if(maturity < 0.0) {
            throw new IllegalArgumentException("Maturity must be non-negative.");
        }
        if(strike <= 0.0) {
            throw new IllegalArgumentException("Strike must be positive.");
        }
        if(lowerBarrier <= 0.0 || upperBarrier <= 0.0) {
            throw new IllegalArgumentException("Barriers must be positive.");
        }
        if(lowerBarrier >= upperBarrier) {
            throw new IllegalArgumentException("lowerBarrier must be < upperBarrier.");
        }
        if(doubleBarrierType == DoubleBarrierType.KIKO || doubleBarrierType == DoubleBarrierType.KOKI) {
            throw new IllegalArgumentException("KIKO/KOKI are not supported for vanilla double-barrier options.");
        }

        this.underlyingName = underlyingName;
        this.maturity = maturity;
        this.strike = strike;
        this.lowerBarrier = lowerBarrier;
        this.upperBarrier = upperBarrier;
        this.callOrPutSign = callOrPutSign;
        this.doubleBarrierType = doubleBarrierType;
        this.exercise = exercise;
    }

    public DoubleBarrierOption(
            final String underlyingName,
            final double maturity,
            final double strike,
            final double lowerBarrier,
            final double upperBarrier,
            final CallOrPut callOrPutSign,
            final DoubleBarrierType doubleBarrierType) {
        this(
                underlyingName,
                maturity,
                strike,
                lowerBarrier,
                upperBarrier,
                callOrPutSign,
                doubleBarrierType,
                new EuropeanExercise(maturity)
        );
    }

    public DoubleBarrierOption(
            final double maturity,
            final double strike,
            final double lowerBarrier,
            final double upperBarrier,
            final CallOrPut callOrPutSign,
            final DoubleBarrierType doubleBarrierType) {
        this(
                null,
                maturity,
                strike,
                lowerBarrier,
                upperBarrier,
                callOrPutSign,
                doubleBarrierType,
                new EuropeanExercise(maturity)
        );
    }

    public DoubleBarrierOption(
            final double maturity,
            final double strike,
            final double lowerBarrier,
            final double upperBarrier,
            final double callOrPutSign,
            final DoubleBarrierType doubleBarrierType) {
        this(
                null,
                maturity,
                strike,
                lowerBarrier,
                upperBarrier,
                mapCallOrPut(callOrPutSign),
                doubleBarrierType,
                new EuropeanExercise(maturity)
        );
    }

    @Override
    public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
        final double[][] values = getValues(model);

        final SpaceTimeDiscretization valuationDiscretization = model.getSpaceTimeDiscretization();
        final double tau = maturity - evaluationTime;
        final int timeIndex = valuationDiscretization.getTimeDiscretization()
                .getTimeIndexNearestLessOrEqual(tau);

        final double[] column = new double[values.length];
        for(int i = 0; i < values.length; i++) {
            column[i] = values[i][timeIndex];
        }
        return column;
    }

    @Override
    public double[][] getValues(final FiniteDifferenceEquityModel model) {
        validateProductConfiguration(model);

        switch(getPricingMode()) {
        case DIRECT_OUT:
            return priceOutOptionDirectly(model);
        case PARITY_IN:
            return priceInOptionByParity(model);
        default:
            throw new IllegalStateException("Unsupported pricing mode.");
        }
    }

    private PricingMode getPricingMode() {
        return doubleBarrierType == DoubleBarrierType.KNOCK_OUT
                ? PricingMode.DIRECT_OUT
                : PricingMode.PARITY_IN;
    }

    private void validateProductConfiguration(final FiniteDifferenceEquityModel model) {
        if(model == null) {
            throw new IllegalArgumentException("Model must not be null.");
        }
        if(!exercise.isEuropean()) {
            throw new IllegalArgumentException("DoubleBarrierOption currently supports only European exercise.");
        }

        final double[] spotGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
        final double gridMin = spotGrid[0];
        final double gridMax = spotGrid[spotGrid.length - 1];

        if(lowerBarrier < gridMin || upperBarrier > gridMax) {
            throw new IllegalArgumentException(
                    "Both double barriers must lie inside the first state-variable grid domain of the supplied model.");
        }
    }

    private double[][] priceOutOptionDirectly(final FiniteDifferenceEquityModel model) {
        final FDMSolver solver = FDMSolverFactory.createSolver(
                model,
                this,
                model.getSpaceTimeDiscretization(),
                exercise
        );

        final boolean isOneDimensional = model.getSpaceTimeDiscretization().getNumberOfSpaceGrids() == 1;

        if(isOneDimensional) {
            final double[] terminalValues =
                    buildCellAveragedTerminalValues(model.getSpaceTimeDiscretization());

            return solver.getValues(maturity, terminalValues);
        }

        return solver.getValues(maturity, this::pointwisePayoffForDirectOutPricing);
    }

    private double[][] priceInOptionByParity(final FiniteDifferenceEquityModel model) {
        final DoubleBarrierOption correspondingOutOption = createCorrespondingOutOption();
        final EuropeanOption vanillaOption = createVanillaEuropeanOption();

        final double[][] outValues = correspondingOutOption.getValues(model);
        final double[][] vanillaValues = vanillaOption.getValues(model);

        final int numberOfRows = outValues.length;
        final int numberOfColumns = outValues[0].length;

        final double[][] inValues = new double[numberOfRows][numberOfColumns];

        for(int i = 0; i < numberOfRows; i++) {
            for(int j = 0; j < numberOfColumns; j++) {
                inValues[i][j] = vanillaValues[i][j] - outValues[i][j];
            }
        }

        return inValues;
    }

    private DoubleBarrierOption createCorrespondingOutOption() {
        return new DoubleBarrierOption(
                underlyingName,
                maturity,
                strike,
                lowerBarrier,
                upperBarrier,
                callOrPutSign,
                DoubleBarrierType.KNOCK_OUT,
                exercise
        );
    }

    private EuropeanOption createVanillaEuropeanOption() {
        return new EuropeanOption(
                underlyingName,
                maturity,
                strike,
                callOrPutSign
        );
    }

    private double[] buildCellAveragedTerminalValues(final SpaceTimeDiscretization discretization) {
        final double[] sGrid = discretization.getSpaceGrid(0).getGrid();
        final double[] terminalValues = new double[sGrid.length];

        for(int i = 0; i < sGrid.length; i++) {
            final double leftEdge = getLeftDualCellEdge(sGrid, i);
            final double rightEdge = getRightDualCellEdge(sGrid, i);
            terminalValues[i] = cellAveragedPayoffForDirectOutPricing(leftEdge, rightEdge);
        }

        return terminalValues;
    }

    private double cellAveragedPayoffForDirectOutPricing(final double leftEdge, final double rightEdge) {
        if(!(leftEdge < rightEdge)) {
            throw new IllegalArgumentException("Require leftEdge < rightEdge.");
        }

        final double aliveLeft = Math.max(leftEdge, lowerBarrier);
        final double aliveRight = Math.min(rightEdge, upperBarrier);

        if(aliveRight <= aliveLeft) {
            return 0.0;
        }

        final double aliveWeight = (aliveRight - aliveLeft) / (rightEdge - leftEdge);
        return aliveWeight * cellAveragedVanillaPayoff(aliveLeft, aliveRight);
    }

    private double cellAveragedVanillaPayoff(final double leftEdge, final double rightEdge) {
        final double cellLength = rightEdge - leftEdge;

        if(callOrPutSign == CallOrPut.CALL) {
            if(rightEdge <= strike) {
                return 0.0;
            }
            if(leftEdge >= strike) {
                return 0.5 * (leftEdge + rightEdge) - strike;
            }
            return 0.5 * Math.pow(rightEdge - strike, 2.0) / cellLength;
        }
        else {
            if(leftEdge >= strike) {
                return 0.0;
            }
            if(rightEdge <= strike) {
                return strike - 0.5 * (leftEdge + rightEdge);
            }
            return 0.5 * Math.pow(strike - leftEdge, 2.0) / cellLength;
        }
    }

    private double pointwisePayoffForDirectOutPricing(final double assetValue) {
        if(!isInsideBarrierBand(assetValue)) {
            return 0.0;
        }

        if(callOrPutSign == CallOrPut.CALL) {
            return Math.max(assetValue - strike, 0.0);
        }
        else {
            return Math.max(strike - assetValue, 0.0);
        }
    }

    private double getLeftDualCellEdge(final double[] grid, final int i) {
        if(i == 0) {
            return grid[0];
        }
        return 0.5 * (grid[i - 1] + grid[i]);
    }

    private double getRightDualCellEdge(final double[] grid, final int i) {
        if(i == grid.length - 1) {
            return grid[grid.length - 1];
        }
        return 0.5 * (grid[i] + grid[i + 1]);
    }

    private boolean isInsideBarrierBand(final double assetValue) {
        return assetValue > lowerBarrier && assetValue < upperBarrier;
    }

    @Override
    public boolean isConstraintActive(final double time, final double... stateVariables) {
        final double underlyingLevel = stateVariables[0];
        return underlyingLevel <= lowerBarrier || underlyingLevel >= upperBarrier;
    }

    @Override
    public double getConstrainedValue(final double time, final double... stateVariables) {
        return 0.0;
    }

    public String getUnderlyingName() {
        return underlyingName;
    }

    public double getMaturity() {
        return maturity;
    }

    public double getStrike() {
        return strike;
    }

    public double getLowerBarrier() {
        return lowerBarrier;
    }

    public double getUpperBarrier() {
        return upperBarrier;
    }

    public CallOrPut getCallOrPut() {
        return callOrPutSign;
    }

    public DoubleBarrierType getDoubleBarrierType() {
        return doubleBarrierType;
    }

    public Exercise getExercise() {
        return exercise;
    }

    private static CallOrPut mapCallOrPut(final double sign) {
        if(sign == 1.0) {
            return CallOrPut.CALL;
        }
        if(sign == -1.0) {
            return CallOrPut.PUT;
        }
        throw new IllegalArgumentException("Unknown option type.");
    }
}