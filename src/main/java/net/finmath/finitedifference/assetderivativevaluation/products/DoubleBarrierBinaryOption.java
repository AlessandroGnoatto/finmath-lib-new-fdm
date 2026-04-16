package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.DoubleBarrierType;

/**
 * Finite-difference valuation of a continuously monitored double-barrier cash binary option.
 *
 * <p>
 * Supported barrier styles:
 * </p>
 * <ul>
 *   <li>{@link DoubleBarrierType#KNOCK_OUT}: pays the cash amount if neither barrier is hit,</li>
 *   <li>{@link DoubleBarrierType#KNOCK_IN}: pays the cash amount once either barrier is hit,</li>
 *   <li>{@link DoubleBarrierType#KIKO}: knock-in at the lower barrier, knock-out at the upper barrier,</li>
 *   <li>{@link DoubleBarrierType#KOKI}: knock-out at the lower barrier, knock-in at the upper barrier.</li>
 * </ul>
 *
 * <p>
 * Current implementation policy:
 * </p>
 * <ul>
 *   <li>cash payoff only,</li>
 *   <li>European only,</li>
 *   <li>direct one-state pricing through internal state constraints,</li>
 *   <li>barrier activation semantics are enforced through the constrained value outside the alive band.</li>
 * </ul>
 * 
 * @author Alessandro Gnoatto
 */
public class DoubleBarrierBinaryOption implements
        FiniteDifferenceProduct,
        FiniteDifferenceInternalStateConstraint {

    private final String underlyingName;
    private final double maturity;
    private final double cashPayoff;
    private final double lowerBarrier;
    private final double upperBarrier;
    private final DoubleBarrierType doubleBarrierType;
    private final Exercise exercise;

    public DoubleBarrierBinaryOption(
            final String underlyingName,
            final double maturity,
            final double cashPayoff,
            final double lowerBarrier,
            final double upperBarrier,
            final DoubleBarrierType doubleBarrierType,
            final Exercise exercise) {

        if(doubleBarrierType == null) {
            throw new IllegalArgumentException("Double barrier type must not be null.");
        }
        if(exercise == null) {
            throw new IllegalArgumentException("Exercise must not be null.");
        }
        if(!exercise.isEuropean()) {
            throw new IllegalArgumentException("DoubleBarrierBinaryOption currently supports only European exercise.");
        }
        if(maturity < 0.0) {
            throw new IllegalArgumentException("Maturity must be non-negative.");
        }
        if(cashPayoff < 0.0) {
            throw new IllegalArgumentException("Cash payoff must be non-negative.");
        }
        if(lowerBarrier <= 0.0 || upperBarrier <= 0.0) {
            throw new IllegalArgumentException("Barriers must be positive.");
        }
        if(lowerBarrier >= upperBarrier) {
            throw new IllegalArgumentException("lowerBarrier must be < upperBarrier.");
        }

        this.underlyingName = underlyingName;
        this.maturity = maturity;
        this.cashPayoff = cashPayoff;
        this.lowerBarrier = lowerBarrier;
        this.upperBarrier = upperBarrier;
        this.doubleBarrierType = doubleBarrierType;
        this.exercise = exercise;
    }

    public DoubleBarrierBinaryOption(
            final String underlyingName,
            final double maturity,
            final double cashPayoff,
            final double lowerBarrier,
            final double upperBarrier,
            final DoubleBarrierType doubleBarrierType) {
        this(
                underlyingName,
                maturity,
                cashPayoff,
                lowerBarrier,
                upperBarrier,
                doubleBarrierType,
                new EuropeanExercise(maturity)
        );
    }

    public DoubleBarrierBinaryOption(
            final double maturity,
            final double cashPayoff,
            final double lowerBarrier,
            final double upperBarrier,
            final DoubleBarrierType doubleBarrierType) {
        this(
                null,
                maturity,
                cashPayoff,
                lowerBarrier,
                upperBarrier,
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

        if(cashPayoff == 0.0) {
            return buildZeroValueSurface(model);
        }

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

        return solver.getValues(maturity, this::pointwiseTerminalPayoff);
    }

    private void validateProductConfiguration(final FiniteDifferenceEquityModel model) {
        if(model == null) {
            throw new IllegalArgumentException("Model must not be null.");
        }
        if(!exercise.isEuropean()) {
            throw new IllegalArgumentException("DoubleBarrierBinaryOption currently supports only European exercise.");
        }

        final double[] spotGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
        final double gridMin = spotGrid[0];
        final double gridMax = spotGrid[spotGrid.length - 1];

        if(lowerBarrier < gridMin || upperBarrier > gridMax) {
            throw new IllegalArgumentException(
                    "Both double barriers must lie inside the first state-variable grid domain of the supplied model.");
        }
    }

    private double[][] buildZeroValueSurface(final FiniteDifferenceEquityModel model) {
        final int numberOfSpacePoints =
                model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid().length;
        final int numberOfTimePoints =
                model.getSpaceTimeDiscretization().getTimeDiscretization().getNumberOfTimeSteps() + 1;

        final double[][] zeroValues = new double[numberOfSpacePoints][numberOfTimePoints];
        for(int i = 0; i < numberOfSpacePoints; i++) {
            for(int j = 0; j < numberOfTimePoints; j++) {
                zeroValues[i][j] = 0.0;
            }
        }
        return zeroValues;
    }

    private double[] buildCellAveragedTerminalValues(final SpaceTimeDiscretization discretization) {
        final double[] sGrid = discretization.getSpaceGrid(0).getGrid();
        final double[] terminalValues = new double[sGrid.length];

        for(int i = 0; i < sGrid.length; i++) {
            final double leftEdge = getLeftDualCellEdge(sGrid, i);
            final double rightEdge = getRightDualCellEdge(sGrid, i);
            terminalValues[i] = cellAveragedTerminalPayoff(leftEdge, rightEdge);
        }

        return terminalValues;
    }

    private double cellAveragedTerminalPayoff(final double leftEdge, final double rightEdge) {
        if(!(leftEdge < rightEdge)) {
            throw new IllegalArgumentException("Require leftEdge < rightEdge.");
        }

        final double cellLength = rightEdge - leftEdge;

        final double belowLowerLength = Math.max(0.0, Math.min(rightEdge, lowerBarrier) - leftEdge);
        final double aboveUpperLength = Math.max(0.0, rightEdge - Math.max(leftEdge, upperBarrier));
        final double insideBandLength = Math.max(0.0, Math.min(rightEdge, upperBarrier) - Math.max(leftEdge, lowerBarrier));

        switch(doubleBarrierType) {
        case KNOCK_OUT:
            return cashPayoff * insideBandLength / cellLength;

        case KNOCK_IN:
            return cashPayoff * (belowLowerLength + aboveUpperLength) / cellLength;

        case KIKO:
            return cashPayoff * belowLowerLength / cellLength;

        case KOKI:
            return cashPayoff * aboveUpperLength / cellLength;

        default:
            throw new IllegalArgumentException("Unsupported double barrier type.");
        }
    }

    private double pointwiseTerminalPayoff(final double assetValue) {
        switch(doubleBarrierType) {
        case KNOCK_OUT:
            return isInsideBarrierBand(assetValue) ? cashPayoff : 0.0;

        case KNOCK_IN:
            return isInsideBarrierBand(assetValue) ? 0.0 : cashPayoff;

        case KIKO:
            return assetValue <= lowerBarrier ? cashPayoff : 0.0;

        case KOKI:
            return assetValue >= upperBarrier ? cashPayoff : 0.0;

        default:
            throw new IllegalArgumentException("Unsupported double barrier type.");
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

    private boolean isBelowLowerBarrier(final double assetValue) {
        return assetValue <= lowerBarrier;
    }

    private boolean isAboveUpperBarrier(final double assetValue) {
        return assetValue >= upperBarrier;
    }

    @Override
    public boolean isConstraintActive(final double time, final double... stateVariables) {
        final double underlyingLevel = stateVariables[0];
        return isBelowLowerBarrier(underlyingLevel) || isAboveUpperBarrier(underlyingLevel);
    }

    @Override
    public double getConstrainedValue(final double time, final double... stateVariables) {
        final double underlyingLevel = stateVariables[0];

        switch(doubleBarrierType) {
        case KNOCK_OUT:
            return 0.0;

        case KNOCK_IN:
            return cashPayoff;

        case KIKO:
            return isBelowLowerBarrier(underlyingLevel) ? cashPayoff : 0.0;

        case KOKI:
            return isAboveUpperBarrier(underlyingLevel) ? cashPayoff : 0.0;

        default:
            throw new IllegalArgumentException("Unsupported double barrier type.");
        }
    }

    public String getUnderlyingName() {
        return underlyingName;
    }

    public double getMaturity() {
        return maturity;
    }

    public double getCashPayoff() {
        return cashPayoff;
    }

    public double getLowerBarrier() {
        return lowerBarrier;
    }

    public double getUpperBarrier() {
        return upperBarrier;
    }

    public DoubleBarrierType getDoubleBarrierType() {
        return doubleBarrierType;
    }

    public Exercise getExercise() {
        return exercise;
    }
}