package net.finmath.finitedifference.assetderivativevaluation.products;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import net.finmath.finitedifference.FiniteDifferenceExerciseUtil;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.finitedifference.solvers.adi.AbstractADI2D;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;

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
 * Exercise semantics implemented here:
 * </p>
 * <ul>
 *   <li>European: payoff is determined at maturity as usual,</li>
 *   <li>Bermudan/American: the holder may take the current binary intrinsic value
 *       (cash-or-zero depending on the current barrier state) at an allowed exercise time.</li>
 * </ul>
 *
 * <p>
 * This means:
 * </p>
 * <ul>
 *   <li>KNOCK_OUT: inside the alive band the immediate exercise payoff is {@code cashPayoff},</li>
 *   <li>KNOCK_IN: outside the alive band the immediate exercise payoff is {@code cashPayoff},</li>
 *   <li>KIKO: below the lower barrier the immediate exercise payoff is {@code cashPayoff},</li>
 *   <li>KOKI: above the upper barrier the immediate exercise payoff is {@code cashPayoff}.</li>
 * </ul>
 *
 * <p>
 * The implementation remains a direct one-state constrained PDE.
 * </p>
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
		if(!exercise.isEuropean() && !exercise.isBermudan() && !exercise.isAmerican()) {
			throw new IllegalArgumentException(
					"DoubleBarrierBinaryOption currently supports only European, Bermudan, and American exercise.");
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

	public DoubleBarrierBinaryOption(
			final double maturity,
			final double cashPayoff,
			final double lowerBarrier,
			final double upperBarrier,
			final DoubleBarrierType doubleBarrierType,
			final Exercise exercise) {
		this(
				null,
				maturity,
				cashPayoff,
				lowerBarrier,
				upperBarrier,
				doubleBarrierType,
				exercise
		);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final double[][] values = getValues(model);

		final SpaceTimeDiscretization valuationDiscretization = getValuationSpaceTimeDiscretization(model);
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

		final FiniteDifferenceEquityModel effectiveModel = getEffectiveModelForValuation(model);
		final SpaceTimeDiscretization valuationDiscretization = effectiveModel.getSpaceTimeDiscretization();

		if(cashPayoff == 0.0) {
			return buildZeroValueSurface(valuationDiscretization);
		}

		final int dims = valuationDiscretization.getNumberOfSpaceGrids();

		if(dims == 1) {
			return getValues1D(effectiveModel, valuationDiscretization);
		}
		else if(dims == 2) {
			return getValues2D(effectiveModel, valuationDiscretization);
		}
		else {
			throw new IllegalArgumentException("DoubleBarrierBinaryOption currently supports only 1D and 2D models.");
		}
	}

	private double[][] getValues1D(
			final FiniteDifferenceEquityModel model,
			final SpaceTimeDiscretization valuationDiscretization) {

		final FDMSolver solver = new FDMThetaMethod1D(
				model,
				this,
				valuationDiscretization,
				exercise
		);

		final double[] terminalValues = buildCellAveragedTerminalValues(valuationDiscretization);

		if(exercise.isEuropean()) {
			return solver.getValues(maturity, terminalValues);
		}

		return solver.getValues(
				maturity,
				terminalValues,
				this::pointwiseExercisePayoff
		);
	}

	private double[][] getValues2D(
			final FiniteDifferenceEquityModel model,
			final SpaceTimeDiscretization valuationDiscretization) {

		final FDMSolver solver = FDMSolverFactory.createSolver(
				model,
				this,
				valuationDiscretization,
				exercise
		);

		final DoubleBinaryOperator terminalPayoff2D =
				(assetValue, secondState) -> pointwiseTerminalPayoff(assetValue);

		if(exercise.isEuropean()) {
			return solver.getValues(maturity, terminalPayoff2D);
		}

		if(!(solver instanceof AbstractADI2D)) {
			throw new IllegalArgumentException(
					"Two-dimensional Bermudan/American double-barrier binary pricing requires an ADI solver.");
		}

		return ((AbstractADI2D)solver).getValues(
				maturity,
				terminalPayoff2D,
				(runningTime, assetValue, secondState) -> pointwiseExercisePayoff(assetValue)
		);
	}

	private void validateProductConfiguration(final FiniteDifferenceEquityModel model) {
		if(model == null) {
			throw new IllegalArgumentException("Model must not be null.");
		}

		final SpaceTimeDiscretization valuationDiscretization = getValuationSpaceTimeDiscretization(model);
		final int dims = valuationDiscretization.getNumberOfSpaceGrids();

		if(dims != 1 && dims != 2) {
			throw new IllegalArgumentException("DoubleBarrierBinaryOption currently supports only 1D and 2D models.");
		}

		final double[] spotGrid = valuationDiscretization.getSpaceGrid(0).getGrid();
		final double gridMin = spotGrid[0];
		final double gridMax = spotGrid[spotGrid.length - 1];

		if(lowerBarrier < gridMin || upperBarrier > gridMax) {
			throw new IllegalArgumentException(
					"Both double barriers must lie inside the first state-variable grid domain of the supplied model.");
		}
	}

	private double[][] buildZeroValueSurface(final SpaceTimeDiscretization discretization) {
		final int numberOfSpacePoints = getTotalNumberOfSpacePoints(discretization);
		final int numberOfTimePoints = discretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;

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

	/**
	 * Immediate-exercise payoff under the current binary state.
	 *
	 * <p>
	 * For this product, the natural exercise value is the current binary intrinsic value.
	 * That matches the terminal state classification.
	 * </p>
	 */
	private double pointwiseExercisePayoff(final double assetValue) {
		return pointwiseTerminalPayoff(assetValue);
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

	private SpaceTimeDiscretization getValuationSpaceTimeDiscretization(final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization base = model.getSpaceTimeDiscretization();

		if(!exercise.isBermudan()) {
			return base;
		}

		final TimeDiscretization refinedTimeDiscretization =
				FiniteDifferenceExerciseUtil.refineTimeDiscretization(
						base.getTimeDiscretization(),
						exercise
				);

		if(base.getNumberOfSpaceGrids() == 1) {
			return new SpaceTimeDiscretization(
					base.getSpaceGrid(0),
					refinedTimeDiscretization,
					base.getTheta(),
					new double[] { base.getCenter(0) }
			);
		}

		final int numberOfSpaceGrids = base.getNumberOfSpaceGrids();
		final Grid[] spaceGrids = new Grid[numberOfSpaceGrids];
		final double[] center = new double[numberOfSpaceGrids];

		for(int i = 0; i < numberOfSpaceGrids; i++) {
			spaceGrids[i] = base.getSpaceGrid(i);
			center[i] = base.getCenter(i);
		}

		return new SpaceTimeDiscretization(
				spaceGrids,
				refinedTimeDiscretization,
				base.getTheta(),
				center
		);
	}

	private FiniteDifferenceEquityModel getEffectiveModelForValuation(final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization effectiveDiscretization = getValuationSpaceTimeDiscretization(model);

		if(effectiveDiscretization == model.getSpaceTimeDiscretization()) {
			return model;
		}

		return model.getCloneWithModifiedSpaceTimeDiscretization(effectiveDiscretization);
	}

	private int getTotalNumberOfSpacePoints(final SpaceTimeDiscretization discretization) {
		final int dims = discretization.getNumberOfSpaceGrids();

		if(dims == 1) {
			return discretization.getSpaceGrid(0).getGrid().length;
		}
		else if(dims == 2) {
			return discretization.getSpaceGrid(0).getGrid().length
					* discretization.getSpaceGrid(1).getGrid().length;
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D grids are supported.");
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