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
 * The contract is defined by a maturity <i>T</i>, a cash amount <i>C</i>, and two barriers
 * <i>L &lt; U</i>. Let <i>S(t)</i> denote the underlying and let
 * </p>
 *
 * <p>
 * <i>&tau;<sub>L</sub> = inf { t in [0,T] : S(t) &le; L }</i>,
 * </p>
 *
 * <p>
 * <i>&tau;<sub>U</sub> = inf { t in [0,T] : S(t) &ge; U }</i>,
 * </p>
 *
 * <p>
 * with the alive band given by
 * </p>
 *
 * <p>
 * <i>L &lt; S(t) &lt; U</i>.
 * </p>
 *
 * <p>
 * Supported barrier styles:
 * </p>
 * <ul>
 *   <li>{@link DoubleBarrierType#KNOCK_OUT}: pays <i>C</i> if neither barrier is hit before maturity,</li>
 *   <li>{@link DoubleBarrierType#KNOCK_IN}: pays <i>C</i> if either barrier is hit before or at maturity,</li>
 *   <li>{@link DoubleBarrierType#KIKO}: pays <i>C</i> if the lower barrier is hit first while the upper barrier acts as knock-out,</li>
 *   <li>{@link DoubleBarrierType#KOKI}: pays <i>C</i> if the upper barrier is hit first while the lower barrier acts as knock-out.</li>
 * </ul>
 *
 * <p>
 * In indicator notation, the terminal payoff can be written as:
 * </p>
 * <ul>
 *   <li>KNOCK_OUT:
 *       <i>C 1<sub>{&tau;<sub>L</sub> &gt; T, &tau;<sub>U</sub> &gt; T}</sub></i>,</li>
 *   <li>KNOCK_IN:
 *       <i>C 1<sub>{min(&tau;<sub>L</sub>,&tau;<sub>U</sub>) &le; T}</sub></i>,</li>
 *   <li>KIKO:
 *       <i>C 1<sub>{&tau;<sub>L</sub> &le; T, &tau;<sub>L</sub> &lt; &tau;<sub>U</sub>}</sub></i>,</li>
 *   <li>KOKI:
 *       <i>C 1<sub>{&tau;<sub>U</sub> &le; T, &tau;<sub>U</sub> &lt; &tau;<sub>L</sub>}</sub></i>.</li>
 * </ul>
 *
 * <p>
 * Exercise semantics implemented here:
 * </p>
 * <ul>
 *   <li>European: payoff is determined at maturity,</li>
 *   <li>Bermudan and American: the holder may exercise into the current binary intrinsic value,
 *       that is, the immediate cash-or-zero value implied by the current barrier state.</li>
 * </ul>
 *
 * <p>
 * Hence the immediate exercise value is:
 * </p>
 * <ul>
 *   <li>KNOCK_OUT: <i>C</i> inside the alive band and <i>0</i> outside,</li>
 *   <li>KNOCK_IN: <i>0</i> inside the alive band and <i>C</i> outside,</li>
 *   <li>KIKO: <i>C</i> below the lower barrier and <i>0</i> elsewhere,</li>
 *   <li>KOKI: <i>C</i> above the upper barrier and <i>0</i> elsewhere.</li>
 * </ul>
 *
 * <p>
 * The implementation remains a direct one-state constrained PDE. In one dimension the terminal
 * condition is cell-averaged for improved barrier resolution. In two dimensions the same barrier
 * logic is applied on the first state variable, while the second state variable is propagated by
 * the corresponding ADI solver.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class DoubleBarrierBinaryOption implements
	FiniteDifferenceEquityProduct,
	FiniteDifferenceInternalStateConstraint {

	private final String underlyingName;
	private final double maturity;
	private final double cashPayoff;
	private final double lowerBarrier;
	private final double upperBarrier;
	private final DoubleBarrierType doubleBarrierType;
	private final Exercise exercise;

	/**
	 * Creates a double-barrier cash binary option.
	 *
	 * @param underlyingName Name of the underlying. May be {@code null}.
	 * @param maturity Option maturity.
	 * @param cashPayoff Cash payoff amount.
	 * @param lowerBarrier Lower barrier.
	 * @param upperBarrier Upper barrier.
	 * @param doubleBarrierType Double-barrier type.
	 * @param exercise Exercise specification.
	 */
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

	/**
	 * Creates a European double-barrier cash binary option.
	 *
	 * @param underlyingName Name of the underlying. May be {@code null}.
	 * @param maturity Option maturity.
	 * @param cashPayoff Cash payoff amount.
	 * @param lowerBarrier Lower barrier.
	 * @param upperBarrier Upper barrier.
	 * @param doubleBarrierType Double-barrier type.
	 */
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

	/**
	 * Creates a European double-barrier cash binary option with anonymous underlying.
	 *
	 * @param maturity Option maturity.
	 * @param cashPayoff Cash payoff amount.
	 * @param lowerBarrier Lower barrier.
	 * @param upperBarrier Upper barrier.
	 * @param doubleBarrierType Double-barrier type.
	 */
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

	/**
	 * Creates a double-barrier cash binary option with anonymous underlying.
	 *
	 * @param maturity Option maturity.
	 * @param cashPayoff Cash payoff amount.
	 * @param lowerBarrier Lower barrier.
	 * @param upperBarrier Upper barrier.
	 * @param doubleBarrierType Double-barrier type.
	 * @param exercise Exercise specification.
	 */
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

	/**
	 * Returns the values at the specified evaluation time on the model space grid.
	 *
	 * @param evaluationTime Evaluation time.
	 * @param model The finite-difference model.
	 * @return The value vector at the requested evaluation time.
	 */
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

	/**
	 * Returns the full value surface.
	 *
	 * @param model The finite-difference model.
	 * @return The value surface indexed by space point and time index.
	 */
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

		return ((AbstractADI2D) solver).getValues(
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
	 * Returns the immediate exercise payoff under the current binary state.
	 *
	 * <p>
	 * For this product, the natural exercise value is the current binary intrinsic value.
	 * This matches the terminal state classification.
	 * </p>
	 *
	 * @param assetValue Current underlying level.
	 * @return The immediate exercise payoff.
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

	/**
	 * Returns whether the internal constrained regime is active.
	 *
	 * @param time Evaluation time.
	 * @param stateVariables State variables.
	 * @return {@code true} if the underlying is outside the alive band.
	 */
	@Override
	public boolean isConstraintActive(final double time, final double... stateVariables) {
		final double underlyingLevel = stateVariables[0];
		return isBelowLowerBarrier(underlyingLevel) || isAboveUpperBarrier(underlyingLevel);
	}

	/**
	 * Returns the constrained value in the barrier region.
	 *
	 * @param time Evaluation time.
	 * @param stateVariables State variables.
	 * @return The constrained value.
	 */
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

	/**
	 * Returns the underlying name.
	 *
	 * @return The underlying name, possibly {@code null}.
	 */
	public String getUnderlyingName() {
		return underlyingName;
	}

	/**
	 * Returns the maturity.
	 *
	 * @return The maturity.
	 */
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Returns the cash payoff.
	 *
	 * @return The cash payoff.
	 */
	public double getCashPayoff() {
		return cashPayoff;
	}

	/**
	 * Returns the lower barrier.
	 *
	 * @return The lower barrier.
	 */
	public double getLowerBarrier() {
		return lowerBarrier;
	}

	/**
	 * Returns the upper barrier.
	 *
	 * @return The upper barrier.
	 */
	public double getUpperBarrier() {
		return upperBarrier;
	}

	/**
	 * Returns the double-barrier type.
	 *
	 * @return The double-barrier type.
	 */
	public DoubleBarrierType getDoubleBarrierType() {
		return doubleBarrierType;
	}

	/**
	 * Returns the exercise specification.
	 *
	 * @return The exercise specification.
	 */
	public Exercise getExercise() {
		return exercise;
	}
}