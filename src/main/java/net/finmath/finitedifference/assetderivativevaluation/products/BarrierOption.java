package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.interpolation.RationalFunctionInterpolation;
import net.finmath.interpolation.RationalFunctionInterpolation.ExtrapolationMethod;
import net.finmath.interpolation.RationalFunctionInterpolation.InterpolationMethod;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;

/**
 * Implements valuation of a standard single-barrier option on one asset.
 *
 * <p>
 * The class currently supports finite-difference valuation under
 * {@link FDMBlackScholesModel}. The barrier type is specified through
 * {@link BarrierType}.
 * </p>
 *
 * <p>
 * For out-options, the price is obtained directly from the finite-difference
 * solver. For in-options, the standard parity relation
 * </p>
 * <p>
 * in = vanilla - out
 * </p>
 * <p>
 * is used in the European case.
 * </p>
 *
 * <p>
 * The current implementation assumes that the barrier coincides with the
 * lower or upper boundary of the spatial grid:
 * </p>
 * <ul>
 *   <li>down-barrier: lower boundary of the grid,</li>
 *   <li>up-barrier: upper boundary of the grid.</li>
 * </ul>
 *
 * <p>
 * The terminal payoff logic follows the standard barrier-option cases.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOption implements FiniteDifferenceProduct {

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final double barrierValue;
	private final double rebate;
	private final CallOrPut callOrPutSign;
	private final BarrierType barrierType;
	private final ExerciseType exercise;

	/**
	 * Creates a barrier option.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity Maturity of the option.
	 * @param strike Strike of the option.
	 * @param barrierValue Barrier level.
	 * @param rebate Rebate paid at knock-out.
	 * @param callOrPutSign Sign of the option payoff: {@code +1.0} for call,
	 *        {@code -1.0} for put.
	 * @param barrierType Barrier type.
	 */
	public BarrierOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final double barrierValue,
			final double rebate,
			final double callOrPutSign,
			final BarrierType barrierType) {
		super();
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.barrierValue = barrierValue;
		this.rebate = rebate;
		this.barrierType = barrierType;

		if(callOrPutSign == 1.0) {
			this.callOrPutSign = CallOrPut.CALL;
		}
		else if(callOrPutSign == -1.0) {
			this.callOrPutSign = CallOrPut.PUT;
		}
		else {
			throw new IllegalArgumentException("Unknown option type.");
		}

		this.exercise = ExerciseType.EUROPEAN;
	}

	/**
	 * Creates a barrier option.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity Maturity of the option.
	 * @param strike Strike of the option.
	 * @param barrierValue Barrier level.
	 * @param rebate Rebate paid at knock-out.
	 * @param callOrPutSign Call/put flag.
	 * @param barrierType Barrier type.
	 */
	public BarrierOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final double barrierValue,
			final double rebate,
			final CallOrPut callOrPutSign,
			final BarrierType barrierType) {
		super();
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.barrierValue = barrierValue;
		this.rebate = rebate;
		this.callOrPutSign = callOrPutSign;
		this.barrierType = barrierType;
		this.exercise = ExerciseType.EUROPEAN;
	}

	/**
	 * Creates a barrier option on the default single underlying.
	 *
	 * @param maturity Maturity of the option.
	 * @param strike Strike of the option.
	 * @param barrierValue Barrier level.
	 * @param rebate Rebate paid at knock-out.
	 * @param callOrPutSign Sign of the option payoff: {@code +1.0} for call,
	 *        {@code -1.0} for put.
	 * @param barrierType Barrier type.
	 */
	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrierValue,
			final double rebate,
			final double callOrPutSign,
			final BarrierType barrierType) {
		this(null, maturity, strike, barrierValue, rebate, callOrPutSign, barrierType);
	}

	/**
	 * Creates a barrier option on the default single underlying.
	 *
	 * @param maturity Maturity of the option.
	 * @param strike Strike of the option.
	 * @param barrierValue Barrier level.
	 * @param rebate Rebate paid at knock-out.
	 * @param callOrPutSign Call/put flag.
	 * @param barrierType Barrier type.
	 */
	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrierValue,
			final CallOrPut callOrPutSign,
			final BarrierType barrierType) {
		this(null, maturity, strike, barrierValue, 0.0, callOrPutSign, barrierType);
	}

	/**
	 * Creates a zero-rebate barrier option.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity Maturity of the option.
	 * @param strike Strike of the option.
	 * @param barrierValue Barrier level.
	 * @param callOrPutSign Sign of the option payoff: {@code +1.0} for call,
	 *        {@code -1.0} for put.
	 * @param barrierType Barrier type.
	 */
	public BarrierOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final double barrierValue,
			final CallOrPut callOrPutSign,
			final BarrierType barrierType) {
		this(underlyingName, maturity, strike, barrierValue, 0.0, callOrPutSign, barrierType);
	}

	/**
	 * Creates a zero-rebate barrier option on the default single underlying.
	 *
	 * @param maturity Maturity of the option.
	 * @param strike Strike of the option.
	 * @param barrierValue Barrier level.
	 * @param callOrPutSign Sign of the option payoff: {@code +1.0} for call,
	 *        {@code -1.0} for put.
	 * @param barrierType Barrier type.
	 */
	public BarrierOption(
			final double maturity,
			final double strike,
			final double barrierValue,
			final double callOrPutSign,
			final BarrierType barrierType) {
		this(null, maturity, strike, barrierValue, 0.0, callOrPutSign, barrierType);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final double[][] values = getValues(model);
		final double tau = maturity - evaluationTime;
		final int timeIndex =
				model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {

		if(!(model instanceof FDMBlackScholesModel)) {
			throw new IllegalArgumentException(
					"BarrierOption currently supports only FDMBlackScholesModel.");
		}

		if(exercise != ExerciseType.EUROPEAN) {
			throw new IllegalArgumentException(
					"BarrierOption currently supports only European exercise.");
		}

		validateBarrierAgainstGrid(model);

		if(isIdenticallyZero()) {
			return getZeroValues(model);
		}

		if(isIdenticallyVanilla()) {
			return getVanillaValues(model);
		}

		if(isOutOption()) {
			return getOutValues(model);
		}
		else {
			return getInValuesByParity((FDMBlackScholesModel) model);
		}
	}

	private double[][] getZeroValues(final FiniteDifferenceEquityModel model) {
		final int numberOfSpacePoints = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid().length;
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

	private double[][] getVanillaValues(final FiniteDifferenceEquityModel model) {
		final EuropeanOption vanillaOption =
				new EuropeanOption(underlyingName, maturity, strike, callOrPutSign);
		return vanillaOption.getValues(model);
	}

	private double[][] getOutValues(final FiniteDifferenceEquityModel model) {
		final FDMSolver solver = new FDMThetaMethod1D(
				model,
				this,
				model.getSpaceTimeDiscretization(),
				exercise);

		return solver.getValues(maturity, this::payoffAtMaturityForOutOption);
	}

	private double[][] getInValuesByParity(final FiniteDifferenceEquityModel barrierModel) {
		final EuropeanOption vanillaOption =
				new EuropeanOption(underlyingName, maturity, strike, callOrPutSign);

		final BarrierType correspondingOutType;
		if(barrierType == BarrierType.DOWN_IN) {
			correspondingOutType = BarrierType.DOWN_OUT;
		}
		else if(barrierType == BarrierType.UP_IN) {
			correspondingOutType = BarrierType.UP_OUT;
		}
		else {
			throw new IllegalArgumentException("Parity for in-options called with non in-type barrier.");
		}

		final BarrierOption outOption =
				new BarrierOption(
						underlyingName,
						maturity,
						strike,
						barrierValue,
						rebate,
						callOrPutSign,
						correspondingOutType);

		final double[][] outValues = outOption.getValues(barrierModel);

		final FiniteDifferenceEquityModel vanillaModel = createAuxiliaryVanillaModel(barrierModel);
		final double[][] vanillaValues = vanillaOption.getValues(vanillaModel);

		final double[] barrierGrid = barrierModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] vanillaGrid = vanillaModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		final int numberOfColumns = outValues[0].length;
		final double[][] inValues = new double[outValues.length][numberOfColumns];

		for(int timeIndex = 0; timeIndex < numberOfColumns; timeIndex++) {
			for(int i = 0; i < barrierGrid.length; i++) {
				final double stock = barrierGrid[i];
				RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(vanillaGrid, getColumn(vanillaValues, timeIndex),
						InterpolationMethod.LINEAR, ExtrapolationMethod.CONSTANT);
				final double vanillaValue = interpolator.getValue(stock);//interpolateLinear(vanillaGrid, getColumn(vanillaValues, timeIndex), stock);
				inValues[i][timeIndex] = vanillaValue - outValues[i][timeIndex];
			}
		}

		return inValues;
	}

	private FiniteDifferenceEquityModel createAuxiliaryVanillaModel(
			final FiniteDifferenceEquityModel barrierModel) {

		final SpaceTimeDiscretization barrierDiscretization = barrierModel.getSpaceTimeDiscretization();
		final TimeDiscretization timeDiscretization = barrierDiscretization.getTimeDiscretization();
		final double thetaValue = barrierDiscretization.getTheta();

		final double[] barrierGrid = barrierDiscretization.getSpaceGrid(0).getGrid();

		if(barrierGrid.length < 2) {
			throw new IllegalArgumentException("Barrier grid must contain at least two points.");
		}

		/*
		 * Reuse the same mesh size as the barrier grid in the first space dimension.
		 */
		final double deltaS = barrierGrid[1] - barrierGrid[0];

		/*
		 * Build a wider auxiliary grid for the vanilla problem.
		 *
		 * Since we no longer want to hard-code a volatility, choose a simple symmetric
		 * enlargement around the initial value, based on the current grid width.
		 * This avoids introducing model-specific assumptions here.
		 */
		final double initialValue = barrierModel.getInitialValue()[0];
		final double currentMin = barrierGrid[0];
		final double currentMax = barrierGrid[barrierGrid.length - 1];
		final double currentHalfWidth = Math.max(initialValue - currentMin, currentMax - initialValue);

		/*
		 * Enlarge the domain symmetrically. A factor of 2 is a reasonable default and
		 * avoids embedding model-specific volatility logic into the product class.
		 */
		final double targetMin = Math.max(1E-8, initialValue - 2.0 * currentHalfWidth);
		final double targetMax = initialValue + 2.0 * currentHalfWidth;

		final double sMin = Math.floor(targetMin / deltaS) * deltaS;
		final double sMax = Math.ceil(targetMax / deltaS) * deltaS;
		final int numberOfSteps = (int)Math.round((sMax - sMin) / deltaS);

		final Grid vanillaGrid = new UniformGrid(numberOfSteps, sMin, sMax);

		final SpaceTimeDiscretization vanillaDiscretization = new SpaceTimeDiscretization(
				vanillaGrid,
				timeDiscretization,
				thetaValue,
				new double[] { initialValue }
		);

		return barrierModel.getCloneWithModifiedSpaceTimeDiscretization(vanillaDiscretization);
	}

	private static double[] getColumn(final double[][] matrix, final int columnIndex) {
		final double[] column = new double[matrix.length];
		for(int i = 0; i < matrix.length; i++) {
			column[i] = matrix[i][columnIndex];
		}
		return column;
	}

	private boolean isOutOption() {
		return barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.UP_OUT;
	}

	private boolean isIdenticallyZero() {
		return (barrierType == BarrierType.UP_OUT
				&& callOrPutSign == CallOrPut.CALL
				&& barrierValue <= strike)
			|| (barrierType == BarrierType.DOWN_OUT
				&& callOrPutSign == CallOrPut.PUT
				&& barrierValue >= strike)
			|| (barrierType == BarrierType.DOWN_IN
				&& callOrPutSign == CallOrPut.CALL
				&& barrierValue >= strike)
			|| (barrierType == BarrierType.UP_IN
				&& callOrPutSign == CallOrPut.PUT
				&& barrierValue <= strike);
	}

	private boolean isIdenticallyVanilla() {
		return (barrierType == BarrierType.UP_IN
				&& callOrPutSign == CallOrPut.CALL
				&& barrierValue <= strike)
			|| (barrierType == BarrierType.DOWN_IN
				&& callOrPutSign == CallOrPut.PUT
				&& barrierValue >= strike);
	}

	private double payoffAtMaturityForOutOption(final double assetValue) {

		if(callOrPutSign == CallOrPut.CALL) {
			if(barrierType == BarrierType.DOWN_OUT) {
				if(barrierValue <= strike) {
					return Math.max(assetValue - strike, 0.0);
				}
				else {
					return assetValue > barrierValue ? Math.max(assetValue - strike, 0.0) : rebate;
				}
			}
			else if(barrierType == BarrierType.UP_OUT) {
				if(barrierValue <= strike) {
					return 0.0;
				}
				else {
					return assetValue < barrierValue ? Math.max(assetValue - strike, 0.0) : rebate;
				}
			}
		}
		else {
			if(barrierType == BarrierType.DOWN_OUT) {
				if(barrierValue >= strike) {
					return 0.0;
				}
				else {
					return assetValue > barrierValue ? Math.max(strike - assetValue, 0.0) : rebate;
				}
			}
			else if(barrierType == BarrierType.UP_OUT) {
				if(barrierValue >= strike) {
					return Math.max(strike - assetValue, 0.0);
				}
				else {
					return assetValue < barrierValue ? Math.max(strike - assetValue, 0.0) : rebate;
				}
			}
		}

		throw new IllegalArgumentException("Unsupported barrier / option type combination.");
	}

	private void validateBarrierAgainstGrid(final FiniteDifferenceEquityModel model) {
		final double[] grid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double lowerBoundary = grid[0];
		final double upperBoundary = grid[grid.length - 1];
		final double tolerance = 1E-12;

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			if(Math.abs(barrierValue - lowerBoundary) > tolerance) {
				throw new IllegalArgumentException(
						"For down-barrier options the barrier must coincide with the lower grid boundary.");
			}
		}
		else if(barrierType == BarrierType.UP_IN || barrierType == BarrierType.UP_OUT) {
			if(Math.abs(barrierValue - upperBoundary) > tolerance) {
				throw new IllegalArgumentException(
						"For up-barrier options the barrier must coincide with the upper grid boundary.");
			}
		}
	}

	/**
	 * Returns the name of the underlying.
	 *
	 * @return Underlying name.
	 */
	public String getUnderlyingName() {
		return underlyingName;
	}

	/**
	 * Returns the maturity.
	 *
	 * @return Maturity.
	 */
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Returns the strike.
	 *
	 * @return Strike.
	 */
	public double getStrike() {
		return strike;
	}

	/**
	 * Returns the barrier level.
	 *
	 * @return Barrier level.
	 */
	public double getBarrierValue() {
		return barrierValue;
	}

	/**
	 * Returns the rebate.
	 *
	 * @return Rebate.
	 */
	public double getRebate() {
		return rebate;
	}

	/**
	 * Returns the call/put flag.
	 *
	 * @return Call/put flag.
	 */
	public CallOrPut getCallOrPut() {
		return callOrPutSign;
	}

	/**
	 * Returns the barrier type.
	 *
	 * @return Barrier type.
	 */
	public BarrierType getBarrierType() {
		return barrierType;
	}

	/**
	 * Returns the exercise type.
	 *
	 * @return Exercise type.
	 */
	public ExerciseType getExercise() {
		return exercise;
	}
}