package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.finitedifference.solvers.FDMThetaMethod1DTwoState;
import net.finmath.finitedifference.solvers.FDMThetaMethod2D;
import net.finmath.interpolation.RationalFunctionInterpolation;
import net.finmath.interpolation.RationalFunctionInterpolation.ExtrapolationMethod;
import net.finmath.interpolation.RationalFunctionInterpolation.InterpolationMethod;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;

/**
 * Finite-difference valuation of a standard single-barrier option on one asset.
 *
 * <p>
 * The barrier acts on the first state variable of the model, which is assumed
 * to represent the underlying level.
 * </p>
 *
 * <p>
 * Current implementation policy:
 * </p>
 * <ul>
 * <li>knock-out options are priced directly by the finite-difference solver,
 * using internal state constraints,</li>
 * <li>knock-in options are priced through an internal activation policy, which
 * currently uses parity but is intentionally encapsulated so that it may later
 * be replaced by a direct solver formulation,</li>
 * <li>exercise is currently European only.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOption implements FiniteDifferenceProduct, FiniteDifferenceInternalStateConstraint {

	private enum PricingMode {
		DIRECT_OUT, ACTIVATION_POLICY_IN
	}

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final double barrierValue;
	private final double rebate;
	private final CallOrPut callOrPutSign;
	private final BarrierType barrierType;
	private final Exercise exercise;

	public BarrierOption(final String underlyingName, final double maturity, final double strike,
			final double barrierValue, final double rebate, final double callOrPutSign, final BarrierType barrierType) {
		this(underlyingName, maturity, strike, barrierValue, rebate, mapCallOrPut(callOrPutSign), barrierType);
	}

	public BarrierOption(final String underlyingName, final double maturity, final double strike,
			final double barrierValue, final double rebate, final CallOrPut callOrPutSign,
			final BarrierType barrierType) {
		super();
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.barrierValue = barrierValue;
		this.rebate = rebate;
		this.callOrPutSign = callOrPutSign;
		this.barrierType = barrierType;
		this.exercise = new EuropeanExercise(maturity);
	}

	public BarrierOption(final double maturity, final double strike, final double barrierValue, final double rebate,
			final double callOrPutSign, final BarrierType barrierType) {
		this(null, maturity, strike, barrierValue, rebate, callOrPutSign, barrierType);
	}

	public BarrierOption(final double maturity, final double strike, final double barrierValue, final double rebate,
			final CallOrPut callOrPutSign, final BarrierType barrierType) {
		this(null, maturity, strike, barrierValue, rebate, callOrPutSign, barrierType);
	}

	public BarrierOption(final String underlyingName, final double maturity, final double strike,
			final double barrierValue, final CallOrPut callOrPutSign, final BarrierType barrierType) {
		this(underlyingName, maturity, strike, barrierValue, 0.0, callOrPutSign, barrierType);
	}

	public BarrierOption(final double maturity, final double strike, final double barrierValue,
			final double callOrPutSign, final BarrierType barrierType) {
		this(null, maturity, strike, barrierValue, 0.0, callOrPutSign, barrierType);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final double[][] values = getValues(model);
		final double tau = maturity - evaluationTime;
		final int timeIndex = model.getSpaceTimeDiscretization().getTimeDiscretization()
				.getTimeIndexNearestLessOrEqual(tau);

		final double[] column = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {

		validateProductConfiguration(model);

		if (isDegenerateZeroCase()) {
			return buildZeroValueSurface(model);
		}

		if (isDegenerateVanillaCase()) {
			return createVanillaOption().getValues(model);
		}

		switch (getPricingMode()) {
		case DIRECT_OUT:
			return priceOutOptionDirectly(model);
		case ACTIVATION_POLICY_IN:
			return priceInOptionThroughActivationPolicy(model);
		default:
			throw new IllegalStateException("Unsupported pricing mode.");
		}
	}

	private PricingMode getPricingMode() {
		return isOutOption() ? PricingMode.DIRECT_OUT : PricingMode.ACTIVATION_POLICY_IN;
	}

	private void validateProductConfiguration(final FiniteDifferenceEquityModel model) {
		if (!exercise.isEuropean()) {
			throw new IllegalArgumentException("BarrierOption currently supports only European exercise.");
		}

		validateBarrierInsideGrid(model);
	}

	private double[][] buildZeroValueSurface(final FiniteDifferenceEquityModel model) {
		final int numberOfSpacePoints = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid().length;
		final int numberOfTimePoints = model.getSpaceTimeDiscretization().getTimeDiscretization().getNumberOfTimeSteps()
				+ 1;

		final double[][] zeroValues = new double[numberOfSpacePoints][numberOfTimePoints];
		for (int i = 0; i < numberOfSpacePoints; i++) {
			for (int j = 0; j < numberOfTimePoints; j++) {
				zeroValues[i][j] = 0.0;
			}
		}
		return zeroValues;
	}

	private double[][] priceOutOptionDirectly(final FiniteDifferenceEquityModel model) {
		return createSolver(model).getValues(maturity, this::getTerminalPayoffForDirectOutPricing);
	}

	/**
	 * Prices a knock-in option through the current activation policy.
	 *
	 * <p>
	 * Today this activation policy is implemented by the identity
	 * </p>
	 * <p>
	 * knock-in = vanilla - corresponding knock-out.
	 * </p>
	 * <p>
	 * This is intentionally encapsulated here so that a future direct coupled-PDE
	 * treatment of knock-in activation can replace this implementation without
	 * changing the public semantics of the product class.
	 * </p>
	 */
	/*private double[][] priceInOptionThroughActivationPolicy(final FiniteDifferenceEquityModel model) {
		return priceInOptionByParity(model);
	}*/
	private double[][] priceInOptionThroughActivationPolicy(final FiniteDifferenceEquityModel model) {

		final int numberOfSpaceDimensions = model.getSpaceTimeDiscretization().getNumberOfSpaceGrids();

		if(numberOfSpaceDimensions != 1) {
			throw new IllegalArgumentException(
					"Direct coupled two-state knock-in pricing is currently implemented only for 1D finite-difference models.");
		}

		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalStateException(
					"priceInOptionThroughActivationPolicy was called for a non knock-in barrier type.");
		}

		final FDMSolver solver = new FDMThetaMethod1DTwoState(
				model,
				this,
				model.getSpaceTimeDiscretization(),
				exercise
		);

		return solver.getValues(
				maturity,
				assetValue -> {
					if(callOrPutSign == CallOrPut.CALL) {
						return Math.max(assetValue - strike, 0.0);
					}
					else {
						return Math.max(strike - assetValue, 0.0);
					}
				}
		);
	}


	private double[][] priceInOptionByParity(final FiniteDifferenceEquityModel barrierModel) {

		final EuropeanOption vanillaOption = createVanillaOption();
		final BarrierOption correspondingOutOption = createCorrespondingOutOption();

		final double[][] outValues = correspondingOutOption.getValues(barrierModel);

		final FiniteDifferenceEquityModel vanillaModel = createAuxiliaryVanillaModel(barrierModel);
		final double[][] vanillaValues = vanillaOption.getValues(vanillaModel);

		final double[] barrierGrid = barrierModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] vanillaGrid = vanillaModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		final int numberOfColumns = outValues[0].length;
		final double[][] inValues = new double[outValues.length][numberOfColumns];

		for (int timeIndex = 0; timeIndex < numberOfColumns; timeIndex++) {
			final RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(vanillaGrid,
					getColumn(vanillaValues, timeIndex), InterpolationMethod.LINEAR, ExtrapolationMethod.CONSTANT);

			for (int i = 0; i < barrierGrid.length; i++) {
				final double stock = barrierGrid[i];
				final double vanillaValue = interpolator.getValue(stock);
				inValues[i][timeIndex] = vanillaValue - outValues[i][timeIndex];
			}
		}

		return inValues;
	}

	private FDMSolver createSolver(final FiniteDifferenceEquityModel model) {
		final int numberOfSpaceDimensions = model.getSpaceTimeDiscretization().getNumberOfSpaceGrids();

		if (numberOfSpaceDimensions == 1) {
			return new FDMThetaMethod1D(model, this, model.getSpaceTimeDiscretization(), exercise);
		} else if (numberOfSpaceDimensions == 2) {
			return new FDMThetaMethod2D(model, this, model.getSpaceTimeDiscretization(), exercise);
		} else {
			throw new IllegalArgumentException(
					"BarrierOption currently supports only 1D or 2D finite-difference models.");
		}
	}

	private EuropeanOption createVanillaOption() {
		return new EuropeanOption(underlyingName, maturity, strike, callOrPutSign);
	}

	private BarrierOption createCorrespondingOutOption() {
		return new BarrierOption(underlyingName, maturity, strike, barrierValue, rebate, callOrPutSign,
				getCorrespondingOutBarrierType());
	}

	private BarrierType getCorrespondingOutBarrierType() {
		if (barrierType == BarrierType.DOWN_IN) {
			return BarrierType.DOWN_OUT;
		}
		if (barrierType == BarrierType.UP_IN) {
			return BarrierType.UP_OUT;
		}
		throw new IllegalArgumentException("No corresponding out barrier type for " + barrierType);
	}

	private FiniteDifferenceEquityModel createAuxiliaryVanillaModel(final FiniteDifferenceEquityModel barrierModel) {

		final SpaceTimeDiscretization barrierDiscretization = barrierModel.getSpaceTimeDiscretization();
		final TimeDiscretization timeDiscretization = barrierDiscretization.getTimeDiscretization();
		final double thetaValue = barrierDiscretization.getTheta();

		final double[] barrierGrid = barrierDiscretization.getSpaceGrid(0).getGrid();

		if (barrierGrid.length < 2) {
			throw new IllegalArgumentException("Barrier grid must contain at least two points.");
		}

		final double deltaS = barrierGrid[1] - barrierGrid[0];

		final double initialValue = barrierModel.getInitialValue()[0];
		final double currentMin = barrierGrid[0];
		final double currentMax = barrierGrid[barrierGrid.length - 1];
		final double currentHalfWidth = Math.max(initialValue - currentMin, currentMax - initialValue);

		final double targetMin = Math.max(1E-8, initialValue - 2.0 * currentHalfWidth);
		final double targetMax = initialValue + 2.0 * currentHalfWidth;

		final double sMin = Math.floor(targetMin / deltaS) * deltaS;
		final double sMax = Math.ceil(targetMax / deltaS) * deltaS;
		final int numberOfSteps = (int) Math.round((sMax - sMin) / deltaS);

		final Grid vanillaGrid = new UniformGrid(numberOfSteps, sMin, sMax);

		final SpaceTimeDiscretization vanillaDiscretization = new SpaceTimeDiscretization(vanillaGrid,
				timeDiscretization, thetaValue, new double[] { initialValue });

		return barrierModel.getCloneWithModifiedSpaceTimeDiscretization(vanillaDiscretization);
	}

	private static double[] getColumn(final double[][] matrix, final int columnIndex) {
		final double[] column = new double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			column[i] = matrix[i][columnIndex];
		}
		return column;
	}

	private boolean isOutOption() {
		return barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.UP_OUT;
	}

	private boolean isDegenerateZeroCase() {
		return (barrierType == BarrierType.UP_OUT && callOrPutSign == CallOrPut.CALL && barrierValue <= strike)
				|| (barrierType == BarrierType.DOWN_OUT && callOrPutSign == CallOrPut.PUT && barrierValue >= strike)
				|| (barrierType == BarrierType.DOWN_IN && callOrPutSign == CallOrPut.CALL && barrierValue >= strike)
				|| (barrierType == BarrierType.UP_IN && callOrPutSign == CallOrPut.PUT && barrierValue <= strike);
	}

	private boolean isDegenerateVanillaCase() {
		return (barrierType == BarrierType.UP_IN && callOrPutSign == CallOrPut.CALL && barrierValue <= strike)
				|| (barrierType == BarrierType.DOWN_IN && callOrPutSign == CallOrPut.PUT && barrierValue >= strike);
	}

	private double getTerminalPayoffForDirectOutPricing(final double assetValue) {

		if (callOrPutSign == CallOrPut.CALL) {
			if (barrierType == BarrierType.DOWN_OUT) {
				if (barrierValue <= strike) {
					return Math.max(assetValue - strike, 0.0);
				}
				return assetValue > barrierValue ? Math.max(assetValue - strike, 0.0) : rebate;
			} else if (barrierType == BarrierType.UP_OUT) {
				if (barrierValue <= strike) {
					return 0.0;
				}
				return assetValue < barrierValue ? Math.max(assetValue - strike, 0.0) : rebate;
			}
		} else {
			if (barrierType == BarrierType.DOWN_OUT) {
				if (barrierValue >= strike) {
					return 0.0;
				}
				return assetValue > barrierValue ? Math.max(strike - assetValue, 0.0) : rebate;
			} else if (barrierType == BarrierType.UP_OUT) {
				if (barrierValue >= strike) {
					return Math.max(strike - assetValue, 0.0);
				}
				return assetValue < barrierValue ? Math.max(strike - assetValue, 0.0) : rebate;
			}
		}

		throw new IllegalArgumentException("Direct terminal payoff requested for non out-option type.");
	}

	private void validateBarrierInsideGrid(final FiniteDifferenceEquityModel model) {
		final double[] grid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double lowerBoundary = grid[0];
		final double upperBoundary = grid[grid.length - 1];

		if (barrierValue < lowerBoundary || barrierValue > upperBoundary) {
			throw new IllegalArgumentException("The barrier must lie inside the first state-variable grid domain.");
		}
	}

	@Override
	public boolean isConstraintActive(final double time, final double... stateVariables) {
		if (!isOutOption()) {
			return false;
		}

		final double underlyingLevel = stateVariables[0];

		switch (barrierType) {
		case DOWN_OUT:
			return underlyingLevel <= barrierValue;
		case UP_OUT:
			return underlyingLevel >= barrierValue;
		default:
			return false;
		}
	}

	@Override
	public double getConstrainedValue(final double time, final double... stateVariables) {
		if (!isOutOption()) {
			throw new IllegalStateException("Internal constrained value requested for a non out-option.");
		}

		return rebate;
	}

	private static CallOrPut mapCallOrPut(final double callOrPutSign) {
		if (callOrPutSign == 1.0) {
			return CallOrPut.CALL;
		}
		if (callOrPutSign == -1.0) {
			return CallOrPut.PUT;
		}
		throw new IllegalArgumentException("Unknown option type.");
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

	public double getBarrierValue() {
		return barrierValue;
	}

	public double getRebate() {
		return rebate;
	}

	public CallOrPut getCallOrPut() {
		return callOrPutSign;
	}

	public BarrierType getBarrierType() {
		return barrierType;
	}

	public Exercise getExercise() {
		return exercise;
	}
}