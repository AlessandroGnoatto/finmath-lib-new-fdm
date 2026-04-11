package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.boundaries.ActiveBoundaryProviderFactory;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.finitedifference.solvers.FDMThetaMethod1DTwoState;
import net.finmath.interpolation.RationalFunctionInterpolation;
import net.finmath.interpolation.RationalFunctionInterpolation.ExtrapolationMethod;
import net.finmath.interpolation.RationalFunctionInterpolation.InterpolationMethod;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
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
 *   <li>knock-out options are priced directly by the finite-difference solver,
 *       using internal state constraints on the original product grid,</li>
 *   <li>1D knock-in options are priced directly through a coupled two-state PDE
 *       on an auxiliary spatial grid where the barrier is placed on an interior node,</li>
 *   <li>2D knock-in options currently fall back to in-out parity,</li>
 *   <li>for 2D parity pricing, the vanilla surface is computed on an auxiliary grid
 *       and interpolated back to the original product grid along the first state variable,</li>
 *   <li>exercise is currently European only.</li>
 * </ul>
 *
 * <p>
 * The auxiliary interior-barrier grid used for direct knock-ins is chosen because
 * the coupled formulation requires an activated state evolving on a full domain,
 * unlike knock-out pricing where it is natural to place the barrier on the
 * outer grid boundary.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOption implements FiniteDifferenceProduct, FiniteDifferenceInternalStateConstraint {

	private enum PricingMode {
		DIRECT_OUT, ACTIVATION_POLICY_IN
	}

	/*
	 * 1D direct knock-in settings.
	 * These were the settings that produced the good Black-Scholes interior-barrier results.
	 */
	private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS_1D = 40;
	private static final int DOWN_IN_PUT_EXTRA_STEPS_1D = 160;
	private static final int UP_IN_CALL_EXTRA_STEPS_1D = 160;

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
		for(int i = 0; i < values.length; i++) {
			column[i] = values[i][timeIndex];
		}
		return column;
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {

		validateProductConfiguration(model);

		if(isDegenerateZeroCase()) {
			return buildZeroValueSurface(model);
		}

		if(isDegenerateVanillaCase()) {
			return createVanillaOption().getValues(model);
		}

		switch(getPricingMode()) {
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
		if(!exercise.isEuropean()) {
			throw new IllegalArgumentException("BarrierOption currently supports only European exercise.");
		}

		validateBarrierInsideGrid(model);
	}

	private double[][] buildZeroValueSurface(final FiniteDifferenceEquityModel model) {
		final int numberOfSpacePoints = getTotalNumberOfSpacePoints(model.getSpaceTimeDiscretization());
		final int numberOfTimePoints = model.getSpaceTimeDiscretization().getTimeDiscretization().getNumberOfTimeSteps() + 1;

		final double[][] zeroValues = new double[numberOfSpacePoints][numberOfTimePoints];
		for(int i = 0; i < numberOfSpacePoints; i++) {
			for(int j = 0; j < numberOfTimePoints; j++) {
				zeroValues[i][j] = 0.0;
			}
		}
		return zeroValues;
	}

	private double[][] priceOutOptionDirectly(final FiniteDifferenceEquityModel model) {
		return createSolver(model).getValues(maturity, this::getTerminalPayoffForDirectOutPricing);
	}

	private double[][] priceInOptionThroughActivationPolicy(final FiniteDifferenceEquityModel model) {

		final int numberOfSpaceDimensions = model.getSpaceTimeDiscretization().getNumberOfSpaceGrids();

		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalStateException(
					"priceInOptionThroughActivationPolicy was called for a non knock-in barrier type.");
		}

		if(numberOfSpaceDimensions == 1) {
			final FiniteDifferenceEquityModel knockInModel = createAuxiliaryKnockInModel1D(model);

			final FDMSolver solver = new FDMThetaMethod1DTwoState(
					knockInModel,
					this,
					knockInModel.getSpaceTimeDiscretization(),
					exercise,
					ActiveBoundaryProviderFactory.createProvider(
							knockInModel,
							strike,
							maturity,
							callOrPutSign
					)
			);

			final double[][] knockInValuesOnAuxiliaryGrid = solver.getValues(
					maturity,
					assetValue -> callOrPutSign == CallOrPut.CALL
							? Math.max(assetValue - strike, 0.0)
							: Math.max(strike - assetValue, 0.0)
			);

			return interpolateSurfaceToOriginalGrid1D(
					knockInValuesOnAuxiliaryGrid,
					knockInModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
					model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid()
			);
		}
		else {
			return priceInOptionByParity(model);
		}
	}

	private double[][] priceInOptionByParity(final FiniteDifferenceEquityModel barrierModel) {

		final EuropeanOption vanillaOption = createVanillaOption();
		final BarrierOption correspondingOutOption = createCorrespondingOutOption();

		final double[][] outValues = correspondingOutOption.getValues(barrierModel);

		final FiniteDifferenceEquityModel vanillaModel = createAuxiliaryVanillaModel(barrierModel);
		final double[][] vanillaValues = vanillaOption.getValues(vanillaModel);

		final SpaceTimeDiscretization barrierDiscretization = barrierModel.getSpaceTimeDiscretization();
		final SpaceTimeDiscretization vanillaDiscretization = vanillaModel.getSpaceTimeDiscretization();

		final int dims = barrierDiscretization.getNumberOfSpaceGrids();

		if(dims == 1) {
			final double[] barrierGrid = barrierDiscretization.getSpaceGrid(0).getGrid();
			final double[] vanillaGrid = vanillaDiscretization.getSpaceGrid(0).getGrid();

			final int numberOfColumns = outValues[0].length;
			final double[][] inValues = new double[outValues.length][numberOfColumns];

			for(int timeIndex = 0; timeIndex < numberOfColumns; timeIndex++) {
				final RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(
						vanillaGrid,
						getColumn(vanillaValues, timeIndex),
						InterpolationMethod.LINEAR,
						ExtrapolationMethod.CONSTANT
				);

				for(int i = 0; i < barrierGrid.length; i++) {
					final double stock = barrierGrid[i];
					final double vanillaValue = interpolator.getValue(stock);
					inValues[i][timeIndex] = vanillaValue - outValues[i][timeIndex];
				}
			}

			return inValues;
		}
		else if(dims == 2) {
			final double[][] vanillaOnBarrierGrid = interpolateSurfaceToOriginalGrid2DAlongFirstState(
					vanillaValues,
					vanillaDiscretization,
					barrierDiscretization
			);

			final int numberOfRows = outValues.length;
			final int numberOfColumns = outValues[0].length;
			final double[][] inValues = new double[numberOfRows][numberOfColumns];

			for(int i = 0; i < numberOfRows; i++) {
				for(int j = 0; j < numberOfColumns; j++) {
					inValues[i][j] = vanillaOnBarrierGrid[i][j] - outValues[i][j];
				}
			}

			return inValues;
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D grids are supported.");
		}
	}

	private double[][] interpolateSurfaceToOriginalGrid1D(
			final double[][] valuesOnAuxiliaryGrid,
			final double[] auxiliaryGrid,
			final double[] originalGrid) {

		final int numberOfColumns = valuesOnAuxiliaryGrid[0].length;
		final double[][] interpolatedValues = new double[originalGrid.length][numberOfColumns];

		for(int timeIndex = 0; timeIndex < numberOfColumns; timeIndex++) {
			final RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(
					auxiliaryGrid,
					getColumn(valuesOnAuxiliaryGrid, timeIndex),
					InterpolationMethod.LINEAR,
					ExtrapolationMethod.CONSTANT
			);

			for(int i = 0; i < originalGrid.length; i++) {
				interpolatedValues[i][timeIndex] = interpolator.getValue(originalGrid[i]);
			}
		}

		return interpolatedValues;
	}

	private double[][] interpolateSurfaceToOriginalGrid2DAlongFirstState(
			final double[][] valuesOnAuxiliaryGrid,
			final SpaceTimeDiscretization auxiliaryDiscretization,
			final SpaceTimeDiscretization originalDiscretization) {

		final double[] auxiliaryX0 = auxiliaryDiscretization.getSpaceGrid(0).getGrid();
		final double[] auxiliaryX1 = auxiliaryDiscretization.getSpaceGrid(1).getGrid();

		final double[] originalX0 = originalDiscretization.getSpaceGrid(0).getGrid();
		final double[] originalX1 = originalDiscretization.getSpaceGrid(1).getGrid();

		if(auxiliaryX1.length != originalX1.length) {
			throw new IllegalArgumentException(
					"2D knock-in interpolation currently requires the second state-variable grid to remain unchanged.");
		}

		for(int j = 0; j < originalX1.length; j++) {
			if(Math.abs(auxiliaryX1[j] - originalX1[j]) > 1E-12) {
				throw new IllegalArgumentException(
						"2D knock-in interpolation currently requires the second state-variable grid to remain unchanged.");
			}
		}

		final int auxiliaryN0 = auxiliaryX0.length;
		final int originalN0 = originalX0.length;
		final int originalN1 = originalX1.length;

		final int numberOfColumns = valuesOnAuxiliaryGrid[0].length;
		final double[][] interpolatedValues = new double[originalN0 * originalN1][numberOfColumns];

		for(int timeIndex = 0; timeIndex < numberOfColumns; timeIndex++) {
			for(int j = 0; j < originalN1; j++) {

				final double[] auxiliarySlice = new double[auxiliaryN0];
				for(int i = 0; i < auxiliaryN0; i++) {
					final int k = flatten(i, j, auxiliaryN0);
					auxiliarySlice[i] = valuesOnAuxiliaryGrid[k][timeIndex];
				}

				final RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(
						auxiliaryX0,
						auxiliarySlice,
						InterpolationMethod.LINEAR,
						ExtrapolationMethod.CONSTANT
				);

				for(int i = 0; i < originalN0; i++) {
					final int k = flatten(i, j, originalN0);
					interpolatedValues[k][timeIndex] = interpolator.getValue(originalX0[i]);
				}
			}
		}

		return interpolatedValues;
	}

	private FDMSolver createSolver(final FiniteDifferenceEquityModel model) {
		return FDMSolverFactory.createSolver(
				model,
				this,
				model.getSpaceTimeDiscretization(),
				exercise
		);
	}

	private EuropeanOption createVanillaOption() {
		return new EuropeanOption(underlyingName, maturity, strike, callOrPutSign);
	}

	private BarrierOption createCorrespondingOutOption() {
		return new BarrierOption(underlyingName, maturity, strike, barrierValue, rebate, callOrPutSign,
				getCorrespondingOutBarrierType());
	}

	private BarrierType getCorrespondingOutBarrierType() {
		if(barrierType == BarrierType.DOWN_IN) {
			return BarrierType.DOWN_OUT;
		}
		if(barrierType == BarrierType.UP_IN) {
			return BarrierType.UP_OUT;
		}
		throw new IllegalArgumentException("No corresponding out barrier type for " + barrierType);
	}

	private FiniteDifferenceEquityModel createAuxiliaryVanillaModel(final FiniteDifferenceEquityModel barrierModel) {

		final SpaceTimeDiscretization barrierDiscretization = barrierModel.getSpaceTimeDiscretization();
		final TimeDiscretization timeDiscretization = barrierDiscretization.getTimeDiscretization();
		final double thetaValue = barrierDiscretization.getTheta();

		final double[] barrierGrid = barrierDiscretization.getSpaceGrid(0).getGrid();

		if(barrierGrid.length < 2) {
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
		final int numberOfSteps = (int)Math.round((sMax - sMin) / deltaS);

		final Grid vanillaSpotGrid = new UniformGrid(numberOfSteps, sMin, sMax);

		if(barrierDiscretization.getNumberOfSpaceGrids() == 1) {
			final SpaceTimeDiscretization vanillaDiscretization = new SpaceTimeDiscretization(
					vanillaSpotGrid,
					timeDiscretization,
					thetaValue,
					new double[] { initialValue }
			);
			return barrierModel.getCloneWithModifiedSpaceTimeDiscretization(vanillaDiscretization);
		}
		else if(barrierDiscretization.getNumberOfSpaceGrids() == 2) {
			final double[] varianceGrid = barrierDiscretization.getSpaceGrid(1).getGrid();
			final Grid preservedVarianceGrid = new UniformGrid(
					varianceGrid.length - 1,
					varianceGrid[0],
					varianceGrid[varianceGrid.length - 1]
			);

			final SpaceTimeDiscretization vanillaDiscretization = new SpaceTimeDiscretization(
					new Grid[] { vanillaSpotGrid, preservedVarianceGrid },
					timeDiscretization,
					thetaValue,
					barrierModel.getInitialValue()
			);
			return barrierModel.getCloneWithModifiedSpaceTimeDiscretization(vanillaDiscretization);
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D grids are supported.");
		}
	}

	private FiniteDifferenceEquityModel createAuxiliaryKnockInModel1D(final FiniteDifferenceEquityModel originalModel) {

		final SpaceTimeDiscretization originalDiscretization = originalModel.getSpaceTimeDiscretization();
		final TimeDiscretization timeDiscretization = originalDiscretization.getTimeDiscretization();
		final double thetaValue = originalDiscretization.getTheta();

		final double[] originalGrid = originalDiscretization.getSpaceGrid(0).getGrid();
		if(originalGrid.length < 2) {
			throw new IllegalArgumentException("Barrier grid must contain at least two points.");
		}

		final double deltaS = originalGrid[1] - originalGrid[0];
		final int numberOfSteps = originalGrid.length - 1;
		final double initialValue = originalModel.getInitialValue()[0];
		final int extraStepsBeyondBarrier = getKnockInExtraStepsBeyondBarrier1D();

		final double sMin;
		final double sMax;

		if(barrierType == BarrierType.DOWN_IN) {
			sMin = barrierValue - extraStepsBeyondBarrier * deltaS;
			sMax = sMin + numberOfSteps * deltaS;
		}
		else if(barrierType == BarrierType.UP_IN) {
			sMax = barrierValue + extraStepsBeyondBarrier * deltaS;
			sMin = sMax - numberOfSteps * deltaS;
		}
		else {
			throw new IllegalArgumentException("Auxiliary knock-in model requested for non knock-in barrier type.");
		}

		validateBarrierIsInteriorGridNode(sMin, sMax, deltaS, numberOfSteps);

		final Grid knockInGrid = new UniformGrid(numberOfSteps, sMin, sMax);

		final SpaceTimeDiscretization knockInDiscretization = new SpaceTimeDiscretization(
				knockInGrid,
				timeDiscretization,
				thetaValue,
				new double[] { initialValue }
		);

		return originalModel.getCloneWithModifiedSpaceTimeDiscretization(knockInDiscretization);
	}

	private int getKnockInExtraStepsBeyondBarrier1D() {
		if(barrierType == BarrierType.DOWN_IN && callOrPutSign == CallOrPut.PUT) {
			return DOWN_IN_PUT_EXTRA_STEPS_1D;
		}
		if(barrierType == BarrierType.UP_IN && callOrPutSign == CallOrPut.CALL) {
			return UP_IN_CALL_EXTRA_STEPS_1D;
		}
		return DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS_1D;
	}

	private void validateBarrierIsInteriorGridNode(
			final double sMin,
			final double sMax,
			final double deltaS,
			final int numberOfSteps) {

		final double barrierIndexReal = (barrierValue - sMin) / deltaS;
		final long barrierIndexRounded = Math.round(barrierIndexReal);

		if(Math.abs(barrierIndexReal - barrierIndexRounded) > 1E-8) {
			throw new IllegalArgumentException("Auxiliary knock-in grid does not place the barrier on a grid node.");
		}

		if(barrierIndexRounded <= 0 || barrierIndexRounded >= numberOfSteps) {
			throw new IllegalArgumentException("Auxiliary knock-in grid must place the barrier on an interior node.");
		}

		if(barrierValue <= sMin || barrierValue >= sMax) {
			throw new IllegalArgumentException(
					"Auxiliary knock-in grid must contain the barrier strictly inside the domain.");
		}
	}

	private static double[] getColumn(final double[][] matrix, final int columnIndex) {
		final double[] column = new double[matrix.length];
		for(int i = 0; i < matrix.length; i++) {
			column[i] = matrix[i][columnIndex];
		}
		return column;
	}

	private static int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
	}

	private int getTotalNumberOfSpacePoints(final SpaceTimeDiscretization discretization) {
		final int dims = discretization.getNumberOfSpaceGrids();
		if(dims == 1) {
			return discretization.getSpaceGrid(0).getGrid().length;
		}
		else if(dims == 2) {
			return discretization.getSpaceGrid(0).getGrid().length * discretization.getSpaceGrid(1).getGrid().length;
		}
		else {
			throw new IllegalArgumentException("Only 1D and 2D grids are supported.");
		}
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

		if(callOrPutSign == CallOrPut.CALL) {
			if(barrierType == BarrierType.DOWN_OUT) {
				if(barrierValue <= strike) {
					return Math.max(assetValue - strike, 0.0);
				}
				return assetValue > barrierValue ? Math.max(assetValue - strike, 0.0) : rebate;
			}
			else if(barrierType == BarrierType.UP_OUT) {
				if(barrierValue <= strike) {
					return 0.0;
				}
				return assetValue < barrierValue ? Math.max(assetValue - strike, 0.0) : rebate;
			}
		}
		else {
			if(barrierType == BarrierType.DOWN_OUT) {
				if(barrierValue >= strike) {
					return 0.0;
				}
				return assetValue > barrierValue ? Math.max(strike - assetValue, 0.0) : rebate;
			}
			else if(barrierType == BarrierType.UP_OUT) {
				if(barrierValue >= strike) {
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

		if(barrierValue < lowerBoundary || barrierValue > upperBoundary) {
			throw new IllegalArgumentException(
					"The barrier must lie inside the first state-variable grid domain of the supplied model.");
		}
	}

	@Override
	public boolean isConstraintActive(final double time, final double... stateVariables) {
		if(!isOutOption()) {
			return false;
		}

		final double underlyingLevel = stateVariables[0];

		switch(barrierType) {
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
		if(!isOutOption()) {
			throw new IllegalStateException("Internal constrained value requested for a non out-option.");
		}

		return rebate;
	}

	private static CallOrPut mapCallOrPut(final double callOrPutSign) {
		if(callOrPutSign == 1.0) {
			return CallOrPut.CALL;
		}
		if(callOrPutSign == -1.0) {
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