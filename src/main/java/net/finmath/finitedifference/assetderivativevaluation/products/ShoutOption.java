package net.finmath.finitedifference.assetderivativevaluation.products;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.FDMThetaMethod1D;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.finitedifference.solvers.adi.AbstractADI2D;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;

/**
 * Implementation of a Shout Option. We adapt, to the present setting, the ideas of
 * H. Windcliff, K.R. Vetzal, P.A. Forsyth, A. Verma, T.F. Coleman, An object-oriented framework for 
 * valuing shout options on high-performance computer architectures. JEDC
 * <p>
 * Locked semantics:
 * </p>
 * <ul>
 *   <li>fixed maturity,</li>
 *   <li>finite total number of shouts,</li>
 *   <li>continuous shout right,</li>
 *   <li>standard reset rule K* = S,</li>
 *   <li>optional constant shout cash adjustment,</li>
 *   <li>no maturity extension,</li>
 *   <li>no yearly counter reset.</li>
 * </ul>
 *
 * <p>
 * The recursion is performed over planes of used shout count U and slices of fixed
 * strike K. Each slice is a standard PDE solve with a continuous floor coming from
 * the next plane.
 * </p>
 * 
 * @author Alessandro Gnoatto
 */
public class ShoutOption implements FiniteDifferenceProduct {

	private final String underlyingName;
	private final double maturity;
	private final double initialStrike;
	private final double[] strikeGrid;
	private final int maximumNumberOfShouts;
	private final CallOrPut callOrPut;
	private final double shoutCashAdjustment;

	public ShoutOption(
			final String underlyingName,
			final double maturity,
			final double initialStrike,
			final double[] strikeGrid,
			final int maximumNumberOfShouts,
			final CallOrPut callOrPut,
			final double shoutCashAdjustment) {

		if(callOrPut == null) {
			throw new IllegalArgumentException("callOrPut must not be null.");
		}
		if(maturity < 0.0) {
			throw new IllegalArgumentException("maturity must be non-negative.");
		}
		if(initialStrike <= 0.0) {
			throw new IllegalArgumentException("initialStrike must be positive.");
		}
		if(strikeGrid == null || strikeGrid.length < 2) {
			throw new IllegalArgumentException("strikeGrid must contain at least two points.");
		}
		if(maximumNumberOfShouts < 0) {
			throw new IllegalArgumentException("maximumNumberOfShouts must be non-negative.");
		}

		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.initialStrike = initialStrike;
		this.strikeGrid = strikeGrid.clone();
		this.maximumNumberOfShouts = maximumNumberOfShouts;
		this.callOrPut = callOrPut;
		this.shoutCashAdjustment = shoutCashAdjustment;

		validateStrikeGrid();
	}

	public ShoutOption(
			final double maturity,
			final double initialStrike,
			final double[] strikeGrid,
			final int maximumNumberOfShouts,
			final CallOrPut callOrPut) {
		this(
				null,
				maturity,
				initialStrike,
				strikeGrid,
				maximumNumberOfShouts,
				callOrPut,
				0.0
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
		validateModel(model);

		final int dimensions = model.getSpaceTimeDiscretization().getNumberOfSpaceGrids();

		if(maximumNumberOfShouts == 0) {
			return createVanillaSliceProduct(initialStrike).getValues(model);
		}

		if(dimensions == 1) {
			return getValues1D(model);
		}
		else if(dimensions == 2) {
			return getValues2D(model);
		}
		else {
			throw new IllegalArgumentException("ShoutOption currently supports only 1D and 2D models.");
		}
	}

	private double[][] getValues1D(final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization discretization = model.getSpaceTimeDiscretization();
		final double[] xGrid = discretization.getSpaceGrid(0).getGrid();

		validateInitialStrikeInsideGrid();
		validateStrikeGridCoversResetRange(xGrid);

		double[][][] nextPlane = solveLastPlane1D(model);

		for(int usedShouts = maximumNumberOfShouts - 1; usedShouts >= 0; usedShouts--) {
			final double[][][] currentPlane = new double[strikeGrid.length][][];

			for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
				final double strike = strikeGrid[strikeIndex];

				final FiniteDifferenceProduct sliceProduct = createVanillaSliceProduct(strike);
				final double[] terminalValues = buildPointwiseTerminalValues(xGrid, strike);

				final FDMThetaMethod1D solver = new FDMThetaMethod1D(
						model,
						sliceProduct,
						discretization,
						new EuropeanExercise(maturity)
				);

				final double[][][] nextPlaneForObstacle = nextPlane;

				final DoubleBinaryOperator continuousObstacle =
						(runningTime, currentSpot) ->
								interpolateNextPlaneAtResetStrike1D(
										nextPlaneForObstacle,
										discretization,
										xGrid,
										currentSpot,
										runningTime
								) + shoutCashAdjustment;

				currentPlane[strikeIndex] = solver.getValues(
						maturity,
						terminalValues,
						continuousObstacle
				);
			}

			nextPlane = currentPlane;
		}

		return interpolatePlaneAtInitialStrike1D(nextPlane, discretization, xGrid, initialStrike);
	}

	private double[][] getValues2D(final FiniteDifferenceEquityModel model) {
		final SpaceTimeDiscretization discretization = model.getSpaceTimeDiscretization();
		final double[] x0Grid = discretization.getSpaceGrid(0).getGrid();
		final double[] x1Grid = discretization.getSpaceGrid(1).getGrid();

		validateInitialStrikeInsideGrid();
		validateStrikeGridCoversResetRange(x0Grid);

		double[][][] nextPlane = solveLastPlane2D(model);

		for(int usedShouts = maximumNumberOfShouts - 1; usedShouts >= 0; usedShouts--) {
			final double[][][] currentPlane = new double[strikeGrid.length][][];

			for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
				final double strike = strikeGrid[strikeIndex];

				final FiniteDifferenceProduct sliceProduct = createVanillaSliceProduct(strike);
				final FDMSolver solver = FDMSolverFactory.createSolver(
						model,
						sliceProduct,
						new EuropeanExercise(maturity)
				);

				if(!(solver instanceof AbstractADI2D)) {
					throw new IllegalArgumentException("2D shout recursion requires an ADI-style solver.");
				}

				final double[][][] nextPlaneForObstacle = nextPlane;

				final AbstractADI2D.DoubleTernaryOperator continuousObstacle =
						(runningTime, currentSpot, secondState) ->
								interpolateNextPlaneAtResetStrike2D(
										nextPlaneForObstacle,
										discretization,
										x0Grid,
										x1Grid,
										currentSpot,
										secondState,
										runningTime
								) + shoutCashAdjustment;

				currentPlane[strikeIndex] = ((AbstractADI2D)solver).getValuesWithContinuousObstacle(
						maturity,
						(assetValue, secondState) -> terminalPayoff(assetValue, strike),
						continuousObstacle
				);
			}

			nextPlane = currentPlane;
		}

		return interpolatePlaneAtInitialStrike2D(nextPlane, discretization, x0Grid, x1Grid, initialStrike);
	}

	private double[][][] solveLastPlane1D(final FiniteDifferenceEquityModel model) {
		final double[][][] plane = new double[strikeGrid.length][][];

		for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
			plane[strikeIndex] = createVanillaSliceProduct(strikeGrid[strikeIndex]).getValues(model);
		}

		return plane;
	}

	private double[][][] solveLastPlane2D(final FiniteDifferenceEquityModel model) {
		final double[][][] plane = new double[strikeGrid.length][][];

		for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
			plane[strikeIndex] = createVanillaSliceProduct(strikeGrid[strikeIndex]).getValues(model);
		}

		return plane;
	}

	private FiniteDifferenceProduct createVanillaSliceProduct(final double strike) {
		return new EuropeanOption(underlyingName, maturity, strike, callOrPut);
	}

	private double[] buildPointwiseTerminalValues(final double[] xGrid, final double strike) {
		final double[] values = new double[xGrid.length];

		for(int i = 0; i < xGrid.length; i++) {
			values[i] = terminalPayoff(xGrid[i], strike);
		}

		return values;
	}

	private double terminalPayoff(final double assetValue, final double strike) {
		if(callOrPut == CallOrPut.CALL) {
			return Math.max(assetValue - strike, 0.0);
		}
		else {
			return Math.max(strike - assetValue, 0.0);
		}
	}

	private double interpolateNextPlaneAtResetStrike1D(
			final double[][][] nextPlane,
			final SpaceTimeDiscretization discretization,
			final double[] xGrid,
			final double currentSpot,
			final double runningTime) {

		final int timeIndex = getTimeIndexForRunningTime(discretization, runningTime);

		final double[] valuesAcrossStrikes = new double[strikeGrid.length];
		for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
			valuesAcrossStrikes[strikeIndex] = interpolate1DAtTime(
					nextPlane[strikeIndex],
					xGrid,
					timeIndex,
					currentSpot
			);
		}

		return interpolateLinearWithConstantExtrapolation(
				strikeGrid,
				valuesAcrossStrikes,
				resetStrike(currentSpot)
		);
	}

	private double interpolateNextPlaneAtResetStrike2D(
			final double[][][] nextPlane,
			final SpaceTimeDiscretization discretization,
			final double[] x0Grid,
			final double[] x1Grid,
			final double currentSpot,
			final double secondState,
			final double runningTime) {

		final int timeIndex = getTimeIndexForRunningTime(discretization, runningTime);

		final double[] valuesAcrossStrikes = new double[strikeGrid.length];
		for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
			valuesAcrossStrikes[strikeIndex] = interpolate2DAtTime(
					nextPlane[strikeIndex],
					x0Grid,
					x1Grid,
					timeIndex,
					currentSpot,
					secondState
			);
		}

		return interpolateLinearWithConstantExtrapolation(
				strikeGrid,
				valuesAcrossStrikes,
				resetStrike(currentSpot)
		);
	}

	private double[][] interpolatePlaneAtInitialStrike1D(
			final double[][][] plane,
			final SpaceTimeDiscretization discretization,
			final double[] xGrid,
			final double strike) {

		final int numberOfTimePoints = discretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final double[][] result = new double[xGrid.length][numberOfTimePoints];

		for(int timeIndex = 0; timeIndex < numberOfTimePoints; timeIndex++) {
			for(int spotIndex = 0; spotIndex < xGrid.length; spotIndex++) {
				final double[] valuesAcrossStrikes = new double[strikeGrid.length];

				for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
					valuesAcrossStrikes[strikeIndex] = plane[strikeIndex][spotIndex][timeIndex];
				}

				result[spotIndex][timeIndex] = interpolateLinearWithConstantExtrapolation(
						strikeGrid,
						valuesAcrossStrikes,
						strike
				);
			}
		}

		return result;
	}

	private double[][] interpolatePlaneAtInitialStrike2D(
			final double[][][] plane,
			final SpaceTimeDiscretization discretization,
			final double[] x0Grid,
			final double[] x1Grid,
			final double strike) {

		final int numberOfTimePoints = discretization.getTimeDiscretization().getNumberOfTimeSteps() + 1;
		final int numberOfStates = x0Grid.length * x1Grid.length;
		final double[][] result = new double[numberOfStates][numberOfTimePoints];

		for(int timeIndex = 0; timeIndex < numberOfTimePoints; timeIndex++) {
			for(int flatIndex = 0; flatIndex < numberOfStates; flatIndex++) {
				final double[] valuesAcrossStrikes = new double[strikeGrid.length];

				for(int strikeIndex = 0; strikeIndex < strikeGrid.length; strikeIndex++) {
					valuesAcrossStrikes[strikeIndex] = plane[strikeIndex][flatIndex][timeIndex];
				}

				result[flatIndex][timeIndex] = interpolateLinearWithConstantExtrapolation(
						strikeGrid,
						valuesAcrossStrikes,
						strike
				);
			}
		}

		return result;
	}

	private int getTimeIndexForRunningTime(
			final SpaceTimeDiscretization discretization,
			final double runningTime) {

		final double tau = maturity - runningTime;
		int timeIndex = discretization.getTimeDiscretization().getTimeIndexNearestLessOrEqual(tau);

		if(timeIndex < 0) {
			timeIndex = 0;
		}
		if(timeIndex >= discretization.getTimeDiscretization().getNumberOfTimeSteps() + 1) {
			timeIndex = discretization.getTimeDiscretization().getNumberOfTimeSteps();
		}

		return timeIndex;
	}

	private double interpolate1DAtTime(
			final double[][] surface,
			final double[] xGrid,
			final int timeIndex,
			final double xQuery) {

		final int i0 = getLowerBracketIndexWithConstantExtrapolation(xGrid, xQuery);
		final int i1 = Math.min(i0 + 1, xGrid.length - 1);

		final double x0 = xGrid[i0];
		final double x1 = xGrid[i1];

		final double v0 = surface[i0][timeIndex];
		final double v1 = surface[i1][timeIndex];

		if(i0 == i1 || Math.abs(x1 - x0) < 1E-14) {
			return v0;
		}

		final double w = (xQuery - x0) / (x1 - x0);
		return (1.0 - w) * v0 + w * v1;
	}

	private double interpolate2DAtTime(
			final double[][] surface,
			final double[] x0Grid,
			final double[] x1Grid,
			final int timeIndex,
			final double x0Query,
			final double x1Query) {

		final int i0 = getLowerBracketIndexWithConstantExtrapolation(x0Grid, x0Query);
		final int i1 = Math.min(i0 + 1, x0Grid.length - 1);

		final int j0 = getLowerBracketIndexWithConstantExtrapolation(x1Grid, x1Query);
		final int j1 = Math.min(j0 + 1, x1Grid.length - 1);

		final double x0L = x0Grid[i0];
		final double x0U = x0Grid[i1];
		final double x1L = x1Grid[j0];
		final double x1U = x1Grid[j1];

		final double f00 = surface[flatten(i0, j0, x0Grid.length)][timeIndex];
		final double f10 = surface[flatten(i1, j0, x0Grid.length)][timeIndex];
		final double f01 = surface[flatten(i0, j1, x0Grid.length)][timeIndex];
		final double f11 = surface[flatten(i1, j1, x0Grid.length)][timeIndex];

		final double wx;
		if(i0 == i1 || Math.abs(x0U - x0L) < 1E-14) {
			wx = 0.0;
		}
		else {
			wx = (x0Query - x0L) / (x0U - x0L);
		}

		final double wy;
		if(j0 == j1 || Math.abs(x1U - x1L) < 1E-14) {
			wy = 0.0;
		}
		else {
			wy = (x1Query - x1L) / (x1U - x1L);
		}

		return (1.0 - wx) * (1.0 - wy) * f00
				+ wx * (1.0 - wy) * f10
				+ (1.0 - wx) * wy * f01
				+ wx * wy * f11;
	}

	private double resetStrike(final double currentSpot) {
		return currentSpot;
	}

	private int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
	}

	private double interpolateLinearWithConstantExtrapolation(
			final double[] x,
			final double[] y,
			final double xQuery) {

		final int i0 = getLowerBracketIndexWithConstantExtrapolation(x, xQuery);
		final int i1 = Math.min(i0 + 1, x.length - 1);

		final double x0 = x[i0];
		final double x1 = x[i1];

		final double y0 = y[i0];
		final double y1 = y[i1];

		if(i0 == i1 || Math.abs(x1 - x0) < 1E-14) {
			return y0;
		}

		final double w = (xQuery - x0) / (x1 - x0);
		return (1.0 - w) * y0 + w * y1;
	}

	private int getLowerBracketIndexWithConstantExtrapolation(final double[] grid, final double x) {
		if(x <= grid[0]) {
			return 0;
		}
		if(x >= grid[grid.length - 1]) {
			return grid.length - 2;
		}

		int upperIndex = 1;
		while(upperIndex < grid.length && grid[upperIndex] < x) {
			upperIndex++;
		}
		return upperIndex - 1;
	}

	private void validateModel(final FiniteDifferenceEquityModel model) {
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}
	}

	private void validateStrikeGrid() {
		for(int i = 0; i < strikeGrid.length; i++) {
			if(strikeGrid[i] <= 0.0) {
				throw new IllegalArgumentException("All strike-grid values must be positive.");
			}
			if(i > 0 && strikeGrid[i] <= strikeGrid[i - 1]) {
				throw new IllegalArgumentException("strikeGrid must be strictly increasing.");
			}
		}
	}

	private void validateInitialStrikeInsideGrid() {
		if(initialStrike < strikeGrid[0] || initialStrike > strikeGrid[strikeGrid.length - 1]) {
			throw new IllegalArgumentException("initialStrike must lie inside strikeGrid.");
		}
	}

	private void validateStrikeGridCoversResetRange(final double[] xGrid) {
		final double effectiveLower = xGrid.length > 2 ? xGrid[1] : xGrid[0];
		final double effectiveUpper = xGrid.length > 2 ? xGrid[xGrid.length - 2] : xGrid[xGrid.length - 1];

		if(strikeGrid[0] > effectiveLower || strikeGrid[strikeGrid.length - 1] < effectiveUpper) {
			throw new IllegalArgumentException(
					"For v1 with K* = S, strikeGrid should cover the interior range of the first-state grid."
			);
		}
	}

	public String getUnderlyingName() {
		return underlyingName;
	}

	public double getMaturity() {
		return maturity;
	}

	public double getInitialStrike() {
		return initialStrike;
	}

	public double[] getStrikeGrid() {
		return strikeGrid.clone();
	}

	public int getMaximumNumberOfShouts() {
		return maximumNumberOfShouts;
	}

	public CallOrPut getCallOrPut() {
		return callOrPut;
	}

	public double getShoutCashAdjustment() {
		return shoutCashAdjustment;
	}

	@Override
	public String toString() {
		return "ShoutOption [maturity=" + maturity
				+ ", initialStrike=" + initialStrike
				+ ", maximumNumberOfShouts=" + maximumNumberOfShouts
				+ ", callOrPut=" + callOrPut
				+ ", strikeGrid=" + Arrays.toString(strikeGrid)
				+ "]";
	}
}